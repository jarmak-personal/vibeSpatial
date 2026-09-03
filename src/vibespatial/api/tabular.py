"""Public numeric reductions that preserve device-backed pandas Series."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.api._native_expression import NativeExpression
from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
from vibespatial.api.geo_base import (
    _attach_native_expression,
    _native_expression_from_public_series,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.runtime import ExecutionMode, get_requested_mode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import (
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)

from ._dense_count_kernels import (
    _DENSE_COUNT_UPDATE_KERNEL_NAMES,
    _DENSE_COUNT_UPDATE_KERNEL_SOURCE,
)

request_warmup(["radix_sort_i64_i32", "segmented_reduce_sum_f64"])

request_nvrtc_warmup(
    [
        (
            "dense-count-update",
            _DENSE_COUNT_UPDATE_KERNEL_SOURCE,
            _DENSE_COUNT_UPDATE_KERNEL_NAMES,
        )
    ]
)

_SAFE_DIVIDE_KERNEL = None


def _sync_tabular_hotpath() -> None:
    """Fence GPU work only when diagnostic stage timing is explicitly enabled."""
    if hotpath_timing_enabled():
        from vibespatial.cuda._runtime import get_cuda_runtime

        get_cuda_runtime().synchronize()


def _compact_streaming_topk_state(result: pd.DataFrame) -> pd.DataFrame:
    """Detach a bounded native winner set from batch ancestry on device."""
    from vibespatial.api._native_result_core import (
        NativeAttributeTable,
        NativeGeometryColumn,
        NativeTabularResult,
    )
    from vibespatial.api._native_rowset import NativeIndexPlan
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(result)
    if state is None:
        return result
    composition_parts = (
        len(state.geometry.composition.parts)
        if state.geometry.composition is not None
        else 1
    )
    _sync_tabular_hotpath()
    with hotpath_stage(
        "tabular.streaming_topk.compact",
        category="emit",
        metadata={
            "output_rows": state.row_count,
            "source_geometry_parts": composition_parts,
        },
    ) as stage_metadata:
        geometry = state.geometry.physicalize_singular_device_rows()
        if geometry is None:
            return result
        secondary_geometry = []
        for column in state.secondary_geometry:
            compact = column.geometry.physicalize_singular_device_rows()
            if compact is None:
                return result
            secondary_geometry.append(NativeGeometryColumn(column.name, compact))
        attributes = NativeAttributeTable.from_value(
            state.attributes
        )._physicalize_device_row_view()
        index = pd.RangeIndex(state.row_count)
        compact_result = NativeTabularResult(
            attributes=attributes.with_index(index),
            geometry=geometry,
            geometry_name=state.geometry_name,
            column_order=state.column_order,
            attrs=state.attrs,
            secondary_geometry=tuple(secondary_geometry),
            provenance=None,
            geometry_metadata=None,
            index_plan=NativeIndexPlan.from_index(index),
        ).to_geodataframe()
        attach_work_amplification(
            stage_metadata,
            operation="streaming_topk_compact",
            metric_family="tabular",
            sums={"compacted_rows": state.row_count},
            maxima={
                "output_rows": state.row_count,
                "source_geometry_parts": composition_parts,
            },
            physical_shape="bounded winner rows -> standalone native carrier",
            consumer_kind="next streaming top-k batch",
        )
        _sync_tabular_hotpath()
    return compact_result


def _safe_divide_device(left, right, fill_value, *, dtype):
    """Launch one cached elementwise expression without a validity buffer."""
    import cupy as cp

    global _SAFE_DIVIDE_KERNEL
    if _SAFE_DIVIDE_KERNEL is None:
        _SAFE_DIVIDE_KERNEL = cp.ElementwiseKernel(
            "T numerator, T denominator, T fill_value",
            "T output",
            "output = denominator != (T)0 "
            "? numerator / denominator : fill_value;",
            "vibespatial_safe_numeric_divide",
        )
    return _SAFE_DIVIDE_KERNEL(
        left.astype(dtype, copy=False),
        right.astype(dtype, copy=False),
        dtype.type(fill_value),
    )


def _streaming_topk(
    frame: pd.DataFrame,
    n: int,
    columns,
    *,
    largest: bool = True,
    keep: str = "first",
    out: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge one input batch into a bounded exact top-k frame.

    Device-backed frames retain their native geometry, attributes, and row
    references.  The accumulator owns at most one batch-local selection plus
    the previous result.  Streaming currently admits ``keep='first'`` because
    pandas' other tie modes are not compositionally bounded across batches;
    use one-shot ``nlargest``/``nsmallest`` for those modes.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("streaming_topk frame must be a pandas DataFrame")
    if out is not None and not isinstance(out, pd.DataFrame):
        raise TypeError("streaming_topk out must be a pandas DataFrame")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("streaming_topk n must be an integer")
    n = int(n)
    if n < 0:
        raise ValueError("streaming_topk n must be non-negative")
    if keep not in {"first", "last", "all"}:
        raise ValueError("streaming_topk keep must be 'first', 'last', or 'all'")
    if keep != "first":
        raise NotImplementedError(
            "streaming_topk currently supports only keep='first'; "
            "use one-shot nlargest/nsmallest for keep='last' or keep='all'"
        )
    sort_columns = [columns] if isinstance(columns, (str, bytes)) else list(columns)
    if not sort_columns:
        raise ValueError("streaming_topk requires at least one ordering column")
    missing = [column for column in sort_columns if column not in frame.columns]
    if missing:
        raise KeyError(missing)
    if out is not None:
        if not frame.columns.equals(out.columns):
            raise ValueError("streaming_topk out columns must match frame columns")
        missing = [column for column in sort_columns if column not in out.columns]
        if missing:
            raise KeyError(missing)

    input_rows = len(frame)
    operation = "nlargest" if largest else "nsmallest"
    selector = getattr(frame, operation)
    timing = hotpath_timing_enabled()
    _sync_tabular_hotpath()
    with hotpath_stage(
        "tabular.streaming_topk.batch_select",
        category="sort",
        metadata={
            "input_rows": input_rows,
            "requested_rows": n,
            "ordering_keys": len(sort_columns),
        },
    ) as stage_metadata:
        batch = selector(n, sort_columns, keep=keep)
        attach_work_amplification(
            stage_metadata,
            operation="streaming_topk_batch_select",
            metric_family="tabular",
            sums={
                "input_rows": input_rows,
                "selected_rows": len(batch),
                "diagnostic_synchronizations": 2 if timing else 0,
            },
            maxima={
                "input_rows": input_rows,
                "selected_rows": len(batch),
                "requested_rows": n,
            },
            physical_shape="batch rowset -> bounded ordered NativeRowSet",
            consumer_kind="streaming top-k accumulator",
            semantic_contract={"keep": keep, "largest": bool(largest)},
        )
        _sync_tabular_hotpath()

    if out is None or out.empty:
        result = batch
        merged_rows = len(batch)
    elif batch.empty:
        # ``out`` may have been created with a larger prior bound.  Reapply
        # the requested bound even though this batch contributes no rows.
        result = getattr(out, operation)(n, sort_columns, keep=keep)
        merged_rows = len(out)
    else:
        merged_rows = len(out) + len(batch)
        merge_inputs = (out, batch)
        _sync_tabular_hotpath()
        with hotpath_stage(
            "tabular.streaming_topk.merge",
            category="sort",
            metadata={
                "retained_rows": len(out),
                "batch_rows": len(batch),
                "requested_rows": n,
            },
        ) as stage_metadata:
            from vibespatial.api._native_state import get_native_state

            retained_state = get_native_state(out)
            batch_state = get_native_state(batch)
            if (
                retained_state is not None
                and batch_state is not None
                and retained_state.geometry_name == batch_state.geometry_name
                and retained_state.column_order == batch_state.column_order
            ):
                from vibespatial.api._native_result_core import NativeTabularResult
                from vibespatial.api._native_results import (
                    _concat_native_tabular_results,
                )
                from vibespatial.api._native_rowset import NativeIndexPlan

                def _result_with_range_index(state):
                    index = pd.RangeIndex(state.row_count)
                    return NativeTabularResult(
                        attributes=state.attributes.with_index(index),
                        geometry=state.geometry,
                        geometry_name=state.geometry_name,
                        column_order=state.column_order,
                        attrs=state.attrs,
                        secondary_geometry=state.secondary_geometry,
                        provenance=state.provenance,
                        geometry_metadata=state.geometry_metadata_cache,
                        index_plan=NativeIndexPlan.from_index(index),
                    )

                merge_states = (
                    retained_state,
                    batch_state,
                )
                merged_native = _concat_native_tabular_results(
                    [
                        _result_with_range_index(state) for state in merge_states
                    ],
                    geometry_name=retained_state.geometry_name,
                    crs=getattr(out, "crs", None),
                    ignore_index=True,
                )
                merged = merged_native.to_geodataframe()
            else:
                merged = pd.concat(merge_inputs, ignore_index=True)
            result = getattr(merged, operation)(n, sort_columns, keep=keep)
            attach_work_amplification(
                stage_metadata,
                operation="streaming_topk_merge",
                metric_family="tabular",
                sums={
                    "merged_rows": merged_rows,
                    "output_rows": len(result),
                    "diagnostic_synchronizations": 2 if timing else 0,
                },
                maxima={
                    "retained_rows": len(out),
                    "batch_rows": len(batch),
                    "merged_rows": merged_rows,
                    "output_rows": len(result),
                },
                physical_shape="bounded NativeFrameState merge",
                consumer_kind="persistent streaming top-k state",
                semantic_contract={"keep": keep, "largest": bool(largest)},
            )
            _sync_tabular_hotpath()

    from vibespatial.api._native_state import get_native_state

    native = get_native_state(result) is not None
    if native:
        result = _compact_streaming_topk_state(result)
    record_dispatch_event(
        surface="vibespatial.api.tabular._streaming_topk",
        operation="streaming_topk",
        implementation=(
            "native_bounded_streaming_topk" if native else "pandas_streaming_topk"
        ),
        reason=(
            "batch-local winners merged into bounded device-resident state"
            if native
            else "host frame winners merged with pandas top-k semantics"
        ),
        detail=(
            f"input_rows={input_rows}, merged_rows={merged_rows}, "
            f"output_rows={len(result)}, n={n}, keep={keep!r}"
        ),
        requested=get_requested_mode(),
        selected=ExecutionMode.GPU if native else ExecutionMode.CPU,
    )
    return result


def _numeric_divide(
    numerator,
    denominator,
    *,
    fill_value: float = np.nan,
    name: Any = None,
) -> pd.Series:
    """Divide positionally aligned vectors with zero fill in one native stage."""
    if (
        isinstance(numerator, pd.Series)
        and isinstance(denominator, pd.Series)
        and not numerator.index.equals(denominator.index)
    ):
        raise ValueError("numeric divide inputs must have identical indexes")
    left_expression = _native_expression_from_public_series(numerator)
    right_expression = _native_expression_from_public_series(denominator)
    requested = get_requested_mode()
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and any(
            expression is not None and expression.is_device
            for expression in (left_expression, right_expression)
        )
    )
    output_index = (
        numerator.index.copy()
        if isinstance(numerator, pd.Series)
        else denominator.index.copy()
        if isinstance(denominator, pd.Series)
        else pd.RangeIndex(len(numerator))
    )
    output_name = getattr(numerator, "name", None) if name is None else name
    if use_device:
        import cupy as cp

        left = cp.asarray(
            left_expression.values
            if left_expression is not None
            else np.asarray(numerator)
        )
        right = cp.asarray(
            right_expression.values
            if right_expression is not None
            else np.asarray(denominator)
        )
        if left.ndim != 1 or right.ndim != 1 or int(left.size) != int(right.size):
            raise ValueError("numeric_divide inputs must be equal-length vectors")
        dtype = np.result_type(left.dtype, right.dtype, np.float64)
        with hotpath_stage(
            "tabular.expression.safe_divide",
            category="other",
            metadata={"input_rows": int(left.size)},
        ) as stage_metadata:
            result = _safe_divide_device(
                left,
                right,
                fill_value,
                dtype=np.dtype(dtype),
            )
            attach_work_amplification(
                stage_metadata,
                operation="numeric_divide",
                metric_family="tabular",
                sums={"expression_rows": int(left.size)},
                maxima={"input_rows": int(left.size), "output_rows": int(left.size)},
                physical_shape="aligned fused elementwise expression",
                consumer_kind="NativeExpression",
            )
        expression = NativeExpression(
            operation="numeric_divide",
            values=result,
            source_token=None,
            source_row_count=int(result.size),
            dtype=str(result.dtype),
            precision="fp64-derived-expression",
        )
        record_dispatch_event(
            surface="vibespatial.api.tabular._numeric_divide",
            operation="numeric_divide",
            implementation="cupy_safe_divide",
            reason="aligned division and zero-denominator fill remain device-native",
            detail=f"input_rows={int(result.size)}, dtype={result.dtype}",
            requested=requested,
            selected=ExecutionMode.GPU,
        )
        return _public_native_series(
            expression,
            index=output_index,
            name=output_name,
            operation="numeric_divide_to_public_array",
        )

    left = np.asarray(numerator)
    right = np.asarray(denominator)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise ValueError("numeric_divide inputs must be equal-length vectors")
    dtype = np.result_type(left.dtype, right.dtype, np.float64)
    result = np.full(left.size, fill_value, dtype=dtype)
    np.divide(left, right, out=result, where=right != 0)
    record_dispatch_event(
        surface="vibespatial.api.tabular._numeric_divide",
        operation="numeric_divide",
        implementation="numpy_safe_divide",
        reason="host aligned division uses explicit zero-denominator fill",
        detail=f"input_rows={int(result.size)}, dtype={result.dtype}",
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return pd.Series(result, index=output_index, name=output_name)


def _count_dtype(dtype: Any) -> np.dtype:
    normalized = np.dtype(dtype)
    if normalized not in {np.dtype(np.uint32), np.dtype(np.uint64)}:
        raise TypeError("dense_count dtype must be uint32 or uint64")
    return normalized


def _dense_count_update_kernels():
    from vibespatial.cuda._runtime import compile_kernel_group

    return compile_kernel_group(
        "dense-count-update",
        _DENSE_COUNT_UPDATE_KERNEL_SOURCE,
        _DENSE_COUNT_UPDATE_KERNEL_NAMES,
    )


def _validate_dense_count_out(
    out: pd.Series,
    *,
    group_count: int,
    output_dtype: np.dtype,
    use_device: bool,
) -> NativeExpression | None:
    if not isinstance(out, pd.Series):
        raise TypeError("dense_count out must be a pandas Series")
    if len(out) != group_count:
        raise ValueError("dense_count out length must match size")
    if not out.index.equals(pd.RangeIndex(group_count)):
        raise ValueError("dense_count out index must be RangeIndex(size)")

    expression = _native_expression_from_public_series(out)
    if use_device:
        if expression is None or not expression.is_device:
            raise TypeError("dense_count device execution requires a device-backed out")
        actual_dtype = np.dtype(expression.values.dtype)
    else:
        if expression is not None and expression.is_device:
            raise TypeError("dense_count host execution requires a host-backed out")
        try:
            actual_dtype = np.dtype(out.dtype)
        except TypeError as exc:
            raise TypeError(
                "dense_count out dtype must exactly match the requested dtype"
            ) from exc
    if actual_dtype != output_dtype:
        raise TypeError("dense_count out dtype must exactly match the requested dtype")
    return expression


def _dense_count_host_update(
    host_codes: np.ndarray,
    out: pd.Series,
    *,
    output_dtype: np.dtype,
) -> None:
    if not host_codes.size:
        return
    unique_codes, increments = np.unique(host_codes, return_counts=True)
    counts = out.to_numpy(copy=False)
    current = counts[unique_codes]
    headroom = np.asarray(np.iinfo(output_dtype).max, dtype=output_dtype) - current
    if np.any(increments.astype(np.uint64, copy=False) > headroom):
        raise OverflowError("dense_count update exceeds the requested count dtype")
    out.iloc[unique_codes] = current + increments.astype(output_dtype, copy=False)


def _dense_count_device_update(
    d_indices,
    d_counts,
    *,
    output_dtype: np.dtype,
) -> tuple[Any, ...]:
    """Transactionally add one code batch without domain-sized scratch."""
    import cupy as cp

    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I64,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )

    row_count = int(d_indices.size)
    if row_count == 0:
        return ()
    code_stride_bytes = int(d_indices.strides[0])
    count_stride_bytes = int(d_counts.strides[0])
    code_itemsize = int(d_indices.dtype.itemsize)
    count_itemsize = int(d_counts.dtype.itemsize)
    if code_stride_bytes % code_itemsize or count_stride_bytes % count_itemsize:
        raise ValueError("dense_count device vectors require element-aligned strides")
    code_stride = code_stride_bytes // code_itemsize
    count_stride = count_stride_bytes // count_itemsize
    if int(d_counts.size) > 1 and count_stride == 0:
        raise ValueError("dense_count out must not alias accumulator elements")
    if cp.shares_memory(d_indices, d_counts):
        raise ValueError("dense_count codes and out must not share device storage")
    runtime = get_cuda_runtime()
    suffix = "u32" if output_dtype == np.dtype(np.uint32) else "u64"
    kernels = _dense_count_update_kernels()

    def launch_config(kernel):
        grid, block = runtime.launch_config(kernel, row_count)
        multiprocessors = int(cp.cuda.Device().attributes["MultiProcessorCount"])
        return (min(grid[0], multiprocessors * 4), 1, 1), block

    risk = cp.zeros(1, dtype=cp.uint32)
    preflight = kernels[f"dense_count_preflight_{suffix}"]
    grid, block = launch_config(preflight)
    runtime.launch(
        preflight,
        grid=grid,
        block=block,
        params=(
            (
                runtime.pointer(d_indices),
                code_stride,
                runtime.pointer(d_counts),
                count_stride,
                row_count,
                runtime.pointer(risk),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    update = kernels[f"dense_count_update_{suffix}"]
    grid, block = launch_config(update)
    runtime.launch(
        update,
        grid=grid,
        block=block,
        params=(
            (
                runtime.pointer(d_indices),
                code_stride,
                runtime.pointer(d_counts),
                count_stride,
                row_count,
                runtime.pointer(risk),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    exact_inputs: tuple[Any, ...] = ()
    if _device_any(
        risk,
        reason="public dense-count transactional overflow preflight fence",
    ):
        exact_required_bytes = max(1, row_count * 112)
        exact_admission = runtime.admit_device_memory(
            stage="tabular-dense-count-update-exact-overflow",
            required_bytes=exact_required_bytes,
            requested_units=row_count,
        )
        if not exact_admission.admitted:
            raise MemoryError(
                "dense_count exact overflow validation requires "
                f"{exact_required_bytes} device bytes with "
                f"{exact_admission.remaining_bytes} available"
            )
        unique_codes, increments = cp.unique(d_indices, return_counts=True)
        current = d_counts[unique_codes]
        headroom = cp.asarray(np.iinfo(output_dtype).max, dtype=output_dtype) - current
        unsafe = increments.astype(cp.uint64, copy=False) > headroom
        if _device_any(
            cp.any(unsafe),
            reason="public dense-count exact overflow validation fence",
        ):
            raise OverflowError("dense_count update exceeds the requested count dtype")
        cp.add.at(
            d_counts,
            unique_codes,
            increments.astype(output_dtype, copy=False),
        )
        exact_inputs = (unique_codes, increments, current, headroom, unsafe)
    return (risk, *exact_inputs)


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


def _dense_grouped_device_required_bytes(
    *,
    row_count: int,
    group_count: int,
    code_dtype: np.dtype,
    value_dtypes: tuple[np.dtype, ...],
    output_dtype: np.dtype,
    output_bytes: int,
    upload_bytes: int,
) -> int:
    """Conservatively admit dense count or stable segmented-reduction workspace."""
    rows = int(row_count)
    groups = int(group_count)
    index_conversion = 0 if code_dtype == np.dtype(np.int64) else rows * 8
    value_conversion = sum(
        0 if dtype == output_dtype else rows * int(output_dtype.itemsize)
        for dtype in value_dtypes
    )
    if not value_dtypes:
        # CuPy bincount owns a row-shaped primitive workspace plus an int64
        # dense output before the requested count dtype is exposed. Both terms
        # matter independently: many rows with few groups and few rows with a
        # very wide fixed domain are valid public shapes.
        bincount_workspace = max(1 << 20, rows * 96)
        int64_count_output = groups * 8
        return max(
            1,
            upload_bytes
            + index_conversion
            + rows
            + bincount_workspace
            + int64_count_output
            + output_bytes,
        )

    # Stable code ordering owns one int64 permutation. The radix-sort and CCCL
    # storage estimates deliberately exceed their observed allocator peaks;
    # admission is a safety bound, not an allocation target. Only one gathered
    # value column and NaN mask are live at a time.
    stable_order = rows * 8
    sort_workspace = rows * 64
    segmented_workspace = max(1 << 20, rows * 8)
    value_workspace = rows * (
        max(int(dtype.itemsize) for dtype in value_dtypes) + 1
    )
    group_workspace = groups * 32
    return max(
        1,
        upload_bytes
        + index_conversion
        + value_conversion
        + rows
        + stable_order
        + sort_workspace
        + segmented_workspace
        + value_workspace
        + group_workspace
        + output_bytes,
    )


def _device_dense_grouped_plan(d_indices, *, group_count: int):
    """Return stable ordering and dense segment offsets for deterministic sums."""
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    counts = cp.bincount(d_indices, minlength=int(group_count))[
        : int(group_count)
    ].astype(cp.int64, copy=False)
    if int(d_indices.size) > int(np.iinfo(np.int32).max):
        # The precompiled CCCL value carrier is int32. Preserve exact source
        # ordering for exceptional wider rowsets instead of narrowing indexes.
        order = cp.argsort(d_indices, kind="stable")
        ends = cp.cumsum(counts, dtype=cp.int64)
        return counts, order, ends - counts, ends
    source_order = cp.arange(int(d_indices.size), dtype=cp.int32)
    order = sort_pairs(
        d_indices,
        source_order,
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values
    ends = cp.cumsum(counts, dtype=cp.int64)
    starts = ends - counts
    return counts, order, starts, ends


def _device_segmented_sum(
    d_values,
    *,
    order,
    starts,
    ends,
    output_dtype: np.dtype,
):
    """Stable grouped sum with explicit input order and no contended fp atomics."""
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import segmented_reduce_sum

    sorted_values = d_values[order]
    if sorted_values.dtype.kind == "f":
        cp.copyto(
            sorted_values,
            cp.zeros((), dtype=sorted_values.dtype),
            where=cp.isnan(sorted_values),
        )
    sorted_values = sorted_values.astype(output_dtype, copy=False)
    return segmented_reduce_sum(
        sorted_values,
        starts,
        ends,
        num_segments=int(starts.size),
        synchronize=False,
    ).values


def dense_count(
    codes,
    *,
    size: int,
    dtype: Any = np.uint32,
    name: Any = None,
    out: pd.Series | None = None,
) -> pd.Series:
    """Count non-negative integer codes into a fixed-size dense Series.

    This is the fixed-domain counterpart of ``numpy.bincount``. ``codes`` must
    be one-dimensional and every value must be in ``[0, size)``. Device-backed
    vibeSpatial expressions produce a device-backed real pandas Series. When
    ``out`` is supplied, the batch is added transactionally to that same Series
    without allocating another fixed-domain vector. The accumulator must have
    exactly the requested unsigned dtype, length, and ``RangeIndex(size)``, and
    its residency must match the selected execution mode. Overflow raises
    before any counters from the batch are changed. Device-backed ``codes``
    and ``out`` must not overlap because the update reads codes while mutating
    counters.
    """
    group_count = int(size)
    if group_count < 0:
        raise ValueError("dense_count size must be non-negative")
    output_dtype = _count_dtype(dtype)
    expression = _native_expression_from_public_series(codes)
    if out is not None and not isinstance(out, pd.Series):
        raise TypeError("dense_count out must be a pandas Series")
    out_expression = (
        None if out is None else _native_expression_from_public_series(out)
    )
    requested = get_requested_mode()
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and any(
            candidate is not None and candidate.is_device
            for candidate in (expression, out_expression)
        )
    )
    if out is not None:
        out_expression = _validate_dense_count_out(
            out,
            group_count=group_count,
            output_dtype=output_dtype,
            use_device=use_device,
        )
    if use_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        source_codes = expression.values if expression is not None else np.asarray(codes)
        source_is_device = expression is not None and expression.is_device
        if source_is_device:
            source_shape = source_codes.shape
            source_dtype = np.dtype(source_codes.dtype)
            source_size = int(source_codes.size)
            upload_bytes = 0
        else:
            source_codes = np.asarray(source_codes)
            source_shape = source_codes.shape
            source_dtype = source_codes.dtype
            source_size = int(source_codes.size)
            upload_bytes = int(source_codes.nbytes)
        if len(source_shape) != 1 or source_dtype.kind not in {"i", "u"}:
            raise TypeError("dense_count codes must be one-dimensional integers")
        if source_size > int(np.iinfo(output_dtype).max):
            raise OverflowError("dense_count input exceeds the requested count dtype")
        output_bytes = (
            0 if out is not None else group_count * int(output_dtype.itemsize)
        )
        index_bytes = (
            0 if source_dtype == np.dtype(np.int64) else source_size * 8
        )
        validation_bytes = source_size * 3
        update_state_bytes = 2 * int(np.dtype(np.uint32).itemsize) if out is not None else 0
        required_bytes = max(
            1,
            output_bytes
            + upload_bytes
            + index_bytes
            + validation_bytes
            + update_state_bytes,
        )
        runtime = get_cuda_runtime()
        admission = runtime.admit_device_memory(
            stage=(
                "tabular-dense-count-update"
                if out is not None
                else "tabular-dense-count"
            ),
            required_bytes=required_bytes,
            requested_units=group_count,
        )
        if not admission.admitted:
            raise MemoryError(
                "dense_count requires "
                f"{required_bytes} device bytes with "
                f"{admission.remaining_bytes} available"
            )
        d_codes = cp.asarray(source_codes)
        invalid = (d_codes < 0) | (d_codes >= group_count)
        if int(d_codes.size) and _device_any(
            cp.any(invalid),
            reason="public dense-count code-domain validation fence",
        ):
            raise ValueError("dense_count codes must be in [0, size)")
        del invalid
        d_indices = d_codes.astype(cp.int64, copy=False)
        retained_inputs: tuple[Any, ...] = (d_codes, d_indices)
        if out is None:
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
            result = _public_native_series(
                result_expression,
                index=pd.RangeIndex(group_count),
                name=name,
                operation="dense_count_to_public_array",
            )
        else:
            d_counts = cp.asarray(out_expression.values)
            _sync_tabular_hotpath()
            with hotpath_stage(
                "tabular.dense_count.update",
                category="other",
                metadata={
                    "input_rows": int(d_indices.size),
                    "group_domain": group_count,
                    "persistent_output_bytes": int(d_counts.nbytes),
                    "output_allocation_bytes": 0,
                    "dtype": output_dtype.name,
                },
            ) as stage_metadata:
                update_inputs = _dense_count_device_update(
                    d_indices,
                    d_counts,
                    output_dtype=output_dtype,
                )
                attach_work_amplification(
                    stage_metadata,
                    operation="dense_count_update",
                    metric_family="tabular",
                    sums={"updated_input_rows": int(d_indices.size)},
                    maxima={
                        "batch_rows": int(d_indices.size),
                        "group_domain": group_count,
                        "persistent_output_bytes": int(d_counts.nbytes),
                        "output_allocation_bytes": 0,
                    },
                    physical_shape="persistent fixed-domain integer scatter accumulator",
                    consumer_kind="next streamed dense-count batch",
                    semantic_contract={
                        "transactional_overflow": True,
                        "output_identity_reused": True,
                    },
                )
                _sync_tabular_hotpath()
            retained_inputs += update_inputs
            result = out
        if retained_inputs and int(d_codes.size):
            from vibespatial.cuda._runtime import get_cuda_completion_retainer

            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                retained_inputs,
                lambda _arrays: None,
            )
        record_dispatch_event(
            surface="vibespatial.api.dense_count",
            operation="dense_count",
            implementation=(
                "cuda_dense_scatter_count_update"
                if out is not None
                else "cupy_dense_scatter_count"
            ),
            reason=(
                "integer codes updated a persistent device count vector in place"
                if out is not None
                else "fixed-domain integer codes reduced into a dense device count vector"
            ),
            detail=(
                f"input_rows={int(d_codes.size)}, groups={group_count}, "
                f"dtype={output_dtype.name}, mode={'update' if out is not None else 'create'}, "
                f"output_allocation_bytes={output_bytes}"
            ),
            requested=requested,
            selected=ExecutionMode.GPU,
        )
        return result

    host_codes = np.asarray(codes)
    if host_codes.ndim != 1 or host_codes.dtype.kind not in {"i", "u"}:
        raise TypeError("dense_count codes must be one-dimensional integers")
    if host_codes.size > int(np.iinfo(output_dtype).max):
        raise OverflowError("dense_count input exceeds the requested count dtype")
    if np.any((host_codes < 0) | (host_codes >= group_count)):
        raise ValueError("dense_count codes must be in [0, size)")
    if out is None:
        counts = np.zeros(group_count, dtype=output_dtype)
        if host_codes.size:
            np.add.at(counts, host_codes.astype(np.int64, copy=False), 1)
        result = pd.Series(counts, index=pd.RangeIndex(group_count), name=name)
    else:
        _dense_count_host_update(
            host_codes.astype(np.int64, copy=False),
            out,
            output_dtype=output_dtype,
        )
        result = out
    record_dispatch_event(
        surface="vibespatial.api.dense_count",
        operation="dense_count",
        implementation=(
            "numpy_dense_scatter_count_update"
            if out is not None
            else "numpy_dense_scatter_count"
        ),
        reason=(
            "integer codes updated a persistent host count vector in place"
            if out is not None
            else "host integer codes reduced into a fixed-domain dense count vector"
        ),
        detail=(
            f"input_rows={int(host_codes.size)}, groups={group_count}, "
            f"dtype={output_dtype.name}, mode={'update' if out is not None else 'create'}, "
            f"output_allocation_bytes={0 if out is not None else counts.nbytes}"
        ),
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return result


def dense_sum(
    codes,
    values,
    *,
    size: int,
    dtype: Any = np.float64,
    name: Any = None,
) -> pd.Series:
    """Sum numeric values by fixed-domain non-negative integer codes.

    This is the weighted counterpart of :func:`dense_count`. Missing floating
    values are skipped, matching pandas grouped-sum behavior, and empty groups
    contain zero. Device-backed expressions remain a device-backed real pandas
    Series so bounded grouped states can be merged before terminal export.
    """
    group_count = int(size)
    if group_count < 0:
        raise ValueError("dense_sum size must be non-negative")
    output_dtype = np.dtype(dtype)
    if output_dtype.kind not in {"f", "i", "u"}:
        raise TypeError("dense_sum dtype must be a real numeric dtype")
    code_expression = _native_expression_from_public_series(codes)
    value_expression = _native_expression_from_public_series(values)
    requested = get_requested_mode()
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and any(
            expression is not None and expression.is_device
            for expression in (code_expression, value_expression)
        )
    )
    if use_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        source_codes = (
            code_expression.values
            if code_expression is not None
            else np.asarray(codes)
        )
        source_values = (
            value_expression.values
            if value_expression is not None
            else np.asarray(values)
        )
        if source_codes.ndim != 1 or source_codes.dtype.kind not in {"i", "u"}:
            raise TypeError("dense_sum codes must be one-dimensional integers")
        if source_values.ndim != 1 or source_values.dtype.kind not in {"f", "i", "u"}:
            raise TypeError("dense_sum values must be one-dimensional real numeric values")
        if int(source_codes.size) != int(source_values.size):
            raise ValueError("dense_sum codes and values must have the same length")
        output_bytes = group_count * int(output_dtype.itemsize)
        upload_bytes = sum(
            int(source.size) * int(source.dtype.itemsize)
            for source, expression in (
                (source_codes, code_expression),
                (source_values, value_expression),
            )
            if expression is None or not expression.is_device
        )
        required_bytes = _dense_grouped_device_required_bytes(
            row_count=int(source_codes.size),
            group_count=group_count,
            code_dtype=np.dtype(source_codes.dtype),
            value_dtypes=(np.dtype(source_values.dtype),),
            output_dtype=output_dtype,
            output_bytes=output_bytes,
            upload_bytes=upload_bytes,
        )
        runtime = get_cuda_runtime()
        admission = runtime.admit_device_memory(
            stage="tabular-dense-sum",
            required_bytes=required_bytes,
            requested_units=group_count,
        )
        if not admission.admitted:
            raise MemoryError(
                "dense_sum requires "
                f"{required_bytes} device bytes with "
                f"{admission.remaining_bytes} available"
            )
        d_codes = cp.asarray(source_codes)
        d_values = cp.asarray(source_values)
        invalid = (d_codes < 0) | (d_codes >= group_count)
        if int(d_codes.size) and _device_any(
            cp.any(invalid),
            reason="public dense-sum code-domain validation fence",
        ):
            raise ValueError("dense_sum codes must be in [0, size)")
        del invalid
        d_indices = d_codes.astype(cp.int64, copy=False)
        _, order, starts, ends = _device_dense_grouped_plan(
            d_indices,
            group_count=group_count,
        )
        d_sums = _device_segmented_sum(
            d_values,
            order=order,
            starts=starts,
            ends=ends,
            output_dtype=output_dtype,
        )
        result_expression = NativeExpression(
            operation="dense_sum",
            values=d_sums,
            source_token=None,
            source_row_count=group_count,
            dtype=str(output_dtype),
            precision="fp64-grouped-sum" if output_dtype == np.dtype(np.float64) else "source",
        )
        record_dispatch_event(
            surface="vibespatial.api.dense_sum",
            operation="dense_sum",
            implementation="cccl_stable_segmented_sum",
            reason="stable code ordering feeds deterministic fixed-domain segmented sums",
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
            operation="dense_sum_to_public_array",
        )

    host_codes = np.asarray(codes)
    host_values = np.asarray(values)
    if host_codes.ndim != 1 or host_codes.dtype.kind not in {"i", "u"}:
        raise TypeError("dense_sum codes must be one-dimensional integers")
    if host_values.ndim != 1 or host_values.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("dense_sum values must be one-dimensional real numeric values")
    if host_codes.size != host_values.size:
        raise ValueError("dense_sum codes and values must have the same length")
    if np.any((host_codes < 0) | (host_codes >= group_count)):
        raise ValueError("dense_sum codes must be in [0, size)")
    if host_values.dtype.kind == "f":
        host_values = np.where(
            np.isnan(host_values),
            np.zeros((), dtype=host_values.dtype),
            host_values,
        )
    weights = host_values.astype(output_dtype, copy=False)
    sums = np.zeros(group_count, dtype=output_dtype)
    if host_codes.size:
        np.add.at(sums, host_codes.astype(np.int64, copy=False), weights)
    record_dispatch_event(
        surface="vibespatial.api.dense_sum",
        operation="dense_sum",
        implementation="numpy_dense_scatter_sum",
        reason="host integer codes reduced numeric values into a fixed-domain dense sum vector",
        detail=(
            f"input_rows={int(host_codes.size)}, groups={group_count}, "
            f"dtype={output_dtype.name}"
        ),
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return pd.Series(sums, index=pd.RangeIndex(group_count), name=name)


def dense_grouped_reduce(
    codes,
    *,
    size: int,
    sums: Mapping[Any, Any],
    count_name: Any | None = None,
    count_dtype: Any = np.uint64,
    sum_dtype: Any = np.float64,
    out: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reduce count and named numeric sums into one fixed-domain device state.

    Code-domain validation and index conversion are shared across every output
    column. The returned object is a real pandas DataFrame whose columns remain
    device-backed until an explicit public export. When ``out`` is an earlier
    device-backed result with the same fixed domain and columns, reductions are
    accumulated into its existing vectors and that same frame is returned.
    """
    group_count = int(size)
    if group_count < 0:
        raise ValueError("dense_grouped_reduce size must be non-negative")
    named_values = dict(sums)
    if not named_values and count_name is None:
        raise ValueError("dense_grouped_reduce requires a count or sum output")
    if count_name is not None and count_name in named_values:
        raise ValueError("dense_grouped_reduce output names must be unique")
    resolved_count_dtype = _count_dtype(count_dtype)
    resolved_sum_dtype = np.dtype(sum_dtype)
    if resolved_sum_dtype.kind not in {"f", "i", "u"}:
        raise TypeError("dense_grouped_reduce sum_dtype must be a real numeric dtype")

    code_expression = _native_expression_from_public_series(codes)
    value_expressions = {
        name: _native_expression_from_public_series(values)
        for name, values in named_values.items()
    }
    if out is not None and not isinstance(out, pd.DataFrame):
        raise TypeError("dense_grouped_reduce out must be a pandas DataFrame")
    output_names = (
        ([] if count_name is None else [count_name])
        + list(named_values)
    )
    out_expressions = (
        {}
        if out is None
        else {
            name: _native_expression_from_public_series(out[name])
            for name in out.columns
        }
    )
    requested = get_requested_mode()
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and any(
            expression is not None and expression.is_device
            for expression in (
                code_expression,
                *value_expressions.values(),
                *out_expressions.values(),
            )
        )
    )
    output_index = pd.RangeIndex(group_count)
    if out is not None:
        if list(out.columns) != output_names:
            raise ValueError(
                "dense_grouped_reduce out columns must match the requested outputs"
            )
        if not out.index.equals(output_index):
            raise ValueError(
                "dense_grouped_reduce out index must match the fixed group domain"
            )
        if not use_device or any(
            expression is None or not expression.is_device
            for expression in out_expressions.values()
        ):
            raise TypeError(
                "dense_grouped_reduce out must contain device-backed native columns"
            )
        expected_dtypes = {
            **(
                {}
                if count_name is None
                else {count_name: resolved_count_dtype}
            ),
            **{name: resolved_sum_dtype for name in named_values},
        }
        if any(
            np.dtype(out_expressions[name].values.dtype) != dtype
            for name, dtype in expected_dtypes.items()
        ):
            raise TypeError(
                "dense_grouped_reduce out column dtypes must match the requested dtypes"
            )
    if use_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        source_codes = (
            code_expression.values
            if code_expression is not None
            else np.asarray(codes)
        )
        if source_codes.ndim != 1 or source_codes.dtype.kind not in {"i", "u"}:
            raise TypeError(
                "dense_grouped_reduce codes must be one-dimensional integers"
            )
        source_values_by_name = {}
        for name, values in named_values.items():
            expression = value_expressions[name]
            source_values = (
                expression.values if expression is not None else np.asarray(values)
            )
            if source_values.ndim != 1 or source_values.dtype.kind not in {
                "f",
                "i",
                "u",
            }:
                raise TypeError(
                    "dense_grouped_reduce values must be one-dimensional real "
                    "numeric values"
                )
            if int(source_values.size) != int(source_codes.size):
                raise ValueError(
                    "dense_grouped_reduce codes and values must have the same length"
                )
            source_values_by_name[name] = source_values

        output_bytes = (
            0
            if out is not None
            else group_count
            * (
                len(named_values) * int(resolved_sum_dtype.itemsize)
                + (
                    0
                    if count_name is None
                    else int(resolved_count_dtype.itemsize)
                )
            )
        )
        upload_bytes = (
            0
            if code_expression is not None and code_expression.is_device
            else int(source_codes.size) * int(source_codes.dtype.itemsize)
        )
        upload_bytes += sum(
            int(source_values.size) * int(source_values.dtype.itemsize)
            for name, source_values in source_values_by_name.items()
            if (expression := value_expressions[name]) is None
            or not expression.is_device
        )
        required_bytes = _dense_grouped_device_required_bytes(
            row_count=int(source_codes.size),
            group_count=group_count,
            code_dtype=np.dtype(source_codes.dtype),
            value_dtypes=tuple(
                np.dtype(values.dtype) for values in source_values_by_name.values()
            ),
            output_dtype=resolved_sum_dtype,
            output_bytes=output_bytes,
            upload_bytes=upload_bytes,
        )
        runtime = get_cuda_runtime()
        admission = runtime.admit_device_memory(
            stage="tabular-dense-grouped-reduce",
            required_bytes=required_bytes,
            requested_units=group_count,
        )
        if not admission.admitted:
            raise MemoryError(
                "dense_grouped_reduce requires "
                f"{required_bytes} device bytes with "
                f"{admission.remaining_bytes} available"
            )
        d_codes = cp.asarray(source_codes)
        d_values_by_name = {
            name: cp.asarray(values)
            for name, values in source_values_by_name.items()
        }
        invalid = (d_codes < 0) | (d_codes >= group_count)
        if int(d_codes.size) and _device_any(
            cp.any(invalid),
            reason="public dense-grouped-reduce code-domain validation fence",
        ):
            raise ValueError("dense_grouped_reduce codes must be in [0, size)")
        del invalid
        d_indices = d_codes.astype(cp.int64, copy=False)
        output: dict[Any, pd.Series] = {}
        accumulation_inputs = []
        d_counts = None
        order = starts = ends = None
        if d_values_by_name:
            _sync_tabular_hotpath()
            with hotpath_stage(
                "tabular.grouped_reduce.ordering",
                category="sort",
                metadata={
                    "input_rows": int(d_indices.size),
                    "group_domain": group_count,
                    "reduction_columns": len(d_values_by_name),
                    "algorithm": "cccl_int64_radix_sort",
                    "workspace_bytes": required_bytes,
                },
            ) as stage_metadata:
                d_counts, order, starts, ends = _device_dense_grouped_plan(
                    d_indices,
                    group_count=group_count,
                )
                attach_work_amplification(
                    stage_metadata,
                    operation="dense_grouped_reduce_ordering",
                    metric_family="tabular",
                    sums={"ordered_rows": int(d_indices.size)},
                    maxima={
                        "input_rows": int(d_indices.size),
                        "group_domain": group_count,
                        "reduction_columns": len(d_values_by_name),
                        "workspace_bytes": required_bytes,
                    },
                    unavailable=("observed_group_cardinality",),
                    physical_shape="integer-key radix ordering",
                    consumer_kind="shared segmented grouped reduction plan",
                    semantic_contract={
                        "algorithm": "cccl_int64_radix_sort",
                        "stable_source_order": True,
                    },
                )
                _sync_tabular_hotpath()
        if count_name is not None:
            if d_counts is None:
                d_counts = cp.bincount(d_indices, minlength=group_count)[
                    :group_count
                ]
            d_counts = d_counts.astype(resolved_count_dtype, copy=False)
            if out is not None:
                cp.add(
                    out_expressions[count_name].values,
                    d_counts,
                    out=out_expressions[count_name].values,
                )
                accumulation_inputs.append(d_counts)
            else:
                output[count_name] = _public_native_series(
                    NativeExpression(
                        operation="dense_grouped_reduce.count",
                        values=d_counts,
                        source_token=None,
                        source_row_count=group_count,
                        dtype=str(resolved_count_dtype),
                        precision="exact-integer-count",
                    ),
                    index=output_index,
                    name=count_name,
                    operation="dense_grouped_reduce_to_public_array",
                )
        for name, d_values in d_values_by_name.items():
            _sync_tabular_hotpath()
            with hotpath_stage(
                "tabular.grouped_reduce.segmented_sum",
                category="other",
                metadata={
                    "input_rows": int(d_indices.size),
                    "group_domain": group_count,
                    "column": str(name),
                },
            ) as stage_metadata:
                d_sums = _device_segmented_sum(
                    d_values,
                    order=order,
                    starts=starts,
                    ends=ends,
                    output_dtype=resolved_sum_dtype,
                )
                attach_work_amplification(
                    stage_metadata,
                    operation="dense_grouped_segmented_sum",
                    metric_family="tabular",
                    sums={"reduced_rows": int(d_indices.size)},
                    maxima={
                        "input_rows": int(d_indices.size),
                        "group_domain": group_count,
                    },
                    unavailable=("observed_group_cardinality",),
                    physical_shape="stable segmented fp64 reduction",
                    consumer_kind="dense grouped output vector",
                )
                _sync_tabular_hotpath()
            if out is not None:
                _sync_tabular_hotpath()
                with hotpath_stage(
                    "tabular.grouped_reduce.accumulate",
                    category="emit",
                    metadata={
                        "group_domain": group_count,
                        "column": str(name),
                    },
                ) as stage_metadata:
                    cp.add(
                        out_expressions[name].values,
                        d_sums,
                        out=out_expressions[name].values,
                    )
                    attach_work_amplification(
                        stage_metadata,
                        operation="dense_grouped_reduce_accumulate",
                        metric_family="tabular",
                        sums={"merged_output_rows": group_count},
                        maxima={
                            "group_domain": group_count,
                            "persistent_output_bytes": group_count
                            * int(resolved_sum_dtype.itemsize),
                        },
                        physical_shape="persistent dense device accumulator",
                        consumer_kind="next streamed reduction batch",
                    )
                    _sync_tabular_hotpath()
                accumulation_inputs.append(d_sums)
            else:
                output[name] = _public_native_series(
                    NativeExpression(
                        operation="dense_grouped_reduce.sum",
                        values=d_sums,
                        source_token=None,
                        source_row_count=group_count,
                        dtype=str(resolved_sum_dtype),
                        precision=(
                            "fp64-grouped-sum"
                            if resolved_sum_dtype == np.dtype(np.float64)
                            else "source"
                        ),
                    ),
                    index=output_index,
                    name=name,
                    operation="dense_grouped_reduce_to_public_array",
                )
        if accumulation_inputs:
            from vibespatial.cuda._runtime import get_cuda_completion_retainer

            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                tuple(accumulation_inputs),
                lambda _arrays: None,
            )
        record_dispatch_event(
            surface="vibespatial.api.dense_grouped_reduce",
            operation="dense_grouped_reduce",
            implementation=(
                "cccl_stable_dense_grouped_reduce_accumulate"
                if out is not None
                else "cccl_stable_dense_grouped_reduce"
            ),
            reason=(
                "fixed-domain reductions update persistent device output vectors"
                if out is not None
                else "fixed-domain count and deterministic sums share stable code ordering"
            ),
            detail=(
                f"input_rows={int(d_codes.size)}, groups={group_count}, "
                f"sum_columns={len(named_values)}, count={count_name is not None}"
            ),
            requested=requested,
            selected=ExecutionMode.GPU,
        )
        return out if out is not None else pd.DataFrame(output, index=output_index)

    host_codes = np.asarray(codes)
    if host_codes.ndim != 1 or host_codes.dtype.kind not in {"i", "u"}:
        raise TypeError("dense_grouped_reduce codes must be one-dimensional integers")
    if np.any((host_codes < 0) | (host_codes >= group_count)):
        raise ValueError("dense_grouped_reduce codes must be in [0, size)")
    host_indices = host_codes.astype(np.int64, copy=False)
    host_output: dict[Any, Any] = {}
    if count_name is not None:
        counts = np.zeros(group_count, dtype=resolved_count_dtype)
        if host_codes.size:
            np.add.at(counts, host_indices, 1)
        host_output[count_name] = counts
    for name, values in named_values.items():
        host_values = np.asarray(values)
        if host_values.ndim != 1 or host_values.dtype.kind not in {"f", "i", "u"}:
            raise TypeError(
                "dense_grouped_reduce values must be one-dimensional real numeric values"
            )
        if host_values.size != host_codes.size:
            raise ValueError(
                "dense_grouped_reduce codes and values must have the same length"
            )
        if host_values.dtype.kind == "f":
            host_values = np.where(
                np.isnan(host_values),
                np.zeros((), dtype=host_values.dtype),
                host_values,
            )
        weights = host_values.astype(resolved_sum_dtype, copy=False)
        reduced = np.zeros(group_count, dtype=resolved_sum_dtype)
        if host_codes.size:
            np.add.at(reduced, host_indices, weights)
        host_output[name] = reduced
    record_dispatch_event(
        surface="vibespatial.api.dense_grouped_reduce",
        operation="dense_grouped_reduce",
        implementation="numpy_dense_grouped_reduce",
        reason="host fixed-domain count and sums share one code validation",
        detail=(
            f"input_rows={int(host_codes.size)}, groups={group_count}, "
            f"sum_columns={len(named_values)}, count={count_name is not None}"
        ),
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return pd.DataFrame(host_output, index=output_index)


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


__all__ = ["dense_count", "dense_grouped_reduce", "dense_sum", "numeric_take"]
