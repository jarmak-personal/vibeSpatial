from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vibespatial as gpd
from vibespatial.api._native_expression import NativeExpression
from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
from vibespatial.runtime import ExecutionMode, has_gpu_runtime, set_requested_mode


def test_dense_count_and_numeric_take_host_contract() -> None:
    codes = pd.Series([2, 0, 2, 1, 2], name="codes")

    counts = gpd.dense_count(codes, size=4, dtype=np.uint32, name="count")
    sums = gpd.dense_sum(
        codes,
        pd.Series([1.5, np.nan, 2.5, 4.0, 8.0]),
        size=4,
        name="total",
    )
    reduced = gpd.dense_grouped_reduce(
        codes,
        size=4,
        count_name="count",
        sums={
            "total": pd.Series([1.5, np.nan, 2.5, 4.0, 8.0]),
            "other": pd.Series([1, 2, 3, 4, 5]),
        },
    )
    gathered = gpd.numeric_take(counts, pd.Series([2, 0, 3], index=[9, 8, 7]))

    assert isinstance(counts, pd.Series)
    assert counts.dtype == np.dtype(np.uint32)
    assert counts.tolist() == [1, 1, 3, 0]
    assert sums.dtype == np.dtype(np.float64)
    assert sums.name == "total"
    assert sums.tolist() == [0.0, 4.0, 12.0, 0.0]
    pd.testing.assert_frame_equal(
        reduced,
        pd.DataFrame(
            {
                "count": np.asarray([1, 1, 3, 0], dtype=np.uint64),
                "total": [0.0, 4.0, 12.0, 0.0],
                "other": [2.0, 4.0, 9.0, 0.0],
            }
        ),
    )
    assert gathered.index.tolist() == [9, 8, 7]
    assert gathered.name == "count"
    assert gathered.tolist() == [3, 1, 0]


def test_dense_sums_skip_float_nan_before_integer_output_cast_on_host() -> None:
    codes = pd.Series([0, 1, 1])
    values = pd.Series([1.0, np.nan, 3.0])

    sums = gpd.dense_sum(codes, values, size=2, dtype=np.int64)
    reduced = gpd.dense_grouped_reduce(
        codes,
        size=2,
        count_name="count",
        sums={"total": values},
        sum_dtype=np.int64,
    )

    assert sums.tolist() == [1, 3]
    assert reduced["count"].tolist() == [1, 2]
    assert reduced["total"].tolist() == [1, 3]


def test_dense_count_empty_uint64_and_empty_take_contract() -> None:
    counts = gpd.dense_count(
        pd.Series([], dtype=np.int64),
        size=0,
        dtype=np.uint64,
        name="count",
    )
    gathered = gpd.numeric_take(
        pd.Series([], dtype=np.float64, name="values"),
        pd.Series([], dtype=np.int64, index=pd.Index([], dtype=np.int64)),
    )

    assert counts.dtype == np.dtype(np.uint64)
    assert counts.empty
    assert counts.name == "count"
    assert gathered.dtype == np.dtype(np.float64)
    assert gathered.empty
    assert gathered.name == "values"


@pytest.mark.gpu
def test_dense_count_and_numeric_take_keep_native_series_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    codes_expression = NativeExpression(
        operation="test.codes",
        values=cp.asarray([2, 0, 2, 1, 2], dtype=cp.int64),
        source_token="rows",
        source_row_count=5,
        dtype="int64",
        precision="exact-integer",
    )
    codes = pd.Series(NativeNumericExpressionArray(codes_expression))

    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)
    gpd.clear_dispatch_events()
    counts = gpd.dense_count(codes, size=4, dtype=np.uint32, name="count")
    values = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.values",
                values=cp.asarray([1.5, cp.nan, 2.5, 4.0, 8.0]),
                source_token="rows",
                source_row_count=5,
                dtype="float64",
                precision="fp64",
            )
        )
    )
    sums = gpd.dense_sum(
        codes,
        values,
        size=4,
        name="total",
    )
    reduced = gpd.dense_grouped_reduce(
        codes,
        size=4,
        count_name="count",
        sums={"total": values},
    )
    gathered = gpd.numeric_take(counts, codes)
    host_index_gathered = gpd.numeric_take(counts, pd.Series([2, 0]))
    selected = gathered > 1
    events = gpd.get_dispatch_events(clear=True)
    admissions = runtime.memory_admission_events(clear=True)

    assert isinstance(counts, pd.Series)
    assert isinstance(counts.array, NativeNumericExpressionArray)
    assert counts.array.expression.is_device
    assert isinstance(sums.array, NativeNumericExpressionArray)
    assert sums.array.expression.is_device
    assert sums.tolist() == [0.0, 4.0, 12.0, 0.0]
    assert all(
        isinstance(reduced[column].array, NativeNumericExpressionArray)
        for column in reduced.columns
    )
    assert reduced["count"].tolist() == [1, 1, 3, 0]
    assert reduced["total"].tolist() == [0.0, 4.0, 12.0, 0.0]
    assert isinstance(gathered.array, NativeNumericExpressionArray)
    assert gathered.array.expression.is_device
    assert gathered.index.equals(codes.index)
    assert isinstance(host_index_gathered.array, NativeNumericExpressionArray)
    assert host_index_gathered.tolist() == [3, 1]
    assert selected.tolist() == [True, False, True, False, True]
    assert any(event.implementation == "cupy_dense_scatter_count" for event in events)
    assert any(event.implementation == "cccl_stable_segmented_sum" for event in events)
    assert any(
        event.implementation == "cccl_stable_dense_grouped_reduce"
        for event in events
    )
    assert any(event.implementation == "cupy_numeric_gather" for event in events)
    assert any(
        event.stage == "tabular-dense-count" and event.admitted
        for event in admissions
    )
    assert any(
        event.stage == "tabular-dense-sum" and event.admitted
        for event in admissions
    )
    assert any(
        event.stage == "tabular-dense-grouped-reduce" and event.admitted
        for event in admissions
    )
    assert any(
        event.stage == "tabular-numeric-take" and event.admitted
        for event in admissions
    )


@pytest.mark.gpu
def test_dense_sum_is_deterministic_for_adversarial_fp64_group() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    repeats = 400_000
    host_values = np.tile(
        np.asarray([1.0e16, 1.0, -1.0e16], dtype=np.float64),
        repeats,
    )
    row_count = int(host_values.size)
    codes = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.adversarial_codes",
                values=cp.zeros(row_count, dtype=cp.int64),
                source_token="rows",
                source_row_count=row_count,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    values = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.adversarial_values",
                values=cp.asarray(host_values),
                source_token="rows",
                source_row_count=row_count,
                dtype="float64",
                precision="fp64",
            )
        )
    )

    results = [
        float(
            cp.asnumpy(
                gpd.dense_sum(codes, values, size=1).array.expression.values
            )[0]
        )
        for _ in range(3)
    ]

    assert results == [0.0, 0.0, 0.0]


@pytest.mark.gpu
def test_dense_sum_auto_uses_device_when_any_input_is_device_backed() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    values = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.mixed_residency_values",
                values=cp.asarray([1.0, 2.0, 3.0]),
                source_token="rows",
                source_row_count=3,
                dtype="float64",
                precision="fp64",
            )
        )
    )
    gpd.clear_fallback_events()
    gpd.clear_materialization_events()
    gpd.clear_dispatch_events()

    result = gpd.dense_sum(pd.Series([0, 1, 1]), values, size=2)
    dispatch = gpd.get_dispatch_events(clear=True)[-1]

    assert isinstance(result.array, NativeNumericExpressionArray)
    assert result.array.expression.is_device
    assert gpd.get_fallback_events(clear=True) == []
    assert gpd.get_materialization_events(clear=True) == []
    assert dispatch.selected is ExecutionMode.GPU
    assert cp.asnumpy(result.array.expression.values).tolist() == [1.0, 5.0]


@pytest.mark.gpu
def test_dense_sums_skip_float_nan_before_integer_output_cast_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    codes = pd.Series([0, 1, 1])
    values = pd.Series([1.0, np.nan, 3.0])
    with set_requested_mode(ExecutionMode.GPU):
        sums = gpd.dense_sum(codes, values, size=2, dtype=np.int64)
        reduced = gpd.dense_grouped_reduce(
            codes,
            size=2,
            count_name="count",
            sums={"total": values},
            sum_dtype=np.int64,
        )

    assert isinstance(sums.array, NativeNumericExpressionArray)
    assert cp.asnumpy(sums.array.expression.values).tolist() == [1, 3]
    assert cp.asnumpy(reduced["count"].array.expression.values).tolist() == [1, 2]
    assert cp.asnumpy(reduced["total"].array.expression.values).tolist() == [1, 3]


@pytest.mark.gpu
def test_dense_grouped_reduce_admission_bounds_operation_peak() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial.spatial_index_knn_device import (
        _OperationAllocationMonitor,
    )

    row_count = 1_000_000
    codes_values = cp.arange(row_count, dtype=cp.int64) % 84
    numeric_values = cp.arange(row_count, dtype=cp.float64)
    codes = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.admission_codes",
                values=codes_values,
                source_token="rows",
                source_row_count=row_count,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    values = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.admission_values",
                values=numeric_values,
                source_token="rows",
                source_row_count=row_count,
                dtype="float64",
                precision="fp64",
            )
        )
    )
    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)

    monitor = _OperationAllocationMonitor()
    result = gpd.dense_grouped_reduce(
        codes,
        size=32_768,
        count_name="count",
        sums={"a": values, "b": values, "c": values},
    )
    cp.cuda.get_current_stream().synchronize()
    peak_bytes, _ = monitor.finish()
    admission = runtime.memory_admission_events(clear=True)[-1]

    assert list(result.columns) == ["count", "a", "b", "c"]
    assert admission.required_bytes >= peak_bytes


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("row_count", "group_count"),
    [(1_000_000, 32_768), (1_000, 1_000_000)],
)
def test_dense_grouped_count_only_admission_bounds_operation_peak(
    row_count: int,
    group_count: int,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial.spatial_index_knn_device import (
        _OperationAllocationMonitor,
    )

    codes_values = cp.arange(row_count, dtype=cp.int64) % min(group_count, 84)
    codes = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.count_only_admission_codes",
                values=codes_values,
                source_token="rows",
                source_row_count=row_count,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)

    monitor = _OperationAllocationMonitor()
    result = gpd.dense_grouped_reduce(
        codes,
        size=group_count,
        count_name="count",
        sums={},
    )
    cp.cuda.get_current_stream().synchronize()
    peak_bytes, _ = monitor.finish()
    admission = runtime.memory_admission_events(clear=True)[-1]

    assert list(result.columns) == ["count"]
    assert admission.required_bytes >= peak_bytes


@pytest.mark.gpu
def test_tabular_operations_respect_explicit_execution_mode() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    expression = NativeExpression(
        operation="test.codes",
        values=cp.asarray([0, 1, 1], dtype=cp.int64),
        source_token="rows",
        source_row_count=3,
        dtype="int64",
        precision="exact-integer",
    )
    device_codes = pd.Series(NativeNumericExpressionArray(expression))

    with set_requested_mode(ExecutionMode.CPU):
        gpd.clear_dispatch_events()
        cpu_counts = gpd.dense_count(device_codes, size=2)
        cpu_event = gpd.get_dispatch_events(clear=True)[-1]
    with set_requested_mode(ExecutionMode.GPU):
        gpd.clear_dispatch_events()
        gpu_counts = gpd.dense_count(pd.Series([0, 1, 1]), size=2)
        gpu_event = gpd.get_dispatch_events(clear=True)[-1]

    assert not isinstance(cpu_counts.array, NativeNumericExpressionArray)
    assert cpu_counts.tolist() == [1, 2]
    assert cpu_event.requested is ExecutionMode.CPU
    assert cpu_event.selected is ExecutionMode.CPU
    assert isinstance(gpu_counts.array, NativeNumericExpressionArray)
    assert gpu_counts.tolist() == [1, 2]
    assert gpu_event.requested is ExecutionMode.GPU
    assert gpu_event.selected is ExecutionMode.GPU


@pytest.mark.parametrize("codes", [[-1], [4]])
def test_dense_count_rejects_codes_outside_fixed_domain(codes) -> None:
    with pytest.raises(ValueError, match=r"\[0, size\)"):
        gpd.dense_count(pd.Series(codes), size=4)


def test_numeric_take_rejects_out_of_bounds_positions() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        gpd.numeric_take(pd.Series([1, 2]), pd.Series([2]))


def test_tabular_operations_reject_non_integer_codes_and_indices() -> None:
    with pytest.raises(TypeError, match="one-dimensional integers"):
        gpd.dense_count(pd.Series([0.0]), size=1)
    with pytest.raises(TypeError, match="one-dimensional integers"):
        gpd.numeric_take(pd.Series([1]), pd.Series([0.0]))
    with pytest.raises(TypeError, match="uint32 or uint64"):
        gpd.dense_count(pd.Series([0]), size=1, dtype=np.int64)
    with pytest.raises(TypeError, match="one-dimensional integers"):
        gpd.dense_sum(pd.Series([0.0]), pd.Series([1.0]), size=1)
    with pytest.raises(TypeError, match="real numeric"):
        gpd.dense_sum(pd.Series([0]), pd.Series(["x"]), size=1)
    with pytest.raises(ValueError, match="same length"):
        gpd.dense_sum(pd.Series([0]), pd.Series([1.0, 2.0]), size=1)
