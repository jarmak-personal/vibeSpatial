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
    gathered = gpd.numeric_take(counts, pd.Series([2, 0, 3], index=[9, 8, 7]))

    assert isinstance(counts, pd.Series)
    assert counts.dtype == np.dtype(np.uint32)
    assert counts.tolist() == [1, 1, 3, 0]
    assert gathered.index.tolist() == [9, 8, 7]
    assert gathered.name == "count"
    assert gathered.tolist() == [3, 1, 0]


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
    gathered = gpd.numeric_take(counts, codes)
    host_index_gathered = gpd.numeric_take(counts, pd.Series([2, 0]))
    selected = gathered > 1
    events = gpd.get_dispatch_events(clear=True)
    admissions = runtime.memory_admission_events(clear=True)

    assert isinstance(counts, pd.Series)
    assert isinstance(counts.array, NativeNumericExpressionArray)
    assert counts.array.expression.is_device
    assert isinstance(gathered.array, NativeNumericExpressionArray)
    assert gathered.array.expression.is_device
    assert gathered.index.equals(codes.index)
    assert isinstance(host_index_gathered.array, NativeNumericExpressionArray)
    assert host_index_gathered.tolist() == [3, 1]
    assert selected.tolist() == [True, False, True, False, True]
    assert any(event.implementation == "cupy_dense_scatter_count" for event in events)
    assert any(event.implementation == "cupy_numeric_gather" for event in events)
    assert any(
        event.stage == "tabular-dense-count" and event.admitted
        for event in admissions
    )
    assert any(
        event.stage == "tabular-numeric-take" and event.admitted
        for event in admissions
    )


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
