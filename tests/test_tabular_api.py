from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vibespatial as gpd
from vibespatial.api._native_expression import NativeExpression
from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
from vibespatial.api.tabular import _numeric_divide, _streaming_topk
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


def test_numeric_divide_fills_zero_denominators_on_host() -> None:
    result = _numeric_divide(
        pd.Series([6.0, 4.0, 9.0], index=[3, 4, 5], name="reported"),
        pd.Series([2.0, 0.0, 3.0], index=[3, 4, 5]),
    )

    assert result.index.tolist() == [3, 4, 5]
    assert result.name == "reported"
    assert result.iloc[[0, 2]].tolist() == [3.0, 3.0]
    assert np.isnan(result.iloc[1])


def test_streaming_topk_matches_one_shot_pandas_across_tied_batches() -> None:
    frame = pd.DataFrame(
        {
            "primary": [5.0, 4.0, 5.0, np.nan, 5.0, 3.0, 5.0],
            "secondary": [2, 0, 1, 9, 1, 4, 0],
            "payload": list("abcdefg"),
        }
    )
    accumulator = None
    for positions in ([0, 1, 2], [3, 4], [5, 6]):
        accumulator = _streaming_topk(
            frame.iloc[positions],
            4,
            ["primary", "secondary"],
            largest=True,
            out=accumulator,
        )

    expected = frame.nlargest(4, ["primary", "secondary"])
    pd.testing.assert_frame_equal(
        accumulator.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


@pytest.mark.parametrize("keep", ["last", "all"])
def test_streaming_topk_rejects_noncompositional_tie_modes(keep: str) -> None:
    frame = pd.DataFrame({"primary": [3, 3, 2], "payload": [0, 1, 2]})

    with pytest.raises(NotImplementedError, match="only keep='first'"):
        _streaming_topk(frame, 1, "primary", keep=keep)


def test_streaming_topk_empty_batch_reapplies_smaller_bound() -> None:
    retained = pd.DataFrame({"primary": [5, 4, 3], "payload": list("abc")})
    empty = retained.iloc[:0]

    reduced = _streaming_topk(empty, 1, "primary", out=retained)
    zero = _streaming_topk(empty, 0, "primary", out=retained)

    pd.testing.assert_frame_equal(reduced, retained.iloc[:1])
    pd.testing.assert_frame_equal(zero, retained.iloc[:0])


def test_numeric_divide_rejects_reordered_indexes() -> None:
    numerator = pd.Series([6.0, 8.0], index=["a", "b"])
    denominator = pd.Series([2.0, 4.0], index=["b", "a"])

    with pytest.raises(ValueError, match="identical indexes"):
        _numeric_divide(numerator, denominator)


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

    updated = gpd.dense_count(
        pd.Series([], dtype=np.int64),
        size=0,
        dtype=np.uint64,
        out=counts,
    )
    assert updated is counts


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_dense_count_updates_host_accumulator_transactionally_in_place(dtype) -> None:
    counts = gpd.dense_count(pd.Series([0, 2, 2]), size=4, dtype=dtype)

    updated = gpd.dense_count(
        pd.Series([1, 2, 3, 2]),
        size=4,
        dtype=dtype,
        out=counts,
    )

    assert updated is counts
    assert counts.tolist() == [1, 1, 4, 1]
    assert gpd.get_dispatch_events(clear=True)[-1].implementation == (
        "numpy_dense_scatter_count_update"
    )


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_dense_count_host_overflow_leaves_entire_accumulator_unchanged(dtype) -> None:
    maximum = np.iinfo(dtype).max
    counts = pd.Series(np.asarray([maximum, 4], dtype=dtype))
    before = counts.copy()

    with pytest.raises(OverflowError, match="exceeds"):
        gpd.dense_count(
            pd.Series([0, 1]),
            size=2,
            dtype=dtype,
            out=counts,
        )

    pd.testing.assert_series_equal(counts, before)


@pytest.mark.parametrize(
    ("out", "error", "message"),
    [
        (np.zeros(2, dtype=np.uint32), TypeError, "pandas Series"),
        (pd.Series([0], dtype=np.uint32), ValueError, "length"),
        (
            pd.Series([0, 0], dtype=np.uint32, index=pd.Index([1, 2])),
            ValueError,
            "RangeIndex",
        ),
        (pd.Series([0, 0], dtype=np.uint64), TypeError, "dtype"),
    ],
)
def test_dense_count_rejects_incompatible_host_accumulator(out, error, message) -> None:
    with pytest.raises(error, match=message):
        gpd.dense_count(
            pd.Series([0]),
            size=2,
            dtype=np.uint32,
            out=out,
        )


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
@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_dense_count_updates_device_accumulator_transactionally_in_place(dtype) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    def native_codes(values, operation):
        array = cp.asarray(values, dtype=cp.int64)
        return pd.Series(
            NativeNumericExpressionArray(
                NativeExpression(
                    operation=operation,
                    values=array,
                    source_token="rows",
                    source_row_count=int(array.size),
                    dtype="int64",
                    precision="exact-integer",
                )
            )
        )

    counts = gpd.dense_count(
        native_codes([0, 2, 2], "test.dense_count.first"),
        size=4,
        dtype=dtype,
    )
    storage = counts.array.expression.values
    gpd.clear_dispatch_events()
    updated = gpd.dense_count(
        native_codes([1, 2, 3, 2], "test.dense_count.second"),
        size=4,
        dtype=dtype,
        out=counts,
    )
    gathered = gpd.numeric_take(
        counts,
        native_codes([2, 0], "test.dense_count.take"),
    )

    assert updated is counts
    assert counts.array.expression.values is storage
    assert isinstance(counts.array, NativeNumericExpressionArray)
    assert cp.asnumpy(storage).tolist() == [1, 1, 4, 1]
    assert cp.asnumpy(gathered.array.expression.values).tolist() == [4, 1]
    assert any(
        event.implementation == "cuda_dense_scatter_count_update"
        for event in gpd.get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_dense_count_device_overflow_leaves_entire_accumulator_unchanged(dtype) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    maximum = np.iinfo(dtype).max
    values = cp.asarray([maximum, 4], dtype=dtype)
    counts = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.saturated",
                values=values,
                source_token=None,
                source_row_count=2,
                dtype=np.dtype(dtype).name,
                precision="exact-integer-count",
            )
        )
    )
    before = values.copy()

    with pytest.raises(OverflowError, match="exceeds"):
        gpd.dense_count(
            pd.Series([0, 1]),
            size=2,
            dtype=dtype,
            out=counts,
        )

    assert cp.array_equal(values, before)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_dense_count_updates_strided_device_input_and_output(dtype) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    code_storage = cp.asarray([1, 99, 0, 99], dtype=cp.int64)
    count_storage = cp.asarray([10, 99, 20, 99], dtype=dtype)
    codes = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.strided.codes",
                values=code_storage[::2],
                source_token="rows",
                source_row_count=2,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    counts = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.strided.counts",
                values=count_storage[::2],
                source_token=None,
                source_row_count=2,
                dtype=np.dtype(dtype).name,
                precision="exact-integer-count",
            )
        )
    )

    updated = gpd.dense_count(codes, size=2, dtype=dtype, out=counts)

    assert updated is counts
    assert cp.asnumpy(count_storage).tolist() == [11, 99, 21, 99]


@pytest.mark.gpu
def test_dense_count_rejects_aliased_device_input_and_output_before_mutation() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    storage = cp.asarray([0, 1, 1], dtype=cp.uint64)
    codes = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.aliased.codes",
                values=storage.view(cp.int64),
                source_token="rows",
                source_row_count=3,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    counts = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.aliased.counts",
                values=storage,
                source_token=None,
                source_row_count=3,
                dtype="uint64",
                precision="exact-integer-count",
            )
        )
    )
    before = storage.copy()

    with pytest.raises(ValueError, match="must not share device storage"):
        gpd.dense_count(codes, size=3, dtype=np.uint64, out=counts)

    assert cp.array_equal(storage, before)


@pytest.mark.gpu
def test_dense_count_admits_host_upload_before_allocating_it(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    host_codes = np.asarray([0, 1, 1], dtype=np.int64)
    runtime = get_cuda_runtime()
    original_admit = runtime.admit_device_memory
    original_asarray = cp.asarray
    admission_seen = False

    def record_admission(**kwargs):
        nonlocal admission_seen
        admission_seen = True
        return original_admit(**kwargs)

    def require_admission_before_upload(values, *args, **kwargs):
        if values is host_codes:
            assert admission_seen
        return original_asarray(values, *args, **kwargs)

    monkeypatch.setattr(runtime, "admit_device_memory", record_admission)
    monkeypatch.setattr(cp, "asarray", require_admission_before_upload)

    with set_requested_mode(ExecutionMode.GPU):
        counts = gpd.dense_count(host_codes, size=2, dtype=np.uint32)

    assert admission_seen
    assert cp.asnumpy(counts.array.expression.values).tolist() == [1, 2]


@pytest.mark.gpu
def test_dense_count_update_fits_envelope_that_rejects_second_domain_vector(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.runtime.hotpath_trace import (
        reset_hotpath_trace,
        summarize_hotpath_trace,
    )

    group_count = 1_000_000
    first = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.envelope.first",
                values=cp.asarray([0, 2, 2], dtype=cp.int64),
                source_token="rows",
                source_row_count=3,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    second = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.dense_count.envelope.second",
                values=cp.asarray([1, 2, 3], dtype=cp.int64),
                source_token="rows",
                source_row_count=3,
                dtype="int64",
                precision="exact-integer",
            )
        )
    )
    counts = gpd.dense_count(first, size=group_count, dtype=np.uint32)
    storage = counts.array.expression.values
    output_bytes = int(storage.nbytes)
    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    reset_hotpath_trace()
    monkeypatch.setattr(
        runtime,
        "query_memory_remaining_bytes",
        lambda: output_bytes - 1,
    )

    updated = gpd.dense_count(
        second,
        size=group_count,
        dtype=np.uint32,
        out=counts,
    )
    update_admission = runtime.memory_admission_events(clear=True)[-1]
    update_stage = next(
        stage
        for stage in summarize_hotpath_trace()
        if stage["name"] == "tabular.dense_count.update"
    )
    with pytest.raises(MemoryError, match="dense_count requires"):
        gpd.dense_count(second, size=group_count, dtype=np.uint32)

    assert updated is counts
    assert counts.array.expression.values is storage
    assert update_admission.stage == "tabular-dense-count-update"
    assert update_admission.required_bytes < output_bytes
    packet = update_stage["metadata"]["work_amplification"]
    assert packet["max"]["persistent_output_bytes"] == output_bytes
    assert packet["max"]["output_allocation_bytes"] == 0
    assert packet["semantic_contract"] == {
        "output_identity_reused": True,
        "transactional_overflow": True,
    }
    assert cp.asnumpy(storage[:4]).tolist() == [1, 1, 3, 1]


@pytest.mark.gpu
def test_dense_grouped_reduce_accumulates_into_persistent_device_outputs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    def native(values, *, operation, dtype):
        array = cp.asarray(values, dtype=dtype)
        return pd.Series(
            NativeNumericExpressionArray(
                NativeExpression(
                    operation=operation,
                    values=array,
                    source_token="rows",
                    source_row_count=int(array.size),
                    dtype=str(array.dtype),
                    precision="test",
                )
            )
        )

    first = gpd.dense_grouped_reduce(
        native([0, 1, 0], operation="test.codes.first", dtype=cp.int64),
        size=3,
        count_name="count",
        sums={
            "total": native(
                [1.5, 2.0, 3.5],
                operation="test.values.first",
                dtype=cp.float64,
            )
        },
    )
    count_values = first["count"].array.expression.values
    total_values = first["total"].array.expression.values
    gpd.clear_dispatch_events()
    accumulated = gpd.dense_grouped_reduce(
        native([2, 0, 2], operation="test.codes.second", dtype=cp.int64),
        size=3,
        count_name="count",
        sums={
            "total": native(
                [4.0, 5.0, cp.nan],
                operation="test.values.second",
                dtype=cp.float64,
            )
        },
        out=first,
    )

    assert accumulated is first
    assert accumulated["count"].array.expression.values is count_values
    assert accumulated["total"].array.expression.values is total_values
    assert accumulated["count"].tolist() == [3, 1, 2]
    assert accumulated["total"].tolist() == [10.0, 2.0, 4.0]
    assert any(
        event.implementation == "cccl_stable_dense_grouped_reduce_accumulate"
        for event in gpd.get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
def test_dense_grouped_reduce_large_small_domain_uses_linear_radix_work(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    import cupy as cp

    from vibespatial.runtime.hotpath_trace import (
        reset_hotpath_trace,
        summarize_hotpath_trace,
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    group_count = 84
    for row_count in (250_000, 1_000_000):
        codes_array = cp.arange(row_count, dtype=cp.int64) % group_count
        values_array = cp.ones(row_count, dtype=cp.float64)
        codes = pd.Series(
            NativeNumericExpressionArray(
                NativeExpression(
                    operation="test.large_small_domain.codes",
                    values=codes_array,
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
                    operation="test.large_small_domain.values",
                    values=values_array,
                    source_token="rows",
                    source_row_count=row_count,
                    dtype="float64",
                    precision="fp64",
                )
            )
        )
        reset_hotpath_trace()
        reduced = gpd.dense_grouped_reduce(
            codes,
            size=group_count,
            count_name="count",
            sums={"total": values},
        )

        ordering = next(
            stage
            for stage in summarize_hotpath_trace()
            if stage["name"] == "tabular.grouped_reduce.ordering"
        )
        packet = ordering["metadata"]["work_amplification"]
        assert packet["semantic_contract"]["algorithm"] == "cccl_int64_radix_sort"
        assert packet["sum"]["ordered_rows"] == row_count
        assert packet["max"]["group_domain"] == group_count
        assert int(reduced["count"].sum()) == row_count


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

    with set_requested_mode(ExecutionMode.CPU):
        with pytest.raises(TypeError, match="host execution requires"):
            gpd.dense_count(device_codes, size=2, out=gpu_counts)
    with set_requested_mode(ExecutionMode.GPU):
        with pytest.raises(TypeError, match="device execution requires"):
            gpd.dense_count(
                pd.Series([0, 1, 1]),
                size=2,
                out=pd.Series([0, 0], dtype=np.uint32),
            )


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
