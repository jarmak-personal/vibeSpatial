from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.cuda.cccl_primitives import (
    CompactionStrategy,
    PairSortStrategy,
    ScanStrategy,
    compact_indices,
    counting_iterator,
    exclusive_sum,
    has_cccl_primitives,
    lower_bound,
    lower_bound_counting,
    segmented_reduce_max,
    segmented_reduce_min,
    segmented_reduce_sum,
    segmented_sort,
    select_pair_sort_strategy,
    sort_pairs,
    three_way_partition,
    transform_iterator,
    unique_sorted_pairs,
    upper_bound,
    upper_bound_counting,
)


def _cupy():
    if not has_cccl_primitives():
        pytest.skip("CCCL Python primitives are not available")
    import cupy as cp

    return cp


def test_compact_indices_matches_flatnonzero() -> None:
    cp = _cupy()
    mask = cp.asarray([0, 1, 0, 1, 1, 0], dtype=cp.uint8)

    result = compact_indices(mask)

    assert result.count == 3
    np.testing.assert_array_equal(result.values.get(), np.flatnonzero(mask.get()))


def test_compact_indices_cccl_select_matches_flatnonzero() -> None:
    cp = _cupy()
    mask = cp.asarray([1, 0, 1, 0, 0, 1], dtype=cp.uint8)

    result = compact_indices(mask, strategy=CompactionStrategy.CCCL_SELECT)

    assert result.count == 3
    np.testing.assert_array_equal(result.values.get(), np.flatnonzero(mask.get()))


def test_exclusive_sum_matches_numpy_prefix_sum() -> None:
    cp = _cupy()
    values = cp.asarray([3, 1, 4, 1, 5], dtype=cp.int32)

    result = exclusive_sum(values)

    np.testing.assert_array_equal(result.get(), np.asarray([0, 3, 4, 8, 9], dtype=np.int32))


def test_exclusive_sum_cccl_scan_matches_numpy_prefix_sum() -> None:
    cp = _cupy()
    values = cp.asarray([3, 1, 4, 1, 5], dtype=cp.int32)

    result = exclusive_sum(values, strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN)

    np.testing.assert_array_equal(result.get(), np.asarray([0, 3, 4, 8, 9], dtype=np.int32))


def test_sort_pairs_auto_routes_numeric_keys_to_radix() -> None:
    cp = _cupy()
    keys = cp.asarray([4, 1, 3, 1], dtype=cp.int32)
    values = cp.asarray([40, 10, 30, 11], dtype=cp.int32)

    result = sort_pairs(keys, values)

    # AUTO may choose RADIX (warm) or CUPY (cold start); both produce correct results
    assert result.strategy in (PairSortStrategy.RADIX, PairSortStrategy.CUPY)
    np.testing.assert_array_equal(result.keys.get(), np.asarray([1, 1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(result.values.get(), np.asarray([10, 11, 30, 40], dtype=np.int32))


def test_sort_pairs_merge_route_supports_descending() -> None:
    cp = _cupy()
    keys = cp.asarray([4, 1, 3, 1], dtype=cp.int32)
    values = cp.asarray([40, 10, 30, 11], dtype=cp.int32)

    result = sort_pairs(keys, values, descending=True, strategy=PairSortStrategy.MERGE)

    assert result.strategy is PairSortStrategy.MERGE
    np.testing.assert_array_equal(result.keys.get(), np.asarray([4, 3, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(result.values.get(), np.asarray([40, 30, 10, 11], dtype=np.int32))


def test_sort_pairs_uses_precompiled_radix_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cp = _cupy()
    keys = cp.asarray([4, 1, 3, 1], dtype=cp.int32)
    values = cp.asarray([40, 10, 30, 11], dtype=cp.int32)

    def _make_callable(temp, in_keys, out_keys, in_values, out_values, item_count):
        if temp is None:  # pragma: no cover - exercised by _ensure_temp query path
            return 1
        idx = cp.argsort(in_keys[:item_count])
        out_keys[:item_count] = in_keys[idx]
        out_values[:item_count] = in_values[idx]

    precompiled = SimpleNamespace(
        make_callable=_make_callable,
        temp_storage=cp.empty(1, dtype=cp.uint8),
        temp_storage_bytes=1,
        high_water_n=128,
    )

    from vibespatial.cuda import cccl_primitives as primitives_module

    monkeypatch.setattr(primitives_module, "_get_precompiled", lambda name: precompiled)
    monkeypatch.setattr(
        primitives_module.algorithms,
        "radix_sort",
        lambda *args, **kwargs: pytest.fail("one-shot radix_sort should not be used"),
    )

    result = sort_pairs(keys, values, strategy=PairSortStrategy.RADIX)

    assert result.strategy is PairSortStrategy.RADIX
    np.testing.assert_array_equal(result.keys.get(), np.asarray([1, 1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(result.values.get(), np.asarray([10, 11, 30, 40], dtype=np.int32))


def test_unique_sorted_pairs_dedupes_runs_and_keeps_first_value() -> None:
    cp = _cupy()
    keys = cp.asarray([1, 1, 3, 4, 4, 9], dtype=cp.int32)
    values = cp.asarray([10, 11, 30, 40, 41, 90], dtype=cp.int32)

    result = unique_sorted_pairs(keys, values)

    assert result.count == 4
    np.testing.assert_array_equal(result.keys.get(), np.asarray([1, 3, 4, 9], dtype=np.int32))
    np.testing.assert_array_equal(result.values.get(), np.asarray([10, 30, 40, 90], dtype=np.int32))


def test_sort_strategy_selection_falls_back_to_merge_for_complex_dtypes() -> None:
    resolved = select_pair_sort_strategy(np.dtype(np.complex64))
    assert resolved is PairSortStrategy.MERGE


def test_segmented_sort_sorts_within_segments() -> None:
    cp = _cupy()
    # Two segments: [0:3) and [3:6)
    keys = cp.asarray([5, 2, 8, 7, 1, 4], dtype=cp.int32)
    values = cp.asarray([50, 20, 80, 70, 10, 40], dtype=cp.int32)
    starts = cp.asarray([0, 3], dtype=cp.int32)
    ends = cp.asarray([3, 6], dtype=cp.int32)

    result = segmented_sort(keys, values, starts=starts, ends=ends)

    assert result.segment_count == 2
    np.testing.assert_array_equal(
        result.keys.get(), np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.values.get(), np.asarray([20, 50, 80, 10, 40, 70], dtype=np.int32),
    )


def test_segmented_sort_descending() -> None:
    cp = _cupy()
    keys = cp.asarray([1, 3, 2, 9, 5, 7], dtype=cp.int32)
    starts = cp.asarray([0, 3], dtype=cp.int32)
    ends = cp.asarray([3, 6], dtype=cp.int32)

    result = segmented_sort(keys, starts=starts, ends=ends, descending=True)

    assert result.segment_count == 2
    assert result.values is None
    np.testing.assert_array_equal(
        result.keys.get(), np.asarray([3, 2, 1, 9, 7, 5], dtype=np.int32),
    )


def test_segmented_sort_empty() -> None:
    cp = _cupy()
    keys = cp.empty(0, dtype=cp.int32)
    starts = cp.empty(0, dtype=cp.int32)
    ends = cp.empty(0, dtype=cp.int32)

    result = segmented_sort(keys, starts=starts, ends=ends)

    assert result.segment_count == 0
    assert result.keys.size == 0


def test_segmented_sort_synchronize_false_composes_with_follow_on_gather() -> None:
    cp = _cupy()
    keys = cp.asarray([5, 2, 8, 7, 1, 4], dtype=cp.int32)
    payload = cp.asarray([50, 20, 80, 70, 10, 40], dtype=cp.int32)
    values = cp.arange(int(keys.size), dtype=cp.int32)
    starts = cp.asarray([0, 3], dtype=cp.int32)
    ends = cp.asarray([3, 6], dtype=cp.int32)

    result = segmented_sort(
        keys,
        values,
        starts=starts,
        ends=ends,
        synchronize=False,
    )
    reordered = payload[result.values]

    np.testing.assert_array_equal(
        result.keys.get(), np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reordered.get(), np.asarray([20, 50, 80, 10, 40, 70], dtype=np.int32),
    )


def test_sort_pairs_synchronize_false_composes_with_follow_on_gather() -> None:
    cp = _cupy()
    keys = cp.asarray([5, 2, 8, 7, 1, 4], dtype=cp.int32)
    payload = cp.asarray([50, 20, 80, 70, 10, 40], dtype=cp.int32)
    values = cp.arange(int(keys.size), dtype=cp.int32)

    result = sort_pairs(keys, values, synchronize=False)
    reordered = payload[result.values]

    np.testing.assert_array_equal(
        result.keys.get(), np.asarray([1, 2, 4, 5, 7, 8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reordered.get(), np.asarray([10, 20, 40, 50, 70, 80], dtype=np.int32),
    )


def test_segmented_reduce_sum_synchronize_false_composes_with_device_math() -> None:
    cp = _cupy()
    values = cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float64)
    starts = cp.asarray([0, 2], dtype=cp.int32)
    ends = cp.asarray([2, 5], dtype=cp.int32)

    result = segmented_reduce_sum(
        values,
        starts,
        ends,
        synchronize=False,
    )
    doubled = result.values * 2.0

    np.testing.assert_array_equal(
        doubled.get(), np.asarray([6.0, 24.0], dtype=np.float64),
    )


def test_segmented_reduce_async_temp_reuse_stays_correct() -> None:
    cp = _cupy()
    first_values = cp.asarray([0.0, 1.0, 10.0, 11.0], dtype=cp.float64)
    second_values = cp.asarray([5.0, 6.0, 20.0, 21.0], dtype=cp.float64)
    starts = cp.asarray([0, 2], dtype=cp.int32)
    ends = cp.asarray([2, 4], dtype=cp.int32)

    first_min = segmented_reduce_min(
        first_values,
        starts,
        ends,
        synchronize=False,
    )
    second_min = segmented_reduce_min(
        second_values,
        starts,
        ends,
        synchronize=False,
    )
    first_max = segmented_reduce_max(
        first_values,
        starts,
        ends,
        synchronize=False,
    )
    second_max = segmented_reduce_max(
        second_values,
        starts,
        ends,
        synchronize=False,
    )

    np.testing.assert_array_equal(first_min.values.get(), np.asarray([0.0, 10.0], dtype=np.float64))
    np.testing.assert_array_equal(second_min.values.get(), np.asarray([5.0, 20.0], dtype=np.float64))
    np.testing.assert_array_equal(first_max.values.get(), np.asarray([1.0, 11.0], dtype=np.float64))
    np.testing.assert_array_equal(second_max.values.get(), np.asarray([6.0, 21.0], dtype=np.float64))


def test_three_way_partition_splits_correctly() -> None:
    cp = _cupy()
    # Partition: negatives first, then zeros, then positives
    values = cp.asarray([3, -1, 0, -2, 5, 0, -3, 1], dtype=cp.int32)

    def is_negative(x):  # pragma: no cover - exercised through CCCL JIT
        return x < 0

    def is_zero(x):  # pragma: no cover - exercised through CCCL JIT
        return x == 0

    result = three_way_partition(values, is_negative, is_zero)

    assert result.first_count == 3  # negatives
    assert result.second_count == 2  # zeros
    h_out = result.values.get()
    # First partition: all negative
    assert all(h_out[i] < 0 for i in range(result.first_count))
    # Second partition: all zero
    assert all(
        h_out[i] == 0
        for i in range(result.first_count, result.first_count + result.second_count)
    )
    # Third partition: all positive
    assert all(h_out[i] > 0 for i in range(result.first_count + result.second_count, len(h_out)))


def test_three_way_partition_empty() -> None:
    cp = _cupy()
    values = cp.empty(0, dtype=cp.int32)

    result = three_way_partition(values, lambda x: x < 0, lambda x: x == 0)

    assert result.first_count == 0
    assert result.second_count == 0


def test_counting_iterator_returns_iterator_object() -> None:
    _cupy()
    it = counting_iterator(42)
    assert it is not None


def test_transform_iterator_returns_iterator_object() -> None:
    cp = _cupy()
    arr = cp.asarray([1, 2, 3], dtype=cp.int32)

    def double(x):  # pragma: no cover - exercised through CCCL JIT
        return x * 2

    it = transform_iterator(arr, double)
    assert it is not None


def test_lower_bound_counting_matches_materialized_queries() -> None:
    cp = _cupy()
    sorted_data = cp.asarray([2, 5, 5, 9, 14], dtype=cp.int32)
    query_values = cp.arange(0, 12, dtype=cp.int32)

    materialized = lower_bound(sorted_data, query_values)
    counted = lower_bound_counting(sorted_data, 0, int(query_values.size), dtype=np.int32)

    np.testing.assert_array_equal(counted.get(), materialized.get())


def test_upper_bound_counting_matches_materialized_queries() -> None:
    cp = _cupy()
    sorted_data = cp.asarray([2, 5, 5, 9, 14], dtype=cp.int32)
    query_values = cp.arange(0, 12, dtype=cp.int32)

    materialized = upper_bound(sorted_data, query_values)
    counted = upper_bound_counting(sorted_data, 0, int(query_values.size), dtype=np.int32)

    np.testing.assert_array_equal(counted.get(), materialized.get())


def test_default_cccl_wrappers_do_not_force_null_stream_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    cp = _cupy()
    original_sync = cp.cuda.Stream.null.synchronize

    def _assert_wrapper_stays_async(fn):
        sync_calls: list[str] = []

        def _record_sync():
            sync_calls.append("sync")

        monkeypatch.setattr(cp.cuda.Stream.null, "synchronize", _record_sync)
        try:
            result = fn()
            assert sync_calls == []
            return result
        finally:
            monkeypatch.setattr(cp.cuda.Stream.null, "synchronize", original_sync)

    values = cp.asarray([3, 1, 4, 1, 5], dtype=cp.int32)
    prefix = _assert_wrapper_stays_async(
        lambda: exclusive_sum(values, strategy=ScanStrategy.CUPY),
    )
    np.testing.assert_array_equal(
        prefix.get(), np.asarray([0, 3, 4, 8, 9], dtype=np.int32),
    )

    keys = cp.asarray([5, 2, 8, 7, 1, 4], dtype=cp.int32)
    payload = cp.asarray([50, 20, 80, 70, 10, 40], dtype=cp.int32)
    order = cp.arange(int(keys.size), dtype=cp.int32)
    sorted_pairs = _assert_wrapper_stays_async(
        lambda: sort_pairs(keys, order, strategy=PairSortStrategy.CUPY),
    )
    np.testing.assert_array_equal(
        payload[sorted_pairs.values].get(),
        np.asarray([10, 20, 40, 50, 70, 80], dtype=np.int32),
    )

    starts = cp.asarray([0, 3], dtype=cp.int32)
    ends = cp.asarray([3, 6], dtype=cp.int32)
    seg_sorted = _assert_wrapper_stays_async(
        lambda: segmented_sort(keys, order, starts=starts, ends=ends),
    )
    np.testing.assert_array_equal(
        seg_sorted.keys.get(), np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )

    reduced = _assert_wrapper_stays_async(
        lambda: segmented_reduce_sum(
            cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float64),
            cp.asarray([0, 2], dtype=cp.int32),
            cp.asarray([2, 5], dtype=cp.int32),
        ),
    )
    np.testing.assert_array_equal(
        reduced.values.get(), np.asarray([3.0, 12.0], dtype=np.float64),
    )

    sorted_data = cp.asarray([2, 5, 5, 9, 14], dtype=cp.int32)
    query_values = np.arange(12, dtype=np.int32)
    bounds = _assert_wrapper_stays_async(
        lambda: lower_bound_counting(sorted_data, 0, 12, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        bounds.get(),
        np.searchsorted(np.asarray([2, 5, 5, 9, 14], dtype=np.int32), query_values, side="left"),
    )

    upper = _assert_wrapper_stays_async(
        lambda: upper_bound_counting(sorted_data, 0, 12, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        upper.get(),
        np.searchsorted(np.asarray([2, 5, 5, 9, 14], dtype=np.int32), query_values, side="right"),
    )


def test_runtime_cuda_stream_exposes_cuda_stream_protocol() -> None:
    _cupy()
    runtime = get_cuda_runtime()
    stream = runtime.create_stream()
    try:
        version, handle = stream.__cuda_stream__()
        assert version == 0
        assert handle == int(stream.handle)
    finally:
        stream.synchronize()
        runtime.destroy_stream(stream)


def test_precompiled_async_reuse_fences_prior_launch_stream_not_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _cupy()
    values_a = cp.asarray([3, 1, 4, 1], dtype=cp.int32)
    values_b = cp.asarray([2, 7, 1, 8], dtype=cp.int32)
    null_sync_calls: list[str] = []

    class _FakeStream:
        def __init__(self, handle: int) -> None:
            self.handle = handle
            self.sync_calls = 0

        def __cuda_stream__(self) -> tuple[int, int]:
            return (0, self.handle)

        def synchronize(self) -> None:
            self.sync_calls += 1

    stream_a = _FakeStream(11)
    stream_b = _FakeStream(12)
    recorded_streams: list[object | None] = []

    def _make_callable(temp, d_in, d_out, op, item_count, init, stream=None):
        if temp is None:  # pragma: no cover - exercised by temp query path
            return 1
        recorded_streams.append(stream)
        d_out[:] = cp.cumsum(d_in[:item_count], dtype=d_in.dtype)
        d_out[:] -= d_in[:item_count]
        return 1

    precompiled = SimpleNamespace(
        make_callable=_make_callable,
        temp_storage=cp.empty(1, dtype=cp.uint8),
        temp_storage_bytes=1,
        high_water_n=128,
    )

    from vibespatial.cuda import cccl_primitives as primitives_module

    monkeypatch.setattr(primitives_module, "_get_precompiled", lambda name: precompiled)
    monkeypatch.setattr(cp.cuda.Stream.null, "synchronize", lambda: null_sync_calls.append("sync"))

    result_a = exclusive_sum(
        values_a,
        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
        synchronize=False,
        stream=stream_a,
    )
    result_b = exclusive_sum(
        values_b,
        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
        synchronize=False,
        stream=stream_b,
    )

    assert recorded_streams == [stream_a, stream_b]
    assert stream_a.sync_calls == 1
    assert stream_b.sync_calls == 0
    assert null_sync_calls == []
    np.testing.assert_array_equal(result_a.get(), np.asarray([0, 3, 4, 8], dtype=np.int32))
    np.testing.assert_array_equal(result_b.get(), np.asarray([0, 2, 9, 10], dtype=np.int32))


def test_lower_bound_counting_passes_explicit_stream_to_cccl(monkeypatch: pytest.MonkeyPatch) -> None:
    cp = _cupy()
    sorted_data = cp.asarray([2, 5, 5, 9, 14], dtype=cp.int32)
    recorded_streams: list[object | None] = []

    class _FakeStream:
        def __cuda_stream__(self) -> tuple[int, int]:
            return (0, 17)

        def synchronize(self) -> None:
            return None

    stream = _FakeStream()

    from vibespatial.cuda import cccl_primitives as primitives_module

    def _fake_lower_bound(d_data, d_values, d_out, num_items, num_values, comp=None, stream=None):
        recorded_streams.append(stream)
        d_out[:] = cp.arange(num_values, dtype=d_out.dtype)

    monkeypatch.setattr(primitives_module.algorithms, "lower_bound", _fake_lower_bound)

    result = lower_bound_counting(
        sorted_data,
        0,
        4,
        dtype=np.int32,
        synchronize=False,
        stream=stream,
    )

    assert recorded_streams == [stream]
    np.testing.assert_array_equal(result.get(), np.asarray([0, 1, 2, 3], dtype=np.uintp))
