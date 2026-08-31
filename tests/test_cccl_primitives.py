from __future__ import annotations

import ast
import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vibespatial import has_gpu_runtime
from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.cuda.cccl_primitives import (
    CompactionStrategy,
    PairSortStrategy,
    ScanStrategy,
    compact_indices,
    counting_iterator,
    exclusive_sum,
    has_cccl_primitives,
    inclusive_max,
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
    if not has_gpu_runtime() or not has_cccl_primitives():
        pytest.skip("CCCL Python primitives require an available CUDA runtime")
    import cupy as cp

    return cp


def test_cccl_primitives_have_no_raw_device_scalar_item_syncs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "cuda" / "cccl_primitives.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    failures: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            failures.append(f"raw .item() at line {node.lineno}")

    assert failures == []


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


def test_inclusive_max_matches_numpy_prefix_maximum() -> None:
    cp = _cupy()
    values = cp.asarray([-3.0, 1.0, -4.0, 5.0, 2.0], dtype=cp.float64)

    result = inclusive_max(values, synchronize=True)

    np.testing.assert_array_equal(
        result.get(),
        np.asarray([-3.0, 1.0, 1.0, 5.0, 5.0], dtype=np.float64),
    )


def test_sort_pairs_auto_routes_numeric_keys_to_radix() -> None:
    cp = _cupy()
    keys = cp.asarray([4, 1, 3, 1], dtype=cp.int32)
    values = cp.asarray([40, 10, 30, 11], dtype=cp.int32)

    result = sort_pairs(keys, values)

    # AUTO may choose RADIX (warm) or CUPY (cold start); both produce correct results
    assert result.strategy in (PairSortStrategy.RADIX, PairSortStrategy.CUPY)
    np.testing.assert_array_equal(result.keys.get(), np.asarray([1, 1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(result.values.get(), np.asarray([10, 11, 30, 40], dtype=np.int32))


def test_sort_pairs_writes_into_reusable_output_buffers() -> None:
    cp = _cupy()
    keys = cp.asarray([4, 1, 3, 1], dtype=cp.uint64)
    values = cp.asarray([40, 10, 30, 11], dtype=cp.int32)
    out_keys = cp.empty_like(keys)
    out_values = cp.empty_like(values)

    result = sort_pairs(
        keys,
        values,
        strategy=PairSortStrategy.RADIX,
        out_keys=out_keys,
        out_values=out_values,
    )

    assert result.keys is out_keys
    assert result.values is out_values
    np.testing.assert_array_equal(out_keys.get(), [1, 1, 3, 4])
    np.testing.assert_array_equal(out_values.get(), [10, 11, 30, 40])


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

    sync_calls: list[object] = []

    def _synchronize(_cp_module, stream) -> None:
        sync_calls.append(stream)
        stream.synchronize()

    monkeypatch.setattr(primitives_module, "_get_precompiled", lambda name: precompiled)
    monkeypatch.setattr(primitives_module, "_stream_synchronize", _synchronize)
    monkeypatch.setattr(
        primitives_module.algorithms,
        "radix_sort",
        lambda *args, **kwargs: pytest.fail("one-shot radix_sort should not be used"),
    )

    result = sort_pairs(
        keys,
        values,
        strategy=PairSortStrategy.RADIX,
        synchronize=True,
    )

    assert result.strategy is PairSortStrategy.RADIX
    assert sync_calls == [cp.cuda.get_current_stream()]
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
        result.keys.get(),
        np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.values.get(),
        np.asarray([20, 50, 80, 10, 40, 70], dtype=np.int32),
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
        result.keys.get(),
        np.asarray([3, 2, 1, 9, 7, 5], dtype=np.int32),
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
        result.keys.get(),
        np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reordered.get(),
        np.asarray([20, 50, 80, 10, 40, 70], dtype=np.int32),
    )


def test_sort_pairs_synchronize_false_composes_with_follow_on_gather() -> None:
    cp = _cupy()
    keys = cp.asarray([5, 2, 8, 7, 1, 4], dtype=cp.int32)
    payload = cp.asarray([50, 20, 80, 70, 10, 40], dtype=cp.int32)
    values = cp.arange(int(keys.size), dtype=cp.int32)

    result = sort_pairs(keys, values, synchronize=False)
    reordered = payload[result.values]

    np.testing.assert_array_equal(
        result.keys.get(),
        np.asarray([1, 2, 4, 5, 7, 8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reordered.get(),
        np.asarray([10, 20, 40, 50, 70, 80], dtype=np.int32),
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
        doubled.get(),
        np.asarray([6.0, 24.0], dtype=np.float64),
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
    np.testing.assert_array_equal(
        second_min.values.get(), np.asarray([5.0, 20.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(first_max.values.get(), np.asarray([1.0, 11.0], dtype=np.float64))
    np.testing.assert_array_equal(
        second_max.values.get(), np.asarray([6.0, 21.0], dtype=np.float64)
    )


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
        h_out[i] == 0 for i in range(result.first_count, result.first_count + result.second_count)
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


def test_default_cccl_wrappers_do_not_force_null_stream_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        prefix.get(),
        np.asarray([0, 3, 4, 8, 9], dtype=np.int32),
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
        seg_sorted.keys.get(),
        np.asarray([2, 5, 8, 1, 4, 7], dtype=np.int32),
    )

    reduced = _assert_wrapper_stays_async(
        lambda: segmented_reduce_sum(
            cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float64),
            cp.asarray([0, 2], dtype=cp.int32),
            cp.asarray([2, 5], dtype=cp.int32),
        ),
    )
    np.testing.assert_array_equal(
        reduced.values.get(),
        np.asarray([3.0, 12.0], dtype=np.float64),
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


def test_precompiled_async_reuse_uses_independent_cross_stream_scratch(
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

    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

        def claim_stream_retirements(self, stream):
            selected = [item[1:] for item in self.retirements if item[0] is stream]
            self.retirements = [item for item in self.retirements if item[0] is not stream]
            return selected

        @staticmethod
        def release_claimed_retirements(retirements) -> None:
            for payload, release in retirements:
                release(payload)

    retainer = _Retainer()

    monkeypatch.setattr(primitives_module, "_get_precompiled", lambda name: precompiled)
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)
    monkeypatch.setattr(cp.cuda.Stream.null, "synchronize", lambda: null_sync_calls.append("sync"))

    result_a = exclusive_sum(
        values_a,
        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
        synchronize=False,
        stream=stream_a,
    )
    retained_a = retainer.retirements[-1][1][3]
    assert any(value is values_a for value in retained_a)
    assert any(value is result_a for value in retained_a)

    result_b = exclusive_sum(
        values_b,
        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
        synchronize=False,
        stream=stream_b,
    )
    retained_b = retainer.retirements[-1][1][3]

    assert recorded_streams == [stream_a, stream_b]
    assert stream_a.sync_calls == 0
    assert stream_b.sync_calls == 0
    assert null_sync_calls == []
    assert len(retainer.retirements) == 2
    assert retainer.retirements[0][1][1] is not retainer.retirements[1][1][1]
    assert all(value is not values_a for value in retained_b)
    assert all(value is not result_a for value in retained_b)
    assert any(value is values_b for value in retained_b)
    assert any(value is result_b for value in retained_b)
    np.testing.assert_array_equal(result_a.get(), np.asarray([0, 3, 4, 8], dtype=np.int32))
    np.testing.assert_array_equal(result_b.get(), np.asarray([0, 2, 9, 10], dtype=np.int32))

    retainer.release_claimed_retirements(retainer.claim_stream_retirements(stream_a))
    retainer.release_claimed_retirements(retainer.claim_stream_retirements(stream_b))
    assert primitives_module._active_precompiled_launch_count(precompiled) == 0


def test_precompiled_ptds_calls_use_thread_distinct_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _cupy()
    invocation_state_lock = threading.Lock()
    active_invocations = 0
    max_active_invocations = 0
    scratch_ids: list[int] = []
    start = threading.Barrier(2)

    class _FakeStream:
        def __init__(self, handle: int) -> None:
            self.handle = handle

        def __cuda_stream__(self) -> tuple[int, int]:
            return (0, self.handle)

        def synchronize(self) -> None:
            return None

    def _make_callable(temp, d_in, d_out, op, item_count, init, stream=None):
        nonlocal active_invocations, max_active_invocations
        if temp is None:
            return 1
        with invocation_state_lock:
            active_invocations += 1
            max_active_invocations = max(max_active_invocations, active_invocations)
            scratch_ids.append(id(temp))
        time.sleep(0.02)
        d_out[:] = cp.cumsum(d_in[:item_count], dtype=d_in.dtype)
        d_out[:] -= d_in[:item_count]
        with invocation_state_lock:
            active_invocations -= 1
        return 1

    precompiled = SimpleNamespace(
        make_callable=_make_callable,
        temp_storage=cp.empty(1, dtype=cp.uint8),
        temp_storage_bytes=1,
        high_water_n=128,
    )

    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Retainer:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            with self.lock:
                self.retirements.append((stream, payload, release))

        def claim_stream_retirements(self, stream):
            with self.lock:
                selected = [item for item in self.retirements if item[0] is stream]
                self.retirements = [item for item in self.retirements if item[0] is not stream]
            return [item[1:] for item in selected]

        @staticmethod
        def release_claimed_retirements(retirements) -> None:
            for payload, release in retirements:
                release(payload)

    retainer = _Retainer()
    monkeypatch.setattr(primitives_module, "_get_precompiled", lambda name: precompiled)
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)

    def _launch(values, stream):
        start.wait()
        return exclusive_sum(
            values,
            strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
            synchronize=False,
            stream=stream,
        )

    values_a = cp.asarray([3, 1, 4, 1], dtype=cp.int32)
    values_b = cp.asarray([2, 7, 1, 8], dtype=cp.int32)
    stream_a = _FakeStream(2)
    stream_b = _FakeStream(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_launch, values_a, stream_a)
        future_b = executor.submit(_launch, values_b, stream_b)
        result_a = future_a.result()
        result_b = future_b.result()

    assert max_active_invocations == 1
    assert len(set(scratch_ids)) == 2
    assert len(retainer.retirements) == 2
    np.testing.assert_array_equal(result_a.get(), np.asarray([0, 3, 4, 8], dtype=np.int32))
    np.testing.assert_array_equal(result_b.get(), np.asarray([0, 2, 9, 10], dtype=np.int32))

    retainer.release_claimed_retirements(retainer.claim_stream_retirements(stream_a))
    retainer.release_claimed_retirements(retainer.claim_stream_retirements(stream_b))
    assert primitives_module._active_precompiled_launch_count(precompiled) == 0


def test_precompiled_sync_waits_overlap_after_serialized_launch_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _ArrayModule:
        uint8 = np.uint8

        @staticmethod
        def empty(size, **_kwargs):
            return bytearray(size)

    class _Stream:
        def __init__(self, handle: int) -> None:
            self.handle = handle

    class _Retainer:
        def defer(self, _stream, _payload, _release) -> None:
            raise AssertionError("successful synchronous calls retire directly")

        @staticmethod
        def claim_stream_retirements(_stream):
            return []

        @staticmethod
        def release_claimed_retirements(retirements) -> None:
            assert retirements == []

    launches: list[int] = []
    sync_barrier = threading.Barrier(2)

    def _make_callable(temp, stream=None):
        if temp is None:
            return 1
        launches.append(stream.handle)
        return 1

    precompiled = SimpleNamespace(
        make_callable=_make_callable,
        temp_storage=bytearray(1),
        temp_storage_bytes=1,
        high_water_n=1,
    )
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", _Retainer)
    monkeypatch.setattr(
        primitives_module,
        "_stream_synchronize",
        lambda _cp_module, _stream: sync_barrier.wait(timeout=2.0),
    )

    def _launch(handle: int) -> None:
        primitives_module._execute_precompiled(
            precompiled,
            _ArrayModule,
            num_items=1,
            args=(),
            stream=_Stream(handle),
            synchronize=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_launch, handle) for handle in (81, 82)]
        for future in futures:
            future.result(timeout=3.0)

    assert sorted(launches) == [81, 82]
    assert primitives_module._active_precompiled_launch_count(precompiled) == 0


def test_invocation_failure_transfers_one_shot_ownership_to_retainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _Stream:
        handle = 75

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

    def _raise_after_enqueue(*_args, **_kwargs):
        raise RuntimeError("one-shot invoke failed after enqueue")

    retainer = _Retainer()
    owners = [_Owner(), _Owner()]
    owner_refs = [weakref.ref(owner) for owner in owners]
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)

    with pytest.raises(RuntimeError, match="one-shot invoke failed after enqueue"):
        primitives_module._execute_one_shot(
            _raise_after_enqueue,
            object(),
            args=(owners[0],),
            references=(owners[1],),
            stream=_Stream(),
            synchronize=False,
        )

    assert len(retainer.retirements) == 1
    del owners
    gc.collect()
    assert all(owner_ref() is not None for owner_ref in owner_refs)

    _stream, payload, release = retainer.retirements.pop()
    release(payload)
    del payload
    gc.collect()
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_successful_async_one_shot_retains_ownership_until_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _Stream:
        handle = 76

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

    retainer = _Retainer()
    owners = [_Owner(), _Owner()]
    owner_refs = [weakref.ref(owner) for owner in owners]
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)

    result = primitives_module._execute_one_shot(
        lambda *_args, **_kwargs: "launched",
        object(),
        args=(owners[0],),
        references=(owners[1],),
        stream=_Stream(),
        synchronize=False,
    )

    assert result == "launched"
    assert len(retainer.retirements) == 1
    del owners
    gc.collect()
    assert all(owner_ref() is not None for owner_ref in owner_refs)
    _stream, payload, release = retainer.retirements.pop()
    release(payload)
    del payload
    gc.collect()
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_one_shot_sync_failure_transfers_ownership_to_retainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _Stream:
        handle = 77

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []
            self.restored = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

        @staticmethod
        def claim_stream_retirements(_stream):
            return []

        def restore_stream_retirements(self, stream, retirements) -> None:
            self.restored.append((stream, retirements))

        def release_claimed_retirements(self, _retirements) -> None:
            raise AssertionError("failed synchronization cannot release payloads")

    retainer = _Retainer()
    owner = _Owner()
    owner_ref = weakref.ref(owner)
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)
    monkeypatch.setattr(
        primitives_module,
        "_stream_synchronize",
        lambda _cp_module, _stream: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        primitives_module._execute_one_shot(
            lambda *_args, **_kwargs: None,
            object(),
            args=(owner,),
            stream=_Stream(),
            synchronize=True,
        )

    assert len(retainer.restored) == 1
    assert len(retainer.retirements) == 1
    del owner
    gc.collect()
    assert owner_ref() is not None
    _stream, payload, release = retainer.retirements.pop()
    release(payload)
    del payload
    gc.collect()
    assert owner_ref() is None


@pytest.mark.parametrize("allocation_path", ["first-use", "growth", "concurrent-slot"])
def test_precompiled_scratch_allocation_failure_releases_pointer_owners(
    allocation_path: str,
) -> None:
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _PointerIterator:
        def __init__(self) -> None:
            self.state = 0

        def is_kind_pointer(self) -> bool:
            return True

    class _InnerCallable:
        def __init__(self) -> None:
            self.data_cccl = _PointerIterator()
            self.values_cccl = _PointerIterator()
            self.output_cccl = _PointerIterator()

    class _Adapter:
        def __init__(self) -> None:
            self._inner = _InnerCallable()

        def __call__(self, temp, data, values, output, *_args, **_kwargs):
            self._inner.data_cccl.state = data
            self._inner.values_cccl.state = values
            self._inner.output_cccl.state = output
            if temp is None:
                return 64
            return 0

    class _FailingArrayModule:
        uint8 = np.uint8

        @staticmethod
        def empty(*_args, **_kwargs):
            raise MemoryError("scratch allocation failed")

    class _Stream:
        handle = 72

    make_callable = _Adapter()
    precompiled = SimpleNamespace(
        make_callable=make_callable,
        temp_storage=bytearray(1),
        temp_storage_bytes=1,
        high_water_n=0,
    )
    slots = None
    slot = None
    if allocation_path != "first-use":
        slots = primitives_module._precompiled_slots(precompiled)
        slot = slots[0]
    if allocation_path == "concurrent-slot":
        assert slot is not None
        slot.token = object()
        slot.stream_key = 71
        precompiled.high_water_n = 128
        slot.high_water_n = 128
    precompiled_snapshot = (
        precompiled.temp_storage,
        precompiled.temp_storage_bytes,
        precompiled.high_water_n,
    )
    slot_snapshot = (
        None
        if slot is None
        else (
            slot.temp,
            slot.temp_storage_bytes,
            slot.high_water_n,
            slot.stream,
            slot.stream_key,
            slot.token,
        )
    )

    owners = [_Owner(), _Owner(), _Owner()]
    owner_refs = [weakref.ref(owner) for owner in owners]
    with pytest.raises(MemoryError, match="scratch allocation failed"):
        primitives_module._execute_precompiled(
            precompiled,
            _FailingArrayModule,
            num_items=64,
            args=(*owners, 32, 32),
            stream=_Stream(),
            synchronize=False,
        )

    assert make_callable._inner.data_cccl.state == 0
    assert make_callable._inner.values_cccl.state == 0
    assert make_callable._inner.output_cccl.state == 0
    assert precompiled.temp_storage is precompiled_snapshot[0]
    assert precompiled.temp_storage_bytes == precompiled_snapshot[1]
    assert precompiled.high_water_n == precompiled_snapshot[2]
    if allocation_path == "first-use":
        assert not hasattr(precompiled, "scratch_slots")
    else:
        assert primitives_module._precompiled_slots(precompiled) is slots
        assert slot is not None
        assert slot_snapshot is not None
        assert slot.temp is slot_snapshot[0]
        assert slot.temp_storage_bytes == slot_snapshot[1]
        assert slot.high_water_n == slot_snapshot[2]
        assert slot.stream is slot_snapshot[3]
        assert slot.stream_key == slot_snapshot[4]
        assert slot.token is slot_snapshot[5]
    del owners
    gc.collect()
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_sync_failure_transfers_precompiled_operand_ownership_to_retainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.cuda import _runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _ArrayModule:
        uint8 = np.uint8

        @staticmethod
        def empty(size, **_kwargs):
            return bytearray(size)

    class _Stream:
        handle = 73

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

        def claim_stream_retirements(self, stream):
            return []

        def restore_stream_retirements(self, stream, retirements) -> None:
            assert retirements == []

        def release_claimed_retirements(self, retirements) -> None:
            raise AssertionError("failed synchronization cannot release payloads")

    precompiled = SimpleNamespace(
        make_callable=lambda temp, *_args, **_kwargs: 1,
        temp_storage=bytearray(1),
        temp_storage_bytes=1,
        high_water_n=64,
    )
    retainer = _Retainer()
    owners = [_Owner(), _Owner()]
    owner_refs = [weakref.ref(owner) for owner in owners]
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)
    monkeypatch.setattr(
        primitives_module,
        "_stream_synchronize",
        lambda cp_module, stream: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        primitives_module._execute_precompiled(
            precompiled,
            _ArrayModule,
            num_items=2,
            args=(*owners, 2),
            stream=_Stream(),
            synchronize=True,
            references=tuple(owners),
        )

    assert len(retainer.retirements) == 1
    slot = primitives_module._precompiled_slots(precompiled)[0]
    assert slot.token is not None
    del owners
    gc.collect()
    assert all(owner_ref() is not None for owner_ref in owner_refs)

    _stream, payload, release = retainer.retirements.pop()
    release(payload)
    del payload
    gc.collect()
    assert slot.token is None
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_invocation_failure_transfers_precompiled_operand_ownership_to_retainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.cuda import _runtime as runtime_module
    from vibespatial.cuda import cccl_primitives as primitives_module

    class _Owner:
        pass

    class _ArrayModule:
        uint8 = np.uint8

    class _Stream:
        handle = 74

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

    def _raise_after_enqueue(temp, *_args, **_kwargs):
        if temp is None:
            return 1
        raise RuntimeError("invoke failed after enqueue")

    precompiled = SimpleNamespace(
        make_callable=_raise_after_enqueue,
        temp_storage=bytearray(1),
        temp_storage_bytes=1,
        high_water_n=64,
    )
    retainer = _Retainer()
    owners = [_Owner(), _Owner()]
    owner_refs = [weakref.ref(owner) for owner in owners]
    monkeypatch.setattr(runtime_module, "get_cuda_completion_retainer", lambda: retainer)

    with pytest.raises(RuntimeError, match="invoke failed after enqueue"):
        primitives_module._execute_precompiled(
            precompiled,
            _ArrayModule,
            num_items=2,
            args=(*owners, 2),
            stream=_Stream(),
            synchronize=False,
            references=tuple(owners),
        )

    assert len(retainer.retirements) == 1
    slot = primitives_module._precompiled_slots(precompiled)[0]
    assert slot.token is not None
    del owners
    gc.collect()
    assert all(owner_ref() is not None for owner_ref in owner_refs)

    _stream, payload, release = retainer.retirements.pop()
    release(payload)
    del payload
    gc.collect()
    assert slot.token is None
    assert all(owner_ref() is None for owner_ref in owner_refs)


@pytest.mark.gpu
def test_async_precompiled_operands_retire_after_stream_completion() -> None:
    cp = _cupy()
    if not has_cccl_primitives():
        pytest.skip("CCCL primitives are unavailable")

    from vibespatial.cuda import cccl_primitives as primitives_module

    precompiled = primitives_module._get_precompiled("exclusive_scan_i32")
    if precompiled is None:
        pytest.skip("precompiled exclusive scan is unavailable")

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        values = cp.arange(2_000_000, dtype=cp.int32)
        result = exclusive_sum(
            values,
            strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
            synchronize=False,
            stream=stream,
        )
    values_ref = weakref.ref(values)
    result_ref = weakref.ref(result)
    del values
    del result

    assert values_ref() is not None
    assert result_ref() is not None
    stream.synchronize()
    deadline = time.monotonic() + 2.0
    while (
        values_ref() is not None
        or result_ref() is not None
        or primitives_module._active_precompiled_launch_count(precompiled) != 0
    ) and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert values_ref() is None
    assert result_ref() is None
    assert primitives_module._active_precompiled_launch_count(precompiled) == 0


@pytest.mark.gpu
def test_async_binary_search_adapter_releases_pointer_iterator_owners() -> None:
    cp = _cupy()
    if not has_cccl_primitives():
        pytest.skip("CCCL primitives are unavailable")

    from vibespatial.cuda import cccl_primitives as primitives_module

    precompiled = primitives_module._get_precompiled("lower_bound_i32")
    if precompiled is None:
        pytest.skip("precompiled lower-bound is unavailable")

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        values = cp.arange(2_000_000, dtype=cp.int32)
        queries = cp.arange(0, 2_000_000, 2, dtype=cp.int32)
        result = lower_bound(values, queries, synchronize=False, stream=stream)
    values_ref = weakref.ref(values)
    queries_ref = weakref.ref(queries)
    result_ref = weakref.ref(result)
    del values
    del queries
    del result

    assert values_ref() is not None
    assert queries_ref() is not None
    assert result_ref() is not None
    stream.synchronize()
    deadline = time.monotonic() + 2.0
    while (
        values_ref() is not None
        or queries_ref() is not None
        or result_ref() is not None
        or primitives_module._active_precompiled_launch_count(precompiled) != 0
    ) and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert values_ref() is None
    assert queries_ref() is None
    assert result_ref() is None
    assert primitives_module._active_precompiled_launch_count(precompiled) == 0


def test_precompiled_default_launch_uses_cupy_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _cupy()
    values = cp.asarray([3, 1, 4, 1], dtype=cp.int32)
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

    result = exclusive_sum(
        values,
        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
        synchronize=True,
    )

    assert recorded_streams == [cp.cuda.get_current_stream()]
    np.testing.assert_array_equal(result.get(), np.asarray([0, 3, 4, 8], dtype=np.int32))


def test_lower_bound_counting_passes_explicit_stream_to_cccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = _cupy()
    sorted_data = cp.asarray([2, 5, 5, 9, 14], dtype=cp.int32)
    recorded_streams: list[object | None] = []

    stream = cp.cuda.Stream(non_blocking=True)

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
    stream.synchronize()
