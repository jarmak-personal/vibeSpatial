from __future__ import annotations

import numpy as np
import pytest

from vibespatial.cccl_primitives import (
    CompactionStrategy,
    PairSortStrategy,
    ScanStrategy,
    compact_indices,
    counting_iterator,
    exclusive_sum,
    has_cccl_primitives,
    segmented_sort,
    select_pair_sort_strategy,
    sort_pairs,
    three_way_partition,
    transform_iterator,
    unique_sorted_pairs,
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

    assert result.strategy is PairSortStrategy.RADIX
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
