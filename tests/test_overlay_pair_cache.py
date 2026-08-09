from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

import vibespatial
import vibespatial.api as gpd
from vibespatial.api._native_relation import NativeRelationSelection
from vibespatial.api._native_result_core import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeTabularResult,
)
from vibespatial.api._native_rowset import NativeRowSet
from vibespatial.api._native_state import NativeFrameState, attach_native_state
from vibespatial.api.geodataframe import _public_frame_from_native_state
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools._pair_cache import (
    _INTERSECTION_PAIR_CACHE,
    cache_intersection_pairs,
    get_cached_intersection_pairs,
)
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime.materialization import (
    clear_materialization_events,
    get_materialization_events,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.spatial.query_types import DeviceSpatialJoinResult


def _frame(index):
    return gpd.GeoDataFrame(
        {"value": np.arange(len(index), dtype=np.int64)},
        geometry=[Point(float(i), float(i)) for i, _ in enumerate(index)],
        index=index,
    )


def _native_device_frame(index):
    owned = from_shapely_geometries(
        [Point(float(i), float(i)) for i, _ in enumerate(index)]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native pair-cache device frame",
    )
    frame = gpd.GeoDataFrame(
        {"value": np.arange(len(index), dtype=np.int64)},
        geometry=gpd.GeoSeries(
            GeometryArray.from_owned(owned, crs=None),
            name="geometry",
            index=index,
        ),
        index=index,
    )
    result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=frame.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(frame, state)
    return frame, state


def test_intersection_pair_cache_preserves_device_relation_after_host_update() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("requires CUDA runtime")
    cp = pytest.importorskip("cupy")

    _INTERSECTION_PAIR_CACHE.clear()
    left = _frame([10, 11, 12])
    right = _frame([100, 101])
    d_left = cp.asarray([0, 2, 1], dtype=cp.int32)
    d_right = cp.asarray([1, 0, 1], dtype=cp.int32)

    cache_intersection_pairs(left, right, d_left, d_right)
    cache_intersection_pairs(
        left,
        right,
        cp.asnumpy(d_left),
        cp.asnumpy(d_right),
    )

    result = get_cached_intersection_pairs(left, right, return_device=True)

    assert isinstance(result, DeviceSpatialJoinResult)
    np.testing.assert_array_equal(cp.asnumpy(result.d_left_idx), [0, 2, 1])
    np.testing.assert_array_equal(cp.asnumpy(result.d_right_idx), [1, 0, 1])


def test_intersection_pair_cache_can_physicalize_subset_rows_to_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("requires CUDA runtime")
    cp = pytest.importorskip("cupy")

    _INTERSECTION_PAIR_CACHE.clear()
    left = _frame([10, 11, 12])
    right = _frame([100, 101])
    subset = left.loc[[10, 12]].copy()

    cache_intersection_pairs(
        left,
        right,
        np.asarray([0, 2, 1], dtype=np.int32),
        np.asarray([1, 0, 1], dtype=np.int32),
    )

    result = get_cached_intersection_pairs(subset, right, return_device=True)

    assert isinstance(result, DeviceSpatialJoinResult)
    np.testing.assert_array_equal(cp.asnumpy(result.d_left_idx), [0, 1])
    np.testing.assert_array_equal(cp.asnumpy(result.d_right_idx), [1, 0])


def test_intersection_pair_cache_remaps_native_subset_without_index_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("requires CUDA runtime")
    cp = pytest.importorskip("cupy")

    _INTERSECTION_PAIR_CACHE.clear()
    left, left_state = _native_device_frame([10, 11, 12])
    right, _right_state = _native_device_frame([100, 101])
    subset_rowset = NativeRowSet.from_positions(
        cp.asarray([0, 2], dtype=cp.int32),
        source_token=left_state.lineage_token,
        source_row_count=left_state.row_count,
        ordered=True,
        unique=True,
    )
    subset_state = left_state.take(subset_rowset, preserve_index=True)
    subset = _public_frame_from_native_state(
        left,
        subset_state,
        geometry_column="geometry",
    )

    cache_intersection_pairs(
        left,
        right,
        cp.asarray([0, 2, 1], dtype=cp.int32),
        cp.asarray([1, 0, 1], dtype=cp.int32),
    )
    clear_materialization_events()

    result = get_cached_intersection_pairs(subset, right, return_device=True)

    assert isinstance(result, NativeRelationSelection)
    assert result.capacity == 3
    np.testing.assert_array_equal(cp.asnumpy(result.logical_count), [2])
    np.testing.assert_array_equal(
        cp.asnumpy(
            result.selection.gather_capacity(
                result.relation.left_indices,
                fill_value=-1,
            )
        ),
        [0, 1, -1],
    )
    np.testing.assert_array_equal(
        cp.asnumpy(
            result.selection.gather_capacity(
                result.relation.right_indices,
                fill_value=-1,
            )
        ),
        [1, 0, -1],
    )
    assert get_materialization_events(clear=True) == []
