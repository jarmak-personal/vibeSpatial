from __future__ import annotations

import importlib
from collections import Counter

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from vibespatial import (
    ExecutionMode,
    build_gpu_atomic_edges,
    build_gpu_split_events,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.overlay.graph import _largest_power_of_two_block_size


def _group_point_counts(source_segment_ids: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in source_segment_ids.tolist())


@pytest.mark.gpu
def test_gpu_split_events_and_atomic_edges_for_proper_cross() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    assert split_events.runtime_selection.selected is ExecutionMode.GPU
    assert split_events.device_state is not None
    assert atomic_edges.device_state is not None
    assert _group_point_counts(split_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert atomic_edges.count == 8
    assert np.allclose(split_events.x[[1, 4]], [2.0, 2.0])
    assert np.allclose(split_events.y[[1, 4]], [2.0, 2.0])
    assert all(event.kind.value != "materialization" for event in left.diagnostics)
    assert all(event.kind.value != "materialization" for event in right.diagnostics)


@pytest.mark.gpu
def test_gpu_atomic_edges_derive_pair_count_from_split_event_cardinality() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])])
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])])
    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    atomic_edges = build_gpu_atomic_edges(split_events)
    events = get_d2h_transfer_events(clear=True)

    expected_pairs = (
        split_events.count
        - split_events.left_segment_count
        - split_events.right_segment_count
    )
    assert atomic_edges.count == expected_pairs * 2
    assert "overlay split atomic-edge pair-count allocation fence" not in {
        event.reason for event in events
    }


@pytest.mark.gpu
def test_gpu_split_events_and_atomic_edges_for_touch_and_overlap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    touch_left = from_shapely_geometries([LineString([(0, 0), (2, 2)])])
    touch_right = from_shapely_geometries([LineString([(2, 2), (4, 0)])])
    touch_events = build_gpu_split_events(touch_left, touch_right, dispatch_mode=ExecutionMode.GPU)
    touch_edges = build_gpu_atomic_edges(touch_events)
    assert _group_point_counts(touch_events.source_segment_ids) == Counter({0: 2, 1: 2})
    assert touch_edges.count == 4

    overlap_left = from_shapely_geometries([LineString([(0, 0), (5, 0)])])
    overlap_right = from_shapely_geometries([LineString([(2, 0), (7, 0)])])
    overlap_events = build_gpu_split_events(overlap_left, overlap_right, dispatch_mode=ExecutionMode.GPU)
    overlap_edges = build_gpu_atomic_edges(overlap_events)
    assert _group_point_counts(overlap_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert overlap_edges.count == 6
    assert np.allclose(overlap_events.x[:3], [0.0, 2.0, 5.0])
    assert np.allclose(overlap_events.x[3:], [2.0, 5.0, 7.0])
    assert np.allclose(overlap_events.y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.gpu
def test_gpu_split_events_dedup_sorted_runs_without_unique_by_key_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    split_module = importlib.import_module("vibespatial.overlay.split")

    monkeypatch.setattr(
        split_module,
        "unique_sorted_pairs",
        lambda *args, **kwargs: pytest.fail("unique_by_key primitive should not be used"),
        raising=False,
    )

    overlap_left = from_shapely_geometries([LineString([(0, 0), (5, 0)])])
    overlap_right = from_shapely_geometries([LineString([(2, 0), (7, 0)])])

    overlap_events = build_gpu_split_events(overlap_left, overlap_right, dispatch_mode=ExecutionMode.GPU)

    assert _group_point_counts(overlap_events.source_segment_ids) == Counter({0: 3, 1: 3})
    assert np.allclose(overlap_events.x[:3], [0.0, 2.0, 5.0])
    assert np.allclose(overlap_events.x[3:], [2.0, 5.0, 7.0])


@pytest.mark.gpu
def test_gpu_split_events_handle_empty_segment_tables() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    empty = from_shapely_geometries([])

    split_events = build_gpu_split_events(empty, empty, dispatch_mode=ExecutionMode.GPU)

    assert split_events.count == 0
    assert split_events.source_segment_ids.shape[0] == 0
    assert split_events.x.shape[0] == 0
    assert split_events.runtime_selection.selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_gpu_split_events_preserve_polygon_hole_ring_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )
    vertical = LineString([(3, -1), (3, 7)])

    left = from_shapely_geometries([donut])
    right = from_shapely_geometries([vertical])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    left_mask = split_events.source_side == 1
    assert {int(value) for value in split_events.ring_indices[left_mask].tolist()} == {0, 1}
    assert atomic_edges.count > 0
    assert all(event.kind.value != "materialization" for event in left.diagnostics)
    assert all(event.kind.value != "materialization" for event in right.diagnostics)


@pytest.mark.gpu
def test_gpu_atomic_edges_use_dense_metadata_lookup_without_sort_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    split_module = importlib.import_module("vibespatial.overlay.split")

    left = from_shapely_geometries([Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])])
    right = from_shapely_geometries([Polygon([(2, -1), (5, -1), (5, 3), (2, 3), (2, -1)])])
    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)

    monkeypatch.setattr(
        split_module,
        "sort_pairs",
        lambda *args, **kwargs: pytest.fail("build_gpu_atomic_edges should not sort source ids"),
    )

    atomic_edges = build_gpu_atomic_edges(split_events)

    assert atomic_edges.count > 0
    assert atomic_edges.row_indices.shape[0] == atomic_edges.count


@pytest.mark.gpu
def test_gpu_atomic_edges_collapse_duplicate_overlap_segments() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 5, 2, 7)])
    right = from_shapely_geometries([box(1, 5, 3, 7)])

    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)

    forward_mask = atomic_edges.direction == 0
    forward_segments = np.column_stack(
        (
            atomic_edges.src_x[forward_mask],
            atomic_edges.src_y[forward_mask],
            atomic_edges.dst_x[forward_mask],
            atomic_edges.dst_y[forward_mask],
        )
    )
    assert atomic_edges.count == 20
    assert np.unique(np.round(forward_segments, 12), axis=0).shape[0] == forward_segments.shape[0]


@pytest.mark.gpu
def test_grouped_right_right_split_events_use_original_right_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 10, 10)])
    right = from_shapely_geometries([box(2, 2, 7, 7), box(5, 0, 9, 9)])

    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=False,
        right_geometry_source_rows=np.asarray([0, 0], dtype=np.int32),
    )

    base_endpoint_count = 2 * (split_events.left_segment_count + split_events.right_segment_count)
    right_extra = (
        (split_events.source_side == 2)
        & (split_events.t > 0.0)
        & (split_events.t < 1.0)
    )

    assert split_events.count > base_endpoint_count
    assert right_extra.any()


def test_face_metrics_kernel_block_size_rounds_down_to_power_of_two() -> None:
    assert _largest_power_of_two_block_size(256) == 256
    assert _largest_power_of_two_block_size(224) == 128
    assert _largest_power_of_two_block_size(192) == 128
    assert _largest_power_of_two_block_size(1) == 1
