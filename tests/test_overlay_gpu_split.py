from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from vibespatial import (
    ExecutionMode,
    build_gpu_atomic_edges,
    build_gpu_split_events,
    from_shapely_geometries,
    has_gpu_runtime,
)


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
    assert overlap_edges.count == 8
    assert np.allclose(overlap_events.x[:3], [0.0, 2.0, 5.0])
    assert np.allclose(overlap_events.x[3:], [2.0, 5.0, 7.0])
    assert np.allclose(overlap_events.y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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
