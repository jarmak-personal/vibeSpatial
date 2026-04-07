from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from vibespatial import (
    ExecutionMode,
    build_gpu_atomic_edges,
    build_gpu_half_edge_graph,
    build_gpu_overlay_faces,
    build_gpu_split_events,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.runtime.hotpath_trace import get_hotpath_trace, reset_hotpath_trace


def _build_face_table(left_geometries, right_geometries):
    left = from_shapely_geometries(left_geometries)
    right = from_shapely_geometries(right_geometries)
    split_events = build_gpu_split_events(left, right, dispatch_mode=ExecutionMode.GPU)
    atomic_edges = build_gpu_atomic_edges(split_events)
    graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(left, right, half_edge_graph=graph)
    return left, right, split_events, atomic_edges, graph, faces


@pytest.mark.gpu
def test_gpu_half_edge_graph_and_face_labels_are_deterministic_for_overlapping_rectangles() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    right_polygon = Polygon([(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)])

    left, right, _, atomic_edges, graph, faces = _build_face_table([left_polygon], [right_polygon])
    _, _, _, _, graph_repeat, faces_repeat = _build_face_table([left_polygon], [right_polygon])

    assert atomic_edges.runtime_selection.selected is ExecutionMode.GPU
    assert graph.node_count == 10
    assert graph.edge_count == atomic_edges.count
    assert graph.device_state is not None
    assert faces.device_state is not None
    assert np.array_equal(graph.next_edge_ids, graph_repeat.next_edge_ids)
    assert np.array_equal(faces.face_offsets, faces_repeat.face_offsets)
    assert np.array_equal(faces.face_edge_ids, faces_repeat.face_edge_ids)

    bounded = faces.bounded_mask.astype(bool, copy=False)
    positive_areas = np.sort(np.round(faces.signed_area[bounded], 6))
    assert np.array_equal(positive_areas, np.asarray([4.0, 12.0, 12.0], dtype=np.float64))

    labels = {
        (int(left_value), int(right_value))
        for left_value, right_value, bounded_value in zip(
            faces.left_covered,
            faces.right_covered,
            faces.bounded_mask,
            strict=True,
        )
        if int(bounded_value) != 0
    }
    assert labels == {(1, 0), (1, 1), (0, 1)}
    assert np.all(graph.next_edge_ids >= 0)


@pytest.mark.gpu
def test_gpu_face_labeling_preserves_hole_semantics() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        holes=[[(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]],
    )
    right_polygon = Polygon([(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)])

    _, _, _, atomic_edges, graph, faces = _build_face_table([donut], [right_polygon])

    assert graph.edge_count == atomic_edges.count
    bounded = faces.bounded_mask.astype(bool, copy=False)
    labels = {
        (int(left_value), int(right_value))
        for left_value, right_value, bounded_value in zip(
            faces.left_covered,
            faces.right_covered,
            faces.bounded_mask,
            strict=True,
        )
        if int(bounded_value) != 0
    }
    assert (1, 1) in labels
    assert (0, 1) in labels
    assert np.all(np.abs(faces.signed_area[bounded]) > 1e-12)


@pytest.mark.gpu
def test_gpu_face_labels_include_overlap_band_for_collinear_rectangle_overlap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_polygon = Polygon([(0, 5), (2, 5), (2, 7), (0, 7), (0, 5)])
    right_polygon = Polygon([(1, 5), (3, 5), (3, 7), (1, 7), (1, 5)])

    _, _, _, atomic_edges, graph, faces = _build_face_table([left_polygon], [right_polygon])

    assert graph.edge_count == atomic_edges.count
    bounded = faces.bounded_mask.astype(bool, copy=False)
    positive_areas = np.sort(np.round(faces.signed_area[bounded], 6))
    assert np.array_equal(positive_areas, np.asarray([2.0, 2.0, 2.0], dtype=np.float64))

    labels = {
        (int(left_value), int(right_value))
        for left_value, right_value, bounded_value in zip(
            faces.left_covered,
            faces.right_covered,
            faces.bounded_mask,
            strict=True,
        )
        if int(bounded_value) != 0
    }
    assert labels == {(1, 0), (1, 1), (0, 1)}


@pytest.mark.gpu
def test_gpu_face_coverage_trace_accounts_for_mixed_family_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    left_multi = MultiPolygon(
        [
            Polygon([(5, 0), (7, 0), (7, 2), (5, 2), (5, 0)]),
            Polygon([(5, 3), (7, 3), (7, 5), (5, 5), (5, 3)]),
        ]
    )
    right_polygon = Polygon([(2, -1), (6, -1), (6, 6), (2, 6), (2, -1)])

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    _build_face_table([left_polygon, left_multi], [right_polygon])
    trace_names = [stage.name for stage in get_hotpath_trace()]

    assert "overlay.faces.coverage.left.mixed_family_overlap" in trace_names
