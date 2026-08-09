from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
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
from vibespatial.runtime.residency import Residency


def test_overlay_coordinate_gather_requires_physical_capacity() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "assemble.py"
    ).read_text()
    gather_start = source.index("def _gather_coords_vectorised(")
    expand_start = source.index("def _expand_by_counts(", gather_start)
    expand_end = source.index("\n\ndef ", expand_start + 1)
    gather_source = source[gather_start:expand_start]
    expand_source = source[expand_start:expand_end]

    assert "total_capacity: int" in gather_source
    assert "_device_int_scalar(" not in gather_source
    assert "total: int" in expand_source
    assert "_device_int_scalar(" not in expand_source
    assert "coordinate-gather total-coords allocation fence" not in source
    assert "expand-by-counts total allocation fence" not in source


def test_admitted_gpu_face_assembly_has_no_host_reconstruction_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "gpu.py"
    ).read_text()
    start = source.index("def _materialize_overlay_execution_plan(")
    end = source.index("\n\ndef _expand_group_pair_positions", start)
    function_source = source[start:end]

    assert "_build_polygon_output_from_faces_gpu(" in function_source
    assert "_build_polygon_output_from_faces(" not in function_source
    assert "_select_overlay_face_indices_gpu(" not in function_source
    assert "overlay gpu CPU fallback selected-face indices export" not in function_source
    assert "except Exception" not in function_source


def test_row_isolated_face_output_stays_capacity_backed() -> None:
    root = Path(__file__).resolve().parents[1]
    assemble_source = (root / "src/vibespatial/overlay/assemble.py").read_text()
    kernel_source = (root / "src/vibespatial/overlay/gpu_kernels.py").read_text()

    branch_start = assemble_source.index("if preserve_row_count is not None:")
    branch_end = assemble_source.index(
        "else:\n            # Dynamic public cardinality",
        branch_start,
    )
    capacity_branch = assemble_source[branch_start:branch_end]

    assert "cp.flatnonzero(" not in capacity_branch
    assert "_device_int_scalar(" not in capacity_branch
    sibling_position = assemble_source.index(
        'kernels["count_sibling_hole_depth"]',
    )
    assert sibling_position < branch_start
    assert "d_explicit_polygon_output_rows" in capacity_branch
    assert "d_explicit_polygon_active" in capacity_branch
    assert "excluded_rings" not in assemble_source
    assert "mark_collapsed_excluded_rings" not in kernel_source
    assert "insert_output_hole_signatures" not in kernel_source
    assert "scatter_output_holes" in kernel_source


def test_half_edge_graph_uses_source_twins_and_stable_radix_passes() -> None:
    graph_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "graph.py"
    ).read_text()

    assert "build_radial_successors" in graph_source
    assert "edge_positions = cp.empty" not in graph_source
    assert "twin_positions" not in graph_source
    assert "cp.concatenate((device.src_x, device.dst_x))" not in graph_source
    assert "cp.concatenate((device.src_y, device.dst_y))" not in graph_source
    assert "_stable_radix_order_pass" in graph_source
    assert "strategy=PairSortStrategy.RADIX" in graph_source
    assert "unsupported device radix key dtype" in graph_source
    assert "normalized_key.astype(cp.uint64" in graph_source
    assert "cp.lexsort(cp.stack" not in graph_source
    assert "cycle_rank_pointer_jump" not in graph_source
    assert "packed_key = face_id" not in graph_source
    assert "face_edge_ids = sorted_edge_ids[src_pos]" in graph_source
    assert "node_x=None" in graph_source
    assert "sorted_edge_ids=None" in graph_source
    assemble_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "assemble.py"
    ).read_text()
    kernel_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "gpu_kernels.py"
    ).read_text()
    assert "list_rank_within_cycle" not in assemble_source
    assert "list_rank_within_cycle" not in kernel_source
    assert "const long long* __restrict__ next_edge_ids" not in kernel_source
    assert "d_edge_face_ids" not in assemble_source
    assert "edge_to_compact" not in assemble_source
    assert "cp.full(edge_count, -1" not in assemble_source
    assert "_stable_radix_order_pass" in assemble_source
    assert "cp.lexsort" not in assemble_source
    assert "cp.stack" not in assemble_source
    assert "scatter_edge_face_selection" in kernel_source
    assert "assign_holes_to_exteriors" in kernel_source
    assert "scatter_boundary_ring_coordinates" in kernel_source


def test_radial_successor_relation_uses_clockwise_predecessor_of_twin() -> None:
    # Groups are already sorted by (source node, exact fp64 angle, edge id).
    sorted_edge_ids = np.asarray([4, 1, 7, 0, 3, 6, 2, 5], dtype=np.int32)
    source_node_ids = np.empty(8, dtype=np.int32)
    source_node_ids[sorted_edge_ids] = np.asarray(
        [0, 0, 0, 1, 1, 2, 2, 2],
        dtype=np.int32,
    )

    successors = np.empty(8, dtype=np.int32)
    group_start = 0
    while group_start < sorted_edge_ids.size:
        node = source_node_ids[sorted_edge_ids[group_start]]
        group_end = group_start + 1
        while (
            group_end < sorted_edge_ids.size
            and source_node_ids[sorted_edge_ids[group_end]] == node
        ):
            group_end += 1
        group = sorted_edge_ids[group_start:group_end]
        for position, outgoing_edge in enumerate(group):
            successors[outgoing_edge ^ np.int32(1)] = group[position - 1]
        group_start = group_end

    assert np.array_equal(
        successors,
        np.asarray([4, 3, 0, 6, 2, 7, 1, 5], dtype=np.int32),
    )


def test_compact_boundary_successors_preserve_full_edge_provenance() -> None:
    boundary_edge_ids = np.asarray([1, 4, 8, 11, 14], dtype=np.int32)
    boundary_next_full = np.asarray([8, 14, 1, 11, 4], dtype=np.int32)

    compact_next = np.searchsorted(boundary_edge_ids, boundary_next_full).astype(
        np.int32,
        copy=False,
    )

    assert np.array_equal(compact_next, np.asarray([2, 4, 0, 3, 1], dtype=np.int32))
    assert np.array_equal(boundary_edge_ids[compact_next], boundary_next_full)


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
def test_gpu_centered_face_metrics_preserve_translated_positive_area_sliver() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    origin = 100_000_000.0
    left_polygon = shapely.box(
        origin,
        origin,
        origin + 0.01,
        origin + 0.01,
    )
    right_polygon = shapely.box(
        origin + 0.009999,
        origin,
        origin + 0.02,
        origin + 0.01,
    )
    expected = left_polygon.intersection(right_polygon)
    assert expected.area > 0.0

    left, right, _, _, _, faces = _build_face_table(
        [left_polygon],
        [right_polygon],
    )
    overlap_faces = (
        faces.bounded_mask.astype(bool, copy=False)
        & faces.left_covered.astype(bool, copy=False)
        & faces.right_covered.astype(bool, copy=False)
    )

    assert int(np.count_nonzero(overlap_faces)) == 1
    assert faces.signed_area[overlap_faces][0] == pytest.approx(
        expected.area,
        rel=1.0e-12,
        abs=0.0,
    )

    result = _overlay_owned(
        from_shapely_geometries([left_polygon], residency=Residency.DEVICE),
        from_shapely_geometries([right_polygon], residency=Residency.DEVICE),
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )
    assert result.row_count == 1
    actual = result.to_shapely()[0]
    assert actual.area == pytest.approx(expected.area, rel=1.0e-12, abs=0.0)
    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_gpu_row_isolated_face_assembly_preserves_holes_and_public_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    left_geometries = [
        Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]],
        ),
        Polygon(
            shell=[(20, 0), (30, 0), (30, 10), (20, 10), (20, 0)],
            holes=[[(23, 3), (27, 3), (27, 7), (23, 7), (23, 3)]],
        ),
    ]
    right_geometries = [
        Polygon([(-1, -1), (11, -1), (11, 11), (-1, 11), (-1, -1)]),
        Polygon([(19, -1), (31, -1), (31, 11), (19, 11), (19, -1)]),
    ]
    left = from_shapely_geometries(left_geometries, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geometries, residency=Residency.DEVICE)

    result = _overlay_owned(
        left,
        right,
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    assert result.row_count == 2
    actual = result.to_shapely()
    assert [len(geometry.interiors) for geometry in actual] == [1, 1]
    assert all(
        shapely.normalize(got).equals_exact(
            shapely.normalize(want),
            tolerance=1.0e-9,
        )
        for got, want in zip(actual, left_geometries, strict=True)
    )


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


@pytest.mark.gpu
def test_gpu_face_coverage_trace_uses_same_row_multipolygon_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    right_multi = MultiPolygon(
        [
            Polygon([(1, -1), (3, -1), (3, 5), (1, 5), (1, -1)]),
            Polygon([(6, -1), (8, -1), (8, 5), (6, 5), (6, -1)]),
        ]
    )
    right = from_shapely_geometries(
        [right_multi, right_multi],
        residency=Residency.DEVICE,
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = _overlay_owned(
        left,
        right,
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )
    trace_names = [stage.name for stage in get_hotpath_trace()]

    assert result.row_count == left.row_count
    assert "overlay.faces.coverage.right.multipolygon_same_row" in trace_names


@pytest.mark.gpu
def test_gpu_face_coverage_trace_uses_warp_for_indexed_polygon_logical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.overlay.faces import _gpu_label_face_coverage

    left = from_shapely_geometries(
        [Polygon([(100, 100), (104, 100), (104, 104), (100, 104), (100, 100)])],
        residency=Residency.DEVICE,
    )
    right_base = from_shapely_geometries(
        [
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]),
            Polygon([(10, 0), (13, 0), (14, 2), (12, 4), (10, 3), (10, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    right = right_base._device_indexed_take(
        cp.asarray([0, 1] * 16, dtype=cp.int64),
    )

    assert right.is_indexed_view

    label_x = cp.asarray([1.0, 12.0, 6.0], dtype=cp.float64)
    label_y = cp.asarray([1.0, 2.0, 6.0], dtype=cp.float64)
    face_source_rows = cp.zeros(3, dtype=cp.int32)
    right_source_rows = cp.zeros(right.row_count, dtype=cp.int32)

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    _left_covered, right_covered = _gpu_label_face_coverage(
        left,
        right,
        label_x,
        label_y,
        3,
        face_source_rows=face_source_rows,
        right_geometry_source_rows=right_source_rows,
    )
    trace_names = [stage.name for stage in get_hotpath_trace()]

    assert np.array_equal(cp.asnumpy(right_covered), np.asarray([1, 1, 0], dtype=np.int8))
    assert "overlay.faces.coverage.right.polygon_logical_rows_warp" in trace_names


@pytest.mark.gpu
def test_gpu_face_coverage_uses_coordinate_cooperative_indexed_and_broadcast_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.overlay.faces import _gpu_label_face_coverage

    left = from_shapely_geometries(
        [Polygon([(100, 100), (104, 100), (104, 104), (100, 104), (100, 100)])],
        residency=Residency.DEVICE,
    )
    complex_polygon = shapely.Point(0.0, 0.0).buffer(10.0, quad_segs=512)
    right_base = from_shapely_geometries(
        [complex_polygon],
        residency=Residency.DEVICE,
    )
    right_indexed = right_base._device_indexed_take(
        cp.zeros(16, dtype=cp.int64),
    )
    label_x = cp.asarray([0.0, 9.0, 11.0], dtype=cp.float64)
    label_y = cp.zeros(3, dtype=cp.float64)
    face_source_rows = cp.zeros(3, dtype=cp.int32)

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    _left_covered, indexed_covered = _gpu_label_face_coverage(
        left,
        right_indexed,
        label_x,
        label_y,
        3,
        face_source_rows=face_source_rows,
        right_geometry_source_rows=cp.zeros(right_indexed.row_count, dtype=cp.int32),
    )
    indexed_trace_names = [stage.name for stage in get_hotpath_trace()]

    assert np.array_equal(
        cp.asnumpy(indexed_covered),
        np.asarray([1, 1, 0], dtype=np.int8),
    )
    assert (
        "overlay.faces.coverage.right.polygon_logical_rows_block"
        in indexed_trace_names
    )

    reset_hotpath_trace()
    _left_covered, broadcast_covered = _gpu_label_face_coverage(
        left,
        right_base,
        label_x,
        label_y,
        3,
        face_source_rows=face_source_rows,
        right_geometry_broadcast=True,
    )
    broadcast_trace_names = [stage.name for stage in get_hotpath_trace()]

    assert np.array_equal(
        cp.asnumpy(broadcast_covered),
        np.asarray([1, 1, 0], dtype=np.int8),
    )
    assert "overlay.faces.coverage.right.polygon_block" in broadcast_trace_names
