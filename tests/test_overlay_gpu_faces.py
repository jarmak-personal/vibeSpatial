from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from vibespatial import (
    ExecutionMode,
    RuntimeSelection,
    build_gpu_atomic_edges,
    build_gpu_half_edge_graph,
    build_gpu_overlay_faces,
    build_gpu_split_events,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.overlay.types import AtomicEdgeDeviceState, AtomicEdgeTable
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
    containment_position = assemble_source.index(
        'kernels["count_boundary_ring_containment_depth"]',
    )
    assert containment_position < branch_start
    assert 'kernels["locate_boundary_ring_group_spans"]' in assemble_source
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
    assert "scatter_node_offsets" in graph_source
    assert "merge_node_edges_robust_pass" in graph_source
    assert "sort_node_edges_robust" not in graph_source
    assert "cp.maximum.accumulate" not in graph_source
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


@pytest.mark.gpu
def test_half_edge_radial_merge_handles_high_degree_node() -> None:
    cp = pytest.importorskip("cupy")

    ray_count = 1024
    angles = np.arange(ray_count, dtype=np.float64) * (2.0 * np.pi / ray_count)
    outer_x = np.cos(angles)
    outer_y = np.sin(angles)
    edge_count = ray_count * 2
    src_x = np.empty(edge_count, dtype=np.float64)
    src_y = np.empty(edge_count, dtype=np.float64)
    dst_x = np.empty(edge_count, dtype=np.float64)
    dst_y = np.empty(edge_count, dtype=np.float64)
    src_x[0::2] = 0.0
    src_y[0::2] = 0.0
    dst_x[0::2] = outer_x
    dst_y[0::2] = outer_y
    src_x[1::2] = outer_x
    src_y[1::2] = outer_y
    dst_x[1::2] = 0.0
    dst_y[1::2] = 0.0
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    atomic_edges = AtomicEdgeTable(
        left_segment_count=ray_count,
        right_segment_count=0,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="high-degree radial merge test",
        ),
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=edge_ids // cp.int32(2),
            direction=cp.where((edge_ids & 1) == 0, cp.int8(1), cp.int8(-1)),
            src_x=cp.asarray(src_x),
            src_y=cp.asarray(src_y),
            dst_x=cp.asarray(dst_x),
            dst_y=cp.asarray(dst_y),
            row_indices=cp.zeros(edge_count, dtype=cp.int32),
            part_indices=cp.zeros(edge_count, dtype=cp.int32),
            ring_indices=cp.zeros(edge_count, dtype=cp.int32),
            source_side=cp.ones(edge_count, dtype=cp.int8),
            source_membership=cp.ones(edge_count, dtype=cp.uint8),
            tangent_x=cp.asarray(dst_x - src_x),
            tangent_y=cp.asarray(dst_y - src_y),
        ),
        _count=edge_count,
    )

    graph = build_gpu_half_edge_graph(atomic_edges)
    actual = cp.asnumpy(graph.device_state.next_edge_ids)
    expected = np.empty(edge_count, dtype=np.int32)
    expected[0::2] = np.arange(1, edge_count, 2, dtype=np.int32)
    expected[1] = edge_count - 2
    expected[3::2] = np.arange(0, edge_count - 2, 2, dtype=np.int32)

    assert np.array_equal(actual, expected)


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
def test_adjacent_ulp_triangle_has_exact_direct_and_public_intersection_labels() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    x0 = np.float64(1.0)
    x1 = np.nextafter(x0, np.float64(np.inf))
    triangle = Polygon([(x0, 0.0), (x1, 0.0), (x0, 1.0), (x0, 0.0)])
    container = shapely.box(0.0, -1.0, 2.0, 2.0)
    assert triangle.area > 0.0
    assert np.nextafter(x0, x1) == x1

    _, _, _, _, _, faces = _build_face_table([triangle], [container])
    overlap = (
        faces.bounded_mask.astype(bool, copy=False)
        & faces.left_covered.astype(bool, copy=False)
        & faces.right_covered.astype(bool, copy=False)
    )
    assert np.count_nonzero(overlap) == 1
    assert faces.signed_area[overlap][0] == triangle.area

    result = _overlay_owned(
        from_shapely_geometries([triangle], residency=Residency.DEVICE),
        from_shapely_geometries([container], residency=Residency.DEVICE),
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    ).to_shapely()[0]
    assert shapely.equals(result, triangle)


@pytest.mark.gpu
def test_reversed_shell_and_hole_orientation_preserves_exact_coverage() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    shell = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
    reversed_donut = Polygon(shell=list(reversed(shell)), holes=[list(reversed(hole))])
    container = shapely.box(-1.0, -1.0, 11.0, 11.0)

    _, _, _, atomic_edges, _, faces = _build_face_table(
        [reversed_donut],
        [container],
    )
    assert np.any(atomic_edges.device_state.left_coverage_delta != 0)
    bounded_labels = {
        (int(left), int(right))
        for left, right, bounded in zip(
            faces.left_covered,
            faces.right_covered,
            faces.bounded_mask,
            strict=True,
        )
        if bounded
    }
    assert bounded_labels == {(0, 1), (1, 1)}

    result = _overlay_owned(
        from_shapely_geometries([reversed_donut], residency=Residency.DEVICE),
        from_shapely_geometries([container], residency=Residency.DEVICE),
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    ).to_shapely()[0]
    assert shapely.equals(result, reversed_donut)


@pytest.mark.gpu
def test_duplicate_minimum_public_overlay_preserves_exact_ring_transition() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.overlay.gpu import _overlay_owned

    duplicate_minimum = Polygon(
        [(0, 0), (0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    )
    container = shapely.box(-1.0, -1.0, 3.0, 3.0)
    _, _, _, atomic_edges, _, faces = _build_face_table(
        [duplicate_minimum],
        [container],
    )

    assert np.any(atomic_edges.device_state.left_coverage_delta != 0)
    assert np.array_equal(
        faces.bounded_mask,
        (faces.cycle_orientation > 0).astype(np.int8),
    )
    result = _overlay_owned(
        from_shapely_geometries(
            [duplicate_minimum],
            residency=Residency.DEVICE,
        ),
        from_shapely_geometries([container], residency=Residency.DEVICE),
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    ).to_shapely()[0]

    assert shapely.equals(result, duplicate_minimum)


@pytest.mark.gpu
def test_nested_disconnected_shell_hole_island_uses_exact_component_containment() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.overlay.gpu import _overlay_owned

    outer = Polygon(
        shell=[(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
        holes=[[(4, 4), (16, 4), (16, 16), (4, 16), (4, 4)]],
    )
    island = shapely.box(8.0, 8.0, 12.0, 12.0)
    nested = MultiPolygon([outer, island])
    container = shapely.box(-1.0, -1.0, 21.0, 21.0)

    producer_stream = cp.cuda.Stream(non_blocking=True)
    with producer_stream:
        _, _, _, _, _, faces = _build_face_table([nested], [container])
    producer_stream.synchronize()
    bounded = faces.bounded_mask.astype(bool, copy=False)
    left_areas = np.sort(faces.signed_area[bounded & (faces.left_covered != 0)])
    assert np.array_equal(left_areas, np.asarray([16.0, 400.0], dtype=np.float64))

    result = _overlay_owned(
        from_shapely_geometries([nested], residency=Residency.DEVICE),
        from_shapely_geometries([container], residency=Residency.DEVICE),
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    ).to_shapely()[0]
    assert shapely.equals(result, nested)


@pytest.mark.gpu
def test_dual_work_queue_labels_long_diameter_connected_strip_faces() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    strip_count = 256
    horizontal = shapely.box(0.0, 0.0, float(strip_count * 2), 2.0)
    vertical_parts = [
        shapely.box(float(index * 2), -1.0, float(index * 2 + 1), 3.0)
        for index in range(strip_count)
    ]
    vertical = MultiPolygon(vertical_parts)
    _, _, _, _, _, faces = _build_face_table([horizontal], [vertical])

    assert faces.face_count > strip_count
    overlap = (
        faces.bounded_mask.astype(bool, copy=False)
        & faces.left_covered.astype(bool, copy=False)
        & faces.right_covered.astype(bool, copy=False)
    )
    assert np.count_nonzero(overlap) == strip_count
    assert faces.signed_area[overlap].sum() == strip_count * 2.0


@pytest.mark.gpu
def test_dual_queue_and_containment_launch_shapes_are_valid_above_65535_faces() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )
    from vibespatial.overlay.gpu import _overlay_face_walk_kernels

    face_count = 65_536
    runtime = get_cuda_runtime()
    kernels = _overlay_face_walk_kernels()
    ptr = runtime.pointer
    orientation = cp.full(face_count, -1, dtype=cp.int8)
    queue = cp.empty(face_count, dtype=cp.int32)
    queue_head = cp.zeros(1, dtype=cp.int32)
    queue_tail = cp.zeros(1, dtype=cp.int32)
    queue_ready = cp.zeros(face_count, dtype=cp.int32)
    pending = cp.zeros(1, dtype=cp.int32)
    left_winding = cp.full(face_count, np.iinfo(np.int32).min, dtype=cp.int32)
    right_winding = cp.full_like(left_winding, np.iinfo(np.int32).min)
    face_component = cp.full(face_count, -1, dtype=cp.int32)
    face_offsets = cp.arange(face_count + 1, dtype=cp.int32)
    face_edge_ids = cp.arange(face_count, dtype=cp.int32)
    edge_face_ids = cp.arange(face_count, dtype=cp.int32)
    zero_delta = cp.zeros(face_count, dtype=cp.int32)

    init_grid, init_block = runtime.launch_config(
        kernels["initialize_dual_face_queue"],
        face_count,
    )
    runtime.launch(
        kernels["initialize_dual_face_queue"],
        grid=init_grid,
        block=init_block,
        params=(
            (
                ptr(orientation),
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(queue),
                ptr(queue_tail),
                ptr(queue_ready),
                ptr(pending),
                ptr(left_winding),
                ptr(right_winding),
                ptr(face_component),
                face_count,
            ),
            (KERNEL_PARAM_PTR,) * 10 + (KERNEL_PARAM_I32,),
        ),
    )
    queue_grid, queue_block = runtime.launch_config(
        kernels["propagate_dual_face_queue"],
        face_count,
    )
    assert queue_grid[0] > 1
    assert queue_grid[1:] == (1, 1)
    runtime.launch(
        kernels["propagate_dual_face_queue"],
        grid=queue_grid,
        block=queue_block,
        params=(
            (
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(edge_face_ids),
                ptr(zero_delta),
                ptr(zero_delta),
                ptr(queue),
                ptr(queue_head),
                ptr(queue_tail),
                ptr(queue_ready),
                ptr(pending),
                ptr(left_winding),
                ptr(right_winding),
                ptr(face_component),
                face_count,
                face_count,
            ),
            (KERNEL_PARAM_PTR,) * 13 + (KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    assert int(cp.asnumpy(queue_tail)[0]) == face_count
    assert int(cp.asnumpy(queue_head)[0]) == face_count
    assert int(cp.asnumpy(pending)[0]) == 0

    indexed_face_count = face_count + 1
    root_faces = cp.arange(face_count, dtype=cp.int32)
    candidate_faces = cp.full(face_count, -1, dtype=cp.int32)
    candidate_faces[0] = face_count
    indexed_offsets = cp.arange(indexed_face_count + 1, dtype=cp.int32)
    indexed_edges = cp.arange(indexed_face_count, dtype=cp.int32)
    indexed_x = cp.full(indexed_face_count, 2.0, dtype=cp.float64)
    indexed_y = cp.zeros(indexed_face_count, dtype=cp.float64)
    indexed_bounds = cp.zeros(indexed_face_count * 4, dtype=cp.float64)
    indexed_bounds[face_count * 4 : face_count * 4 + 4] = cp.asarray(
        [0.0, -1.0, 1.0, 1.0],
        dtype=cp.float64,
    )
    interval_max_x = cp.full(face_count * 2, -cp.inf, dtype=cp.float64)
    interval_max_x[face_count] = 1.0
    level_start = face_count >> 1
    while level_start:
        interval_max_x[level_start : level_start * 2] = cp.maximum(
            interval_max_x[level_start * 2 : level_start * 4 : 2],
            interval_max_x[level_start * 2 + 1 : level_start * 4 : 2],
        )
        level_start >>= 1
    indexed_components = cp.arange(indexed_face_count, dtype=cp.int32)
    source_rows = cp.zeros(indexed_face_count, dtype=cp.int32)
    left_baseline = cp.zeros(face_count, dtype=cp.int32)
    right_baseline = cp.zeros(face_count, dtype=cp.int32)
    component_depth = cp.zeros(face_count, dtype=cp.int32)
    containment_grid = (face_count, 1, 1)
    containment_block = (256, 1, 1)
    assert containment_grid[0] > 65_535
    runtime.launch(
        kernels["reduce_indexed_component_containment"],
        grid=containment_grid,
        block=containment_block,
        params=(
            (
                ptr(root_faces),
                ptr(candidate_faces),
                ptr(interval_max_x),
                ptr(indexed_offsets),
                ptr(indexed_edges),
                ptr(indexed_bounds),
                ptr(indexed_x),
                ptr(indexed_y),
                ptr(source_rows),
                ptr(indexed_components),
                ptr(None),
                ptr(None),
                ptr(left_baseline),
                ptr(right_baseline),
                ptr(component_depth),
                face_count,
                face_count,
                0,
                0,
            ),
            (KERNEL_PARAM_PTR,) * 15 + (KERNEL_PARAM_I32,) * 4,
        ),
    )
    assert not bool(cp.any(component_depth))


def test_exact_face_labeling_has_no_probe_refinement_or_host_convergence() -> None:
    root = Path(__file__).resolve().parents[1]
    faces_source = (root / "src/vibespatial/overlay/faces.py").read_text()
    kernels_source = (root / "src/vibespatial/overlay/gpu_kernels.py").read_text()
    types_source = (root / "src/vibespatial/overlay/types.py").read_text()
    split_source = (root / "src/vibespatial/overlay/split.py").read_text()
    walk_start = kernels_source.index("_OVERLAY_FACE_WALK_KERNEL_SOURCE")
    walk_end = kernels_source.index("_OVERLAY_FACE_WALK_KERNEL_NAMES", walk_start)
    walk_source = kernels_source[walk_start:walk_end]
    propagation_start = faces_source.index("def _gpu_propagate_face_coverage(")
    propagation_end = faces_source.index("\ndef _overlay_face_selection_mask_gpu", propagation_start)
    propagation_source = faces_source[propagation_start:propagation_end]
    carrier_start = faces_source.index(
        "def _build_indexed_component_containment_device_state(",
    )
    carrier_end = faces_source.index("\ndef _gpu_propagate_face_coverage(", carrier_start)
    carrier_source = faces_source[carrier_start:carrier_end]
    graph_source = (root / "src/vibespatial/overlay/graph.py").read_text()
    split_kernel_start = kernels_source.index("derive_source_ring_transition_signs(")
    split_kernel_end = kernels_source.index("\n}\n\"\"\"", split_kernel_start)
    split_kernel_source = kernels_source[split_kernel_start:split_kernel_end]

    assert "compute_face_sample_points" not in kernels_source
    assert "count_boundary_face_nesting_depth" not in kernels_source
    assert "collapsed_triangle" not in faces_source
    assert "area_epsilon" not in walk_source
    assert "copy_device_to_host(" not in propagation_source
    assert "copy_device_to_host_async(" not in carrier_source
    assert "overlay indexed component relation allocation fence" not in carrier_source
    assert "producer_stream.synchronize()" not in carrier_source
    assert "runtime.synchronize()" not in propagation_source
    assert "grid=(1, 1, 1)" not in propagation_source
    assert "accumulate_component_containment_baseline" not in kernels_source
    assert "blockIdx.y" not in carrier_source
    assert "_build_indexed_component_containment_device_state" in propagation_source
    assert "reduce_indexed_component_containment" in carrier_source
    assert "select_indexed_component_containment_parent" in carrier_source
    assert "relation_" not in carrier_source
    assert "count_indexed_component_containment_candidates" not in kernels_source
    assert "scatter_indexed_component_containment_candidates" not in kernels_source
    assert "refine_component_containment_relation_exact" not in kernels_source
    assert "reduce_component_containment_segments" not in kernels_source
    assert "reduce_component_containment_nesting" not in kernels_source
    containment_type = types_source[
        types_source.index("class IndexedComponentContainmentDeviceState:") :
        types_source.index("\n\n@dataclass", types_source.index("class IndexedComponentContainmentDeviceState:"))
    ]
    assert "relation_" not in containment_type
    assert "face_capacity: int" in containment_type
    assert "interval_max_x: DeviceArray" in containment_type
    assert "interval_max_x" in carrier_source
    assert "root_block = (256, 1, 1)" in carrier_source
    assert "split_depth = tree_depth < 8 ? tree_depth : 8" in kernels_source
    assert "component_depth[candidate_component] == target_depth" in kernels_source
    assert "cycle_orientation > 0" in graph_source
    assert "signed_area > 0.0" not in graph_source
    assert "candidate_orientation == 0" in split_kernel_source
    assert "source_x0[prior] != cx" in split_kernel_source
    assert "max_passes" not in split_source
    assert "_OVERLAY_FACE_LABEL_KERNEL_SOURCE" not in kernels_source
