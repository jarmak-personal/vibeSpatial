from __future__ import annotations

import os

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from vibespatial.api import read_file
from vibespatial.constructive.binary_constructive import (
    _dispatch_overlay_gpu,
    binary_constructive_owned,
)
from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.owned import (
    build_empty_polygon_rows_device,
    from_shapely_geometries,
)
from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
from vibespatial.overlay.boundary_graph import build_polygon_output_from_boundary_segments_gpu
from vibespatial.overlay.faces import _select_overlay_face_indices_gpu, build_gpu_overlay_faces
from vibespatial.overlay.gpu import (
    _build_overlay_execution_plan,
    _materialize_overlay_execution_plan,
)
from vibespatial.overlay.graph import build_gpu_half_edge_graph
from vibespatial.overlay.host_fallback import _build_polygon_output_from_faces
from vibespatial.overlay.split import (
    build_gpu_atomic_edges,
    build_gpu_split_events,
    renode_grouped_boundary_segments_gpu,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency


def _river_lines(count: int, *, seed: int, vertices: int = 12) -> list[LineString]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1000.0, vertices)
    amplitude = 1000.0 / 8.0
    geoms: list[LineString] = []
    for offset in rng.uniform(0.0, 1000.0, count):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        coords = [
            (
                float(x),
                float(np.clip(offset + amplitude * np.sin(phase + i / 2.0), 0.0, 1000.0)),
            )
            for i, x in enumerate(xs)
        ]
        geoms.append(LineString(coords))
    return geoms


def _buffered_line_reduction_pair(
    *,
    count: int = 200,
    seed: int = 10,
    round_index: int,
    pair_index: int,
    distance: float = 10.0,
):
    buffered = shapely.buffer(np.asarray(_river_lines(count, seed=seed), dtype=object), distance)
    current = list(buffered)
    target_left = None
    target_right = None

    for reduction_round in range(round_index + 1):
        next_round = []
        for i in range(0, len(current), 2):
            if i + 1 >= len(current):
                next_round.append(current[i])
                continue
            if reduction_round == round_index and i // 2 == pair_index:
                target_left = current[i]
                target_right = current[i + 1]
                break
            got = binary_constructive_owned(
                "union",
                from_shapely_geometries([current[i]]),
                from_shapely_geometries([current[i + 1]]),
                dispatch_mode=ExecutionMode.GPU,
            ).to_shapely()[0]
            next_round.append(got)
        if target_left is not None and target_right is not None:
            return target_left, target_right
        current = next_round

    raise AssertionError(
        f"failed to locate reduction pair round={round_index} pair={pair_index}",
    )


@pytest.mark.gpu
def test_polygonize_classifies_nested_positive_face_as_hole_on_device() -> None:
    source_geometry = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    source = from_shapely_geometries([source_geometry])
    empty = build_empty_polygon_rows_device(1)
    split_events = build_gpu_split_events(
        source,
        empty,
        require_same_row=True,
        use_same_row_fast_path=True,
        same_row_single_group=True,
        same_row_span_summary=(8, 0, 0),
        include_same_side_splits=True,
    )
    atomic_edges = build_gpu_atomic_edges(split_events, isolate_rows=True)
    graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(
        source,
        empty,
        half_edge_graph=graph,
        row_isolated=True,
    )
    selected = _select_overlay_face_indices_gpu(faces, operation="polygonize")

    actual = _build_polygon_output_from_faces_gpu(graph, faces, selected).to_shapely()[0]

    assert actual.is_valid
    assert len(actual.interiors) == 1
    assert shapely.area(shapely.symmetric_difference(actual, source_geometry)) == 0.0


@pytest.mark.gpu
def test_nybb_pair_gpu_face_assembly_matches_host() -> None:
    data = os.path.join(
        os.path.dirname(__file__),
        "upstream",
        "geopandas",
        "tests",
        "data",
    )
    overlay_data = os.path.join(data, "overlay", "nybb_qgis")
    left = read_file(f"zip://{os.path.join(data, 'nybb_16a.zip')}").iloc[[4]].copy()
    right = read_file(os.path.join(overlay_data, "polydf2.shp")).iloc[[8]].copy()

    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    split_events = build_gpu_split_events(left_owned, right_owned)
    atomic_edges = build_gpu_atomic_edges(split_events)
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(left_owned, right_owned, half_edge_graph=half_edge_graph)
    selected = _select_overlay_face_indices_gpu(faces, operation="intersection")

    gpu_result = _build_polygon_output_from_faces_gpu(half_edge_graph, faces, selected)
    host_result = _build_polygon_output_from_faces(half_edge_graph, faces, selected.get())

    host_geom = host_result.to_shapely()[0]
    gpu_geom = gpu_result.to_shapely()[0]
    assert host_geom.is_valid
    assert gpu_geom.is_valid
    assert gpu_geom.geom_type == host_geom.geom_type
    assert gpu_geom.normalize().equals_exact(host_geom.normalize(), tolerance=1e-6)


@pytest.mark.gpu
def test_buffered_line_union_gpu_partition_plan_matches_host() -> None:
    buffered = shapely.buffer(np.asarray(_river_lines(200, seed=10), dtype=object), 10.0)
    left = from_shapely_geometries([buffered[14]])
    right = from_shapely_geometries([buffered[15]])
    result = binary_constructive_owned(
        "union",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    actual = result.to_shapely()[0]
    expected = shapely.union(buffered[14], buffered[15])

    assert actual.is_valid
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.gpu
def test_buffered_line_union_gpu_repairs_invalid_batched_rows() -> None:
    buffered = shapely.buffer(np.asarray(_river_lines(200, seed=10), dtype=object), 10.0)
    left = from_shapely_geometries([buffered[13], buffered[14]])
    right = from_shapely_geometries([buffered[14], buffered[15]])
    expected = [
        shapely.union(buffered[13], buffered[14]),
        shapely.union(buffered[14], buffered[15]),
    ]

    raw_batch = _dispatch_overlay_gpu(
        "union",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )
    raw_actual = raw_batch.to_shapely()
    assert [geom.is_valid for geom in raw_actual] == [True, True]
    for got, want in zip(raw_actual, expected, strict=True):
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0, abs=1e-6)

    repaired = binary_constructive_owned(
        "union",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    actual = repaired.to_shapely()
    assert [geom.is_valid for geom in actual] == [True, True]
    for got, want in zip(actual, expected, strict=True):
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.gpu
def test_overlay_execution_plan_materializes_multiple_ops_from_one_topology() -> None:
    left = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 4.0, 4.0),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.box(2.0, 1.0, 5.0, 3.0),
        ]
    )

    plan = _build_overlay_execution_plan(left, right)
    intersection, intersection_mode = _materialize_overlay_execution_plan(
        plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
    )
    difference, difference_mode = _materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
    )
    right_difference, right_difference_mode = _materialize_overlay_execution_plan(
        plan,
        operation="right_difference",
        requested=ExecutionMode.GPU,
    )

    assert intersection_mode is ExecutionMode.GPU
    assert difference_mode is ExecutionMode.GPU
    assert right_difference_mode is ExecutionMode.GPU

    expected_intersection = shapely.intersection(left.to_shapely()[0], right.to_shapely()[0])
    expected_difference = shapely.difference(left.to_shapely()[0], right.to_shapely()[0])
    expected_right_difference = shapely.difference(right.to_shapely()[0], left.to_shapely()[0])

    actual_intersection = intersection.to_shapely()[0]
    actual_difference = difference.to_shapely()[0]
    actual_right_difference = right_difference.to_shapely()[0]

    assert actual_intersection.normalize().equals_exact(
        expected_intersection.normalize(),
        tolerance=1e-6,
    )
    assert actual_difference.normalize().equals_exact(
        expected_difference.normalize(),
        tolerance=1e-6,
    )
    assert actual_right_difference.normalize().equals_exact(
        expected_right_difference.normalize(),
        tolerance=1e-6,
    )


@pytest.mark.gpu
def test_row_isolated_union_preserves_isolated_input_interior_ring() -> None:
    left_geom = shapely.from_wkt(
        "MULTIPOLYGON (((760 400, 760 390, 770 390, 770 400, "
        "770 402.34933765836877, 769.2591269559479 402.4192905377805, "
        "760 403.1665826929884, 760 400)), ((760 420, "
        "760 415.4180185219829, 760.6889332337587 420, "
        "761.2955140267919 424.034268957292, 769.0632999422907 "
        "427.9901186102625, 764.572987725914 430, 760 432.0468872726247, "
        "760 430.6359251988752, 760 430, 760 428.3860518879259, "
        "760 420)), ((767.8200952458151 580, 770 576.2829775332077, "
        "770 580, 767.8200952458151 580)))"
    )
    right_geom = shapely.from_wkt(
        "POLYGON ((760 590, 760 587.1970902119367, "
        "762.3450433556148 589.3356789298515, 767.8200952458151 580, "
        "770 580, 770 590, 770 600, 770 610, 760 610, 760 600, "
        "760 593.1254694820173, 760 592.5510137807898, "
        "760 591.7934595572564, 760 590), "
        "(760.5097004377169 592.4651784398636, "
        "760.3128597825861 592.205768058526, "
        "761.3019467765761 591.1142949046449, "
        "760.5097004377169 592.4651784398636))"
    )
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    result = _dispatch_overlay_gpu(
        "union",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    actual = result.to_shapely()[0]
    expected = shapely.union(left_geom, right_geom)

    assert actual.is_valid
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    polygon_with_hole = [
        geom for geom in getattr(actual, "geoms", [actual]) if getattr(geom, "interiors", ())
    ]
    assert len(polygon_with_hole) == 1


@pytest.mark.gpu
def test_row_isolated_difference_preserves_sparse_multipolygon_rows() -> None:
    left = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 10.0, 10.0),
            shapely.box(20.0, 20.0, 30.0, 30.0),
            shapely.box(40.0, 40.0, 50.0, 50.0),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 10.0, 10.0),
            shapely.box(20.0, 20.0, 30.0, 30.0),
            shapely.box(44.0, 39.0, 46.0, 51.0),
        ]
    )

    result = _dispatch_overlay_gpu(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    got = result.to_shapely()
    expected = shapely.difference(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()

    assert result.row_count == 3
    assert got[0] is None or shapely.is_empty(got[0])
    assert got[1] is None or shapely.is_empty(got[1])
    assert got[2] is not None
    assert got[2].geom_type == "MultiPolygon"
    assert got[2].normalize().equals_exact(expected[2].normalize(), tolerance=1e-6)


@pytest.mark.gpu
def test_multipolygon_polygon_intersection_packs_disjoint_fragments_on_gpu() -> None:
    left_geom = shapely.MultiPolygon(
        [
            shapely.box(0.0, 0.0, 2.0, 2.0),
            shapely.box(4.0, 0.0, 6.0, 2.0),
        ]
    )
    right_geom = shapely.box(1.0, -1.0, 5.0, 3.0)
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    clear_dispatch_events()
    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = shapely.intersection(left_geom, right_geom)

    assert got is not None
    assert got.geom_type == "MultiPolygon"
    assert got.normalize().equals_exact(expected.normalize(), tolerance=1e-9)
    events = get_dispatch_events(clear=True)
    assert any(
        event.implementation == "polygon_intersection_partitioned_capacity_gpu" for event in events
    ), [(event.operation, event.implementation) for event in events]


@pytest.mark.gpu
def test_row_isolated_difference_preserves_all_empty_rows() -> None:
    left = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 2.0, 2.0),
            shapely.box(4.0, 0.0, 6.0, 2.0),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 2.0, 2.0),
            shapely.box(4.0, 0.0, 6.0, 2.0),
        ]
    )

    result = _dispatch_overlay_gpu(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    assert result.row_count == 2
    assert np.asarray(result.validity, dtype=bool).tolist() == [False, False]


@pytest.mark.gpu
def test_gpu_face_assembly_uses_runtime_launch_config(monkeypatch) -> None:
    left = from_shapely_geometries(
        [
            shapely.box(0.0, 0.0, 4.0, 4.0),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.box(2.0, 1.0, 5.0, 3.0),
        ]
    )

    split_events = build_gpu_split_events(left, right)
    atomic_edges = build_gpu_atomic_edges(split_events)
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(left, right, half_edge_graph=half_edge_graph)
    selected = _select_overlay_face_indices_gpu(faces, operation="intersection")

    runtime = get_cuda_runtime()
    original_launch_config = runtime.launch_config
    launch_config_calls = 0

    def _wrapped_launch_config(kernel, item_count, shared_mem_bytes=0):
        nonlocal launch_config_calls
        launch_config_calls += 1
        return original_launch_config(kernel, item_count, shared_mem_bytes)

    monkeypatch.setattr(runtime, "launch_config", _wrapped_launch_config)

    gpu_result = _build_polygon_output_from_faces_gpu(half_edge_graph, faces, selected)
    assert launch_config_calls >= 5
    assert gpu_result.to_shapely()[0].is_valid


@pytest.mark.gpu
def test_disconnected_overlap_intersection_gpu_matches_host() -> None:
    left_geom, right_geom = _buffered_line_reduction_pair(round_index=2, pair_index=16)
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    result = _dispatch_overlay_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    actual = result.to_shapely()[0]
    expected = shapely.intersection(left_geom, right_geom)

    assert actual.is_valid
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.gpu
def test_exact_split_parameters_preserve_projected_coordinate_sliver() -> None:
    origin_x = 360_000.0
    origin_y = 3_076_000.0
    width = float(np.spacing(origin_x))
    left_geom = shapely.box(origin_x - 10.0, origin_y, origin_x, origin_y + 10.0)
    right_geom = shapely.Polygon(
        [
            (origin_x - width, origin_y + 1.0),
            (origin_x + 10.0, origin_y + 5.0),
            (origin_x - width, origin_y + 9.0),
        ]
    )
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    split_events = build_gpu_split_events(
        left,
        right,
        require_same_row=True,
    )
    source_ids = split_events.source_segment_ids
    event_t = split_events.t
    has_subnanoparameter_split = any(
        np.any(
            (event_t[source_ids == source_id] > 0.0) & (event_t[source_ids == source_id] < 1.0e-9)
        )
        for source_id in np.unique(source_ids)
    )

    result = _dispatch_overlay_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )
    actual = result.to_shapely()[0]
    expected = shapely.intersection(left_geom, right_geom)

    assert has_subnanoparameter_split
    assert expected.geom_type == "Polygon"
    assert expected.area > 0.0
    assert actual is not None
    assert actual.geom_type == "Polygon"
    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_constructive_events_do_not_snap_distinct_nearby_coordinates() -> None:
    left_geom = shapely.from_wkt(
        "POLYGON ((-10.08426784565221 -33.51578049243587, "
        "-10.121566987362815 -33.502529627435585, "
        "-10.121566987362844 -33.50252962743558, "
        "-10.159963703906177 -33.492911750627314, "
        "-10.1715587322087 -33.48876298316187, "
        "-10.183502471459713 -33.48576230898401, "
        "-10.2432442975097 -33.46433896306574, "
        "-10.2432442975097 -33.39709974391177, "
        "-10.0432442975097 -33.39709974391177, "
        "-10.0432442975097 -33.52595791268441, "
        "-10.08426784565221 -33.51578049243587))"
    )
    right_geom = shapely.from_wkt(
        "POLYGON ((-10.127010190354717 -33.50289039179234, "
        "-10.143244297509707 -33.49709974391177, "
        "-10.14324429750973 -33.49709974391176, "
        "-10.159963703906177 -33.492911750627314, "
        "-10.2432442975097 -33.4631134777615, "
        "-10.2432442975097 -33.39709974391177, "
        "-10.0432442975097 -33.39709974391177, "
        "-10.0432442975097 -33.52378510226201, "
        "-10.127010190354717 -33.50289039179234))"
    )
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    split_events = build_gpu_split_events(left, right)
    near_x = -10.1432442975097
    near_y = -33.49709974391177
    near = np.isclose(split_events.x, near_x, rtol=0.0, atol=1e-10) & np.isclose(
        split_events.y,
        near_y,
        rtol=0.0,
        atol=1e-10,
    )
    assert near.sum() >= 4
    assert np.unique(
        np.column_stack((split_events.x[near], split_events.y[near])),
        axis=0,
    ).shape[0] == 3

    result = binary_constructive_owned(
        "union",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    ).to_shapely()[0]
    expected = shapely.union(left_geom, right_geom)
    assert result.is_valid
    assert shapely.area(shapely.symmetric_difference(result, expected)) == pytest.approx(
        0.0,
        abs=1e-12,
    )


@pytest.mark.gpu
def test_grouped_boundary_renoding_splits_residual_constructed_node_crossing() -> None:
    import cupy as cp

    start_x = cp.asarray([1034.8518449359565, 1034.831465433527])
    start_y = cp.asarray([996.9842343903816, 996.5694000884654])
    end_x = cp.asarray([1034.90302686536, 1034.8552737057707])
    end_y = cp.asarray([997.396403326943, 997.0540286033288])
    rows = cp.zeros(2, dtype=cp.int32)
    placeholder = build_empty_polygon_rows_device(1)

    noded = renode_grouped_boundary_segments_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        rows,
        placeholder_owned=placeholder,
    )

    noded_start_x, noded_start_y, noded_end_x, noded_end_y, noded_rows = noded
    assert int(noded_start_x.size) == 4
    assert cp.asnumpy(noded_rows).tolist() == [0, 0, 0, 0]
    lines = np.asarray(
        [
            LineString(pair)
            for pair in zip(
                zip(cp.asnumpy(noded_start_x), cp.asnumpy(noded_start_y), strict=True),
                zip(cp.asnumpy(noded_end_x), cp.asnumpy(noded_end_y), strict=True),
                strict=True,
            )
        ],
        dtype=object,
    )
    pairs = shapely.STRtree(lines).query(lines, predicate="crosses")
    assert not np.any(pairs[0] < pairs[1])


@pytest.mark.gpu
def test_grouped_boundary_renoding_preserves_unmatched_fp64_crossing() -> None:
    import cupy as cp

    source_lines = [
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        LineString([(0.0, 0.3), (1.0, 0.0)]),
    ]
    expected_node = shapely.intersection(*source_lines)
    decimal_grid = 1.0e-12
    quantized_x = round(expected_node.x / decimal_grid) * decimal_grid
    assert quantized_x != expected_node.x

    noded = renode_grouped_boundary_segments_gpu(
        cp.asarray([line.coords[0][0] for line in source_lines]),
        cp.asarray([line.coords[0][1] for line in source_lines]),
        cp.asarray([line.coords[1][0] for line in source_lines]),
        cp.asarray([line.coords[1][1] for line in source_lines]),
        cp.zeros(2, dtype=cp.int32),
        placeholder_owned=build_empty_polygon_rows_device(1),
    )

    start_x, start_y, end_x, end_y, _rows = noded
    all_x = cp.asnumpy(cp.concatenate((start_x, end_x)))
    all_y = cp.asnumpy(cp.concatenate((start_y, end_y)))
    internal = (
        (all_x != 0.0)
        & (all_x != 1.0)
        & (all_y != 0.0)
        & (all_y != 0.3)
        & (all_y != 1.0)
    )
    nodes = np.unique(np.column_stack((all_x[internal], all_y[internal])), axis=0)

    assert nodes.shape == (1, 2)
    assert nodes[0, 0] == pytest.approx(
        expected_node.x,
        rel=0.0,
        abs=np.spacing(expected_node.x),
    )
    assert nodes[0, 1] == pytest.approx(
        expected_node.y,
        rel=0.0,
        abs=np.spacing(expected_node.y),
    )
    assert nodes[0, 0] != quantized_x


@pytest.mark.gpu
def test_boundary_graph_peels_dangle_chain_deeper_than_64_edges() -> None:
    import cupy as cp

    square_segments = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
    ]
    chain_segments = [
        ((float(index), 0.0), (float(index + 1), 0.0))
        for index in range(1, 66)
    ]
    segments = [*square_segments, *chain_segments]
    runtime_selection = plan_dispatch_selection(
        kernel_name="overlay_faces",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=len(segments),
        requested_mode=ExecutionMode.GPU,
        current_residency=Residency.DEVICE,
    )

    actual = build_polygon_output_from_boundary_segments_gpu(
        cp.asarray([start[0] for start, _ in segments]),
        cp.asarray([start[1] for start, _ in segments]),
        cp.asarray([end[0] for _, end in segments]),
        cp.asarray([end[1] for _, end in segments]),
        row_indices=cp.zeros(len(segments), dtype=cp.int32),
        row_count=1,
        runtime_selection=runtime_selection,
    ).to_shapely()[0]

    assert shapely.equals(actual, shapely.box(0.0, 0.0, 1.0, 1.0))


def _assemble_boundary_segments(
    segments,
    *,
    rows=None,
    row_count: int = 1,
    valid_empty_rows: bool = False,
):
    import cupy as cp

    if rows is None:
        rows = [0] * len(segments)
    runtime_selection = plan_dispatch_selection(
        kernel_name="overlay_faces",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=len(segments),
        requested_mode=ExecutionMode.GPU,
        current_residency=Residency.DEVICE,
    )
    return build_polygon_output_from_boundary_segments_gpu(
        cp.asarray([start[0] for start, _ in segments]),
        cp.asarray([start[1] for start, _ in segments]),
        cp.asarray([end[0] for _, end in segments]),
        cp.asarray([end[1] for _, end in segments]),
        row_indices=cp.asarray(rows, dtype=cp.int32),
        row_count=row_count,
        runtime_selection=runtime_selection,
        d_valid_empty_rows=(
            cp.ones(row_count, dtype=cp.bool_) if valid_empty_rows else None
        ),
    )


def _closed_ring_segments(coordinates):
    return list(zip(coordinates[:-1], coordinates[1:], strict=True))


@pytest.mark.gpu
def test_boundary_graph_pure_tree_returns_valid_empty_rows() -> None:
    segments = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (2.0, 1.0)),
        ((1.0, 0.0), (2.0, -1.0)),
        ((2.0, 1.0), (3.0, 1.0)),
    ]

    actual = _assemble_boundary_segments(
        segments,
        valid_empty_rows=True,
    ).to_shapely()[0]

    assert actual.is_empty


@pytest.mark.gpu
def test_boundary_graph_leaf_frontier_handles_branched_race() -> None:
    square = _closed_ring_segments(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
    )
    branches = [
        ((1.0, 0.0), (2.0, 0.0)),
        ((2.0, 0.0), (3.0, 1.0)),
        ((2.0, 0.0), (3.0, 0.0)),
        ((2.0, 0.0), (3.0, -1.0)),
    ]

    actual = _assemble_boundary_segments([*square, *branches]).to_shapely()[0]

    assert shapely.equals(actual, shapely.box(0.0, 0.0, 1.0, 1.0))


@pytest.mark.gpu
def test_boundary_graph_degree_core_isolates_equal_endpoints_by_row() -> None:
    segments = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
    ]

    actual = _assemble_boundary_segments(
        segments,
        rows=[0, 0, 0, 1],
        row_count=2,
        valid_empty_rows=True,
    ).to_shapely()

    assert all(geometry.is_empty for geometry in actual)


@pytest.mark.gpu
def test_boundary_graph_cycle_with_multiple_tendrils_keeps_only_cycle() -> None:
    square = _closed_ring_segments(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)],
    )
    tendrils = [
        ((0.0, 0.0), (-1.0, 0.0)),
        ((-1.0, 0.0), (-2.0, 1.0)),
        ((2.0, 2.0), (3.0, 2.0)),
        ((3.0, 2.0), (4.0, 3.0)),
    ]

    actual = _assemble_boundary_segments([*square, *tendrils]).to_shapely()[0]

    assert shapely.equals(actual, shapely.box(0.0, 0.0, 2.0, 2.0))


@pytest.mark.gpu
def test_boundary_graph_two_cycles_joined_by_bridge_preserves_both_cycles() -> None:
    left = _closed_ring_segments(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ],
    )
    right = _closed_ring_segments(
        [
            (3.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (3.0, 1.0),
            (3.0, 0.5),
            (3.0, 0.0),
        ],
    )
    bridge = [((1.0, 0.5), (3.0, 0.5))]

    actual = _assemble_boundary_segments([*left, *right, *bridge]).to_shapely()[0]
    expected = shapely.multipolygons(
        [shapely.box(0.0, 0.0, 1.0, 1.0), shapely.box(3.0, 0.0, 4.0, 1.0)],
    )

    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_boundary_graph_component_hints_preserve_shell_hole_island() -> None:
    shell = _closed_ring_segments(
        [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)],
    )
    hole = _closed_ring_segments(
        [(4.0, 4.0), (4.0, 16.0), (16.0, 16.0), (16.0, 4.0), (4.0, 4.0)],
    )
    island = _closed_ring_segments(
        [(8.0, 8.0), (12.0, 8.0), (12.0, 12.0), (8.0, 12.0), (8.0, 8.0)],
    )
    expected = shapely.multipolygons(
        [
            shapely.Polygon(
                [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
                holes=[[(4.0, 4.0), (4.0, 16.0), (16.0, 16.0), (16.0, 4.0)]],
            ),
            shapely.box(8.0, 8.0, 12.0, 12.0),
        ],
    )

    actual = _assemble_boundary_segments([*shell, *hole, *island]).to_shapely()[0]

    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_boundary_graph_indexed_containment_preserves_long_nesting_chain() -> None:
    ring_coordinates = []
    segments = []
    for depth in range(17):
        lower = float(depth)
        upper = 40.0 - float(depth)
        coordinates = [
            (lower, lower),
            (upper, lower),
            (upper, upper),
            (lower, upper),
            (lower, lower),
        ]
        ring_coordinates.append(coordinates)
        segments.extend(_closed_ring_segments(coordinates))

    expected_parts = []
    for depth in range(0, len(ring_coordinates), 2):
        holes = (
            [ring_coordinates[depth + 1]]
            if depth + 1 < len(ring_coordinates)
            else None
        )
        expected_parts.append(shapely.Polygon(ring_coordinates[depth], holes=holes))
    expected = shapely.multipolygons(expected_parts)

    actual = _assemble_boundary_segments(segments).to_shapely()[0]

    assert actual.is_valid
    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_boundary_graph_builds_half_edge_graph_once(monkeypatch) -> None:
    import importlib

    graph_module = importlib.import_module("vibespatial.overlay.graph")

    original = graph_module.build_gpu_half_edge_graph
    calls = 0

    def _counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_module, "build_gpu_half_edge_graph", _counted_build)
    square = _closed_ring_segments(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
    )
    _assemble_boundary_segments(square)

    assert calls == 1


def test_boundary_graph_uses_device_core_and_indexed_nesting_only() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    boundary_source = (root / "src/vibespatial/overlay/boundary_graph.py").read_text()
    kernels_source = (root / "src/vibespatial/overlay/gpu_kernels.py").read_text()
    assemble_source = (root / "src/vibespatial/overlay/assemble.py").read_text()
    function_start = boundary_source.index(
        "def build_polygon_output_from_boundary_segments_gpu(",
    )
    function_source = boundary_source[function_start:]

    assert "while True" not in function_source
    assert function_source.count("build_gpu_half_edge_graph(") == 1
    assert "initialize_degree_two_core_frontier" in function_source
    assert "peel_degree_two_core_frontier" in function_source
    assert "_build_indexed_component_containment_device_state" in function_source
    assert "component_nesting=nesting" in function_source
    assert "accumulate_component_containment_baseline" not in kernels_source
    assert "relation_" not in function_source
    assert "reduce_component_containment_nesting" not in kernels_source
    assert "select_indexed_component_containment_parent" in kernels_source
    assert "grid=((face_count" not in function_source
    assert "if component_nesting is not None:" in assemble_source


@pytest.mark.gpu
def test_boundary_graph_uses_exact_radial_order_for_near_collinear_rays() -> None:
    import cupy as cp

    origin = (0.0, 0.0)
    outer_left = (-0.1523828637565714, -0.6083464929280353)
    inner_left = (-0.09064249771765276, -0.3618651352087454)
    inner_right = (-0.09064249771762434, -0.3618651352086317)
    outer_right = (0.03311138804998137, 0.09254013023701191)
    segments = [
        (origin, outer_left),
        (outer_left, outer_right),
        (outer_right, origin),
        (origin, inner_left),
        (inner_left, inner_right),
        (inner_right, origin),
    ]
    runtime_selection = plan_dispatch_selection(
        kernel_name="overlay_faces",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=len(segments),
        requested_mode=ExecutionMode.GPU,
        current_residency=Residency.DEVICE,
    )
    actual = build_polygon_output_from_boundary_segments_gpu(
        cp.asarray([start[0] for start, _ in segments]),
        cp.asarray([start[1] for start, _ in segments]),
        cp.asarray([end[0] for _, end in segments]),
        cp.asarray([end[1] for _, end in segments]),
        row_indices=cp.zeros(len(segments), dtype=cp.int32),
        row_count=1,
        runtime_selection=runtime_selection,
    ).to_shapely()[0]
    expected = shapely.union(
        shapely.Polygon([origin, outer_left, outer_right, origin]),
        shapely.Polygon([origin, inner_left, inner_right, origin]),
    )

    assert actual.is_valid
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == 0.0


@pytest.mark.gpu
def test_atomic_edge_dedup_collapses_opposite_orientation_overlap_segments() -> None:
    left_geom, right_geom = _buffered_line_reduction_pair(round_index=2, pair_index=16)
    split_events = build_gpu_split_events(
        from_shapely_geometries([left_geom]),
        from_shapely_geometries([right_geom]),
    )
    atomic_edges = build_gpu_atomic_edges(split_events)

    forward = atomic_edges.direction == 0
    coords = np.column_stack(
        (
            np.rint(atomic_edges.src_x[forward] * 1_000_000_000.0).astype(np.int64, copy=False),
            np.rint(atomic_edges.src_y[forward] * 1_000_000_000.0).astype(np.int64, copy=False),
            np.rint(atomic_edges.dst_x[forward] * 1_000_000_000.0).astype(np.int64, copy=False),
            np.rint(atomic_edges.dst_y[forward] * 1_000_000_000.0).astype(np.int64, copy=False),
        ),
    )
    unique_count = np.unique(coords, axis=0).shape[0]

    assert unique_count == coords.shape[0]
