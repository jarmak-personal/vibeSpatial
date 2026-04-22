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
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
from vibespatial.overlay.faces import _select_overlay_face_indices_gpu, build_gpu_overlay_faces
from vibespatial.overlay.gpu import (
    _build_overlay_execution_plan,
    _materialize_overlay_execution_plan,
)
from vibespatial.overlay.graph import build_gpu_half_edge_graph
from vibespatial.overlay.host_fallback import _build_polygon_output_from_faces
from vibespatial.overlay.split import build_gpu_atomic_edges, build_gpu_split_events
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events


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
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0, abs=1e-6)


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
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 4.0, 4.0),
    ])
    right = from_shapely_geometries([
        shapely.box(2.0, 1.0, 5.0, 3.0),
    ])

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
def test_row_isolated_difference_preserves_sparse_multipolygon_rows() -> None:
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 10.0, 10.0),
        shapely.box(20.0, 20.0, 30.0, 30.0),
        shapely.box(40.0, 40.0, 50.0, 50.0),
    ])
    right = from_shapely_geometries([
        shapely.box(0.0, 0.0, 10.0, 10.0),
        shapely.box(20.0, 20.0, 30.0, 30.0),
        shapely.box(44.0, 39.0, 46.0, 51.0),
    ])

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
    left_geom = shapely.MultiPolygon([
        shapely.box(0.0, 0.0, 2.0, 2.0),
        shapely.box(4.0, 0.0, 6.0, 2.0),
    ])
    right_geom = shapely.box(1.0, -1.0, 5.0, 3.0)
    left = from_shapely_geometries([left_geom])
    right = from_shapely_geometries([right_geom])

    clear_dispatch_events()
    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _prefer_exact_polygon_intersection=True,
    )

    got = result.to_shapely()[0]
    expected = shapely.intersection(left_geom, right_geom)

    assert got is not None
    assert got.geom_type == "MultiPolygon"
    assert got.normalize().equals_exact(expected.normalize(), tolerance=1e-9)
    assert any(
        event.implementation == "direct_multipart_intersection_pack_gpu"
        for event in get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
def test_row_isolated_difference_preserves_all_empty_rows() -> None:
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 2.0, 2.0),
        shapely.box(4.0, 0.0, 6.0, 2.0),
    ])
    right = from_shapely_geometries([
        shapely.box(0.0, 0.0, 2.0, 2.0),
        shapely.box(4.0, 0.0, 6.0, 2.0),
    ])

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
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 4.0, 4.0),
    ])
    right = from_shapely_geometries([
        shapely.box(2.0, 1.0, 5.0, 3.0),
    ])

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
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0, abs=1e-6)


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
