from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from vibespatial import benchmark_bounds_pairs, from_shapely_geometries, generate_bounds_pairs
from vibespatial.runtime import ExecutionMode, has_gpu_runtime


def test_generate_bounds_pairs_finds_intersections_across_geometry_families() -> None:
    left = from_shapely_geometries(
        [
            Point(0, 0),
            LineString([(5, 5), (7, 7)]),
            Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
        ]
    )
    right = from_shapely_geometries(
        [
            Point(0, 0),
            Polygon([(6, 6), (8, 6), (8, 8), (6, 6)]),
            Polygon([(20, 20), (22, 20), (22, 22), (20, 20)]),
        ]
    )

    pairs = generate_bounds_pairs(left, right, tile_size=2)

    assert set(zip(pairs.left_indices.tolist(), pairs.right_indices.tolist(), strict=True)) == {
        (0, 0),
        (1, 1),
    }
    assert pairs.pairs_examined == 9


def test_generate_bounds_pairs_ignores_null_and_empty() -> None:
    owned = from_shapely_geometries([Point(1, 1), None, Point()])

    pairs = generate_bounds_pairs(owned, include_self=False)

    assert pairs.count == 0
    assert pairs.same_input is True
    if has_gpu_runtime():
        assert pairs.device_left_indices is not None
        assert pairs.device_right_indices is not None


def test_generate_bounds_pairs_same_input_uses_upper_triangle() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(0, 0), Point(10, 10)])

    pairs = generate_bounds_pairs(owned, include_self=False)

    assert set(zip(pairs.left_indices.tolist(), pairs.right_indices.tolist(), strict=True)) == {(0, 1)}


def test_generate_bounds_pairs_honors_explicit_cpu_relation_dispatch() -> None:
    left = from_shapely_geometries([Point(0, 0)])
    right = from_shapely_geometries([Point(0, 0)])

    pairs = generate_bounds_pairs(
        left,
        right,
        requested_mode=ExecutionMode.CPU,
    )

    assert pairs.device_left_indices is None
    assert pairs.device_right_indices is None
    assert pairs.left_indices.tolist() == [0]
    assert pairs.right_indices.tolist() == [0]


@pytest.mark.gpu
def test_generate_bounds_pairs_honors_explicit_gpu_relation_dispatch() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    left = from_shapely_geometries([Point(0, 0)])
    right = from_shapely_geometries([Point(0, 0)])

    pairs = generate_bounds_pairs(
        left,
        right,
        requested_mode=ExecutionMode.GPU,
    )

    assert pairs.device_left_indices is not None
    assert pairs.device_right_indices is not None
    assert pairs._host_left_indices is None
    assert pairs._host_right_indices is None


@pytest.mark.gpu
def test_generate_bounds_pairs_keeps_bounded_pair_cardinality_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left = from_shapely_geometries([Point(0, 0), Point(10, 10)])
    right = from_shapely_geometries([Point(0, 0), Point(5, 5), Point(10, 10)])
    reset_d2h_transfer_count()

    pairs = generate_bounds_pairs(
        left,
        right,
        requested_mode=ExecutionMode.GPU,
        capacity_output=True,
    )
    events = get_d2h_transfer_events(clear=True)

    assert pairs.device_selection is not None
    assert pairs.device_selection.capacity == 6
    assert cp.asnumpy(pairs.device_selection.logical_count).tolist() == [2]
    active = pairs.device_selection.active_capacity_mask()
    assert set(
        zip(
            cp.asnumpy(pairs.device_left_indices[active]).tolist(),
            cp.asnumpy(pairs.device_right_indices[active]).tolist(),
            strict=True,
        )
    ) == {(0, 0), (1, 2)}
    assert not any(
        event.reason == "spatial index sweep candidate-pair allocation fence"
        for event in events
    )


def test_benchmark_bounds_pairs_reports_dataset_stats() -> None:
    owned = from_shapely_geometries([Point(float(index), float(index)) for index in range(32)])

    benchmark = benchmark_bounds_pairs(owned, dataset="uniform", tile_size=8)

    assert benchmark.dataset == "uniform"
    assert benchmark.rows == 32
    assert benchmark.pairs_examined > 0
