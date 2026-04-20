from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon

from vibespatial import benchmark_bounds_pairs, from_shapely_geometries, generate_bounds_pairs
from vibespatial.runtime import has_gpu_runtime


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


def test_benchmark_bounds_pairs_reports_dataset_stats() -> None:
    owned = from_shapely_geometries([Point(float(index), float(index)) for index in range(32)])

    benchmark = benchmark_bounds_pairs(owned, dataset="uniform", tile_size=8)

    assert benchmark.dataset == "uniform"
    assert benchmark.rows == 32
    assert benchmark.pairs_examined > 0
