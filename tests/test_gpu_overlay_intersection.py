from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial import ExecutionMode, from_shapely_geometries, has_gpu_runtime, overlay_intersection_owned


def _normalize(values):
    return [None if value is None else shapely.normalize(value) for value in values]


def _assert_geometry_lists_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(_normalize(actual), _normalize(expected), strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert bool(shapely.equals(left, right))


@pytest.mark.gpu
def test_gpu_overlay_intersection_matches_shapely_for_overlapping_rectangles() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])])
    right = from_shapely_geometries([Polygon([(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)])])

    actual = overlay_intersection_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.intersection(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.runtime_history[-1].selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_gpu_overlay_intersection_preserves_holes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )
    left = from_shapely_geometries([donut])
    right = from_shapely_geometries([box(1, 1, 5, 5)])

    actual = overlay_intersection_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.intersection(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())


@pytest.mark.gpu
def test_gpu_overlay_intersection_handles_duplicate_vertices() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    duplicate_vertex = Polygon([(0, 0), (4, 0), (4, 4), (4, 4), (0, 4), (0, 0)])
    left = from_shapely_geometries([duplicate_vertex])
    right = from_shapely_geometries([box(2, 2, 5, 5)])

    actual = overlay_intersection_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.intersection(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())


@pytest.mark.gpu
def test_gpu_overlay_intersection_drops_null_and_empty_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([None, Polygon()])
    right = from_shapely_geometries([box(0, 0, 2, 2)])

    actual = overlay_intersection_owned(left, right, dispatch_mode=ExecutionMode.GPU)

    assert actual.row_count == 0
    assert actual.families == {}
