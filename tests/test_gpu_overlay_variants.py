from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial import (
    ExecutionMode,
    from_shapely_geometries,
    has_gpu_runtime,
    overlay_difference_owned,
    overlay_identity_owned,
    overlay_symmetric_difference_owned,
    overlay_union_owned,
)


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
def test_gpu_overlay_union_matches_shapely_for_overlapping_rectangles() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.union(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.runtime_history[-1].selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_gpu_overlay_difference_matches_shapely() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        holes=[[(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]],
    )
    left = from_shapely_geometries([donut])
    right = from_shapely_geometries([box(1, 1, 7, 7)])

    actual = overlay_difference_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.difference(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())


@pytest.mark.gpu
def test_gpu_overlay_symmetric_difference_groups_row_outputs_into_multipolygon() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_symmetric_difference_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.symmetric_difference(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    )

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.row_count == 1
    assert actual.to_shapely()[0].geom_type == "MultiPolygon"


@pytest.mark.gpu
def test_gpu_overlay_union_of_disjoint_boxes_emits_single_multipolygon_row() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 2, 2)])
    right = from_shapely_geometries([box(4, 0, 6, 2)])

    actual = overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.union(np.asarray(left.to_shapely(), dtype=object), np.asarray(right.to_shapely(), dtype=object))

    _assert_geometry_lists_equal(actual.to_shapely(), expected.tolist())
    assert actual.row_count == 1
    assert actual.to_shapely()[0].geom_type == "MultiPolygon"


@pytest.mark.gpu
def test_gpu_overlay_identity_geometry_matches_left_input() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([box(2, 2, 6, 6)])

    actual = overlay_identity_owned(left, right, dispatch_mode=ExecutionMode.GPU)

    _assert_geometry_lists_equal(actual.to_shapely(), left.to_shapely())


@pytest.mark.gpu
def test_gpu_overlay_variants_remain_polygon_only() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([shapely.LineString([(0, 0), (1, 1)])])
    right = from_shapely_geometries([box(0, 0, 2, 2)])

    with pytest.raises(NotImplementedError, match="polygon-only"):
        overlay_union_owned(left, right, dispatch_mode=ExecutionMode.GPU)
