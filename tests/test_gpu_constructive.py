from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from vibespatial import ExecutionMode, clip_by_rect_owned, from_shapely_geometries, has_gpu_runtime
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.point_constructive import clip_points_rect_owned, point_buffer_owned_array


def _assert_geometries_equal(actual: list[object | None], expected: list[object | None]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.geom_type == right.geom_type
        assert bool(shapely.equals(left, right))


@pytest.mark.gpu
def test_gpu_clip_by_rect_matches_shapely_for_point_only_inputs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(3, 3), Point(), None])

    result = clip_by_rect_owned(
        values,
        0.0,
        0.0,
        1.5,
        1.5,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.clip_by_rect(
        np.asarray([Point(0, 0), Point(1, 1), Point(3, 3), Point(), None], dtype=object),
        0.0,
        0.0,
        1.5,
        1.5,
    )
    _assert_geometries_equal(result.geometries.tolist(), list(expected))
    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0


@pytest.mark.gpu
def test_gpu_point_buffer_matches_shapely_for_quad_segs_1() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 2), Point(-1, 4)])

    gpu = point_buffer_owned_array(points, 1.5, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    expected = [
        Polygon(((1.5, 0.0), (0.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 0.0))),
        Polygon(((3.5, 2.0), (2.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 2.0))),
        Polygon(((0.5, 4.0), (-1.0, 2.5), (-2.5, 4.0), (-1.0, 5.5), (0.5, 4.0))),
    ]
    _assert_geometries_equal(gpu.to_shapely(), expected)


@pytest.mark.gpu
def test_gpu_clip_by_rect_accepts_device_backed_point_input_without_full_host_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2), Point(4, 4)])
    device_backed = clip_points_rect_owned(
        points,
        -1.0,
        -1.0,
        5.0,
        5.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert device_backed.families[GeometryFamily.POINT].host_materialized is False

    result = clip_by_rect_owned(
        device_backed,
        0.5,
        0.5,
        3.0,
        3.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.clip_by_rect(
        np.asarray([Point(0, 0), Point(1, 1), Point(2, 2), Point(4, 4)], dtype=object),
        0.5,
        0.5,
        3.0,
        3.0,
    )
    _assert_geometries_equal(result.geometries.tolist(), list(expected))
    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0
