from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from vibespatial.constructive.point import (
    clip_points_rect_owned,
    point_buffer_owned_array,
    point_owned_from_xy_device,
)
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.io.arrow import geoseries_from_owned
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency


def _assert_geometries_equal(actual: list[object | None], expected: list[object | None]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.geom_type == right.geom_type
        assert bool(shapely.equals(left, right))


def test_clip_points_rect_owned_cpu_filters_to_points_inside_rect() -> None:
    points = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(3, 3), None, Point()])

    clipped = clip_points_rect_owned(points, 0.0, 0.0, 1.5, 1.5, dispatch_mode=ExecutionMode.CPU)

    _assert_geometries_equal(clipped.to_shapely(), [Point(0, 0), Point(1, 1)])


def test_point_buffer_owned_array_cpu_matches_expected_diamonds() -> None:
    points = from_shapely_geometries([Point(0, 0), Point(2, 2)])

    buffered = point_buffer_owned_array(points, np.asarray([1.0, 2.0]), quad_segs=1, dispatch_mode=ExecutionMode.CPU)

    expected = [
        Polygon(((1, 0), (0, -1), (-1, 0), (0, 1), (1, 0))),
        Polygon(((4, 2), (2, 0), (0, 2), (2, 4), (4, 2))),
    ]
    _assert_geometries_equal(buffered.to_shapely(), expected)


@pytest.mark.gpu
def test_clip_points_rect_owned_gpu_matches_cpu_subset() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(3, 3), Point(1.5, 0.5)])

    gpu = clip_points_rect_owned(points, 0.0, 0.0, 1.5, 1.5, dispatch_mode=ExecutionMode.GPU)
    assert gpu.residency is Residency.DEVICE
    assert gpu.device_state is not None
    assert gpu._validity is None
    assert gpu._tags is None
    assert gpu._family_row_offsets is None
    assert gpu.families[next(iter(gpu.families))].host_materialized is False

    _assert_geometries_equal(gpu.to_shapely(), [Point(0, 0), Point(1, 1), Point(1.5, 0.5)])
    assert gpu.families[next(iter(gpu.families))].host_materialized is True


@pytest.mark.gpu
def test_point_buffer_owned_array_gpu_matches_cpu_diamonds() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 2), Point(-1, 4)])

    gpu = point_buffer_owned_array(points, 1.5, quad_segs=1, dispatch_mode=ExecutionMode.GPU)
    assert gpu.residency is Residency.DEVICE
    assert gpu.device_state is not None
    assert gpu._validity is None
    assert gpu._tags is None
    assert gpu._family_row_offsets is None
    assert gpu.families[next(iter(gpu.families))].host_materialized is False

    expected = [
        Polygon(((1.5, 0.0), (0.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 0.0))),
        Polygon(((3.5, 2.0), (2.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 2.0))),
        Polygon(((0.5, 4.0), (-1.0, 2.5), (-2.5, 4.0), (-1.0, 5.5), (0.5, 4.0))),
    ]
    _assert_geometries_equal(gpu.to_shapely(), expected)
    assert gpu.families[next(iter(gpu.families))].host_materialized is True


@pytest.mark.gpu
def test_point_buffer_auto_sticks_to_device_residency_below_threshold() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0)], residency=Residency.DEVICE)

    buffered = point_buffer_owned_array(points, 1.0, quad_segs=1, dispatch_mode=ExecutionMode.AUTO)

    assert buffered.residency is Residency.DEVICE


def test_point_buffer_owned_array_cpu_quad16_matches_shapely() -> None:
    points = from_shapely_geometries([Point(0, 0), Point(2, 3), Point(-1, 4)])
    radii = np.asarray([1.0, 2.0, 0.5])

    buffered = point_buffer_owned_array(points, radii, quad_segs=16, dispatch_mode=ExecutionMode.CPU)

    for i, (geom, r) in enumerate(zip(buffered.to_shapely(), radii)):
        expected = shapely.buffer(points.to_shapely()[i], r, quad_segs=16)
        assert shapely.equals_exact(geom, expected, tolerance=1e-10), f"row {i} mismatch"


def test_point_buffer_owned_array_cpu_quad2_matches_shapely() -> None:
    points = from_shapely_geometries([Point(1, 1)])

    buffered = point_buffer_owned_array(points, 1.0, quad_segs=2, dispatch_mode=ExecutionMode.CPU)

    expected = shapely.buffer(Point(1, 1), 1.0, quad_segs=2)
    assert shapely.equals_exact(buffered.to_shapely()[0], expected, tolerance=1e-10)


@pytest.mark.gpu
def test_point_buffer_owned_array_gpu_quad16_matches_cpu() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 3), Point(-1, 4)])
    radii = np.asarray([1.0, 2.0, 0.5])

    cpu = point_buffer_owned_array(points, radii, quad_segs=16, dispatch_mode=ExecutionMode.CPU)
    gpu = point_buffer_owned_array(points, radii, quad_segs=16, dispatch_mode=ExecutionMode.GPU)

    assert gpu.residency is Residency.DEVICE
    for i, (g, c) in enumerate(zip(gpu.to_shapely(), cpu.to_shapely())):
        assert shapely.equals_exact(g, c, tolerance=1e-10), f"row {i} GPU/CPU mismatch"


@pytest.mark.gpu
def test_point_buffer_owned_array_gpu_quad4_matches_shapely() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(5, -3)])

    gpu = point_buffer_owned_array(points, 3.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    for i, geom in enumerate(gpu.to_shapely()):
        expected = shapely.buffer(points.to_shapely()[i], 3.0, quad_segs=4)
        assert shapely.equals_exact(geom, expected, tolerance=1e-10), f"row {i} mismatch"


@pytest.mark.gpu
def test_clip_then_buffer_gpu_stays_device_resident_until_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(3, 3), Point(1.5, 0.5)])
    clipped = clip_points_rect_owned(points, 0.0, 0.0, 1.5, 1.5, dispatch_mode=ExecutionMode.GPU)
    buffered = point_buffer_owned_array(clipped, 1.0, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    assert clipped.residency is Residency.DEVICE
    assert buffered.residency is Residency.DEVICE
    assert clipped.families[next(iter(clipped.families))].host_materialized is False
    assert buffered.families[next(iter(buffered.families))].host_materialized is False

    expected = [
        Polygon(((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 0.0))),
        Polygon(((2.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 1.0))),
        Polygon(((2.5, 0.5), (1.5, -0.5), (0.5, 0.5), (1.5, 1.5), (2.5, 0.5))),
    ]
    _assert_geometries_equal(buffered.to_shapely(), expected)


@pytest.mark.gpu
def test_point_owned_from_xy_device_keeps_structural_metadata_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily

    owned = point_owned_from_xy_device(
        np.asarray([0.0, 2.0], dtype=np.float64),
        np.asarray([1.0, 3.0], dtype=np.float64),
    )

    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None

    point_dev_buf = owned.device_state.families[GeometryFamily.POINT]
    assert isinstance(point_dev_buf.geometry_offsets, cp.ndarray)
    assert isinstance(point_dev_buf.empty_mask, cp.ndarray)
    assert isinstance(point_dev_buf.bounds, cp.ndarray)


@pytest.mark.gpu
def test_geoseries_from_owned_materializes_device_backed_buffer_via_geoarrow() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 2)])
    buffered = point_buffer_owned_array(points, 1.0, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    series = geoseries_from_owned(buffered, crs="EPSG:4326")

    expected = [
        Polygon(((1, 0), (0, -1), (-1, 0), (0, 1), (1, 0))),
        Polygon(((3, 2), (2, 1), (1, 2), (2, 3), (3, 2))),
    ]
    _assert_geometries_equal(list(series), expected)
