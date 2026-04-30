from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from vibespatial.api._native_result_core import NativeGeometryProvenance
from vibespatial.constructive.boundary import boundary_owned
from vibespatial.constructive.point import (
    clip_points_rect_owned,
    point_buffer_native_tabular_result,
    point_buffer_owned_array,
    point_owned_from_xy_device,
)
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.io.arrow import geoseries_from_owned
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.materialization import (
    clear_materialization_events,
    get_materialization_events,
)
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


def test_point_buffer_native_tabular_result_carries_source_provenance() -> None:
    points = from_shapely_geometries([Point(0, 0), Point(2, 2)])
    source_rows = np.asarray([7, 11], dtype=np.int32)

    result = point_buffer_native_tabular_result(
        points,
        np.asarray([1.0, 2.0], dtype=np.float64),
        quad_segs=1,
        dispatch_mode=ExecutionMode.CPU,
        crs="EPSG:3857",
        source_rows=source_rows,
        source_tokens=("points",),
    )

    assert result.geometry.row_count == 2
    assert result.geometry.crs == "EPSG:3857"
    assert result.column_order == ("geometry",)
    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.operation == "buffer"
    assert result.provenance.source_tokens == ("points",)
    assert result.provenance.source_rows.tolist() == [7, 11]
    assert result.geometry_metadata is not None
    assert result.geometry_metadata.row_count == 2


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
def test_point_buffer_native_tabular_device_provenance_survives_rowset_take() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_profile,
        reset_d2h_transfer_count,
    )

    points = point_owned_from_xy_device(
        np.asarray([0.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 2.0], dtype=np.float64),
    )
    result = point_buffer_native_tabular_result(
        points,
        1.0,
        quad_segs=1,
        dispatch_mode=ExecutionMode.GPU,
    )
    state = result.to_native_frame_state()

    assert result.geometry.owned.residency is Residency.DEVICE
    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.is_device
    assert result.geometry_metadata is not None
    assert result.geometry_metadata.is_device
    assert result.geometry_metadata.bounds is not None

    reset_d2h_transfer_count()
    clear_materialization_events()
    area_expression = state.geometry_area_expression()
    rowset = area_expression.greater_than(1.5)
    filtered = state.take(rowset, preserve_index=False)
    transfer_count, transfer_bytes, _transfer_seconds = get_d2h_transfer_profile()

    assert area_expression.is_device
    assert rowset.is_device
    assert filtered.row_count == 2
    assert isinstance(filtered.provenance, NativeGeometryProvenance)
    assert filtered.provenance.is_device
    assert filtered.geometry_metadata_cache is not None
    assert filtered.geometry_metadata_cache.is_device
    assert transfer_count <= 2
    assert transfer_bytes <= 16
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(filtered.provenance.source_rows).tolist() == [0, 1]
    reset_d2h_transfer_count()


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
        assert shapely.equals(g, c), f"row {i} GPU/CPU topology mismatch"
        assert shapely.equals_exact(g, c, tolerance=1e-10), f"row {i} GPU/CPU mismatch"


@pytest.mark.gpu
def test_point_buffer_gpu_boundary_single_ring_polygons_are_linestrings() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(0, 0), Point(2, 3)])

    buffered = point_buffer_owned_array(points, 1.0, quad_segs=16, dispatch_mode=ExecutionMode.GPU)
    boundary = boundary_owned(buffered, dispatch_mode=ExecutionMode.GPU)

    assert boundary.residency is Residency.DEVICE
    assert [geom.geom_type for geom in boundary.to_shapely()] == ["LineString", "LineString"]


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
def test_point_buffer_owned_array_gpu_round_buffer_matches_shapely_topology() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries(
        [
            Point(0.8783596961872845, 0.424285920013213),
            Point(0.11344929019320247, 0.9479274767330191),
        ]
    )

    for quad_segs in (16, 25):
        gpu = point_buffer_owned_array(points, 0.1, quad_segs=quad_segs, dispatch_mode=ExecutionMode.GPU)
        for i, geom in enumerate(gpu.to_shapely()):
            expected = shapely.buffer(points.to_shapely()[i], 0.1, quad_segs=quad_segs)
            assert shapely.equals(geom, expected), f"row {i} quad_segs={quad_segs} topology mismatch"


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
