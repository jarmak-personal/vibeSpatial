from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from vibespatial import ExecutionMode, clip_by_rect_owned, from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.boundary import boundary_owned
from vibespatial.constructive.exterior import exterior_owned
from vibespatial.constructive.normalize import normalize_owned
from vibespatial.constructive.point import clip_points_rect_owned, point_buffer_owned_array
from vibespatial.constructive.polygon import polygon_buffer_owned_array
from vibespatial.geometry.buffers import GeometryFamily


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


@pytest.mark.gpu
def test_exterior_gpu_coordinates_stay_device_resident() -> None:
    """Exterior ring extraction keeps coordinates on device."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.runtime.residency import Residency

    polys = from_shapely_geometries([
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
        ),
    ])

    result = exterior_owned(polys, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE, "output should be device-resident"
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    assert result.device_state is not None
    ls_buf = result.device_state.families[GeometryFamily.LINESTRING]
    assert isinstance(ls_buf.x, cp.ndarray), "x should be CuPy"
    assert isinstance(ls_buf.y, cp.ndarray), "y should be CuPy"

    # Verify correctness: exterior_owned returns LineStrings with the same
    # coordinates as the exterior ring.
    expected_coords = [
        list(Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]).exterior.coords),
        list(Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
        ).exterior.coords),
    ]
    actual = result.to_shapely()
    if hasattr(actual, 'tolist'):
        actual = actual.tolist()
    for a, ec in zip(list(actual), expected_coords, strict=True):
        assert a.geom_type == "LineString"
        assert list(a.coords) == ec


@pytest.mark.gpu
def test_exterior_gpu_moderate_scale_with_nulls() -> None:
    """Exterior ring at moderate scale with interleaved None rows."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # 100+ polygons with varying ring sizes + None rows
    geoms: list[Polygon | None] = []
    for i in range(120):
        if i % 7 == 0:
            geoms.append(None)
        else:
            s = float(i)
            geoms.append(Polygon([(s, s), (s + 1, s), (s + 1, s + 1), (s, s + 1), (s, s)]))

    polys = from_shapely_geometries(geoms)
    result = exterior_owned(polys, dispatch_mode=ExecutionMode.GPU)

    actual = result.to_shapely()
    if hasattr(actual, 'tolist'):
        actual = actual.tolist()
    for i, (a, g) in enumerate(zip(list(actual), geoms, strict=True)):
        if g is None:
            assert a is None, f"row {i} should be None"
        else:
            assert a is not None and a.geom_type == "LineString", f"row {i}"
            assert list(a.coords) == list(g.exterior.coords), f"row {i} coords"


@pytest.mark.gpu
def test_exterior_gpu_all_nonpolygon_rows_stay_device_resident(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    owned = from_shapely_geometries([Point(0, 0), None], residency=Residency.DEVICE)
    result = exterior_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@pytest.mark.gpu
def test_boundary_gpu_metadata_stays_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.residency import Residency

    geoms = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        shapely.MultiPolygon([
            Polygon([(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)]),
            Polygon([(20, 0), (24, 0), (24, 4), (20, 4), (20, 0)]),
        ]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@pytest.mark.gpu
def test_boundary_gpu_mixed_polygon_ring_counts_preserves_row_types() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    single_ring = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    with_hole = Polygon(
        [(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)],
        [[(11, 1), (13, 1), (13, 3), (11, 3), (11, 1)]],
    )
    owned = from_shapely_geometries([single_ring, with_hole], residency=Residency.DEVICE)

    with assert_no_transfers():
        result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)
    geometries = result.to_shapely()

    assert result.residency is Residency.DEVICE
    assert geometries[0].geom_type == "LineString"
    assert geometries[1].geom_type == "MultiLineString"
    assert bool(shapely.equals(geometries[0], single_ring.boundary))
    assert bool(shapely.equals(geometries[1], with_hole.boundary))


@pytest.mark.gpu
def test_boundary_gpu_empty_lineal_rows_stay_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    geoms = [
        LineString(),
        LineString([(0, 0), (1, 1)]),
        MultiLineString([]),
        MultiLineString([[(2, 2), (3, 3)]]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)

    with assert_no_transfers():
        result = boundary_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    actual = result.to_shapely()
    _assert_geometries_equal(actual, [geom.boundary for geom in geoms])


@pytest.mark.gpu
def test_normalize_owned_device_resident_polygon_input_uses_device_stats() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    source_poly = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    source = from_shapely_geometries([source_poly])
    device_polys = polygon_buffer_owned_array(source, 0.25, dispatch_mode=ExecutionMode.GPU)

    assert device_polys.device_state is not None
    assert device_polys.families[GeometryFamily.POLYGON].host_materialized is False

    normalized = normalize_owned(device_polys, dispatch_mode=ExecutionMode.GPU)
    expected = shapely.normalize(shapely.buffer(source_poly, 0.25, quad_segs=8))
    actual = shapely.normalize(normalized.to_shapely()[0])

    assert bool(shapely.equals_exact(actual, expected, tolerance=1e-12))
