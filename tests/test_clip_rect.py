from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import (
    ExecutionMode,
    Residency,
    benchmark_clip_by_rect,
    clip_by_rect_owned,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.geometry.device_array import DeviceGeometryArray


def _assert_geometries_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.geom_type == right.geom_type
        assert bool(shapely.equals(left, right))


def _make_device_resident_with_host_stubs_cleared(geoms: list[object | None]):
    from vibespatial.geometry.owned import FamilyGeometryBuffer

    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    owned.families = {
        family: FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )
        for family, buffer in owned.families.items()
    }
    return owned


def test_clip_by_rect_owned_matches_shapely_for_points_lines_and_polygons() -> None:
    values = [
        Point(1, 1),
        MultiPoint([(1, 1), (5, 5)]),
        LineString([(0, 0), (4, 4)]),
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
    ]

    result = clip_by_rect_owned(values, 0, 0, 2, 2)
    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0, 0, 2, 2)

    _assert_geometries_match(result.geometries.tolist(), list(expected))
    assert result.fallback_rows.size == 0


def test_clip_by_rect_owned_preserves_polygon_holes() -> None:
    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )

    result = clip_by_rect_owned([donut], 1, 1, 5, 5)
    expected = shapely.clip_by_rect(np.asarray([donut], dtype=object), 1, 1, 5, 5)

    _assert_geometries_match(result.geometries.tolist(), list(expected))
    assert result.fallback_rows.size == 0


def test_clip_by_rect_owned_falls_back_for_invalid_polygon_rows() -> None:
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])

    result = clip_by_rect_owned([invalid], 0, 0, 2, 2)
    expected = shapely.clip_by_rect(np.asarray([invalid], dtype=object), 0, 0, 2, 2)

    _assert_geometries_match(result.geometries.tolist(), list(expected))
    # Vectorized shapely.clip_by_rect handles invalid polygons internally,
    # so no per-row fallback is needed.
    assert result.fallback_rows.size == 0


def test_geopandas_clip_by_rect_surface_is_observable_when_row_fallback_happens() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries(
        [
            LineString([(0, 0), (4, 4)]),
            box(0, 0, 4, 4),
        ]
    )

    result = series.clip_by_rect(0, 0, 2, 2)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)
    expected = shapely.clip_by_rect(np.asarray(series.values._data, dtype=object), 0, 0, 2, 2)

    assert len(result) == 2
    _assert_geometries_match(result.values._data.tolist(), list(expected))
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.array.clip_by_rect"
    assert dispatch_events[-1].implementation == "owned_clip_by_rect"


def test_clip_by_rect_benchmark_reports_candidate_and_fallback_counts() -> None:
    values = from_shapely_geometries(
        [LineString([(0, 0), (4, 4)]), LineString([(10, 10), (11, 11)]), LineString([(1, 3), (3, 1)])]
    )

    benchmark = benchmark_clip_by_rect(values, 0, 0, 2, 2, dataset="lines")

    assert benchmark.rows == 3
    assert benchmark.candidate_rows == 2
    assert benchmark.fast_rows == 2
    assert benchmark.fallback_rows == 0


@pytest.mark.gpu
def test_clip_polygon_gpu_coordinates_stay_device_resident() -> None:
    """Clip coordinates stay on device — output is device-resident CuPy.

    Verifies that the GPU clip pipeline produces device-resident coordinate
    buffers (CuPy arrays) in the output OwnedGeometryArray, proving no
    premature D2H materialization of the coordinate data.

    Note: strict_device_guard is not used here because CuPy's flatnonzero
    internally calls .get() (a CuPy implementation detail we cannot avoid),
    and the function has justified small-metadata D2H transfers annotated
    with zcopy:ok.  Instead, we directly verify coordinate residency.
    """
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )
    values = from_shapely_geometries([
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        donut,
    ])

    result = clip_by_rect_owned(
        values,
        1.0, 1.0, 5.0, 5.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    # Verify GPU dispatch was used.
    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0

    # Verify coordinate buffers are device-resident CuPy arrays in
    # the device_state (not yet materialized to host).
    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.runtime.residency import Residency

    owned = result.owned_result
    assert owned is not None
    assert owned.residency is Residency.DEVICE, "output should be device-resident"
    assert owned.device_state is not None, "device_state should be populated"
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None
    poly_dev_buf = owned.device_state.families[GeometryFamily.POLYGON]
    assert isinstance(poly_dev_buf.x, cp.ndarray), "x coordinates should be device-resident CuPy"
    assert isinstance(poly_dev_buf.y, cp.ndarray), "y coordinates should be device-resident CuPy"

    # Verify correctness against Shapely oracle.
    expected = shapely.clip_by_rect(
        np.asarray([
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            donut,
        ], dtype=object),
        1.0, 1.0, 5.0, 5.0,
    )

    _assert_geometries_match(result.geometries.tolist(), list(expected))


@pytest.mark.gpu
def test_clip_polygon_rings_gpu_device_does_not_force_runtime_sync(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.constructive.clip_rect import _clip_polygon_rings_gpu_device
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()

    def _fail_sync():
        raise AssertionError("clip polygon device path should not force runtime.synchronize()")

    monkeypatch.setattr(runtime, "synchronize", _fail_sync)

    ring_x = np.asarray([0.0, 4.0, 4.0, 0.0, 0.0], dtype=np.float64)
    ring_y = np.asarray([0.0, 0.0, 4.0, 4.0, 0.0], dtype=np.float64)
    ring_offsets = np.asarray([0, 5], dtype=np.int32)

    d_out_x, d_out_y, d_out_offsets = _clip_polygon_rings_gpu_device(
        ring_x,
        ring_y,
        ring_offsets,
        (1.0, 1.0, 3.0, 3.0),
    )

    assert d_out_x is not None
    assert d_out_y is not None
    assert d_out_offsets is not None
    assert int(d_out_offsets.size) == 2


@pytest.mark.gpu
def test_clip_lines_gpu_uses_device_family_buffers_when_host_stubs_are_empty() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily

    values = [LineString([(0, 0), (4, 4)])]
    owned = _make_device_resident_with_host_stubs_cleared(values)

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        2.0,
        2.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.fallback_rows.size == 0
    assert result.owned_result is not None
    assert result.owned_result.residency is Residency.DEVICE
    assert result.owned_result.device_state is not None
    assert result.owned_result._validity is None
    assert result.owned_result._tags is None
    assert result.owned_result._family_row_offsets is None

    line_dev_buf = result.owned_result.device_state.families[GeometryFamily.LINESTRING]
    assert isinstance(line_dev_buf.x, cp.ndarray)
    assert isinstance(line_dev_buf.y, cp.ndarray)

    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 2.0, 2.0)
    _assert_geometries_match(result.geometries.tolist(), list(expected))


@pytest.mark.gpu
def test_device_geometry_array_clip_by_rect_preserves_device_residency_for_polygons() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        Polygon([(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)]),
        Polygon([(1, 1), (5, 1), (5, 5), (1, 5), (1, 1)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    array = DeviceGeometryArray._from_owned(owned)

    result = array.clip_by_rect(0.0, 0.0, 2.0, 2.0)

    assert isinstance(result, DeviceGeometryArray)
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.device_state is not None
    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 2.0, 2.0)
    expected_list = [None if geom is not None and geom.is_empty else geom for geom in expected.tolist()]
    _assert_geometries_match(list(result.owned.to_shapely()), expected_list)
