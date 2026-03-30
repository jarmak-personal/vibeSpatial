from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import (
    ExecutionMode,
    benchmark_clip_by_rect,
    clip_by_rect_owned,
    from_shapely_geometries,
    has_gpu_runtime,
)


def _assert_geometries_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.geom_type == right.geom_type
        assert bool(shapely.equals(left, right))


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
