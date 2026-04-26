from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon, box

import vibespatial.api as geopandas
import vibespatial.geometry.owned as owned_mod
from vibespatial import (
    ExecutionMode,
    Residency,
    benchmark_clip_by_rect,
    clip_by_rect_owned,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
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


@pytest.mark.gpu
def test_clip_by_rect_cpu_materializes_device_backed_multipoints_before_host_bridge() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            MultiPoint([(-12.0, -15.0), (2.0, 2.0), (3.0, 4.0), (9.0, 8.0)]),
            Point(-11.0, -14.0),
        ]
    )

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        10.0,
        10.0,
        dispatch_mode=ExecutionMode.CPU,
    )
    expected = shapely.clip_by_rect(
        np.asarray(
            [
                MultiPoint([(-12.0, -15.0), (2.0, 2.0), (3.0, 4.0), (9.0, 8.0)]),
                Point(-11.0, -14.0),
            ],
            dtype=object,
        ),
        0.0,
        0.0,
        10.0,
        10.0,
    )

    _assert_geometries_match(result.geometries.tolist(), list(expected))


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


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("values", "bounds"),
    [
        ([Point(1, 1), Point(5, 5)], (0.0, 0.0, 2.0, 2.0)),
        ([LineString([(0, 0), (3, 3)]), LineString([(5, 5), (6, 6)])], (0.0, 0.0, 2.0, 2.0)),
        ([box(0, 0, 4, 4), box(10, 10, 12, 12)], (0.0, 0.0, 2.0, 2.0)),
    ],
)
def test_geometry_array_clip_by_rect_promotes_supported_host_families_to_device(
    values,
    bounds,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    xmin, ymin, xmax, ymax = bounds
    array = geopandas.GeoSeries(values, crs="EPSG:4326").values

    result = array.clip_by_rect(xmin, ymin, xmax, ymax)

    assert isinstance(result, DeviceGeometryArray)
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.device_state is not None
    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), xmin, ymin, xmax, ymax)
    expected_list = [None if geom is not None and geom.is_empty else geom for geom in expected.tolist()]
    _assert_geometries_match(list(result.owned.to_shapely()), expected_list)


@pytest.mark.gpu
def test_geometry_array_clip_by_rect_preserves_multiline_parts_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        MultiLineString(
            [
                [(1.0, 1.0), (2.0, 2.0), (3.0, 2.0), (5.0, 3.0)],
                [(3.0, 4.0), (5.0, 7.0), (12.0, 2.0), (10.0, 5.0), (9.0, 7.5)],
            ]
        )
    ]
    array = geopandas.GeoSeries(values, crs="EPSG:4326").values

    result = array.clip_by_rect(0.0, 0.0, 10.0, 10.0)

    assert isinstance(result, DeviceGeometryArray)
    assert result.owned.residency is Residency.DEVICE
    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 10.0, 10.0)
    expected_list = [None if geom is not None and geom.is_empty else geom for geom in expected.tolist()]
    _assert_geometries_match(list(result.owned.to_shapely()), expected_list)


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
def test_clip_by_rect_auto_sticks_to_gpu_buffers_even_when_residency_marker_is_host() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 12.0, 12.0)],
        residency=Residency.DEVICE,
    )
    # Emulate GPU-native readers that preserve device_state while the public
    # residency marker still reads as host.
    owned.residency = Residency.HOST

    result = clip_by_rect_owned(owned, 0.0, 0.0, 2.0, 2.0)

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.owned_result is not None


@pytest.mark.gpu
def test_clip_by_rect_polygon_auto_uses_pair_owned_rectangle_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial
    import vibespatial.constructive.clip_rect as clip_rect_mod

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 4.0, 4.0),
            box(1.0, 1.0, 5.0, 5.0),
            box(10.0, 10.0, 12.0, 12.0),
        ],
        residency=Residency.DEVICE,
    )
    vibespatial.clear_dispatch_events()

    def _fail_generic_polygon_clip(*_args, **_kwargs):
        raise AssertionError("dense polygon rectangle clip should route through polygon_rect_intersection")

    monkeypatch.setattr(
        clip_rect_mod,
        "_clip_polygon_rings_gpu_device",
        _fail_generic_polygon_clip,
    )

    result = clip_by_rect_owned(owned, 0.0, 0.0, 2.0, 2.0)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result.runtime_selection.selected is ExecutionMode.GPU
    assert result.owned_result is not None
    assert any(
        event.surface == "vibespatial.kernels.constructive.polygon_rect_intersection"
        and event.selected is ExecutionMode.GPU
        for event in dispatch_events
    )


@pytest.mark.gpu
def test_clip_by_rect_gpu_line_path_avoids_host_metadata_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            LineString([(0.0, 0.0), (4.0, 4.0), (8.0, 0.0)]),
            LineString([(10.0, 10.0), (12.0, 12.0)]),
        ]
    )

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("GPU line clip should classify rows from device metadata")

    monkeypatch.setattr(owned_mod.OwnedGeometryArray, "_ensure_host_metadata", _fail_host_metadata)

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        6.0,
        6.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.owned_result is not None
    assert result.fallback_rows.size == 0


@pytest.mark.gpu
def test_clip_by_rect_gpu_line_row_map_stays_lazy_until_requested() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            LineString([(0.0, 0.0), (4.0, 4.0), (8.0, 0.0)]),
            LineString([(20.0, 20.0), (22.0, 22.0)]),
        ]
    )

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        6.0,
        6.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.owned_result is not None
    assert result._owned_result_rows is None
    assert result._owned_result_rows_factory is not None

    row_map = result.owned_result_rows

    assert isinstance(row_map, np.ndarray)
    assert row_map.dtype == np.int32
    assert np.array_equal(row_map, np.asarray([0], dtype=np.int32))


@pytest.mark.gpu
def test_clip_by_rect_gpu_row_classification_stays_lazy_until_requested() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            LineString([(0.0, 0.0), (4.0, 4.0), (8.0, 0.0)]),
            LineString([(20.0, 20.0), (22.0, 22.0)]),
        ]
    )

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        6.0,
        6.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result._candidate_rows is None
    assert result._candidate_rows_factory is not None
    assert result._fast_rows is None
    assert result._fast_rows_factory is not None
    assert result._fallback_rows is None
    assert result._fallback_rows_factory is not None
    assert np.array_equal(result.fast_rows, np.asarray([0, 1], dtype=np.int32))
    assert result.fallback_rows.size == 0


@pytest.mark.gpu
def test_clip_by_rect_gpu_dense_linestring_path_builds_mixed_outputs_without_generic_regroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.clip_rect as clip_rect_mod

    values = [
        LineString([(-1.0, 0.5), (0.5, 0.5), (1.5, 0.5)]),
        LineString(
            [
                (-1.0, 0.5),
                (0.5, 0.5),
                (0.5, 1.5),
                (1.5, 1.5),
                (1.5, 0.5),
                (3.0, 0.5),
            ]
        ),
        LineString([(10.0, 10.0), (12.0, 12.0)]),
    ]
    owned = _make_device_resident_with_host_stubs_cleared(values)

    def _fail_generic(*_args, **_kwargs):
        raise AssertionError("dense LineString rectangle clip should not use generic regroup/scatter")

    total_calls: list[int] = []
    original_totals = clip_rect_mod.count_scatter_totals

    def _record_count_scatter_totals(runtime, count_offset_pairs):
        total_calls.append(len(count_offset_pairs))
        return original_totals(runtime, count_offset_pairs)

    monkeypatch.setattr(clip_rect_mod, "_build_line_clip_device_result", _fail_generic)
    monkeypatch.setattr(clip_rect_mod, "concat_owned_scatter", _fail_generic)
    monkeypatch.setattr(clip_rect_mod, "count_scatter_totals", _record_count_scatter_totals)

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        2.0,
        1.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.owned_result is not None
    assert set(result.owned_result.families) == {
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
    }
    assert result.fallback_rows.size == 0
    assert result.owned_result_rows.tolist() == [0, 1]
    assert total_calls == [3]

    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 2.0, 1.0)
    expected_compact = [
        geom for geom in expected.tolist()
        if geom is not None and not geom.is_empty
    ]
    _assert_geometries_match(list(result.owned_result.to_shapely()), expected_compact)


@pytest.mark.gpu
def test_clip_by_rect_gpu_mixed_polygon_line_owned_result_keeps_both_families() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        LineString([(0.0, 0.0), (4.0, 4.0)]),
        box(0.0, 0.0, 4.0, 4.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        2.0,
        2.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.owned_result is not None
    assert set(result.owned_result.families) == {
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
    }
    assert np.array_equal(result.owned_result_rows, np.asarray([0, 1], dtype=np.int32))

    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 2.0, 2.0)
    expected_compact = [
        geom for geom in expected.tolist()
        if geom is not None and not geom.is_empty
    ]
    _assert_geometries_match(list(result.owned_result.to_shapely()), expected_compact)


@pytest.mark.gpu
def test_clip_by_rect_gpu_mixed_polygon_line_geometries_preserve_source_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        box(0.0, 0.0, 4.0, 4.0),
        LineString([(0.0, 0.0), (4.0, 4.0)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    result = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        2.0,
        2.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.clip_by_rect(np.asarray(values, dtype=object), 0.0, 0.0, 2.0, 2.0)
    _assert_geometries_match(list(result.geometries), expected.tolist())


@pytest.mark.gpu
def test_clip_by_rect_gpu_dense_linestring_path_reuses_cached_segment_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.clip_rect as clip_rect_mod

    owned = from_shapely_geometries(
        [
            LineString([(0.0, 0.0), (4.0, 4.0), (8.0, 0.0)]),
            LineString([(1.0, 0.0), (1.0, 4.0), (4.0, 4.0), (4.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )

    first = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        6.0,
        6.0,
        dispatch_mode=ExecutionMode.GPU,
    )
    assert first.owned_result is not None

    def _fail_extract(*_args, **_kwargs):
        raise AssertionError("repeat viewport clip should reuse cached line segments")

    monkeypatch.setattr(clip_rect_mod, "_extract_segments_vectorized", _fail_extract)

    second = clip_by_rect_owned(
        owned,
        0.0,
        0.0,
        2.0,
        2.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert second.owned_result is not None
    expected = shapely.clip_by_rect(
        np.asarray(
            [
                LineString([(0.0, 0.0), (4.0, 4.0), (8.0, 0.0)]),
                LineString([(1.0, 0.0), (1.0, 4.0), (4.0, 4.0), (4.0, 0.0)]),
            ],
            dtype=object,
        ),
        0.0,
        0.0,
        2.0,
        2.0,
    )
    expected_compact = [
        geom for geom in expected.tolist()
        if geom is not None and not geom.is_empty
    ]
    _assert_geometries_match(list(second.owned_result.to_shapely()), expected_compact)


def test_clip_rectangle_polygon_boundary_rows_bypass_generic_lower_dim_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    overlay_mod = importlib.import_module("vibespatial.api.tools.overlay")

    polygons = geopandas.GeoSeries(
        [
            box(0.2, 0.2, 0.8, 0.8),
            box(1.0, 0.2, 1.4, 0.8),
            box(1.0, 1.0, 1.4, 1.4),
        ],
        crs="EPSG:4326",
    )
    mask = box(0.0, 0.0, 1.0, 1.0)

    def _fail_lower_dim(*args, **kwargs):
        raise AssertionError("rectangle-only boundary rows should bypass generic lower-dim assembly")

    monkeypatch.setattr(
        overlay_mod,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        _fail_lower_dim,
    )

    result = geopandas.clip(polygons, mask)
    raw_expected = np.asarray(
        shapely.intersection(
            np.asarray(polygons.values._data, dtype=object),
            np.full(len(polygons), mask, dtype=object),
        ),
        dtype=object,
    )
    keep = ~shapely.is_missing(raw_expected) & ~shapely.is_empty(raw_expected)
    expected = raw_expected[keep]

    _assert_geometries_match(result.values._data.tolist(), expected.tolist())
    assert result.geom_type.tolist() == ["Polygon", "LineString", "Point"]


@pytest.mark.gpu
def test_device_rectangle_clip_boundary_rows_bypass_host_rectangle_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import importlib

    clip_mod = importlib.import_module("vibespatial.api.tools.clip")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            box(0.2, 0.2, 0.8, 0.8),
            box(1.0, 0.2, 1.4, 0.8),
            box(1.0, 1.0, 1.4, 1.4),
        ]
    )
    polygons = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(owned, crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    mask = box(0.0, 0.0, 1.0, 1.0)

    def _fail_rectangle_probe(*args, **kwargs):
        raise AssertionError("device rectangle rows should bypass the host rectangle probe")

    monkeypatch.setattr(
        clip_mod,
        "_exact_rectangle_clip_boundary_rows",
        _fail_rectangle_probe,
    )

    result = geopandas.clip(polygons, mask)
    raw_expected = np.asarray(
        shapely.intersection(
            np.asarray(polygons.values, dtype=object),
            np.full(len(polygons), mask, dtype=object),
        ),
        dtype=object,
    )
    keep = ~shapely.is_missing(raw_expected) & ~shapely.is_empty(raw_expected)
    expected = raw_expected[keep]

    _assert_geometries_match(result.values.tolist(), expected.tolist())
    assert result.geom_type.tolist() == ["Polygon", "LineString", "Point"]


@pytest.mark.gpu
def test_device_rectangle_clip_boundary_rows_stay_on_device_without_host_bounds_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import importlib

    cp = pytest.importorskip("cupy")
    clip_mod = importlib.import_module("vibespatial.api.tools.clip")

    owned = _make_device_resident_with_host_stubs_cleared(
        [
            box(0.2, 0.2, 0.8, 0.8),
            box(1.0, 0.2, 1.4, 0.8),
            box(1.0, 1.0, 1.4, 1.4),
        ]
    )
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:4326")
    bounds = np.asarray(
        [
            (0.2, 0.2, 0.8, 0.8),
            (1.0, 0.2, 1.4, 0.8),
            (1.0, 1.0, 1.4, 1.4),
        ],
        dtype=np.float64,
    )
    mask = box(0.0, 0.0, 1.0, 1.0)

    def _fail_asnumpy(*_args, **_kwargs):
        raise AssertionError("device rectangle boundary rows should not transfer bounds to host")

    original_asnumpy = cp.asnumpy
    monkeypatch.setattr(cp, "asnumpy", _fail_asnumpy)

    result = clip_mod._exact_rectangle_clip_boundary_owned_rows(
        values,
        bounds,
        (0.0, 0.0, 1.0, 1.0),
    )

    assert result is not None
    monkeypatch.setattr(cp, "asnumpy", original_asnumpy)
    actual = list(result.to_shapely())
    expected = list(
        shapely.intersection(
            np.asarray(values, dtype=object),
            np.full(len(values), mask, dtype=object),
        )
    )
    _assert_geometries_match(actual, expected)


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


@pytest.mark.gpu
def test_public_clip_device_rectangle_boundary_rows_stay_native(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        box(0.2, 0.2, 0.8, 0.8),
        box(1.0, 0.2, 1.4, 0.8),
        box(1.0, 1.0, 1.4, 1.4),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    series = geopandas.GeoSeries(DeviceGeometryArray._from_owned(owned), crs="EPSG:4326")
    mask = box(0.0, 0.0, 1.0, 1.0)

    original = owned_mod.from_shapely_geometries

    def _fail_rectangle_boundary_rewrap(*args, **kwargs):
        raise AssertionError("device rectangle boundary rows should not rewrap through from_shapely_geometries")

    monkeypatch.setattr(owned_mod, "from_shapely_geometries", _fail_rectangle_boundary_rewrap)
    try:
        result = geopandas.clip(series, mask)
    finally:
        monkeypatch.setattr(owned_mod, "from_shapely_geometries", original)

    assert isinstance(result.values, DeviceGeometryArray)
    expected = np.asarray(
        shapely.intersection(
            np.asarray(values, dtype=object),
            np.full(len(values), mask, dtype=object),
        ),
        dtype=object,
    )
    keep = ~shapely.is_missing(expected) & ~shapely.is_empty(expected)
    _assert_geometries_match(list(result.values.owned.to_shapely()), expected[keep].tolist())
