from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

import vibespatial
from benchmarks.shootout._data import setup_fixtures
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime.fallbacks import (
    StrictNativeFallbackError,
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.testing import run_strict_api_matrix, strict_native_environment


def _require_gpu_runtime() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU required")


def _build_viewport_fixture(kind: str) -> vibespatial.GeoDataFrame:
    if kind == "lines":
        geometry = [
            LineString([(0.0, 0.0), (10.0, 10.0)]),
            LineString([(20.0, 20.0), (30.0, 30.0)]),
        ]
    elif kind == "mixed":
        geometry = [
            LineString([(0.0, 0.0), (10.0, 10.0)]),
            Polygon([(2.0, 2.0), (9.0, 2.0), (9.0, 7.0), (2.0, 7.0), (2.0, 2.0)]),
            Point(4.0, 4.0),
        ]
    else:
        raise ValueError(f"unsupported fixture kind: {kind}")

    return vibespatial.GeoDataFrame({"geometry": geometry}, crs="EPSG:4326")


def _load_viewport_fixture(tmp_path: Path, kind: str):
    path = tmp_path / f"{kind}.geojson"
    _build_viewport_fixture(kind).to_file(path, driver="GeoJSON")
    return vibespatial.read_file(path, build_index=True)


def _viewport_call_matrix():
    rect = (1.0, 1.0, 6.0, 6.0)
    bbox = box(*rect)
    return rect, {
        "geometry.clip_by_rect": lambda gdf: gdf.geometry.clip_by_rect(*rect),
        "geodataframe.clip_by_rect": lambda gdf: gdf.clip_by_rect(*rect),
        "geodataframe.cx": lambda gdf: gdf.cx[rect[0] : rect[2], rect[1] : rect[3]],
        "geometry.intersects(box)": lambda gdf: gdf.geometry.intersects(bbox),
        "sindex.query(box, intersects)": lambda gdf: gdf.sindex.query(bbox, predicate="intersects"),
        "geodataframe.clip(box)": lambda gdf: gdf.clip(bbox),
    }


def test_strict_native_mode_disabled_by_default() -> None:
    assert strict_native_mode_enabled() is False


def test_record_fallback_event_raises_in_strict_native_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")

    with pytest.raises(StrictNativeFallbackError):
        record_fallback_event(surface="geopandas.array.contains", reason="explicit CPU fallback")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("fixture_kind", "expected_bounds", "expected_total_bounds"),
    [
        (
            "lines",
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            [0.0, 0.0, 30.0, 30.0],
        ),
        (
            "mixed",
            [[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 9.0, 7.0], [4.0, 4.0, 4.0, 4.0]],
            [0.0, 0.0, 10.0, 10.0],
        ),
    ],
)
def test_strict_bounds_surfaces_succeed_for_viewport_fixtures(
    tmp_path: Path,
    fixture_kind: str,
    expected_bounds: list[list[float]],
    expected_total_bounds: list[float],
) -> None:
    _require_gpu_runtime()
    with strict_native_environment():
        gdf = _load_viewport_fixture(tmp_path, fixture_kind)
        bounds = gdf.geometry.bounds
        total_bounds = gdf.geometry.total_bounds

    assert bounds.values.tolist() == expected_bounds
    assert total_bounds.tolist() == expected_total_bounds


@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "fixture_kind",
        "expected_geometry_types",
        "expected_clip_by_rect_ok",
        "expected_clip_by_rect_len",
        "expected_clip_by_rect_error_type",
        "expected_clip_by_rect_strict_fallback",
        "expected_sindex_len",
        "expected_cx_ok",
        "expected_cx_len",
        "expected_cx_error_type",
        "expected_cx_strict_fallback",
        "expected_intersects_ok",
        "expected_intersects_len",
        "expected_intersects_error_type",
        "expected_intersects_strict_fallback",
        "expected_clip_ok",
        "expected_clip_len",
        "expected_clip_error_type",
        "expected_clip_strict_fallback",
    ),
    [
        ("lines", ("LineString",), True, 2, None, False, 1, True, 1, None, False, True, 2, None, False, True, 1, None, False),
        ("mixed", ("LineString", "Point", "Polygon"), True, 3, None, False, 3, True, 3, None, False, True, 3, None, False, True, 3, None, False),
    ],
)
def test_strict_viewport_matrix_documents_current_public_api_behavior(
    tmp_path: Path,
    fixture_kind: str,
    expected_geometry_types: tuple[str, ...],
    expected_clip_by_rect_ok: bool,
    expected_clip_by_rect_len: int | None,
    expected_clip_by_rect_error_type: str | None,
    expected_clip_by_rect_strict_fallback: bool,
    expected_sindex_len: int,
    expected_cx_ok: bool,
    expected_cx_len: int | None,
    expected_cx_error_type: str | None,
    expected_cx_strict_fallback: bool,
    expected_intersects_ok: bool,
    expected_intersects_len: int | None,
    expected_intersects_error_type: str | None,
    expected_intersects_strict_fallback: bool,
    expected_clip_ok: bool,
    expected_clip_len: int | None,
    expected_clip_error_type: str | None,
    expected_clip_strict_fallback: bool,
) -> None:
    _require_gpu_runtime()
    _, calls = _viewport_call_matrix()

    with strict_native_environment():
        gdf = _load_viewport_fixture(tmp_path, fixture_kind)
        report = run_strict_api_matrix(
            fixture_kind,
            gdf,
            calls,
            geometry_types=tuple(sorted({str(value) for value in gdf.geometry.geom_type})),
        )

    payload = report.to_dict()
    by_surface = report.by_surface()

    assert tuple(payload["geometry_types"]) == expected_geometry_types

    assert by_surface["geometry.clip_by_rect"].ok is expected_clip_by_rect_ok
    assert by_surface["geometry.clip_by_rect"].result_len == expected_clip_by_rect_len
    assert by_surface["geometry.clip_by_rect"].error_type == expected_clip_by_rect_error_type
    assert by_surface["geometry.clip_by_rect"].strict_fallback is expected_clip_by_rect_strict_fallback
    if expected_clip_by_rect_ok:
        assert by_surface["geometry.clip_by_rect"].result_type == "GeoSeries"

    assert by_surface["geodataframe.clip_by_rect"].ok is expected_clip_by_rect_ok
    assert by_surface["geodataframe.clip_by_rect"].result_len == expected_clip_by_rect_len
    assert by_surface["geodataframe.clip_by_rect"].error_type == expected_clip_by_rect_error_type
    assert by_surface["geodataframe.clip_by_rect"].strict_fallback is expected_clip_by_rect_strict_fallback
    if expected_clip_by_rect_ok:
        assert by_surface["geodataframe.clip_by_rect"].result_type == "GeoSeries"

    assert by_surface["geodataframe.cx"].ok is expected_cx_ok
    assert by_surface["geodataframe.cx"].result_len == expected_cx_len
    assert by_surface["geodataframe.cx"].error_type == expected_cx_error_type
    assert by_surface["geodataframe.cx"].strict_fallback is expected_cx_strict_fallback
    if expected_cx_ok:
        assert by_surface["geodataframe.cx"].result_type == "GeoDataFrame"

    assert by_surface["geometry.intersects(box)"].ok is expected_intersects_ok
    assert by_surface["geometry.intersects(box)"].result_len == expected_intersects_len
    assert by_surface["geometry.intersects(box)"].error_type == expected_intersects_error_type
    assert by_surface["geometry.intersects(box)"].strict_fallback is expected_intersects_strict_fallback
    if expected_intersects_ok:
        assert by_surface["geometry.intersects(box)"].result_type == "Series"

    assert by_surface["sindex.query(box, intersects)"].ok is True
    assert by_surface["sindex.query(box, intersects)"].result_type == "ndarray"
    assert by_surface["sindex.query(box, intersects)"].result_len == expected_sindex_len

    assert by_surface["geodataframe.clip(box)"].ok is expected_clip_ok
    assert by_surface["geodataframe.clip(box)"].strict_fallback is expected_clip_strict_fallback
    assert by_surface["geodataframe.clip(box)"].result_len == expected_clip_len
    assert by_surface["geodataframe.clip(box)"].error_type == expected_clip_error_type
    if expected_clip_ok:
        assert by_surface["geodataframe.clip(box)"].result_type == "GeoDataFrame"


@pytest.mark.gpu
def test_strict_clip_box_line_fixture_uses_direct_bbox_candidates_before_gpu_intersection(
    tmp_path: Path,
) -> None:
    _require_gpu_runtime()
    _, calls = _viewport_call_matrix()

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        gdf = _load_viewport_fixture(tmp_path, "lines")
        sindex = gdf.sindex

        assert sindex._tree is None

        report = run_strict_api_matrix(
            "lines",
            gdf,
            {"geodataframe.clip(box)": calls["geodataframe.clip(box)"]},
            geometry_types=tuple(sorted({str(value) for value in gdf.geometry.geom_type})),
        )
        events = vibespatial.get_dispatch_events(clear=True)

    by_surface = report.by_surface()

    assert by_surface["geodataframe.clip(box)"].ok is True
    assert by_surface["geodataframe.clip(box)"].error_type is None
    assert by_surface["geodataframe.clip(box)"].strict_fallback is False
    assert by_surface["geodataframe.clip(box)"].result_len == 1
    assert sindex._tree is None
    assert not any(event.surface == "geopandas.sindex.query" for event in events)
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.implementation == "binary_constructive_gpu"
        for event in events
    )


@pytest.mark.gpu
def test_strict_clip_polygon_mask_keeps_polygon_postprocess_on_device() -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {
            "building_id": [1, 2],
            "geometry": vibespatial.GeoSeries(
                DeviceGeometryArray._from_owned(
                    from_shapely_geometries(
                        [
                            Polygon(
                                [
                                    (0.0, 0.0),
                                    (8.0, 0.0),
                                    (8.0, 0.0),
                                    (8.0, 8.0),
                                    (0.0, 8.0),
                                    (0.0, 0.0),
                                ]
                            ),
                            Polygon(
                                [
                                    (20.0, 20.0),
                                    (28.0, 20.0),
                                    (28.0, 28.0),
                                    (20.0, 28.0),
                                    (20.0, 20.0),
                                ]
                            ),
                        ]
                    )
                ),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    admin = vibespatial.GeoDataFrame(
        {"geometry": [box(2.0, 2.0, 10.0, 10.0)]},
        crs="EPSG:3857",
    )

    assert isinstance(buildings.geometry.values, DeviceGeometryArray)

    with strict_native_environment():
        result = vibespatial.clip(buildings, admin)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(box(2.0, 2.0, 8.0, 8.0))


@pytest.mark.gpu
def test_strict_clip_concave_polygon_mask_matches_shapely_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_runtime()
    monkeypatch.setenv("VSBENCH_SCALE", "100")
    fixtures = setup_fixtures(tmp_path)

    buildings = vibespatial.read_parquet(fixtures["buildings"])
    admin = vibespatial.read_file(fixtures["admin_boundary"])
    mask_geom = admin.geometry.iloc[0]

    source_geoms = np.asarray(buildings.geometry.values, dtype=object)
    expected_geoms = np.asarray(shapely.intersection(source_geoms, mask_geom), dtype=object)
    keep = ~shapely.is_empty(expected_geoms)
    expected_index = buildings.index.to_numpy()[keep]
    expected_norm = shapely.normalize(expected_geoms[keep])

    with strict_native_environment():
        result = vibespatial.clip(buildings, admin, sort=True)

    assert int(result.geometry.isna().sum()) == 0
    assert result.index.to_numpy().tolist() == expected_index.tolist()
    assert len(result) == len(expected_norm)
    actual = shapely.normalize(np.asarray(result.geometry.values, dtype=object))
    assert np.asarray(
        shapely.equals(actual, expected_norm),
        dtype=bool,
    ).tolist() == [True] * len(expected_norm)


@pytest.mark.gpu
def test_strict_clip_concave_polygon_mask_drops_bbox_false_positives() -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {
            "geometry": [
                box(0.5, 0.5, 1.5, 1.5),
                box(2.4, 2.4, 2.8, 2.8),
                box(1.4, 1.4, 2.2, 2.2),
            ]
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
            (0.0, 0.0),
        ]
    )
    admin = vibespatial.GeoDataFrame({"geometry": [mask]}, crs="EPSG:3857")

    with strict_native_environment():
        result = vibespatial.clip(buildings, admin)

    expected_geoms = np.asarray(
        shapely.intersection(np.asarray(buildings.geometry.values, dtype=object), mask),
        dtype=object,
    )
    keep = ~shapely.is_empty(expected_geoms)
    expected_index = buildings.index.to_numpy()[keep]
    expected_norm = shapely.normalize(expected_geoms[keep])

    assert int(result.geometry.isna().sum()) == 0
    assert result.index.to_numpy().tolist() == expected_index.tolist()
    actual = shapely.normalize(np.asarray(result.geometry.values, dtype=object))
    assert np.asarray(shapely.equals(actual, expected_norm), dtype=bool).tolist() == [True] * len(expected_norm)


@pytest.mark.gpu
def test_strict_clip_polygon_mask_uses_rectangle_kernel_fast_path() -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 14.0, 14.0)]},
        crs="EPSG:3857",
    )
    admin = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 1.0),
                        (12.0, 2.0),
                        (11.0, 12.0),
                        (2.0, 11.0),
                        (1.0, 1.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = vibespatial.clip(buildings, admin)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert any(
        event.surface == "vibespatial.kernels.constructive.polygon_rect_intersection"
        and event.selected.value == "gpu"
        for event in events
    )


@pytest.mark.gpu
def test_strict_clip_concave_polygon_mask_keeps_exact_subset_on_gpu() -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 14.0, 14.0)]},
        crs="EPSG:3857",
    )
    admin = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (0.0, 0.0),
                        (12.0, 0.0),
                        (12.0, 4.0),
                        (4.0, 4.0),
                        (4.0, 12.0),
                        (0.0, 12.0),
                        (0.0, 0.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = vibespatial.clip(buildings, admin)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert any(
        event.surface == "vibespatial.predicates.binary"
        and event.operation == "intersects"
        and event.selected.value == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "vibespatial.kernels.constructive.polygon_rect_intersection"
        and event.selected.value == "gpu"
        for event in events
    )
    assert all(
        getattr(getattr(event, "selected", None), "value", None) != "cpu"
        for event in events
    )


@pytest.mark.gpu
def test_strict_clip_polygon_mask_preserves_geometry_array_after_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 14.0, 14.0)]},
        crs="EPSG:3857",
    )
    admin = vibespatial.GeoDataFrame(
        {"geometry": [box(1.0, 1.0, 12.0, 12.0)]},
        crs="EPSG:3857",
    )

    clip_module = importlib.import_module("vibespatial.api.tools.clip")

    called = False
    original = clip_module._geometryarray_from_shapely

    def _wrapped_from_shapely(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        clip_module,
        "_geometryarray_from_shapely",
        _wrapped_from_shapely,
    )

    with strict_native_environment():
        result = vibespatial.clip(buildings, admin)

    assert not called
    assert isinstance(result.geometry.values, GeometryArray | DeviceGeometryArray)


@pytest.mark.gpu
def test_strict_clip_concave_polygon_mask_preserves_multipart_results() -> None:
    _require_gpu_runtime()
    source = vibespatial.GeoDataFrame(
        {"geometry": [box(4.0, 0.5, 7.0, 5.5)]},
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 1.0),
            (1.0, 1.0),
            (1.0, 5.0),
            (6.0, 5.0),
            (6.0, 6.0),
            (0.0, 6.0),
            (0.0, 0.0),
        ]
    )

    with strict_native_environment():
        result = vibespatial.clip(source, mask)

    expected = shapely.intersection(source.geometry.iloc[0], mask)
    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == "MultiPolygon"
    assert result.geometry.iloc[0].equals(expected)


@pytest.mark.gpu
def test_strict_clip_rectangular_polygon_mask_keeps_exact_polygon_semantics() -> None:
    _require_gpu_runtime()
    source = vibespatial.GeoSeries(
        [LineString([(-1.0, -1.0), (0.0, 0.0)])],
        crs="EPSG:3857",
    )
    mask = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])

    with strict_native_environment():
        result = vibespatial.clip(source, mask)

    assert len(result) == 1
    assert result.iloc[0].geom_type == "Point"
    assert result.iloc[0].equals(Point(0.0, 0.0))


@pytest.mark.gpu
def test_strict_clip_box_skips_internal_geoseries_area_warning() -> None:
    _require_gpu_runtime()
    buildings = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 14.0, 14.0)]},
        crs="EPSG:4326",
    )
    mask = box(1.0, 1.0, 12.0, 12.0)

    with strict_native_environment():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = vibespatial.clip(buildings, mask)

    messages = [str(entry.message) for entry in caught]
    assert not any("Results from 'area' are likely incorrect" in message for message in messages)
    assert len(result) == 2


@pytest.mark.gpu
def test_strict_overlay_polygon_query_requests_device_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_runtime()
    left = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 14.0, 14.0)]},
        crs="EPSG:3857",
    )
    right = vibespatial.GeoDataFrame(
        {"geometry": [box(1.0, 1.0, 12.0, 12.0)]},
        crs="EPSG:3857",
    )

    sindex_module = importlib.import_module("vibespatial.api.sindex")
    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
    original_query = sindex_module.SpatialIndex.query
    seen_return_device = []

    def _wrapped_query(self, geometry, *args, **kwargs):
        seen_return_device.append(kwargs.get("return_device", False))
        return original_query(self, geometry, *args, **kwargs)

    monkeypatch.setattr(
        sindex_module.SpatialIndex,
        "query",
        _wrapped_query,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS",
        0,
    )

    with strict_native_environment():
        result = vibespatial.overlay(left, right, how="intersection")

    assert seen_return_device
    assert any(seen_return_device)
    assert len(result) == 2


@pytest.mark.gpu
def test_strict_clip_box_mixed_fixture_uses_direct_bbox_candidates_before_gpu_intersection(
    tmp_path: Path,
) -> None:
    _require_gpu_runtime()
    _, calls = _viewport_call_matrix()

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        gdf = _load_viewport_fixture(tmp_path, "mixed")
        sindex = gdf.sindex

        assert sindex._tree is None

        report = run_strict_api_matrix(
            "mixed",
            gdf,
            {"geodataframe.clip(box)": calls["geodataframe.clip(box)"]},
            geometry_types=tuple(sorted({str(value) for value in gdf.geometry.geom_type})),
        )
        events = vibespatial.get_dispatch_events(clear=True)

    by_surface = report.by_surface()

    assert by_surface["geodataframe.clip(box)"].ok is True
    assert by_surface["geodataframe.clip(box)"].error_type is None
    assert by_surface["geodataframe.clip(box)"].strict_fallback is False
    assert by_surface["geodataframe.clip(box)"].result_len == 3
    assert sindex._tree is None
    assert not any(event.surface == "geopandas.sindex.query" for event in events)
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.implementation == "binary_constructive_gpu"
        for event in events
    )


@pytest.mark.gpu
def test_strict_api_matrix_report_is_structured_by_surface(tmp_path: Path) -> None:
    _require_gpu_runtime()
    _, calls = _viewport_call_matrix()

    with strict_native_environment():
        gdf = _load_viewport_fixture(tmp_path, "lines")
        report = run_strict_api_matrix(
            "lines",
            gdf,
            calls,
            geometry_types=tuple(sorted({str(value) for value in gdf.geometry.geom_type})),
        )

    payload = report.to_dict()

    assert payload["fixture"] == "lines"
    assert set(payload["calls"]) == set(calls)
    assert payload["calls"]["geodataframe.cx"]["strict_fallback"] is False
    assert payload["calls"]["geometry.clip_by_rect"]["ok"] is True
    assert payload["calls"]["geometry.clip_by_rect"]["error_type"] is None
