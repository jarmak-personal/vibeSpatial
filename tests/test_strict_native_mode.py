from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon, box

import vibespatial
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import (
    StrictNativeFallbackError,
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.testing import run_strict_api_matrix, strict_native_environment

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


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


@requires_gpu
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
    with strict_native_environment():
        gdf = _load_viewport_fixture(tmp_path, fixture_kind)
        bounds = gdf.geometry.bounds
        total_bounds = gdf.geometry.total_bounds

    assert bounds.values.tolist() == expected_bounds
    assert total_bounds.tolist() == expected_total_bounds


@requires_gpu
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
        ("lines", ("LineString",), True, 2, None, False, 1, True, 1, None, False, True, 2, None, False, False, None, "StrictNativeFallbackError", True),
        ("mixed", ("LineString", "Point", "Polygon"), True, 3, None, False, 3, False, None, "NotImplementedError", False, False, None, "NotImplementedError", False, False, None, "StrictNativeFallbackError", True),
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


@requires_gpu
def test_strict_clip_box_line_fixture_reaches_public_sindex_query_before_intersection_fallback(
    tmp_path: Path,
) -> None:
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

    assert by_surface["geodataframe.clip(box)"].ok is False
    assert by_surface["geodataframe.clip(box)"].error_type == "StrictNativeFallbackError"
    assert by_surface["geodataframe.clip(box)"].strict_fallback is True
    assert sindex._tree is None
    assert any(event.surface == "geopandas.sindex.query" for event in events)


@requires_gpu
def test_strict_clip_box_mixed_fixture_reaches_public_sindex_query_before_intersection_fallback(
    tmp_path: Path,
) -> None:
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

    assert by_surface["geodataframe.clip(box)"].ok is False
    assert by_surface["geodataframe.clip(box)"].error_type == "StrictNativeFallbackError"
    assert by_surface["geodataframe.clip(box)"].strict_fallback is True
    assert sindex._tree is None
    assert any(event.surface == "geopandas.sindex.query" for event in events)


@requires_gpu
def test_strict_api_matrix_report_is_structured_by_surface(tmp_path: Path) -> None:
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
