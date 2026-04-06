from __future__ import annotations

import importlib

import pandas as pd
import pandas.testing as pdt
import pytest
from shapely.geometry import Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial.api.testing import assert_geodataframe_equal

dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")


def _build_lazy_frame():
    return geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["a", "a", "b", "b"]),
            "value": [1, 2, 3, 4],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(10, 10, 11, 11),
                box(11, 10, 12, 11),
            ],
        },
        crs="EPSG:3857",
    )


def test_dissolve_lazy_intersects_scalar_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _build_lazy_frame()
    lazy = frame.dissolve_lazy(by="group", method="coverage")
    expected = frame.dissolve(by="group", method="coverage")

    calls = 0
    real_fn = dissolve_module.execute_grouped_union

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "execute_grouped_union", _counting)
    result = lazy.intersects(box(0.5, 0.25, 0.75, 0.75))

    assert calls == 0
    expected_result = expected.geometry.intersects(box(0.5, 0.25, 0.75, 0.75), align=False)
    pdt.assert_series_equal(result, expected_result)


def test_dissolve_lazy_contains_point_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _build_lazy_frame()
    lazy = frame.dissolve_lazy(by="group", method="coverage")
    expected = frame.dissolve(by="group", method="coverage")

    calls = 0
    real_fn = dissolve_module.execute_grouped_union

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "execute_grouped_union", _counting)
    result = lazy.contains(Point(0.5, 0.5))

    assert calls == 0
    expected_result = expected.geometry.contains(Point(0.5, 0.5), align=False)
    pdt.assert_series_equal(result, expected_result)


def test_dissolve_lazy_contains_boundary_point_materializes_for_exact_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["a", "a"]),
            "value": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (0, 1)]),
                Polygon([(1, 0), (1, 1), (0, 1)]),
            ],
        },
        crs="EPSG:3857",
    )
    lazy = frame.dissolve_lazy(by="group", method="coverage")
    expected = frame.dissolve(by="group", method="coverage")

    calls = 0
    real_fn = dissolve_module.execute_grouped_union

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "execute_grouped_union", _counting)
    result = lazy.contains(Point(0.5, 0.5))

    assert calls == 1
    expected_result = expected.geometry.contains(Point(0.5, 0.5), align=False)
    pdt.assert_series_equal(result, expected_result)


def test_dissolve_lazy_contains_polygon_materializes_for_exact_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _build_lazy_frame()
    lazy = frame.dissolve_lazy(by="group", method="coverage")
    expected = frame.dissolve(by="group", method="coverage")

    calls = 0
    real_fn = dissolve_module.execute_grouped_union

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "execute_grouped_union", _counting)
    result = lazy.contains(box(0, 0, 2, 1))

    assert calls == 1
    expected_result = expected.geometry.contains(box(0, 0, 2, 1), align=False)
    pdt.assert_series_equal(result, expected_result)


def test_dissolve_lazy_materialize_matches_dissolve() -> None:
    frame = _build_lazy_frame()
    lazy = frame.dissolve_lazy(by="group", method="coverage")
    expected = frame.dissolve(by="group", method="coverage")

    actual = lazy.materialize()

    assert_geodataframe_equal(actual, expected)
