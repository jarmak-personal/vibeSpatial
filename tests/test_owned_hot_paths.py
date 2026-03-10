from __future__ import annotations

import pytest
from shapely.geometry import Point

import vibespatial.api as geopandas
from vibespatial.api import geo_base as base_module


def test_geometry_array_to_owned_is_cached_until_mutation() -> None:
    series = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
    array = series.geometry.values

    first = array.to_owned()
    second = array.to_owned()

    assert first is second

    array[0] = Point(2, 2)
    third = array.to_owned()

    assert third is not first


def test_geo_method_delegate_reuses_existing_geometry_array(monkeypatch: pytest.MonkeyPatch) -> None:
    series = geopandas.GeoSeries([Point(0, 0)])

    def _unexpected_constructor(*args, **kwargs):
        raise AssertionError("delegate should reuse existing GeometryArray instead of reconstructing it")

    monkeypatch.setattr(base_module, "GeometryArray", _unexpected_constructor)

    result = series.buffer(1.0, quad_segs=1)

    assert len(result) == 1
