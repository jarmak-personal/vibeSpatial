from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import (
    clear_fallback_events,
    get_dispatch_events,
    get_fallback_events,
    has_gpu_runtime,
)


def test_geopandas_binary_methods_route_through_vibespatial_engine(monkeypatch) -> None:
    import vibespatial.api.geometry_array as geopandas_array

    calls: list[str] = []
    original = geopandas_array.evaluate_geopandas_binary_predicate

    def spy(predicate, left, right, **kwargs):
        calls.append(predicate)
        return original(predicate, left, right, **kwargs)

    monkeypatch.setattr(geopandas_array, "evaluate_geopandas_binary_predicate", spy)
    left = geopandas.GeoSeries([box(0, 0, 2, 2), box(10, 10, 11, 11)])
    right = geopandas.GeoSeries([Point(1, 1), Point(20, 20)])

    result = left.contains(right)

    assert calls == ["contains"]
    pd.testing.assert_series_equal(result, pd.Series([True, False]))


def test_geopandas_binary_predicates_use_owned_engine_without_fallback() -> None:
    geopandas.clear_dispatch_events()
    clear_fallback_events()

    left = geopandas.GeoSeries([box(0, 0, 2, 2)] * 20_000)
    right = geopandas.GeoSeries([Point(1, 1)] * 20_000)

    result = left.contains(right)

    assert bool(result.iloc[0]) is True
    assert get_fallback_events(clear=True) == []
    events = get_dispatch_events(clear=True)
    assert events[-1].surface == "geopandas.array.contains"
    assert events[-1].implementation in {"owned_gpu_predicate", "owned_cpu_predicate"}


@pytest.mark.gpu
def test_geopandas_binary_predicates_select_gpu_for_large_supported_batches() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geopandas.clear_dispatch_events()
    clear_fallback_events()

    left = geopandas.GeoSeries([box(0, 0, 2, 2)] * 20_000)
    right = geopandas.GeoSeries([Point(1, 1)] * 20_000)

    result = left.contains(right)

    assert bool(result.iloc[0]) is True
    assert get_fallback_events(clear=True) == []
    events = get_dispatch_events(clear=True)
    assert events[-1].surface == "geopandas.array.contains"
    assert events[-1].implementation == "owned_gpu_predicate"


@pytest.mark.gpu
def test_geopandas_binary_predicates_select_gpu_for_small_supported_batches() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geopandas.clear_dispatch_events()

    left = geopandas.GeoSeries([box(0, 0, 2, 2), box(10, 10, 11, 11)])
    right = geopandas.GeoSeries([Point(1, 1), Point(20, 20)])

    result = left.contains(right)

    assert result.tolist() == [True, False]
    events = get_dispatch_events(clear=True)
    assert events[-1].surface == "geopandas.array.contains"
    assert events[-1].implementation == "owned_gpu_predicate"


@pytest.mark.gpu
def test_geopandas_polygon_predicates_respect_hole_boundary_on_gpu() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    hole = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    inner = Polygon(hole)
    polygon_with_hole = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], [hole])
    left = geopandas.GeoSeries([inner])
    right = geopandas.GeoSeries([polygon_with_hole])

    geopandas.clear_dispatch_events()

    assert left.within(right).tolist() == [False]
    assert left.covered_by(right).tolist() == [False]
    assert left.touches(right).tolist() == [True]

    events = [
        event for event in get_dispatch_events(clear=True)
        if event.surface.startswith("geopandas.array.")
        and event.operation in {"within", "covered_by", "touches"}
    ]
    assert [event.implementation for event in events] == [
        "owned_gpu_predicate",
        "owned_gpu_predicate",
        "owned_gpu_predicate",
    ]
