from __future__ import annotations

import pytest
from shapely.geometry import GeometryCollection, LineString, Point, box

import vibespatial.api as geopandas
import vibespatial.spatial_query as spatial_query_module
from vibespatial.runtime import has_gpu_runtime


def test_geopandas_buffer_unsupported_surface_uses_host_dispatch() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), LineString([(0, 0), (1, 0)])])

    result = series.buffer(1.0, quad_segs=1)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 2
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.array.buffer"
    assert dispatch_events[-1].implementation == "shapely_host"


def test_geopandas_sindex_query_owned_dispatch_is_observable() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(10, 10)])

    result = series.sindex.query(box(-1, -1, 1, 1), predicate="intersects")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == [0]
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    assert dispatch_events[-1].implementation in ("owned_cpu_spatial_query", "owned_gpu_spatial_query")
    assert dispatch_events[-1].selected in (geopandas.ExecutionMode.CPU, geopandas.ExecutionMode.GPU)


def test_geopandas_sindex_query_contains_uses_owned_dispatch() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(10, 10)])

    result = series.sindex.query(box(-1, -1, 1, 1), predicate="contains")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == [0]
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    assert dispatch_events[-1].implementation in {"owned_cpu_spatial_query", "owned_gpu_spatial_query"}


@pytest.mark.parametrize("predicate", [None, "intersects", "contains", "covers"])
def test_geopandas_sindex_query_selects_gpu_for_large_point_tree_box_queries(predicate: str | None) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(float(index), 0.0) for index in range(2048)])

    result = series.sindex.query(box(99.5, -1.0, 199.5, 1.0), predicate=predicate)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == list(range(100, 200))
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    if has_gpu_runtime():
        assert dispatch_events[-1].implementation == "owned_gpu_spatial_query"
        assert dispatch_events[-1].selected is geopandas.ExecutionMode.GPU
    else:
        assert dispatch_events[-1].implementation == "owned_cpu_spatial_query"
        assert dispatch_events[-1].selected is geopandas.ExecutionMode.CPU


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("contains_properly", list(range(101, 199))),
        ("touches", [100, 199]),
    ],
)
def test_geopandas_sindex_query_selects_gpu_for_boundary_sensitive_box_predicates(
    predicate: str,
    expected: list[int],
) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(float(index), 0.0) for index in range(2048)])

    result = series.sindex.query(box(100.0, -1.0, 199.0, 1.0), predicate=predicate)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == expected
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    if has_gpu_runtime():
        assert dispatch_events[-1].implementation == "owned_gpu_spatial_query"
        assert dispatch_events[-1].selected is geopandas.ExecutionMode.GPU
    else:
        assert dispatch_events[-1].implementation == "owned_cpu_spatial_query"
        assert dispatch_events[-1].selected is geopandas.ExecutionMode.CPU


def test_geopandas_sindex_query_within_uses_owned_dispatch() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(10, 10)])

    result = series.sindex.query(box(-1, -1, 1, 1), predicate="within")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == []
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    assert dispatch_events[-1].implementation in {"owned_cpu_spatial_query", "owned_gpu_spatial_query"}


def test_geopandas_sindex_query_scalar_box_avoids_query_owned_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for the raw scalar box fast path")

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(float(index), 0.0) for index in range(2048)])

    def _fail(values):
        raise AssertionError("public sindex.query scalar box path should not normalize query input to owned")

    monkeypatch.setattr(spatial_query_module, "_to_owned", _fail)

    result = series.sindex.query(box(99.5, -1.0, 199.5, 1.0), predicate="contains")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == list(range(100, 200))
    assert not events
    assert dispatch_events[-1].implementation == "owned_gpu_spatial_query"
    assert dispatch_events[-1].selected is geopandas.ExecutionMode.GPU


def test_geopandas_sindex_query_host_fallback_is_observable_for_unsupported_input() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(10, 10)])

    result = series.sindex.query(GeometryCollection([Point(0, 0)]), predicate="intersects")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == [0]
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    assert dispatch_events[-1].implementation == "strtree_host"
    assert dispatch_events[-1].selected is geopandas.ExecutionMode.CPU


def test_geopandas_sindex_query_multipoint_uses_gpu_dispatch() -> None:
    """MULTIPOINT query geometries should dispatch to GPU, not Shapely fallback."""
    from shapely.geometry import MultiPoint

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(1, 1), Point(10, 10)])

    result = series.sindex.query(MultiPoint([(0, 0), (1, 1)]), predicate="intersects")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert set(result.tolist()) == {0, 1}
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.query"
    assert dispatch_events[-1].implementation in ("owned_cpu_spatial_query", "owned_gpu_spatial_query")


def test_geopandas_sindex_nearest_fallback_is_observable() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    series = geopandas.GeoSeries([Point(0, 0), Point(10, 10)])

    result = series.sindex.nearest([Point(1, 1)])
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == [[0], [0]]
    assert not events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.sindex.nearest"
    assert dispatch_events[-1].implementation == "strtree_host"
