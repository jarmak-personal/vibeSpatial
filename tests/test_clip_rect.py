from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import benchmark_clip_by_rect, clip_by_rect_owned, from_shapely_geometries


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
