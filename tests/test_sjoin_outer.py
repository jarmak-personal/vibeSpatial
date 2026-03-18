from __future__ import annotations

import shapely
from shapely.geometry import Point, box

import vibespatial.api as geopandas


def test_outer_sjoin_uses_owned_query_dispatch_for_supported_inputs() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    left = geopandas.GeoDataFrame({"left": [0, 1], "geometry": [Point(0, 0), Point(10, 10)]})
    right = geopandas.GeoDataFrame(
        {"right": [2, 3], "geometry": [box(-1, -1, 1, 1), box(20, 20, 21, 21)]}
    )

    result = geopandas.sjoin(left, right, how="outer", predicate="intersects")
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 3
    assert not fallback_events
    assert dispatch_events
    assert dispatch_events[-1].surface == "geopandas.tools.sjoin"
    assert dispatch_events[-1].implementation == "owned_spatial_query"


def test_outer_sjoin_preserves_unmatched_right_geometry_and_index_columns() -> None:
    left = geopandas.GeoDataFrame({"left": [0], "geometry": [Point(0, 0)]})
    right = geopandas.GeoDataFrame(
        {"right": [1, 2], "geometry": [box(-1, -1, 1, 1), box(10, 10, 11, 11)]}
    )

    result = geopandas.sjoin(left, right, how="outer", predicate="intersects")

    assert list(result.columns) == ["index_left", "left", "index_right", "right", "geometry"]
    assert result.index.tolist() == [0, 1]
    assert result["index_left"].iloc[0] == 0.0
    assert result["index_left"].isna().iloc[1]
    assert result["index_right"].tolist() == [0, 1]
    assert result["right"].tolist() == [1, 2]
    assert shapely.equals(result.geometry.iloc[0], Point(0, 0))
    assert shapely.equals(result.geometry.iloc[1], box(10, 10, 11, 11))


def test_geometry_array_owned_spatial_index_is_cached_for_outer_sjoin() -> None:
    right = geopandas.GeoDataFrame({"geometry": [box(-1, -1, 1, 1), box(10, 10, 11, 11)]})

    owned1, flat1 = right.geometry.values.owned_flat_sindex()
    owned2, flat2 = right.geometry.values.owned_flat_sindex()

    assert owned1 is owned2
    assert flat1 is flat2
