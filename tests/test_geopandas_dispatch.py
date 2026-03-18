from __future__ import annotations

from shapely.geometry import Point, Polygon

import vibespatial.api as geopandas


def test_public_dispatch_events_are_observable() -> None:
    geopandas.clear_dispatch_events()

    geopandas.GeoSeries([Point(0, 0)]).buffer(1.0, quad_segs=1)
    geopandas.GeoSeries([Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])]).make_valid()
    geopandas.GeoDataFrame(
        {
            "group": [0, 0],
            "value": [1, 2],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    ).dissolve("group")

    events = geopandas.get_dispatch_events(clear=True)

    assert [event.surface for event in events[-3:]] == [
        "geopandas.array.buffer",
        "geopandas.array.make_valid",
        "geopandas.geodataframe.dissolve",
    ]
    assert events[-3].implementation == "owned_stroke_kernel"
    assert events[-2].implementation == "owned_compact_invalid_rows"
    assert events[-1].implementation == "grouped_union_pipeline"
