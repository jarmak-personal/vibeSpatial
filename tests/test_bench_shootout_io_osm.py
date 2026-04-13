from __future__ import annotations

import importlib.util
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import geopandas as gpd


def _load_osm_helpers():
    path = Path("benchmarks/shootout/io/_osm.py")
    spec = importlib.util.spec_from_file_location("shootout_io_osm", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combine_supported_osm_layers_separates_way_and_relation_multipolygons() -> None:
    helpers = _load_osm_helpers()
    points = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    multilinestrings = gpd.GeoDataFrame(
        geometry=[MultiPolygon([Polygon([(4, 4), (5, 4), (5, 5), (4, 4)])]).boundary],
        crs="EPSG:4326",
    )
    multipolygons = gpd.GeoDataFrame(
        {
            "osm_way_id": [11, None],
        },
        geometry=[
            MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]),
            MultiPolygon([Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])]),
        ],
        crs="EPSG:4326",
    )
    other_relations = gpd.GeoDataFrame(
        geometry=[MultiPolygon([Polygon([(6, 6), (7, 6), (7, 7), (6, 6)])]).boundary],
        crs="EPSG:4326",
    )

    combined = helpers.combine_supported_osm_layers(
        points,
        lines,
        multilinestrings,
        multipolygons,
        other_relations,
    )

    assert list(combined["osm_element"]) == [
        "node",
        "way",
        "relation",
        "way",
        "relation",
        "relation",
    ]


def test_osm_fingerprint_groups_linear_and_area_geometry_families() -> None:
    helpers = _load_osm_helpers()
    frame = gpd.GeoDataFrame(
        {
            "osm_element": ["node", "way", "way", "relation"],
        },
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            MultiPolygon([Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])]),
        ],
        crs="EPSG:4326",
    )

    fp = helpers.fingerprint(frame)

    assert "rows=4" in fp
    assert "elements=(node=1,way=2,relation=1,unknown=0)" in fp
    assert "geom=(point=1,linear=1,area=2,other=0)" in fp
