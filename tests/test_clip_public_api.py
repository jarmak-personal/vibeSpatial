from __future__ import annotations

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

import vibespatial
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.clip import clip


def _build_mixed_viewport_fixture() -> vibespatial.GeoDataFrame:
    return vibespatial.GeoDataFrame(
        {
            "geometry": [
                LineString([(0.0, 0.0), (10.0, 10.0)]),
                Polygon([(2.0, 2.0), (9.0, 2.0), (9.0, 7.0), (2.0, 7.0), (2.0, 2.0)]),
                Point(4.0, 4.0),
            ]
        },
        crs="EPSG:3857",
    )


def test_clip_scalar_mask_rectangle_fast_path_still_keeps_mixed_rows_stable(
    monkeypatch,
) -> None:
    gdf = _build_mixed_viewport_fixture()
    mask = box(1.0, 1.0, 6.0, 6.0)
    seen: list[tuple[str, ...]] = []

    original = GeometryArray._constructive_or_fallback

    def wrapped(self, op, other, **kwargs):
        if op == "intersection":
            seen.append(tuple(self.geom_type.tolist()))
        return original(self, op, other, **kwargs)

    monkeypatch.setattr(GeometryArray, "_constructive_or_fallback", wrapped)

    result = clip(gdf, mask)

    assert len(result) == 3
    assert set(result.geometry.to_wkt().tolist()) == {
        "LINESTRING (1 1, 6 6)",
        "POLYGON ((2 2, 2 6, 6 6, 6 2, 2 2))",
        "POINT (4 4)",
    }
    assert seen[0] == ("LineString",)
    assert all(types == ("LineString",) for types in seen)
    assert isinstance(result.geometry.values, GeometryArray)


def test_clip_polygon_rectangle_mask_routes_multilinestring_rows_through_rect_fast_path(
    monkeypatch,
) -> None:
    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": [
                MultiLineString(
                    [
                        [(1.0, 1.0), (2.0, 2.0), (3.0, 2.0), (5.0, 3.0)],
                        [(3.0, 4.0), (5.0, 7.0), (12.0, 2.0), (10.0, 5.0), (9.0, 7.5)],
                    ]
                ),
                LineString([(2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 2.0)]),
            ]
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 10.0, 10.0)]},
        crs="EPSG:3857",
    )
    seen: list[tuple[str, str]] = []

    original_clip_by_rect = GeometryArray.clip_by_rect

    def wrapped_clip_by_rect(self, xmin, ymin, xmax, ymax):
        seen.append(("clip_by_rect", ",".join(self.geom_type.tolist())))
        return original_clip_by_rect(self, xmin, ymin, xmax, ymax)

    monkeypatch.setattr(GeometryArray, "clip_by_rect", wrapped_clip_by_rect)

    result = clip(gdf, mask)

    assert set(result.geom_type.tolist()) == {"MultiLineString", "LineString"}
    assert ("clip_by_rect", "MultiLineString") in seen


def test_clip_polygon_rectangle_mask_preserves_polygon_line_slivers_as_geometry_collection() -> None:
    points = vibespatial.GeoDataFrame(
        {"geometry": [Point(2.0, 2.0), Point(3.0, 4.0), Point(9.0, 8.0), Point(-12.0, -15.0)]},
        crs="EPSG:3857",
    )
    buffered = points.copy()
    buffered["geometry"] = buffered.buffer(4.0)
    mask = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 10.0, 10.0)]},
        crs="EPSG:3857",
    )

    donut = vibespatial.overlay(buffered, mask, how="symmetric_difference")
    multi_poly = vibespatial.GeoDataFrame(
        {"geometry": vibespatial.GeoSeries([donut.union_all()], crs="EPSG:3857")},
        crs="EPSG:3857",
    )

    result = clip(multi_poly, mask)

    assert result.geom_type.iloc[0] == "GeometryCollection"
    assert tuple(result.total_bounds) == tuple(mask.total_bounds)
