from __future__ import annotations

import importlib

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

import vibespatial
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.clip import clip

clip_module = importlib.import_module("vibespatial.api.tools.clip")


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

    original = GeometryArray.clip_by_rect

    def wrapped(self, xmin, ymin, xmax, ymax):
        seen.append(tuple(self.geom_type.tolist()))
        return original(self, xmin, ymin, xmax, ymax)

    monkeypatch.setattr(GeometryArray, "clip_by_rect", wrapped)

    result = clip(gdf, mask)

    assert len(result) == 3
    assert set(result.geometry.to_wkt().tolist()) == {
        "LINESTRING (1 1, 6 6)",
        "POLYGON ((2 2, 2 6, 6 6, 6 2, 2 2))",
        "POINT (4 4)",
    }
    assert seen[0] == ("LineString",)
    assert set(seen) == {("LineString",), ("Polygon",)}
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


def test_clip_polygon_mask_zero_area_filter_copies_keep_mask_before_mutation(
    monkeypatch,
) -> None:
    gdf = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 2.0, 2.0)]},
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 2.0, 2.0)]},
        crs="EPSG:3857",
    )

    real_asarray = clip_module.np.asarray

    def _readonly_bool_asarray(value, *args, **kwargs):
        arr = real_asarray(value, *args, **kwargs)
        dtype = kwargs.get("dtype", args[0] if args else None)
        if dtype is bool and getattr(arr, "ndim", 0) == 1 and getattr(arr, "size", -1) == len(gdf):
            readonly = np.array(arr, copy=True)
            readonly.setflags(write=False)
            return readonly
        return arr

    monkeypatch.setattr(clip_module.np, "asarray", _readonly_bool_asarray)
    monkeypatch.setattr(
        clip_module.shapely,
        "area",
        lambda values: np.zeros(len(real_asarray(values, dtype=object)), dtype=np.float64),
    )

    result = clip(gdf, mask)

    assert len(result) == 0
