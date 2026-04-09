from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

import vibespatial
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.clip import clip
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

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


def test_clip_scalar_polygon_rectangle_mask_keeps_mixed_rows_stable(
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
    actual = shapely.normalize(np.asarray(result.geometry.values, dtype=object))
    expected = shapely.normalize(
        np.asarray(
            [
                LineString([(1.0, 1.0), (6.0, 6.0)]),
                Polygon([(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0), (2.0, 2.0)]),
                Point(4.0, 4.0),
            ],
            dtype=object,
        )
    )
    assert {geom.wkb for geom in actual} == {geom.wkb for geom in expected}
    assert seen == []
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


def test_clip_polygon_rectangle_mask_multilinestring_survives_strict_native_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        return

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

    with strict_native_environment():
        result = clip(gdf, mask)

    assert set(result.geom_type.tolist()) == {"MultiLineString", "LineString"}


def test_clip_polygon_rectangle_mask_donut_preserves_geometry_collection_slivers() -> None:
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


def test_clip_polygon_mask_preserves_device_backing_for_polygon_workloads() -> None:
    if not vibespatial.has_gpu_runtime():
        return

    buildings_owned = from_shapely_geometries(
        [
            box(2.0, 2.0, 4.0, 4.0),
            box(4.0, 4.0, 6.0, 6.0),
            box(6.0, 6.0, 8.0, 8.0),
            box(8.0, 8.0, 10.0, 10.0),
        ],
        residency=Residency.DEVICE,
    )
    buildings = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                buildings_owned,
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 3.0),
                        (5.0, 1.0),
                        (9.0, 3.0),
                        (7.0, 7.0),
                        (3.0, 7.0),
                        (1.0, 3.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    vibespatial.clear_dispatch_events()
    result = clip(buildings, mask)
    events = vibespatial.get_dispatch_events(clear=True)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    assert result.geometry.values._owned.device_state is not None
    assert any(
        event.surface == "vibespatial.kernels.constructive.polygon_rect_intersection"
        and event.selected.value == "gpu"
        for event in events
    )

    expected = clip(
        vibespatial.GeoDataFrame({"geometry": [box(2.0, 2.0, 4.0, 4.0), box(4.0, 4.0, 6.0, 6.0), box(6.0, 6.0, 8.0, 8.0), box(8.0, 8.0, 10.0, 10.0)]}, crs="EPSG:3857"),
        mask,
    )
    assert set(result.geometry.to_wkt().tolist()) == set(expected.geometry.to_wkt().tolist())


def test_clip_polygon_mask_boundary_filter_uses_gpu_predicate_for_device_backing() -> None:
    if not vibespatial.has_gpu_runtime():
        return

    values = from_shapely_geometries(
        [
            box(0.2, 0.2, 0.8, 0.8),
            box(2.2, 2.2, 2.8, 2.8),
            box(1.0, 1.2, 1.2, 1.6),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                values,
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
            (0.0, 0.0),
        ]
    )

    vibespatial.clear_dispatch_events()
    result = clip(gdf, mask)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert any(
        event.surface == "vibespatial.predicates.binary"
        and event.operation == "intersects"
        and event.selected.value == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "DeviceGeometryArray.intersects"
        and event.selected.value == "cpu"
        for event in events
    )


def test_clip_polygon_mask_boundary_assembly_skips_bbox_false_positives(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": [
                box(0.2, 0.2, 0.8, 0.8),
                box(2.2, 2.2, 2.8, 2.8),
                box(1.0, 1.2, 1.2, 1.6),
            ]
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
            (0.0, 0.0),
        ]
    )

    seen_rows: list[int] = []
    original = overlay_module._assemble_polygon_intersection_rows_with_lower_dim

    def _wrapped(left_pairs, right_pairs, area_pairs):
        seen_rows.append(len(left_pairs))
        return original(left_pairs, right_pairs, area_pairs)

    monkeypatch.setattr(
        overlay_module,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        _wrapped,
    )

    result = clip(gdf, mask)

    assert seen_rows == [1]
    assert len(result) == 2


def test_clip_multipart_result_preserves_duplicate_source_index_order() -> None:
    gdf = _build_mixed_viewport_fixture()
    gdf.index = pd.Index(["dup", "dup", "uniq"])
    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(1.0, 1.0, 6.0, 6.0),
    )

    result = native_result.to_spatial()

    assert len(result) == len(native_result.ordered_index)
    assert list(result.index) == list(native_result.ordered_index)
    assert result.index.tolist().count("dup") == 2
    assert result.index.tolist().count("uniq") == 1


def test_clip_polygon_area_intersection_owned_records_fallback_when_many_vs_one_fails(
    monkeypatch,
) -> None:
    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    left_owned = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0)],
        residency=Residency.HOST,
    )
    mask_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.HOST,
    )
    expected = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.HOST,
    )
    fallback_calls: list[dict[str, object]] = []
    constructive_calls: list[dict[str, object]] = []

    def _raise_many_vs_one(*args, **kwargs):
        raise RuntimeError("forced many-vs-one failure")

    def _record_fallback(**kwargs):
        fallback_calls.append(kwargs)
        return None

    def _binary_constructive(op, left, right, **kwargs):
        constructive_calls.append(
            {
                "op": op,
                "left": left,
                "right": right,
                **kwargs,
            }
        )
        return expected

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        _raise_many_vs_one,
    )
    monkeypatch.setattr(clip_module, "record_fallback_event", _record_fallback)
    monkeypatch.setattr(
        binary_module,
        "binary_constructive_owned",
        _binary_constructive,
    )

    result = clip_module._clip_polygon_area_intersection_owned(left_owned, mask_owned)

    assert result is expected
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["surface"] == "geopandas.clip"
    assert "many-vs-one polygon clip helper failed" in fallback_calls[0]["reason"]
    assert "forced many-vs-one failure" in fallback_calls[0]["detail"]
    assert fallback_calls[0]["selected"].value == "cpu"
    assert fallback_calls[0]["pipeline"] == "_clip_polygon_area_intersection_owned"
    assert len(constructive_calls) == 1
    assert constructive_calls[0]["op"] == "intersection"
    assert constructive_calls[0]["left"] is left_owned
    assert constructive_calls[0]["right"] is mask_owned
    assert constructive_calls[0]["_prefer_exact_polygon_intersection"] is True
    assert constructive_calls[0]["dispatch_mode"].value == "cpu"


def test_clip_polygon_rect_kernel_failure_is_not_silently_swallowed(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
    kernel_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    buildings_owned = from_shapely_geometries(
        [
            box(2.0, 2.0, 4.0, 4.0),
            box(4.0, 4.0, 6.0, 6.0),
        ],
        residency=Residency.DEVICE,
    )
    buildings = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                buildings_owned,
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 3.0),
                        (5.0, 1.0),
                        (9.0, 3.0),
                        (7.0, 7.0),
                        (3.0, 7.0),
                        (1.0, 3.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        kernel_module,
        "polygon_rect_intersection_can_handle",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        kernel_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced clip rect kernel failure")
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "_many_vs_one_intersection_owned should not run after a clip kernel failure"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="forced clip rect kernel failure"):
        clip(buildings, mask)
