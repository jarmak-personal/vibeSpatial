from __future__ import annotations

import ast
import importlib
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial
from vibespatial.api._native_results import NativeTabularResult
from vibespatial.api.geometry_array import POLYGON_GEOM_TYPES, GeometryArray
from vibespatial.api.tools.clip import clip
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

clip_module = importlib.import_module("vibespatial.api.tools.clip")


def test_clip_public_tool_has_no_raw_cupy_scalar_syncs() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "api"
        / "tools"
        / "clip.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    failures: list[str] = []

    cupy_reductions = {
        "all",
        "any",
        "sum",
        "count_nonzero",
        "max",
        "min",
        "nanmax",
        "nanmin",
    }

    def _contains_cupy_reduction(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "cp"
            and child.func.attr in cupy_reductions
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            failures.append(f"raw .item() at line {node.lineno}")
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_reduction(node.args[0])
        ):
            failures.append(f"raw {func.id}(cp reduction) at line {node.lineno}")

    assert failures == []


def _materialize_native_clip_result(
    result: NativeTabularResult,
    *,
    source: vibespatial.GeoDataFrame | vibespatial.GeoSeries,
):
    return clip_module._clip_native_tabular_to_spatial(result, source=source)


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


def _benchmark_admin_star_mask() -> Polygon:
    coords = []
    for i in range(24):
        angle = math.pi * i / 12.0
        radius = 200.0 if i % 2 == 0 else 80.0
        coords.append((500.0 + radius * math.cos(angle), 500.0 + radius * math.sin(angle)))
    return Polygon(coords)


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
    assert isinstance(result.geometry.values, DeviceGeometryArray)


def test_clip_polygon_rectangle_mask_routes_multilinestring_rows_through_rect_fast_path() -> None:
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
    vibespatial.clear_dispatch_events()

    result = clip(gdf, mask)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert set(result.geom_type.tolist()) == {"MultiLineString", "LineString"}
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert any(
        event.surface == "DeviceGeometryArray.clip_by_rect"
        and event.implementation == "owned_clip_by_rect"
        and event.selected.value == "gpu"
        for event in dispatch_events
    )


def test_clip_equivalent_wkt_and_epsg_crs_does_not_warn() -> None:
    pyproj = pytest.importorskip("pyproj")

    source = vibespatial.GeoDataFrame(
        {"geometry": [Point(0.0, 0.0)]},
        crs=pyproj.CRS.from_wkt(pyproj.CRS.from_epsg(4326).to_wkt(version="WKT1_GDAL")),
    )
    mask = vibespatial.GeoDataFrame(
        {"geometry": [box(-1.0, -1.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = clip(source, mask)

    mismatch_warnings = [
        warning
        for warning in caught
        if "CRS mismatch between the CRS" in str(warning.message)
    ]

    assert mismatch_warnings == []
    assert len(result) == 1


def test_clip_polygon_rectangle_mask_multilinestring_survives_strict_native_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        clip_module.shapely,
        "length",
        lambda *_args, **_kwargs: pytest.fail(
            "multiline rectangle clip should not probe host line degeneracy"
        ),
    )

    with strict_native_environment():
        result = clip(gdf, mask)

    assert set(result.geom_type.tolist()) == {"MultiLineString", "LineString"}
    assert isinstance(result.geometry.values, DeviceGeometryArray)


def test_clip_polygon_boundary_touch_mask_uses_scalar_gpu_mask_without_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    values = [
        box(0.0, 0.0, 2.0, 2.0),
        box(3.0, 0.0, 5.0, 2.0),
    ]
    gdf = vibespatial.GeoDataFrame(
        {"geometry": values},
        crs="EPSG:3857",
    )
    source_values = gdf.geometry.values
    left_owned = source_values.to_owned()
    mask = Polygon([(1.0, -1.0), (4.0, -1.0), (4.0, 1.0), (1.0, 1.0), (1.0, -1.0)])
    mask_owned = from_shapely_geometries([mask], residency=left_owned.residency)
    boundary_rows = np.asarray([0, 1], dtype=np.intp)

    import vibespatial.geometry.owned as owned_module

    def _fail_broadcast(*_args, **_kwargs):
        raise AssertionError("boundary touch mask should use a scalar GPU mask, not broadcast it")

    monkeypatch.setattr(owned_module, "materialize_broadcast", _fail_broadcast)

    result = clip_module._clip_polygon_boundary_touch_mask(
        source_values,
        left_owned,
        boundary_rows,
        mask=mask,
        mask_owned=mask_owned,
    )

    expected = np.asarray(
        shapely.intersects(
            np.asarray(values, dtype=object),
            np.full(len(values), mask, dtype=object),
        ),
        dtype=bool,
    )
    np.testing.assert_array_equal(result, expected)


def test_clip_scalar_rectangle_mask_survives_strict_native_mode_without_sindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": [
                box(0.0, 0.0, 2.0, 2.0),
                box(3.0, 3.0, 5.0, 5.0),
                box(10.0, 10.0, 12.0, 12.0),
            ]
        },
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        GeometryArray,
        "sindex",
        property(
            lambda self: pytest.fail(
                "scalar rectangle clip should avoid GeometryArray.sindex in strict native mode"
            )
        ),
    )

    with strict_native_environment():
        result = clip(gdf, box(0.0, 0.0, 6.0, 6.0))

    assert len(result) == 2


def test_clip_mask_covering_source_bounds_returns_device_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                box(0.0, 0.0, 1.0, 1.0),
                box(1.0, 1.0, 2.0, 2.0),
            ],
        },
        crs="EPSG:3857",
    )
    mask = Polygon([(1.0, -2.0), (4.0, 1.0), (1.0, 4.0), (-2.0, 1.0), (1.0, -2.0)])

    monkeypatch.setattr(
        clip_module,
        "_clip_gdf_with_mask_native",
        lambda *_args, **_kwargs: pytest.fail(
            "mask-cover clip should return a passthrough native result"
        ),
    )

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = clip(gdf, mask)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result["value"].tolist() == [1, 2]
    assert [geom.wkb for geom in result.geometry] == [geom.wkb for geom in gdf.geometry]
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert any(
        event.surface == "geopandas.clip"
        and event.implementation == "mask_covers_source_bounds_passthrough"
        and event.selected is ExecutionMode.GPU
        for event in dispatch_events
    )


def test_clip_mask_covering_source_bounds_passthrough_drops_empty_rows() -> None:
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": [
                box(0.0, 0.0, 1.0, 1.0),
                Polygon(),
                None,
            ],
        },
        crs="EPSG:3857",
    )
    mask = box(-1.0, -1.0, 2.0, 2.0)

    vibespatial.clear_dispatch_events()
    result = clip(gdf, mask)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result["value"].tolist() == [1]
    assert result.index.tolist() == [0]
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])
    assert any(
        event.surface == "geopandas.clip"
        and event.implementation == "mask_covers_source_bounds_passthrough"
        and "kept_rows=1" in event.detail
        for event in dispatch_events
    )


def test_clip_mask_covering_source_bounds_uses_native_state_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result

    owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0), Polygon()])
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": vibespatial.GeoSeries(
                GeometryArray.from_owned(owned, crs="EPSG:3857"),
                name="geometry",
            ),
        },
        crs="EPSG:3857",
    )
    attach_native_state_from_native_tabular_result(
        gdf,
        clip_module._spatial_to_native_tabular_result(gdf),
    )
    mask = box(-1.0, -1.0, 2.0, 2.0)

    monkeypatch.setattr(
        clip_module,
        "_take_spatial_rows",
        lambda *_args, **_kwargs: pytest.fail(
            "native mask-cover passthrough should take NativeFrameState directly"
        ),
    )
    monkeypatch.setattr(
        GeometryArray,
        "is_empty",
        property(lambda self: pytest.fail("owned structural metadata should decide empties")),
    )

    native_result = clip_module.evaluate_geopandas_clip_native(gdf, mask)

    assert native_result.geometry.owned is not None
    assert native_result.geometry.owned.row_count == 1
    assert native_result.attributes.to_pandas()["value"].tolist() == [1]


def test_clip_polygon_mask_keep_geom_type_sort_false_strict_preserves_input_order() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
        crs="EPSG:3857",
    )
    mask = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)])

    with strict_native_environment():
        result = clip(gdf, mask, keep_geom_type=True, sort=False)

    assert list(result["col1"]) == [1, 2]


def test_clip_polygon_mask_strict_keeps_polygon_cleanup_off_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
        crs="EPSG:3857",
    )
    mask = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)])

    monkeypatch.setattr(
        clip_module.shapely,
        "area",
        lambda *_args, **_kwargs: pytest.fail(
            "strict polygon clip cleanup should stay on the device path"
        ),
    )

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        result = clip(gdf, mask, keep_geom_type=True, sort=False)

    assert list(result["col1"]) == [1, 2]
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "clip.to_spatial"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_scalar_rectangle_polygon_mask_auto_preserves_device_cleanup_path() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {"parcel_id": [1, 2]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(2.0, 0.0, 4.0, 2.0),
            ],
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    mask = box(1.0, -1.0, 3.0, 1.0)

    vibespatial.clear_fallback_events()
    result = clip(gdf, mask, keep_geom_type=True, sort=False)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    actual = np.asarray(result.geometry.values, dtype=object)
    expected = np.asarray(
        [
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        dtype=object,
    )
    assert len(actual) == len(expected)
    assert all(any(shapely.equals(geom, candidate) for candidate in expected) for geom in actual)
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "clip.to_spatial"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_scalar_rectangle_device_multipolygon_keep_geom_type_stays_off_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source_geom = MultiPolygon(
        [
            box(-1.0, -1.0, 2.0, 2.0),
            box(4.0, 4.0, 12.0, 12.0),
        ]
    )
    source_owned = from_shapely_geometries([source_geom], residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "parcel_id": [1],
            "geometry": DeviceGeometryArray._from_owned(
                source_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = box(0.0, 0.0, 10.0, 10.0)

    monkeypatch.setattr(
        clip_module,
        "_host_polygonal_area_intersection_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "device multipolygon rectangle keep_geom_type clip should stay off host"
        ),
    )

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        result = clip(gdf, mask, keep_geom_type=True, sort=False)

    assert len(result) == 1
    assert result.geom_type.iloc[0] in POLYGON_GEOM_TYPES
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    assert shapely.equals(result.geometry.iloc[0], shapely.intersection(source_geom, mask))
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "_clip_polygon_rectangle_area_intersection_owned"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_scalar_rectangle_device_multipolygon_rescue_failure_uses_generic_device_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source_geom = MultiPolygon(
        [
            box(-1.0, -1.0, 2.0, 2.0),
            box(4.0, 4.0, 12.0, 12.0),
        ]
    )
    source_owned = from_shapely_geometries([source_geom], residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "parcel_id": [1],
            "geometry": DeviceGeometryArray._from_owned(
                source_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = box(0.0, 0.0, 10.0, 10.0)
    expected = shapely.intersection(source_geom, mask)
    calls: list[tuple[str, int]] = []

    def _rescue_unavailable(left_owned, rectangle_bounds):
        assert left_owned.residency is Residency.DEVICE
        calls.append(("rescue", left_owned.row_count))
        return None

    def _generic_area_path(left_owned, rectangle_bounds):
        assert left_owned.residency is Residency.DEVICE
        calls.append(("generic", left_owned.row_count))
        return from_shapely_geometries([expected], residency=Residency.DEVICE)

    monkeypatch.setattr(
        clip_module,
        "_clip_multipolygon_rectangle_keep_geom_type_owned",
        _rescue_unavailable,
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_rectangle_area_intersection_owned",
        _generic_area_path,
    )
    monkeypatch.setattr(
        clip_module,
        "_host_polygonal_area_intersection_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "device multipolygon rescue failure should not use host area cleanup"
        ),
    )

    with strict_native_environment():
        result = clip(gdf, mask, keep_geom_type=True, sort=False)

    assert calls == [("rescue", 1), ("generic", 1)]
    assert len(result) == 1
    assert result.geom_type.iloc[0] in POLYGON_GEOM_TYPES
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    assert shapely.equals(result.geometry.iloc[0], expected)


def test_clip_scalar_rectangle_simple_polygons_skips_generic_cleanup() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {"parcel_id": [1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(2.0, 0.0, 4.0, 2.0),
                box(5.0, 5.0, 6.0, 6.0),
            ],
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    mask = box(1.0, -1.0, 3.0, 1.0)

    vibespatial.clear_dispatch_events()
    result = clip(gdf, mask, sort=False)

    actual = np.asarray(result.geometry.values, dtype=object)
    expected = np.asarray(
        [
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        dtype=object,
    )
    assert list(result["parcel_id"]) == [1, 2]
    assert len(actual) == len(expected)
    assert all(any(shapely.equals(geom, candidate) for candidate in expected) for geom in actual)
    owned = result.geometry.values._owned
    assert owned._cached_is_valid_mask is not None
    assert owned._cached_is_valid_mask.tolist() == [True, True]

    cleanup_events = [
        event
        for event in vibespatial.get_dispatch_events(clear=True)
        if event.surface in {"geopandas.array.area", "geopandas.array.length"}
    ]
    assert cleanup_events == []


def test_clip_polygon_mask_exact_stage_skips_predicate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.9, 0.9),
                box(0.5, 0.5, 1.5, 1.5),
                box(2.0, 2.0, 2.5, 2.5),
                box(3.0, 0.2, 4.0, 0.8),
            ],
            crs="EPSG:3857",
        ),
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

    original = clip_module._clip_polygon_area_intersection_owned
    exact_row_counts: list[int] = []

    def wrapped(left_owned, mask_owned, **kwargs):
        exact_row_counts.append(left_owned.row_count)
        return original(left_owned, mask_owned, **kwargs)

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        wrapped,
    )

    result = clip(gdf, mask, sort=False)

    assert exact_row_counts == [2]
    assert list(result["row"]) == [0, 1, 3]


def test_clip_polygon_mask_predicate_split_exports_rowsets_not_full_bool_masks() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.9, 0.9),
                box(0.5, 0.5, 1.5, 1.5),
                box(2.0, 2.0, 2.5, 2.5),
                box(3.0, 0.2, 4.0, 0.8),
            ],
            crs="EPSG:3857",
        ),
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

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    result = clip(gdf, mask, sort=False)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_surfaces = {
        event.surface for event in get_materialization_events(clear=True)
    }

    assert list(result["row"]) == [0, 1, 3]
    assert "binary predicate fused predicate-results host export" not in runtime_reasons
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" in runtime_reasons
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" in runtime_reasons
    assert {
        "vibespatial.api.tools.clip.polygon_mask_inside_rows",
        "vibespatial.api.tools.clip.polygon_mask_exact_rows",
    }.issubset(materialization_surfaces)


def test_clip_polygon_single_mask_split_exports_rowsets_not_full_bool_mask() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2, 3]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.8, 0.8),
                box(2.0, 0.2, 3.0, 0.8),
                box(0.2, 2.0, 0.8, 3.0),
                box(2.0, 2.0, 3.0, 3.0),
            ],
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (1.0, 1.0),
            (1.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    result = clip(gdf, mask, sort=False)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_surfaces = {
        event.surface for event in get_materialization_events(clear=True)
    }

    assert set(result["row"]) == {0, 1, 2}
    assert "binary predicate covered-by single-mask result host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows::rowset_to_host"
        in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows::rowset_to_host"
        in runtime_reasons
    )
    assert {
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows",
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows",
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows",
    }.issubset(materialization_surfaces)


def test_clip_records_fallback_before_line_make_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    source = vibespatial.GeoDataFrame(
        {
            "geometry": [LineString([(0.0, 0.0), (0.0, 0.0)])],
        },
        crs="EPSG:3857",
    )
    owned = from_shapely_geometries(
        [LineString([(0.0, 0.0), (0.0, 0.0)])],
        residency=Residency.HOST,
    )
    native_result = clip_module.ClipNativeResult(
        source=source,
        parts=(
            clip_module._clip_native_part(
                source,
                np.asarray([0], dtype=np.intp),
                GeometryArray.from_owned(owned, crs=source.crs),
            ),
        ),
        ordered_index=source.index,
        ordered_row_positions=np.asarray([0], dtype=np.intp),
        clipping_by_rectangle=False,
        has_non_point_candidates=True,
        keep_geom_type=False,
    )

    monkeypatch.setattr(
        clip_module.shapely,
        "make_valid",
        lambda *_args, **_kwargs: pytest.fail(
            "strict native line cleanup should stay off the host make_valid path"
        ),
    )

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        result = native_result.to_spatial()

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    assert shapely.equals(result.geometry.iloc[0], Point(0.0, 0.0))
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "clip.to_spatial"
        for event in vibespatial.get_fallback_events(clear=True)
    )


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


def test_clip_polygon_rectangle_mask_donut_keep_geom_type_strips_collection_slivers() -> None:
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

    result = clip(multi_poly, mask, keep_geom_type=True)

    assert result.geom_type.isin(POLYGON_GEOM_TYPES).all()
    assert tuple(result.total_bounds) == tuple(mask.total_bounds)


def test_clip_polygon_mask_zero_area_filter_copies_keep_mask_before_mutation(
    monkeypatch,
) -> None:
    source = vibespatial.GeoDataFrame(
        {
            "geometry": [
                box(0.0, 0.0, 2.0, 2.0),
                GeometryCollection([Point(5.0, 5.0)]),
            ]
        },
        crs="EPSG:3857",
    )
    native_result = clip_module.ClipNativeResult(
        source=source,
        parts=(
            clip_module._clip_native_part(
                source,
                np.asarray([0, 1], dtype=np.intp),
                GeometryArray(
                    np.asarray(
                        [
                            box(0.0, 0.0, 2.0, 2.0),
                            GeometryCollection([Point(5.0, 5.0)]),
                        ],
                        dtype=object,
                    ),
                    crs=source.crs,
                ),
            ),
        ),
        ordered_index=source.index,
        ordered_row_positions=np.asarray([0, 1], dtype=np.intp),
        clipping_by_rectangle=False,
        has_non_point_candidates=True,
        keep_geom_type=False,
    )

    real_asarray = clip_module.np.asarray

    def _readonly_bool_asarray(value, *args, **kwargs):
        arr = real_asarray(value, *args, **kwargs)
        dtype = kwargs.get("dtype", args[0] if args else None)
        if dtype is bool and getattr(arr, "ndim", 0) == 1 and getattr(arr, "size", -1) == len(source):
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

    result = native_result.to_spatial()

    assert len(result) == 1
    assert result.geom_type.tolist() == ["GeometryCollection"]


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


def test_clip_polygon_rectangle_mask_uses_source_polygon_as_kernel_left_operand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    kernel_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    source_owned = from_shapely_geometries(
        [
            Polygon([(0.0, 0.0), (4.0, 0.0), (3.0, 3.0), (0.0, 4.0), (0.0, 0.0)]),
            Polygon([(5.0, 0.0), (9.0, 0.0), (8.0, 3.0), (5.0, 4.0), (5.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    mask_owned = from_shapely_geometries([box(1.0, 1.0, 8.0, 3.0)], residency=Residency.DEVICE)
    sentinel = from_shapely_geometries(
        [box(1.0, 1.0, 2.0, 2.0), box(6.0, 1.0, 7.0, 2.0)],
        residency=Residency.DEVICE,
    )
    can_handle_calls: list[bool] = []
    kernel_calls: list[bool] = []

    def _fake_can_handle(left_arg, right_arg):
        can_handle_calls.append(left_arg is source_owned)
        return left_arg is source_owned and right_arg.row_count == source_owned.row_count

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        kernel_calls.append(left_arg is source_owned)
        assert dispatch_mode is ExecutionMode.GPU
        assert right_arg.row_count == source_owned.row_count
        return sentinel

    monkeypatch.setattr(
        kernel_module,
        "polygon_rect_intersection_can_handle",
        _fake_can_handle,
    )
    monkeypatch.setattr(
        kernel_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )

    result = clip_module._clip_polygon_area_intersection_gpu_owned(source_owned, mask_owned)

    assert result is sentinel
    assert can_handle_calls == [True]
    assert kernel_calls == [True]


def test_clip_records_fallback_before_host_semantic_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    source = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": vibespatial.GeoSeries([box(0.0, 0.0, 2.0, 2.0)]),
        },
        crs="EPSG:3857",
    )
    owned = from_shapely_geometries([box(0.5, 0.5, 1.5, 1.5)], residency=Residency.HOST)
    part = clip_module._clip_native_part(
        source,
        np.asarray([0], dtype=np.intp),
        GeometryArray.from_owned(owned, crs=source.crs),
    )
    native_result = clip_module.ClipNativeResult(
        source=source,
        parts=(part,),
        ordered_index=source.index,
        ordered_row_positions=np.asarray([0], dtype=np.intp),
        clipping_by_rectangle=False,
        has_non_point_candidates=True,
        keep_geom_type=False,
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("strict native polygon cleanup should stay off the host")

    monkeypatch.setattr(clip_module.shapely, "area", _fail)
    monkeypatch.setattr(clip_module.shapely, "length", _fail)

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        result = native_result.to_spatial()

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE
    assert shapely.equals(result.geometry.iloc[0], box(0.5, 0.5, 1.5, 1.5))
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "clip.to_spatial"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_exact_rectangle_clip_boundary_rows_uses_owned_rectangle_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(2.0, 2.0, 4.0, 5.0),
            ],
            residency=Residency.HOST,
        )
    )
    bounds = np.asarray(
        [
            (0.0, 0.0, 2.0, 2.0),
            (2.0, 2.0, 4.0, 5.0),
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        clip_module,
        "_is_axis_aligned_rectangle_polygon",
        lambda *_args, **_kwargs: pytest.fail(
            "owned rectangle metadata should avoid per-row Shapely rectangle checks"
        ),
    )

    result = clip_module._exact_rectangle_clip_boundary_rows(
        values,
        bounds,
        (1.0, 1.0, 3.0, 4.0),
    )

    assert result is not None
    assert [geom.geom_type if geom is not None else None for geom in result] == [
        "Polygon",
        "Polygon",
    ]


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


def test_clip_polygon_mask_rectangle_cells_repair_disconnected_fast_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    mask = Polygon(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (4.0, 3.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]
    )
    cells = np.asarray(
        [
            box(1.5, 0.25, 3.5, 3.75),
            box(0.1, 0.5, 0.9, 3.5),
        ],
        dtype=object,
    )
    left_owned = from_shapely_geometries(cells, residency=Residency.DEVICE)
    mask_owned = from_shapely_geometries([mask], residency=Residency.DEVICE)
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row

    tiled_mask = materialize_broadcast(tile_single_row(mask_owned, left_owned.row_count))

    vibespatial.clear_dispatch_events()
    vibespatial.clear_fallback_events()
    result_owned = clip_module._clip_validated_polygon_rect_mask_intersection_owned(
        left_owned,
        mask_owned,
        tiled_mask,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    actual = np.asarray(result_owned.to_shapely(), dtype=object)
    expected = np.asarray(
        shapely.intersection(
            cells,
            np.full(cells.size, mask, dtype=object),
        ),
        dtype=object,
    )

    assert actual[0].geom_type == "MultiPolygon"
    assert all(shapely.is_valid(actual))
    assert all(
        shapely.equals(shapely.normalize(actual_geom), shapely.normalize(expected_geom))
        for actual_geom, expected_geom in zip(actual, expected, strict=True)
    )
    assert any(
        event.surface == "geopandas.clip"
        and event.operation == "validated_polygon_rect_mask_clip"
        and event.selected.value == "gpu"
        and "repair_rows=1" in event.detail
        for event in events
    )
    assert vibespatial.get_fallback_events(clear=True) == []


def test_clip_polygon_mask_rectangle_cells_split_star_notch_before_make_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    mask = _benchmark_admin_star_mask()
    cell = box(470.0, 410.0, 480.0, 420.0)
    left_owned = from_shapely_geometries([cell], residency=Residency.DEVICE)
    mask_owned = from_shapely_geometries([mask], residency=Residency.DEVICE)
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row

    tiled_mask = materialize_broadcast(tile_single_row(mask_owned, left_owned.row_count))
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    monkeypatch.setattr(
        make_valid_module,
        "make_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "star-notch rectangle clip rows should split on GPU before generic make_valid"
        ),
    )

    vibespatial.clear_dispatch_events()
    result_owned = clip_module._clip_validated_polygon_rect_mask_intersection_owned(
        left_owned,
        mask_owned,
        tiled_mask,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    actual = np.asarray(result_owned.to_shapely(), dtype=object)[0]
    expected = shapely.intersection(cell, mask)

    assert actual.geom_type == "MultiPolygon"
    assert shapely.is_valid(actual)
    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected))
    assert any(
        event.surface == "geopandas.clip"
        and event.operation == "validated_polygon_rect_mask_clip"
        and event.selected.value == "gpu"
        and "repair_impl=polygon_rect_boundary_split_gpu" in event.detail
        for event in events
    )


def test_clip_polygon_mask_preserves_boundary_touch_rows_and_exact_dimension() -> None:
    mask = _benchmark_admin_star_mask()
    gdf = vibespatial.GeoDataFrame(
        {
            "building_id": [3239, 3967, 6760],
            "geometry": [
                box(390.0, 320.0, 400.0, 330.0),
                box(670.0, 390.0, 680.0, 400.0),
                box(600.0, 670.0, 610.0, 680.0),
            ],
        },
        crs="EPSG:4326",
    )

    expected = shapely.intersection(
        np.asarray(gdf.geometry.values, dtype=object),
        np.full(len(gdf), mask, dtype=object),
    )
    expected_by_id = {
        int(building_id): shapely.to_wkt(shapely.normalize(geom), rounding_precision=6)
        for building_id, geom in zip(gdf["building_id"], expected, strict=True)
        if geom is not None and not getattr(geom, "is_empty", False)
    }

    result = clip(gdf, mask)
    result_by_id = {
        int(row.building_id): shapely.to_wkt(shapely.normalize(row.geometry), rounding_precision=6)
        for row in result.itertuples(index=False)
    }

    assert result_by_id == expected_by_id


def test_clip_polygon_mask_reuses_correction_boundary_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-backed clip materialization canary")

    mask = _benchmark_admin_star_mask()
    owned = from_shapely_geometries(
        [
            box(390.0, 320.0, 400.0, 330.0),
            box(670.0, 390.0, 680.0, 400.0),
            box(600.0, 670.0, 610.0, 680.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "building_id": [3239, 3967, 6760],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:4326"),
        },
        crs="EPSG:4326",
    )
    materialized_rows: list[int] = []
    original_array = DeviceGeometryArray.__array__

    def _record_array(self, dtype=None, copy=None):
        materialized_rows.append(len(self))
        return original_array(self, dtype=dtype, copy=copy)

    monkeypatch.setattr(DeviceGeometryArray, "__array__", _record_array)

    result = clip(gdf, mask)

    assert len(result) == 3
    assert materialized_rows == [3]
    assert all(rows == 3 for rows in materialized_rows)


def test_clip_rectangle_filter_avoids_device_array_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(3.0, 3.0, 5.0, 5.0),
            box(10.0, 10.0, 12.0, 12.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:3857",
            ),
            "value": [1, 2, 3],
        },
        crs="EPSG:3857",
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("rectangle clip filter should not materialize DeviceGeometryArray")

    monkeypatch.setattr(DeviceGeometryArray, "__array__", _fail)

    result = clip(gdf, box(0.0, 0.0, 6.0, 6.0))

    assert list(result["value"]) == [1, 2]
    assert isinstance(result.geometry.values, DeviceGeometryArray)


def test_clip_device_backed_routing_avoids_public_frame_metadata_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    def _fail_public_metadata(self):
        raise AssertionError("clip routing must use private native geometry metadata")

    monkeypatch.setattr(vibespatial.GeoDataFrame, "geom_type", property(_fail_public_metadata))
    monkeypatch.setattr(vibespatial.GeoDataFrame, "total_bounds", property(_fail_public_metadata))
    monkeypatch.setattr(vibespatial.GeoSeries, "geom_type", property(_fail_public_metadata))
    monkeypatch.setattr(vibespatial.GeoSeries, "total_bounds", property(_fail_public_metadata))

    source = vibespatial.GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [
                        box(0.0, 0.0, 2.0, 2.0),
                        box(3.0, 3.0, 5.0, 5.0),
                        box(10.0, 10.0, 12.0, 12.0),
                    ],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [box(0.0, 0.0, 6.0, 6.0)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )

    result = clip(source, mask)

    assert list(result["value"]) == [1, 2]
    assert isinstance(result.geometry.values, DeviceGeometryArray)


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

    assert seen_rows == []
    assert len(result) == 2


def test_clip_multipart_result_preserves_duplicate_source_index_order() -> None:
    gdf = _build_mixed_viewport_fixture()
    gdf.index = pd.Index(["dup", "dup", "uniq"])
    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(1.0, 1.0, 6.0, 6.0),
    )

    assert isinstance(native_result, NativeTabularResult)
    result = _materialize_native_clip_result(native_result, source=gdf)

    assert len(result) == len(gdf)
    assert list(result.index) == list(gdf.index)
    assert result.index.tolist().count("dup") == 2
    assert result.index.tolist().count("uniq") == 1


def test_clip_single_polygon_mask_uses_direct_bbox_candidates_before_sindex(
    monkeypatch,
) -> None:
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                box(0.0, 0.0, 2.0, 2.0),
                box(10.0, 10.0, 12.0, 12.0),
            ],
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {"geometry": [Polygon([(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0), (1.0, 1.0)])]},
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "sorted single-row polygon clip should use direct bbox candidates before sindex.query"
        ),
    )

    native_result = clip_module.evaluate_geopandas_clip_native(gdf, mask, sort=True)
    result = _materialize_native_clip_result(native_result, source=gdf)

    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(box(1.0, 1.0, 2.0, 2.0))


def test_clip_promoted_single_polygon_mask_still_uses_direct_bbox_candidates(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                box(0.0, 0.0, 2.0, 2.0),
                box(10.0, 10.0, 12.0, 12.0),
            ],
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 1.0),
                        (4.0, 1.0),
                        (4.0, 2.0),
                        (2.0, 2.0),
                        (2.0, 4.0),
                        (1.0, 4.0),
                        (1.0, 1.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "promoted scalar polygon clip should keep using direct bbox candidates before sindex.query"
        ),
    )

    native_result = clip_module.evaluate_geopandas_clip_native(gdf, mask, sort=True)
    result = _materialize_native_clip_result(native_result, source=gdf)

    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(
        Polygon(
            [
                (1.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (1.0, 2.0),
                (1.0, 1.0),
            ]
        )
    )


def test_clip_promoted_single_polygon_mask_strict_uses_device_bbox_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module(
        "vibespatial.kernels.core.geometry_analysis"
    )
    original_compute_bounds_device = geometry_analysis_module.compute_geometry_bounds_device
    seen: dict[str, bool] = {"called": False}

    def _wrapped_compute_bounds_device(*args, **kwargs):
        seen["called"] = True
        return original_compute_bounds_device(*args, **kwargs)

    monkeypatch.setattr(
        geometry_analysis_module,
        "compute_geometry_bounds_device",
        _wrapped_compute_bounds_device,
    )

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                box(0.0, 0.0, 2.0, 2.0),
                box(10.0, 10.0, 12.0, 12.0),
            ],
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 1.0),
                        (4.0, 1.0),
                        (4.0, 2.0),
                        (2.0, 2.0),
                        (2.0, 4.0),
                        (1.0, 4.0),
                        (1.0, 1.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "strict scalar polygon clip should use device bbox candidates before sindex.query"
        ),
    )

    with strict_native_environment():
        native_result = clip_module.evaluate_geopandas_clip_native(gdf, mask, sort=True)
        result = _materialize_native_clip_result(native_result, source=gdf)

    assert seen["called"] is True
    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(
        Polygon(
            [
                (1.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (1.0, 2.0),
                (1.0, 1.0),
            ]
        )
    )


def test_clip_device_backed_single_polygon_mask_uses_device_bbox_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module(
        "vibespatial.kernels.core.geometry_analysis"
    )
    original_compute_bounds_device = geometry_analysis_module.compute_geometry_bounds_device
    seen: dict[str, bool] = {"called": False}

    def _wrapped_compute_bounds_device(*args, **kwargs):
        seen["called"] = True
        return original_compute_bounds_device(*args, **kwargs)

    monkeypatch.setattr(
        geometry_analysis_module,
        "compute_geometry_bounds_device",
        _wrapped_compute_bounds_device,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(10.0, 10.0, 12.0, 12.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (1.0, 1.0),
                        (4.0, 1.0),
                        (4.0, 2.0),
                        (2.0, 2.0),
                        (2.0, 4.0),
                        (1.0, 4.0),
                        (1.0, 1.0),
                    ]
                )
            ]
        },
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "device-backed scalar polygon clip should use device bbox candidates before sindex.query"
        ),
    )

    native_result = clip_module.evaluate_geopandas_clip_native(gdf, mask, sort=True)
    result = _materialize_native_clip_result(native_result, source=gdf)

    assert seen["called"] is True
    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(
        Polygon(
            [
                (1.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (1.0, 2.0),
                (1.0, 1.0),
            ]
        )
    )


def test_clip_polygon_result_seeds_validity_cache_on_owned_output() -> None:
    residency = Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 12.0, 12.0)],
        residency=residency,
    )
    geometry_values = (
        DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
        if residency is Residency.DEVICE
        else GeometryArray.from_owned(owned, crs="EPSG:3857")
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": geometry_values,
        },
        crs="EPSG:3857",
    )

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(1.0, 1.0, 3.0, 3.0),
        sort=True,
    )
    result = _materialize_native_clip_result(native_result, source=gdf)

    result_owned = getattr(result.geometry.values, "_owned", None)
    assert result_owned is not None
    assert result_owned._cached_is_valid_mask is not None
    np.testing.assert_array_equal(
        result_owned._cached_is_valid_mask,
        np.ones(len(result), dtype=bool),
    )
    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(box(1.0, 1.0, 3.0, 3.0))


def test_clip_polygon_result_preserves_validity_cache_through_public_filter_copy() -> None:
    residency = Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 4.0, 4.0), box(10.0, 10.0, 12.0, 12.0)],
        residency=residency,
    )
    geometry_values = (
        DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
        if residency is Residency.DEVICE
        else GeometryArray.from_owned(owned, crs="EPSG:3857")
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": geometry_values,
        },
        crs="EPSG:3857",
    )

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(1.0, 1.0, 3.0, 3.0),
        sort=True,
    )
    result = _materialize_native_clip_result(native_result, source=gdf)
    filtered = result[result.geometry.geom_type.isin(POLYGON_GEOM_TYPES)].copy()

    filtered_owned = getattr(filtered.geometry.values, "_owned", None)
    assert filtered_owned is not None
    assert filtered_owned._cached_is_valid_mask is not None
    np.testing.assert_array_equal(
        filtered_owned._cached_is_valid_mask,
        np.ones(len(filtered), dtype=bool),
    )
    assert filtered["value"].tolist() == [1]


def test_clip_device_backed_single_polygon_mask_strict_uses_device_bbox_candidates_without_sindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module(
        "vibespatial.kernels.core.geometry_analysis"
    )
    original_compute_bounds_device = geometry_analysis_module.compute_geometry_bounds_device
    seen: dict[str, bool] = {"called": False}

    def _wrapped_compute_bounds_device(*args, **kwargs):
        seen["called"] = True
        return original_compute_bounds_device(*args, **kwargs)

    monkeypatch.setattr(
        geometry_analysis_module,
        "compute_geometry_bounds_device",
        _wrapped_compute_bounds_device,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(10.0, 10.0, 12.0, 12.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (1.0, 1.0),
            (4.0, 1.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (1.0, 4.0),
            (1.0, 1.0),
        ]
    )

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "strict device-backed scalar polygon clip should not fall back to sindex.query"
        ),
    )

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        result = clip(gdf, mask, sort=True)

    assert seen["called"] is True
    assert result["value"].tolist() == [1]
    assert result.geometry.iloc[0].normalize().equals(
        Polygon(
            [
                (1.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (1.0, 2.0),
                (1.0, 1.0),
            ]
        )
    )
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "_bbox_candidate_rows_for_scalar_clip_mask"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_large_scalar_rectangle_mask_promotes_supported_host_candidates_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module(
        "vibespatial.kernels.core.geometry_analysis"
    )
    original_compute_bounds_device = geometry_analysis_module.compute_geometry_bounds_device
    seen: dict[str, bool] = {"called": False}

    def _wrapped_compute_bounds_device(*args, **kwargs):
        seen["called"] = True
        return original_compute_bounds_device(*args, **kwargs)

    monkeypatch.setattr(
        geometry_analysis_module,
        "compute_geometry_bounds_device",
        _wrapped_compute_bounds_device,
    )

    row_count = 50_001
    gdf = vibespatial.GeoDataFrame(
        {
            "value": np.arange(row_count, dtype=np.int32),
            "geometry": [Point(float(index), 0.0) for index in range(row_count)],
        },
        crs="EPSG:3857",
    )
    mask = box(-0.5, -0.5, 0.5, 0.5)

    monkeypatch.setattr(
        gdf.sindex.__class__,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "large supported scalar clip should use device bbox candidates before sindex.query"
        ),
    )

    vibespatial.clear_fallback_events()
    result = clip(gdf, mask, sort=True)

    assert seen["called"] is True
    assert result["value"].tolist() == [0]
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "_bbox_candidate_rows_for_scalar_clip_mask"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_take_spatial_rows_preserves_device_backing_after_row_filter() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(10.0, 10.0, 12.0, 12.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    keep_mask = np.asarray([True, False], dtype=bool)

    result = clip_module._take_spatial_rows(gdf, keep_mask)

    assert result["value"].tolist() == [1]
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result.geometry.values._owned.residency is Residency.DEVICE


def test_clip_polygon_area_intersection_owned_records_fallback_when_many_vs_one_fails(
    monkeypatch,
) -> None:
    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
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
    host_calls: list[tuple[object, object]] = []

    def _raise_many_vs_one(*args, **kwargs):
        raise RuntimeError("forced many-vs-one failure")

    def _record_fallback(**kwargs):
        fallback_calls.append(kwargs)
        return None

    def _host_polygonal_area_intersection(left, right):
        host_calls.append((left, right))
        return expected

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        _raise_many_vs_one,
    )
    monkeypatch.setattr(clip_module, "record_fallback_event", _record_fallback)
    monkeypatch.setattr(
        clip_module,
        "_host_polygonal_area_intersection_owned",
        _host_polygonal_area_intersection,
    )

    result = clip_module._clip_polygon_area_intersection_owned(left_owned, mask_owned)

    assert result is expected
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["surface"] == "geopandas.clip"
    assert "many-vs-one polygon clip helper failed" in fallback_calls[0]["reason"]
    assert "host exact polygonal area extraction" in fallback_calls[0]["reason"]
    assert "forced many-vs-one failure" in fallback_calls[0]["detail"]
    assert fallback_calls[0]["selected"].value == "cpu"
    assert fallback_calls[0]["pipeline"] == "_clip_polygon_area_intersection_owned"
    assert host_calls == [(left_owned, mask_owned)]


def test_clip_polygon_area_intersection_owned_uses_direct_exact_gpu_for_tiny_device_batches(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
    left_owned = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    mask_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    expected = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    direct_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        lambda *args, **kwargs: pytest.fail(
            "tiny device-backed polygon clip should bypass the many-vs-one planner"
        ),
    )

    def _direct_exact(left, mask, *, allow_rectangle_kernel=True):
        direct_calls.append(
            {
                "left_rows": left.row_count,
                "mask_rows": mask.row_count,
                "allow_rectangle_kernel": allow_rectangle_kernel,
            }
        )
        return expected

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_gpu_owned",
        _direct_exact,
    )

    result = clip_module._clip_polygon_area_intersection_owned(left_owned, mask_owned)

    assert result is expected
    assert direct_calls == [
        {
            "left_rows": 1,
            "mask_rows": 1,
            "allow_rectangle_kernel": True,
        }
    ]
def test_clip_polygon_partition_polygon_mask_routes_exact_rows_through_owned_helper(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [
            Polygon(
                [
                    (-0.5, -0.5),
                    (0.4, -0.5),
                    (0.5, 0.1),
                    (0.0, 0.5),
                    (-0.5, 0.2),
                    (-0.5, -0.5),
                ]
            )
        ],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.6),
            (0.6, 0.6),
            (0.6, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]
    )
    expected_owned = from_shapely_geometries(
        [Polygon([(0.0, 0.0), (0.5, 0.1), (0.4, 0.4), (0.0, 0.2), (0.0, 0.0)])],
        residency=Residency.DEVICE,
    )
    calls: list[int] = []

    def _owned_helper(
        left_owned,
        mask_owned,
        *,
        allow_rectangle_kernel=True,
        prefer_exact_polygon_rect_batch=False,
        prefer_many_vs_one_planner=False,
    ):
        calls.append(left_owned.row_count)
        return expected_owned

    def _fail(*_args, **_kwargs):
        raise AssertionError("exact clip rows should not route through the generic rowwise GPU helper")

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        _owned_helper,
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_gpu_owned",
        _fail,
    )
    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: np.zeros(1, dtype=bool),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert calls == [1]
    assert len(result) == 1
    assert np.asarray(result, dtype=object)[0] is not None


def test_clip_polygon_partition_polygon_mask_avoids_host_covered_by_for_rectangle_batch(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (2.0, -1.0),
            (2.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )
    expected_geom = shapely.intersection(
        np.asarray([box(-0.5, -0.5, 0.75, 0.75)], dtype=object),
        np.asarray([mask], dtype=object),
    )[0]
    expected_owned = from_shapely_geometries(
        [expected_geom],
        residency=Residency.DEVICE,
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: pytest.fail(
            "rectangle polygon-mask clip should avoid host covered_by"
        ),
    )

    def _direct_exact(left, right, *, allow_rectangle_kernel=True):
        calls.append(
            {
                "left_rows": left.row_count,
                "mask_rows": right.row_count,
                "allow_rectangle_kernel": allow_rectangle_kernel,
            }
        )
        return expected_owned

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_gpu_owned",
        _direct_exact,
    )

    vibespatial.clear_fallback_events()
    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert calls == [
        {
            "left_rows": 1,
            "mask_rows": 1,
            "allow_rectangle_kernel": False,
        }
    ]
    assert len(result) == 1
    assert isinstance(result, DeviceGeometryArray)
    assert shapely.equals(np.asarray(result, dtype=object)[0], expected_geom)
    assert not vibespatial.get_fallback_events(clear=True)


def test_clip_polygon_partition_polygon_mask_auto_avoids_host_intersects_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    import vibespatial.predicates.binary as predicate_module

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (2.0, -1.0),
            (2.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )
    expected_owned = from_shapely_geometries(
        [
            Polygon(
                [
                    (-0.5, -0.5),
                    (0.75, -0.5),
                    (0.75, 0.5),
                    (0.5, 0.5),
                    (0.5, 0.75),
                    (-0.5, 0.75),
                    (-0.5, -0.5),
                ]
            )
        ],
        residency=Residency.DEVICE,
    )

    def _fake_evaluate_binary_predicate(predicate, left, right, **kwargs):
        row_count = left.row_count
        if predicate in {"intersects", "touches", "covered_by"}:
            return pd.Series(np.zeros(row_count, dtype=bool))
        raise AssertionError(f"unexpected predicate: {predicate}")

    monkeypatch.setattr(
        predicate_module,
        "evaluate_binary_predicate",
        _fake_evaluate_binary_predicate,
    )
    monkeypatch.setattr(
        shapely,
        "intersects",
        lambda *args, **kwargs: pytest.fail(
            "polygon-mask clip should not repair missed GPU intersects on the host"
        ),
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_owned,
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 1
    assert shapely.equals(np.asarray(result, dtype=object)[0], expected_owned.to_shapely()[0])


def test_clip_polygon_partition_polygon_mask_auto_avoids_host_covered_by_for_non_rectangle_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (2.0, -1.0),
            (2.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )
    expected_geom = Polygon(
        [
            (-0.5, -0.5),
            (0.75, -0.5),
            (0.75, 0.5),
            (0.5, 0.5),
            (0.5, 0.75),
            (-0.5, 0.75),
            (-0.5, -0.5),
        ]
    )
    expected_owned = from_shapely_geometries(
        [expected_geom],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: pytest.fail(
            "non-rectangle polygon-mask clip should not validate covered_by on the host"
        ),
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_owned,
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 1
    assert shapely.equals(np.asarray(result, dtype=object)[0], expected_geom)


def test_clip_polygon_partition_single_row_polygon_mask_skips_predicate_refine(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    import vibespatial.predicates.binary as predicate_module

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.5, 0.5)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (2.0, -1.0),
            (2.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )
    expected_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.5, 0.5)],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        predicate_module,
        "evaluate_binary_predicate",
        lambda *args, **kwargs: pytest.fail(
            "single-row polygon clip should skip GPU predicate refinement before exact intersection"
        ),
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_owned,
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 1
    assert shapely.equals(np.asarray(result, dtype=object)[0], box(-0.5, -0.5, 0.5, 0.5))


def test_clip_polygon_partition_single_row_polygon_mask_returns_source_via_containment_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.5, 0.5)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-2.0, -2.0),
            (2.0, -2.0),
            (2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, -2.0),
        ]
    )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: pytest.fail(
            "single-row polygon clip should bypass exact intersection when the source polygon is fully inside the mask"
        ),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 1
    assert shapely.equals(np.asarray(result, dtype=object)[0], box(-0.5, -0.5, 0.5, 0.5))


def test_clip_polygon_partition_single_row_polygon_mask_returns_mask_via_containment_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [box(-3.0, -3.0, 3.0, 3.0)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
    )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: pytest.fail(
            "single-row polygon clip should bypass exact intersection when the mask polygon is fully inside the source"
        ),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 1
    assert shapely.equals(np.asarray(result, dtype=object)[0], mask)


def test_clip_polygon_partition_polygon_mask_returns_direct_exact_owned_when_all_rows_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    owned_module = importlib.import_module("vibespatial.geometry.owned")

    partition_owned = from_shapely_geometries(
        [
            box(-0.5, -0.5, 0.5, 0.5),
            box(1.0, -0.5, 2.0, 0.5),
        ],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (3.0, -1.0),
            (3.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
    )
    expected_owned = from_shapely_geometries(
        [
            box(-0.5, -0.5, 0.5, 0.5),
            box(1.0, -0.5, 2.0, 0.5),
        ],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_owned,
    )
    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: np.zeros(2, dtype=bool),
    )
    monkeypatch.setattr(
        owned_module,
        "build_null_owned_array",
        lambda *args, **kwargs: pytest.fail(
            "all-positive polygon clip should not build a null owned scatter target"
        ),
    )
    monkeypatch.setattr(
        owned_module,
        "concat_owned_scatter",
        lambda *args, **kwargs: pytest.fail(
            "all-positive polygon clip should not scatter exact rows back into a null owned array"
        ),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 2
    actual = shapely.normalize(np.asarray(result, dtype=object))
    expected = shapely.normalize(np.asarray(expected_owned.to_shapely(), dtype=object))
    assert [geom.wkb for geom in actual] == [geom.wkb for geom in expected]


def test_clip_polygon_partition_polygon_mask_mixed_inside_and_exact_positive_skips_null_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    owned_module = importlib.import_module("vibespatial.geometry.owned")

    partition_owned = from_shapely_geometries(
        [
            box(-0.5, -0.5, 0.5, 0.5),
            box(1.0, -0.5, 2.0, 0.5),
        ],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (1.5, -1.0),
            (1.5, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
    )
    expected_owned = from_shapely_geometries(
        [
            box(-0.5, -0.5, 0.5, 0.5),
            box(1.0, -0.5, 1.5, 0.5),
        ],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_owned.take(np.asarray([1], dtype=np.intp)),
    )
    monkeypatch.setattr(
        owned_module,
        "build_null_owned_array",
        lambda *args, **kwargs: pytest.fail(
            "mixed inside/exact polygon clip should not build a null owned scatter target"
        ),
    )
    monkeypatch.setattr(
        owned_module,
        "concat_owned_scatter",
        lambda *args, **kwargs: pytest.fail(
            "mixed inside/exact polygon clip should not scatter rows back into a null owned array"
        ),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert len(result) == 2
    actual = shapely.normalize(np.asarray(result, dtype=object))
    expected = shapely.normalize(np.asarray(expected_owned.to_shapely(), dtype=object))
    assert [geom.wkb for geom in actual] == [geom.wkb for geom in expected]


def test_clip_semantically_clean_owned_part_skips_valid_nonempty_host_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device clip mask canary")

    native_results_module = importlib.import_module("vibespatial.api._native_results")
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE,
    )
    owned._clip_semantically_clean = True
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
    source = vibespatial.GeoDataFrame(
        {"value": [1, 2], "geometry": values},
        crs="EPSG:3857",
    )
    part = clip_module._clip_native_part(
        source,
        np.asarray([0, 1], dtype=np.intp),
        values,
    )

    real_host_array = native_results_module._host_array

    def _guard_host_array(values, *args, **kwargs):
        if kwargs.get("operation") == "clip_valid_nonempty_mask_to_host":
            raise AssertionError(
                "semantically clean clip fragments should not recopy a valid/non-empty mask"
            )
        return real_host_array(values, *args, **kwargs)

    monkeypatch.setattr(native_results_module, "_host_array", _guard_host_array)

    result = native_results_module._clip_constructive_parts_to_native_tabular_result(
        source=source,
        parts=(part,),
        ordered_row_positions=np.asarray([0, 1], dtype=np.intp),
        clipping_by_rectangle=False,
        has_non_point_candidates=True,
        keep_geom_type=False,
    )

    assert result.geometry.owned is not None
    assert result.provenance.source_rows.tolist() == [0, 1]


def test_clip_polygon_keep_geom_type_true_skips_boundary_reconstruction(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    partition_owned = from_shapely_geometries(
        [box(-0.5, -0.5, 0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                partition_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (2.0, -1.0),
            (2.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )

    monkeypatch.setattr(
        clip_module,
        "_exact_polygon_clip_boundary_rows",
        lambda *args, **kwargs: pytest.fail(
            "keep_geom_type polygon clip should not reconstruct lower-dimensional boundary rows"
        ),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(
        partition,
        mask,
        keep_geom_type_only=True,
    )

    assert len(result) == 1
    assert shapely.equals(
        np.asarray(result, dtype=object)[0],
        Polygon(
            [
                (-0.5, -0.5),
                (0.75, -0.5),
                (0.75, 0.5),
                (0.5, 0.5),
                (0.5, 0.75),
                (-0.5, 0.75),
                (-0.5, -0.5),
            ]
        ),
    )


def test_clip_public_rectangle_keep_geom_type_routes_through_polygon_area_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": [box(-0.5, -0.5, 0.75, 0.75)],
        },
        crs="EPSG:3857",
    )
    mask = (-1.0, -1.0, 0.5, 0.5)
    calls: list[bool] = []
    original = clip_module._clip_polygon_partition_with_rectangle_mask

    def _wrapped(partition, rectangle_bounds, *, keep_geom_type_only=False):
        calls.append(keep_geom_type_only)
        return original(
            partition,
            rectangle_bounds,
            keep_geom_type_only=keep_geom_type_only,
        )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_partition_with_rectangle_mask",
        _wrapped,
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_complex_polygon_partition_with_rectangle_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle keep_geom_type polygon clip should not route through host collection reconstruction"
        ),
    )

    result = clip(gdf, mask, keep_geom_type=True)

    assert calls == [True]
    assert result["value"].tolist() == [1]
    assert isinstance(result.geometry.values, DeviceGeometryArray)


def test_clip_rectangle_keep_geom_type_multipolygon_stays_off_host_boundary_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": [
                shapely.MultiPolygon(
                    [
                        Polygon(
                            [
                                (-2.0, -2.0),
                                (4.0, -2.0),
                                (4.0, 4.0),
                                (-2.0, 4.0),
                                (-2.0, -2.0),
                            ]
                        ),
                        Polygon(
                            [
                                (-2.0, 6.0),
                                (0.0, 6.0),
                                (0.0, 9.0),
                                (-2.0, 7.5),
                                (-2.0, 6.0),
                            ]
                        ),
                    ]
                )
            ],
        },
        crs="EPSG:3857",
    )
    mask = (0.0, 0.0, 10.0, 10.0)

    monkeypatch.setattr(
        clip_module,
        "_exact_polygon_clip_boundary_rows",
        lambda *args, **kwargs: pytest.fail(
            "rectangle keep_geom_type multipolygon clip should not recover "
            "polygonal collection parts on the host"
        ),
    )
    monkeypatch.setattr(
        clip_module,
        "_record_clip_host_cleanup_fallback",
        lambda *args, **kwargs: pytest.fail(
            "rectangle keep_geom_type multipolygon clip should stay off host cleanup"
        ),
    )

    result = clip(gdf, mask, keep_geom_type=True)

    assert result["value"].tolist() == [1]
    assert result.geom_type.tolist() == ["Polygon"]
    assert shapely.equals(result.geometry.iloc[0], box(0.0, 0.0, 4.0, 4.0))
    assert isinstance(result.geometry.values, DeviceGeometryArray)


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
