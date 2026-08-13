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
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial
from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeTabularResult,
    NativeTabularSelection,
)
from vibespatial.api.geometry_array import LINE_GEOM_TYPES, POLYGON_GEOM_TYPES, GeometryArray
from vibespatial.api.tools.clip import clip
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

clip_module = importlib.import_module("vibespatial.api.tools.clip")


def _assert_native_geometry_device_resident(values: DeviceGeometryArray) -> None:
    """Accept either contiguous owned storage or a device-native composition."""
    owned = values.cached_owned()
    if owned is not None:
        assert owned.residency is Residency.DEVICE
        assert owned.device_state is not None
        return
    composition = values.native_composition
    assert composition is not None
    assert composition.residency is Residency.DEVICE


def _assert_native_geometry_trusted_valid(values) -> None:
    """Require an all-valid proof without imposing a contiguous layout."""
    if isinstance(values, DeviceGeometryArray):
        owned = values.cached_owned()
        if owned is None:
            composition = values.native_composition
            assert composition is not None
            assert composition.trusted_all_ogc_valid is True
            return
    else:
        owned = getattr(values, "_owned", None)
    assert owned is not None
    cached = owned._cached_is_valid_mask
    if cached is not None:
        np.testing.assert_array_equal(cached, np.ones(len(values), dtype=bool))
        return
    assert owned.device_state is not None
    assert owned.device_state.trusted_all_ogc_valid is True


def _clip_partition_objects(result) -> np.ndarray:
    if isinstance(result, GeometryNativeResult):
        return np.asarray(
            result.to_geoseries(
                index=pd.RangeIndex(result.row_count),
                name="geometry",
            ),
            dtype=object,
        )
    values = result.geometry_values if hasattr(result, "geometry_values") else result
    return np.asarray(values, dtype=object)


def test_clip_public_tool_has_no_raw_cupy_scalar_syncs() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
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


def test_clip_relation_admission_uses_device_rowsets() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    retired_fences = {
        "clip active family admission scalar fence",
        "clip single-mask candidate bounds coverage scalar fence",
        "clip lazy grouped-union relation fanout scalar fence",
        "clip lazy grouped-union uncovered source-row scalar fence",
        "clip invalid polygon candidate scalar fence",
        "clip rectangle source positive-span scalar fence",
        "clip rectangle polygonal mixed-family row collision scalar fence",
        "clip exact-topology admission count packet",
    }

    assert all(fence not in source for fence in retired_fences)


def test_clip_bounds_filter_dispatch_uses_physical_shape_estimate(monkeypatch) -> None:
    class _Buffer:
        x = np.arange(60_000)

    class _Owned:
        row_count = 10
        families = {"polygon": _Buffer()}
        residency = Residency.HOST

    monkeypatch.setattr(clip_module, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(clip_module, "strict_native_mode_enabled", lambda: False)

    assert clip_module._clip_bounds_filter_selects_device(10, None) is False
    assert clip_module._clip_bounds_filter_selects_device(10, _Owned()) is True


def test_densified_axis_aligned_rectangle_mask_bounds_are_recognized() -> None:
    mask = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (1.0, 2.0),
            (0.0, 2.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]
    )

    assert clip_module._rectangle_bounds_from_mask(mask) == (0.0, 0.0, 2.0, 2.0)


def test_nonrectilinear_boundary_mask_bounds_are_not_rectangle() -> None:
    mask = Polygon(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (1.0, 1.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]
    )

    assert clip_module._rectangle_bounds_from_mask(mask) is None


def _materialize_native_clip_result(
    result: NativeTabularResult,
    *,
    source: vibespatial.GeoDataFrame | vibespatial.GeoSeries,
):
    return clip_module._clip_native_tabular_to_spatial(result, source=source)


@pytest.mark.skipif(not vibespatial.has_gpu_runtime(), reason="GPU runtime required")
def test_clip_terminal_selected_geometry_export_avoids_device_array_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 1.0, 2.0, 2.0),
        ],
        residency=Residency.DEVICE,
    )
    values = DeviceGeometryArray._from_owned(owned)

    monkeypatch.setattr(
        DeviceGeometryArray,
        "__array__",
        lambda *_args, **_kwargs: pytest.fail(
            "clip selected-row terminal export must not use generic DGA __array__"
        ),
    )
    clear_materialization_events()

    selected = clip_module._take_geometry_object_values(
        values,
        np.asarray([1], dtype=np.int64),
    )
    events = get_materialization_events(clear=True)

    assert selected.shape == (1,)
    assert selected[0].equals(box(1.0, 1.0, 2.0, 2.0))
    assert any(
        event.surface == "vibespatial.api.tools.clip._take_geometry_object_values"
        and event.operation == "clip_selected_geometry_rows_to_shapely"
        for event in events
    )
    assert not any(
        event.surface == "vibespatial.geometry.DeviceGeometryArray.__array__" for event in events
    )


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
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device rectangle clip dispatch")
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
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device rectangle clip dispatch")
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
        warning for warning in caught if "CRS mismatch between the CRS" in str(warning.message)
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


def test_clip_polygon_boundary_touch_mask_is_host_compatibility_only() -> None:
    values = [
        box(0.0, 0.0, 2.0, 2.0),
        box(3.0, 0.0, 5.0, 2.0),
    ]
    gdf = vibespatial.GeoDataFrame(
        {"geometry": values},
        crs="EPSG:3857",
    )
    source_values = gdf.geometry.values
    mask = Polygon([(1.0, -1.0), (4.0, -1.0), (4.0, 1.0), (1.0, 1.0), (1.0, -1.0)])
    boundary_rows = np.asarray([0, 1], dtype=np.intp)

    result = clip_module._clip_polygon_boundary_touch_mask(
        source_values,
        boundary_rows,
        mask=mask,
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

    if isinstance(native_result, NativeTabularSelection):
        assert native_result.capacity_result.geometry.owned is not None
        assert native_result.capacity_result.geometry.owned.row_count == 2
        assert native_result.to_geodataframe()["value"].tolist() == [1]
    else:
        assert native_result.geometry.owned is not None
        assert native_result.geometry.owned.row_count == 1
        assert native_result.attributes.to_pandas()["value"].tolist() == [1]


def test_clip_mask_covering_source_bounds_device_passthrough_keeps_rowset_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.runtime.residency import TransferTrigger

    geometries = [box(0, 0, 1, 1), Polygon()]
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test clip mask-cover native rowset passthrough",
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = box(-1.0, -1.0, 2.0, 2.0)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module._clip_mask_covers_source_bounds_passthrough_native(
        gdf,
        mask,
        (0.0, 0.0, 1.0, 1.0),
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert native_result is not None
    assert isinstance(native_result, NativeTabularSelection)
    assert native_result.capacity_result.geometry.owned is not None
    assert native_result.capacity_result.geometry.owned.row_count == 2
    assert native_result.capacity_result.attributes.device_table is not None
    assert native_result.capacity_result.index_plan is not None
    assert native_result.capacity_result.index_plan.kind == "range"
    assert cp.asnumpy(native_result.logical_count).tolist() == [1]
    assert cp.asnumpy(native_result.selection.positions[:1]).tolist() == [0]
    assert get_materialization_events(clear=True) == []
    assert "clip mask-cover passthrough valid-nonempty row mask" not in runtime_reasons

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [1]
    assert exported.index.tolist() == [0]


def test_clip_rowset_terminal_export_reuses_public_source_attributes() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import (
        attach_native_state_from_native_tabular_result,
        get_native_state,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.8, 0.8),
            box(0.5, 0.5, 1.5, 1.5),
            box(3.2, 3.2, 3.8, 3.8),
            box(4.5, 4.5, 5.0, 5.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30, 40],
            "label": ["inside", "crossing", "outside", "far"],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            dataframe=pd.DataFrame(
                {
                    "value": [10, 20, 30, 40],
                    "label": ["inside", "crossing", "outside", "far"],
                },
                index=gdf.index,
            ),
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "label", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        Polygon(
            [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        ),
        sort=False,
    )

    assert native_result is not None
    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.loader is not None
    assert capacity_result.terminal_geodataframe_materializer_owns_export is True

    clear_materialization_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    exported = native_result.to_geodataframe()
    events = get_materialization_events(clear=True)
    d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    exported_state = get_native_state(exported)

    assert exported[["value", "label"]].to_dict("list") == {
        "value": [10, 20],
        "label": ["inside", "crossing"],
    }
    assert exported.index.tolist() == [0, 1]
    assert exported_state is not None
    assert exported_state.attributes.loader is not None
    assert exported_state.attributes.index.equals(exported.index)
    assert any(
        reason.endswith("clip_terminal_source_rows_to_host")
        or reason == "clip terminal source attribute rows export"
        for reason in d2h_reasons
    )
    assert any(event.operation == "clip_terminal_source_rows_to_host" for event in events)
    assert not any(event.operation == "index_plan_to_host" for event in events)
    assert not any(
        event.surface == "vibespatial.api.NativeAttributeTable.to_arrow" for event in events
    )


def test_clip_device_attribute_source_skips_terminal_source_materializer_probe() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import (
        attach_native_state_from_native_tabular_result,
        get_native_state,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(0.1, 0.1, 0.8, 0.8),
        box(0.5, 0.5, 1.5, 1.5),
        box(3.2, 3.2, 3.8, 3.8),
        box(4.5, 4.5, 5.0, 5.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30, 40],
            "label": ["inside", "crossing", "outside", "far"],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table(
        {
            "value": pa.array([10, 20, 30, 40], type=pa.int64()),
            "label": pa.array(
                ["inside", "crossing", "outside", "far"],
                type=pa.string(),
            ),
        }
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "label", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    clear_materialization_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        Polygon(
            [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        ),
        sort=False,
    )

    assert native_result is not None
    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.terminal_geodataframe_materializer_owns_export is False
    assert get_materialization_events(clear=True) == []

    clear_materialization_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    exported = native_result.to_geodataframe()
    events = get_materialization_events(clear=True)
    d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    exported_state = get_native_state(exported)

    assert exported_state is not None
    assert exported_state.attributes.device_table is not None
    assert not any(
        event.operation == "native_attribute_column_to_public_series" for event in events
    )
    assert not any(event.operation == "index_plan_to_host" for event in events)
    assert "clip terminal source attribute rows export" not in d2h_reasons
    assert exported["value"].tolist() == [10, 20]


def test_clip_secondary_geometry_probe_does_not_materialize_lazy_device_attributes() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(0.1, 0.1, 0.8, 0.8),
        box(0.5, 0.5, 1.5, 1.5),
        box(3.2, 3.2, 3.8, 3.8),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attributes = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attributes),
            column_override=tuple(attributes.column_names),
            schema_override=attributes.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(source, native_payload)
    source = native_payload.to_geodataframe()
    lazy_value = source["value"]
    assert isinstance(lazy_value.array, NativeNumericExpressionArray)
    source["bucket"] = (lazy_value % 2).astype(str)

    clear_materialization_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    native_result = clip_module.evaluate_geopandas_clip_native(
        source,
        Polygon(
            [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        ),
        sort=False,
    )
    events = get_materialization_events(clear=True)
    d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert native_result is not None
    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert tuple(capacity_result.attributes.columns) == ("value", "bucket")
    assert not any(
        event.operation == "native_attribute_column_to_public_series" for event in events
    )
    assert not any(
        reason
        == "vibespatial.api.GeoDataFrame.__getitem__::native_attribute_column_to_public_series"
        for reason in d2h_reasons
    )


def test_clip_homogeneous_polygon_device_candidates_skip_candidate_rows_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(0.1, 0.1, 0.8, 0.8),
        box(0.5, 0.5, 1.5, 1.5),
        box(3.2, 3.2, 3.8, 3.8),
        box(4.5, 4.5, 5.0, 5.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30, 40],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
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

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    assert capacity_result.geometry.residency is Residency.DEVICE
    if capacity_result.geometry.composition is not None:
        assert all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in capacity_result.geometry.composition.parts
        )
    else:
        assert capacity_result.geometry.owned is not None
        assert capacity_result.geometry.owned.residency is Residency.DEVICE
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    assert not any(reason.startswith("owned geometry host metadata") for reason in runtime_reasons)
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert not any("candidate" in surface for surface in materialization_surfaces)
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 20]


def test_clip_homogeneous_polygon_device_candidate_boundary_rows_stay_device_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import clear_materialization_events

    geometries = [
        box(1.0, 0.2, 2.0, 0.8),
        box(0.2, 0.2, 0.8, 0.8),
        box(3.0, 3.0, 4.0, 4.0),
        box(1.0, 1.0, 2.0, 2.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30, 40],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()
    monkeypatch.setattr(
        clip_module,
        "_take_geometry_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "simple polygon boundary clip should stay on native boundary carriers"
        ),
    )

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(0.0, 0.0, 1.0, 1.0),
        sort=False,
    )
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    assert capacity_result.provenance is not None
    assert capacity_result.provenance.is_device
    assert "clip polygon device-candidate boundary rows host export" not in runtime_reasons
    assert "clip selected geometry rows host export" not in runtime_reasons
    assert "clip polygon device-candidate output rows host export" not in runtime_reasons
    assert "clip polygon boundary unsupported-family scalar fence" not in runtime_reasons
    assert "clip polygon boundary mixed-family row collision scalar fence" not in runtime_reasons
    assert any(
        event.implementation == "polygon_device_candidate_direct_rowset_assembly_gpu"
        for event in dispatch_events
    )

    exported = native_result.to_geodataframe()

    assert set(exported["value"].tolist()) == {10, 20, 40}
    values_by_type = dict(zip(exported.geom_type.tolist(), exported["value"].tolist(), strict=True))
    assert values_by_type["LineString"] == 10
    assert values_by_type["Polygon"] == 20
    assert values_by_type["Point"] == 40


def test_clip_polygon_area_and_boundary_remnants_use_native_composition() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geometry = MultiPolygon(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(3.0, 0.0, 4.0, 1.0),
        ]
    )
    mask = Polygon(
        [
            (1.0, -1.0),
            (3.0, -1.0),
            (3.0, 1.0),
            (2.5, 1.0),
            (2.5, 2.5),
            (1.0, 2.5),
            (1.0, -1.0),
        ]
    )
    owned = from_shapely_geometries(
        [source_geometry],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [7],
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:3857",
            ),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attach_native_state_from_native_tabular_result(
        source,
        NativeTabularResult(
            attributes=NativeAttributeTable(
                dataframe=pd.DataFrame({"value": [7]}),
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        ),
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    native_result = clip_module._clip_homogeneous_polygon_device_candidates_native(
        source,
        mask,
        cp.asarray([0], dtype=cp.int64),
        mask_owned=from_shapely_geometries([mask], residency=Residency.DEVICE),
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert native_result is not None
    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.geometry.composition is not None
    assert capacity_result.geometry.residency is Residency.DEVICE
    assert all(
        part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
        for part in capacity_result.geometry.composition.parts
    )
    d2h_events = get_d2h_transfer_events(clear=True)
    assert d2h_events == []

    exported = native_result.to_geodataframe()
    expected = shapely.intersection(source_geometry, mask)
    assert exported.geom_type.tolist() == ["GeometryCollection"]
    assert shapely.equals(exported.geometry.iloc[0], expected)


@pytest.mark.gpu
def test_clip_geometrycollection_ingress_uses_native_grouped_union() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.api._native_result_core import NativeTabularSelection
    from vibespatial.api.tools.clip import evaluate_geopandas_clip_native
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.residency import Residency

    source_geometry = GeometryCollection(
        [
            Point(2.0, 3.0),
            Polygon(
                [
                    (3.0, 4.0),
                    (5.0, 2.0),
                    (12.0, 2.0),
                    (10.0, 5.0),
                    (9.0, 7.5),
                    (3.0, 4.0),
                ]
            ),
        ]
    )
    source = vibespatial.GeoDataFrame(
        {"value": [7], "geometry": [source_geometry]},
        crs="EPSG:3857",
    )
    mask = box(0.0, 0.0, 10.0, 10.0)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    native_result = evaluate_geopandas_clip_native(source, mask)

    assert isinstance(native_result, NativeTabularSelection)
    composition = native_result.capacity_result.geometry.composition
    assert composition is not None
    assert composition.residency is Residency.DEVICE
    assert all(part.collection_position is None for part in composition.parts)
    d2h_events = get_d2h_transfer_events(clear=True)
    assert [event.reason for event in d2h_events] == [
        "fused multi-root capacity scatter exact allocation packet",
    ]
    assert sum(event.bytes_transferred for event in d2h_events) <= 440

    with pytest.warns(UserWarning, match="GeometryCollection"):
        result = clip(source, mask, keep_geom_type=True)
    expected = shapely.intersection(source_geometry, mask)
    assert result["value"].tolist() == [7]
    assert result.geometry.iloc[0].geom_type == "GeometryCollection"
    assert shapely.equals(result.geometry.iloc[0], expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "source_geometry",
    [
        GeometryCollection(
            [
                Point(1.0, 1.0),
                box(0.0, 0.0, 2.0, 2.0),
                GeometryCollection([Point(4.0, 4.0)]),
            ]
        ),
        GeometryCollection(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(1.0, 0.0, 3.0, 2.0),
            ]
        ),
    ],
    ids=["covered-point-and-nested-member", "overlapping-polygons"],
)
def test_clip_geometrycollection_grouped_union_matches_geos(source_geometry) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source = vibespatial.GeoDataFrame(
        {"value": [11], "geometry": [source_geometry]},
        crs="EPSG:3857",
    )
    mask = box(-1.0, -1.0, 5.0, 5.0)

    result = clip(source, mask)
    expected = shapely.intersection(source_geometry, mask)

    assert result["value"].tolist() == [11]
    assert shapely.equals_exact(
        shapely.normalize(result.geometry.iloc[0]),
        shapely.normalize(expected),
        tolerance=1e-12,
    )


@pytest.mark.gpu
def test_clip_geometrycollection_drops_fully_covered_point_members() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source_geometry = GeometryCollection(
        [
            box(0.0, 0.0, 2.0, 2.0),
            Point(1.0, 1.0),
        ]
    )
    source = vibespatial.GeoDataFrame(
        {"value": [11], "geometry": [source_geometry]},
        crs="EPSG:3857",
    )
    mask = box(-1.0, -1.0, 3.0, 3.0)

    result = clip(source, mask)
    expected = shapely.intersection(source_geometry, mask)

    assert result["value"].tolist() == [11]
    assert shapely.equals_exact(
        shapely.normalize(result.geometry.iloc[0]),
        shapely.normalize(expected),
        tolerance=1e-12,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "source_geometry",
    [
        GeometryCollection([Point(9.0, 9.0)]),
        GeometryCollection([Point()]),
    ],
    ids=["fully-excluded", "empty-member"],
)
def test_clip_geometrycollection_zero_output_skips_grouped_reduction(
    source_geometry,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source = vibespatial.GeoDataFrame(
        {"value": [11], "geometry": [source_geometry]},
        crs="EPSG:3857",
    )

    result = clip(source, box(0.0, 0.0, 1.0, 1.0))

    assert result.empty
    assert result.columns.tolist() == ["value", "geometry"]


@pytest.mark.gpu
def test_clip_geometrycollection_grouped_union_honors_public_index_sort() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    source = vibespatial.GeoDataFrame(
        {
            "value": [20, 10],
            "geometry": [
                GeometryCollection([box(4.0, 0.0, 5.0, 1.0), Point(6.0, 0.5)]),
                GeometryCollection([box(0.0, 0.0, 1.0, 1.0), Point(2.0, 0.5)]),
            ],
        },
        index=[20, 10],
        crs="EPSG:3857",
    )

    result = clip(source, box(-1.0, -1.0, 7.0, 2.0), sort=True)

    assert result.index.tolist() == [10, 20]
    assert result["value"].tolist() == [10, 20]


@pytest.mark.gpu
def test_clip_geometrycollection_probe_does_not_materialize_device_source() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime.residency import Residency

    owned = from_shapely_geometries(
        [box(-1.0, -1.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = clip(source, box(0.0, 0.0, 1.0, 1.0))
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result["value"].tolist() == [1]
    assert not any(reason.startswith("owned geometry") for reason in reasons)


def test_clip_rectangle_keep_geom_type_device_candidates_use_native_rowsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    geometries = [
        box(0.2, 0.2, 0.8, 0.8),
        Polygon(
            [
                (-1.0, 0.1),
                (0.6, 0.1),
                (0.6, 0.4),
                (1.2, 0.4),
                (1.2, 0.9),
                (-1.0, 0.9),
                (-1.0, 0.1),
            ]
        ),
        MultiPolygon([box(-0.5, 0.3, 0.3, 0.7), box(3.0, 3.0, 4.0, 4.0)]),
        box(3.0, 3.0, 4.0, 4.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30, 40],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_partition_with_rectangle_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle keep_geom_type should not build a host candidate partition"
        ),
    )
    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        (0.0, 0.0, 1.0, 1.0),
        keep_geom_type=True,
        sort=False,
    )
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.provenance is not None
    assert capacity_result.provenance.is_device
    assert any(
        event.implementation == "polygon_device_candidate_direct_rowset_assembly_gpu"
        and "keep_geom_type=True" in event.detail
        for event in dispatch_events
    )
    assert not any("keep-geometry-type" in reason for reason in runtime_reasons)
    assert "clip selected geometry rows host export" not in runtime_reasons

    exported = native_result.to_geodataframe()
    expected = [
        shapely.intersection(geometry, box(0.0, 0.0, 1.0, 1.0)) for geometry in geometries[:3]
    ]
    expected_by_value = dict(zip((10, 20, 30), expected, strict=True))
    assert set(exported["value"].tolist()) == set(expected_by_value)
    assert all(
        shapely.equals(actual, expected_by_value[value])
        for value, actual in zip(
            exported["value"],
            exported.geometry,
            strict=True,
        )
    )


def test_clip_boundary_line_parts_pack_at_output_capacity() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import LinePartCapacitySelection
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    base_line_parts = from_shapely_geometries(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(2.0, 0.0), (3.0, 0.0)]),
            LineString([(10.0, 0.0), (10.5, 0.5), (11.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    line_parts = base_line_parts.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))
    assert line_parts.is_indexed_view
    d_output_rows = cp.asarray([7, 5, 5], dtype=cp.int64)
    part_capacity = LinePartCapacitySelection(
        geometry=line_parts,
        source_rows=cp.arange(3, dtype=cp.int32),
        selection=NativeDeviceSelection.from_mask(
            cp.ones(3, dtype=cp.bool_),
        ).as_capacity_prefix(),
        coord_capacity=7,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result_owned = clip_module._clip_pack_line_boundary_part_capacity_device(
        part_capacity,
        d_output_rows,
        output_row_count=9,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result_owned is not None
    assert result_owned.residency is Residency.DEVICE
    assert "owned geometry offset-slice allocation fence" not in runtime_reasons

    values = result_owned.to_shapely()
    first, second = values[5], values[7]
    assert first.geom_type == "MultiLineString"
    assert second.geom_type == "LineString"
    assert shapely.equals_exact(
        shapely.normalize(first),
        shapely.normalize(
            MultiLineString(
                [
                    [(0.0, 0.0), (1.0, 0.0)],
                    [(2.0, 0.0), (3.0, 0.0)],
                ],
            ),
        ),
        tolerance=0.0,
    )
    assert shapely.equals_exact(
        shapely.normalize(second),
        shapely.normalize(LineString([(10.0, 0.0), (10.5, 0.5), (11.0, 0.0)])),
        tolerance=0.0,
    )


def test_linestring_polygon_clip_consumes_indexed_line_rows_without_take_fence() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_polygon_intersection,
    )

    base_lines = from_shapely_geometries(
        [
            LineString([(-1.0, 0.0), (2.0, 0.0)]),
            LineString([(0.2, 0.2), (0.8, 0.2), (1.2, 0.2)]),
            LineString([(-0.5, 0.7), (0.5, 0.7), (1.5, 0.7)]),
        ],
        residency=Residency.DEVICE,
    )
    indexed_lines = base_lines.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))
    assert indexed_lines.is_indexed_view
    clip_polygons = from_shapely_geometries(
        [box(0.0, -0.25, 1.0, 1.0)] * 3,
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clipped = linestring_polygon_intersection(indexed_lines, clip_polygons)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert clipped.residency is Residency.DEVICE
    assert "owned geometry device-take slice-size allocation fence" not in runtime_reasons
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    assert [geom.geom_type for geom in _clip_partition_objects(clipped)] == [
        "LineString",
        "LineString",
        "LineString",
    ]


def test_clip_boundary_point_parts_pack_at_output_capacity() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import PointPartCapacitySelection
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    point_parts = from_shapely_geometries(
        [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(9.0, 9.0),
        ],
        residency=Residency.DEVICE,
    )
    d_output_rows = cp.asarray([3, 3, 8], dtype=cp.int64)
    part_capacity = PointPartCapacitySelection(
        geometry=point_parts,
        source_rows=cp.arange(3, dtype=cp.int32),
        selection=NativeDeviceSelection.from_mask(
            cp.ones(3, dtype=cp.bool_),
        ).as_capacity_prefix(),
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result_owned = clip_module._clip_pack_point_boundary_part_capacity_device(
        part_capacity,
        d_output_rows,
        output_row_count=9,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result_owned is not None
    assert result_owned.residency is Residency.DEVICE
    assert "owned geometry offset-slice allocation fence" not in runtime_reasons

    values = result_owned.to_shapely()
    first, second = values[3], values[8]
    assert first.geom_type == "MultiPoint"
    assert second.geom_type == "Point"
    assert shapely.equals_exact(
        shapely.normalize(first),
        shapely.normalize(MultiPoint([(0.0, 0.0), (1.0, 0.0)])),
        tolerance=0.0,
    )
    assert shapely.equals_exact(
        shapely.normalize(second),
        shapely.normalize(Point(9.0, 9.0)),
        tolerance=0.0,
    )


def test_clip_boundary_mixed_families_remain_native_composition() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        LinePartCapacitySelection,
        PointPartCapacitySelection,
    )

    line_parts = from_shapely_geometries(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(2.0, 0.0), (3.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    point_parts = from_shapely_geometries(
        [Point(9.0, 9.0), Point(10.0, 10.0)],
        residency=Residency.DEVICE,
    )
    all_active = NativeDeviceSelection.from_mask(
        cp.ones(2, dtype=cp.bool_),
    ).as_capacity_prefix()
    line_capacity = clip_module._clip_pack_line_boundary_part_capacity_device(
        LinePartCapacitySelection(
            geometry=line_parts,
            source_rows=cp.arange(2, dtype=cp.int32),
            selection=all_active,
            coord_capacity=4,
        ),
        cp.asarray([5, 5], dtype=cp.int64),
        output_row_count=9,
    )
    point_capacity = clip_module._clip_pack_point_boundary_part_capacity_device(
        PointPartCapacitySelection(
            geometry=point_parts,
            source_rows=cp.arange(2, dtype=cp.int32),
            selection=all_active,
        ),
        cp.asarray([5, 8], dtype=cp.int64),
        output_row_count=9,
    )

    result = clip_module._clip_geometry_composition_at_capacity(
        [
            (line_capacity, cp.arange(9, dtype=cp.int64)),
            (point_capacity, cp.arange(9, dtype=cp.int64)),
        ],
        row_count=9,
        crs=None,
    )

    assert result is not None
    assert result.composition is not None
    values = result.to_geoseries(index=pd.RangeIndex(9), name="geometry")
    mixed = values.iloc[5]
    assert isinstance(mixed, GeometryCollection)
    assert [part.geom_type for part in mixed.geoms] == [
        "MultiLineString",
        "Point",
    ]
    assert values.iloc[8].geom_type == "Point"


def test_clip_boundary_degenerate_line_part_becomes_point_capacity() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import LinePartCapacitySelection

    line_parts = from_shapely_geometries(
        [
            LineString([(1.0, 1.0), (1.0, 1.0)]),
            LineString([(2.0, 0.0), (3.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    partitioned = clip_module._clip_partition_degenerate_line_part_capacity_device(
        LinePartCapacitySelection(
            geometry=line_parts,
            source_rows=cp.arange(2, dtype=cp.int32),
            selection=NativeDeviceSelection.identity(2),
            coord_capacity=4,
        ),
        cp.asarray([2, 3], dtype=cp.int64),
    )

    assert partitioned is not None
    line_selection, d_line_rows, point_selection, d_point_rows = partitioned
    line_capacity = clip_module._clip_pack_line_boundary_part_capacity_device(
        line_selection,
        d_line_rows,
        output_row_count=4,
    )
    point_capacity = clip_module._clip_pack_point_boundary_part_capacity_device(
        point_selection,
        d_point_rows,
        output_row_count=4,
    )
    line_values = line_capacity.to_shapely()
    point_values = point_capacity.to_shapely()
    assert point_values[2].geom_type == "Point"
    assert shapely.equals_exact(point_values[2], Point(1.0, 1.0), tolerance=0.0)
    assert shapely.is_missing(line_values[2])
    assert line_values[3].geom_type == "LineString"


def test_clip_homogeneous_polygon_device_candidates_keep_geom_type_skips_candidate_rows_export() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(-0.5, 0.2, 0.5, 0.8),
        box(2.5, 0.2, 3.5, 0.8),
        box(0.2, 2.5, 0.8, 3.5),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
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

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        keep_geom_type=True,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    assert capacity_result.geometry.residency is Residency.DEVICE
    if capacity_result.geometry.composition is not None:
        assert all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in capacity_result.geometry.composition.parts
        )
    else:
        assert capacity_result.geometry.owned is not None
        assert capacity_result.geometry.owned.residency is Residency.DEVICE
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip polygon device-candidate output rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows::rowset_to_host"
        not in runtime_reasons
    )
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert not any("candidate" in surface for surface in materialization_surfaces)

    exported = native_result.to_geodataframe()

    assert set(exported["value"]) == {10, 20, 30}
    assert exported.geom_type.isin(POLYGON_GEOM_TYPES).all()


def test_clip_homogeneous_polygon_device_candidates_accept_rectangular_polygon_mask() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(0.1, 0.1, 0.8, 0.8),
        box(0.5, 0.5, 1.5, 1.5),
        box(3.2, 3.2, 3.8, 3.8),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame({"value": [10, 20, 30]})),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = box(0.0, 0.0, 1.0, 1.0)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.geometry.residency is Residency.DEVICE
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.attributes.loader is None
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip polygon device-candidate output rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 20]


def test_clip_polygon_device_candidates_host_label_index_keeps_attributes_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    index = pd.Index(["a", "b", "c"], name="parcel")
    owned = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.8, 0.8),
            box(0.5, 0.5, 1.5, 1.5),
            box(3.2, 3.2, 3.8, 3.8),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
        index=index,
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            dataframe=pd.DataFrame({"value": [10, 20, 30]}, index=index),
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        box(0.0, 0.0, 1.0, 1.0),
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "host-labels-take"
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.attributes.loader is None
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip polygon device-candidate output rows host export" not in runtime_reasons
    assert "index_plan_take_positions_to_host" not in materialization_operations
    assert "row_positions_to_host" not in materialization_operations

    exported = native_result.to_geodataframe()
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assert exported.index.tolist() == ["a", "b"]
    assert exported.index.name == "parcel"
    assert exported["value"].tolist() == [10, 20]
    assert "native_tabular_to_geodataframe" in materialization_operations
    assert "index_plan_take_positions_to_host" not in materialization_operations
    assert "row_positions_to_host" not in materialization_operations


def test_clip_source_rowset_marks_device_all_rows_identity_for_host_label_index() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import (
        attach_native_state_from_native_tabular_result,
        get_native_state,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    index = pd.Index([0], name="group")
    owned = from_shapely_geometries(
        [box(0.1, 0.1, 0.9, 0.9)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
        index=index,
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            dataframe=pd.DataFrame({"value": [10]}, index=index),
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    rowset = clip_module._clip_source_rowset_for_positions(
        gdf,
        np.asarray([0], dtype=np.int64),
        device_row_positions=cp.asarray([0], dtype=cp.int64),
    )

    assert rowset is not None
    assert rowset.identity is True

    taken = get_native_state(gdf).take(rowset, preserve_index=True)
    assert taken.index_plan.kind == "host-labels"

    clear_materialization_events()
    exported = taken.to_native_tabular_result().to_geodataframe()
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assert exported.index.tolist() == [0]
    assert exported.index.name == "group"
    assert "index_plan_take_positions_to_host" not in materialization_operations


def test_clip_single_host_label_polygon_keeps_identity_through_parquet_export(
    tmp_path,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    index = pd.Index([0], name="group")
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
        index=index,
    )
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            dataframe=pd.DataFrame({"value": [10]}, index=index),
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = Polygon([(1.0, -1.0), (3.0, 1.0), (1.0, 3.0), (1.0, -1.0)])

    clipped = clip(gdf, mask)
    assert clipped.index.tolist() == [0]

    clear_materialization_events()
    clipped.to_parquet(tmp_path / "clip.parquet")
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assert "geodataframe_to_parquet" in materialization_operations
    assert "index_plan_take_positions_to_host" not in materialization_operations


def test_clip_take_geometry_object_values_device_takes_sparse_rows_first() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    geometries = [box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(64)]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
    rows = np.asarray([7, 61], dtype=np.intp)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip_module._take_geometry_object_values(values, rows)
    runtime_events = get_d2h_transfer_events(clear=True)

    assert [geom.wkb for geom in result] == [geometries[7].wkb, geometries[61].wkb]
    assert not any(
        "owned geometry host metadata" in event.reason and event.item_count == len(geometries)
        for event in runtime_events
    )


def test_clip_polygon_mask_all_inside_rows_skip_rowset_host_exports() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.9, 0.9),
                box(0.1, 3.1, 0.9, 3.9),
                box(3.1, 0.1, 3.9, 0.9),
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
    assert not shapely.covers(mask, box(*gdf.total_bounds))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    with strict_native_environment():
        result = clip(gdf, mask, sort=False)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert result["row"].tolist() == [0, 1, 2]
    assert [geom.wkb for geom in result.geometry] == [geom.wkb for geom in gdf.geometry]
    assert not any("polygon_single_mask_inside_rows" in reason for reason in runtime_reasons)
    assert not any("polygon_single_mask_remaining_rows" in reason for reason in runtime_reasons)
    assert not any("polygon_mask_inside_rows" in reason for reason in runtime_reasons)
    assert not any("polygon_mask_exact_rows" in reason for reason in runtime_reasons)
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows" not in materialization_surfaces
    )
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces


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
        event.surface == "geopandas.clip" and event.pipeline == "clip.to_spatial"
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
    _assert_native_geometry_device_resident(result.geometry.values)
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
        event.surface == "geopandas.clip" and event.pipeline == "clip.to_spatial"
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
    _assert_native_geometry_device_resident(result.geometry.values)
    assert shapely.equals(result.geometry.iloc[0], shapely.intersection(source_geom, mask))
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "_clip_polygon_rectangle_area_intersection_owned"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_multipolygon_rectangle_keep_geom_type_sparse_parts_stay_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geom = MultiPolygon(
        [
            box(-1.0, -1.0, 2.0, 2.0),
            box(20.0, 20.0, 21.0, 21.0),
        ]
    )
    source_owned = from_shapely_geometries([source_geom], residency=Residency.DEVICE)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result_owned = clip_module._clip_multipolygon_rectangle_keep_geom_type_owned(
        source_owned,
        (0.0, 0.0, 10.0, 10.0),
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result_owned is not None
    assert result_owned.residency is Residency.DEVICE
    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)

    actual = np.asarray(result_owned.to_shapely(), dtype=object)[0]
    expected = shapely.intersection(source_geom, box(0.0, 0.0, 10.0, 10.0))
    assert shapely.equals(actual, expected)


def test_clip_rectangle_keep_geom_type_multipolygon_rescue_scatter_stays_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        MultiPolygon(
            [
                box(-1.0, -1.0, 2.0, 2.0),
                box(20.0, 20.0, 21.0, 21.0),
            ]
        ),
        MultiPolygon(
            [
                box(30.0, 30.0, 31.0, 31.0),
                box(40.0, 40.0, 41.0, 41.0),
            ]
        ),
    ]
    owned = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    partition = vibespatial.GeoDataFrame(
        {"geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")},
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    vibespatial.clear_dispatch_events()
    result_values = clip_module._clip_polygon_partition_with_rectangle_mask(
        partition,
        (0.0, 0.0, 10.0, 10.0),
        keep_geom_type_only=True,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)

    actual = np.asarray(result_values, dtype=object)
    expected = shapely.intersection(source_geoms[0], box(0.0, 0.0, 10.0, 10.0))
    assert shapely.equals(actual[0], expected)
    assert shapely.is_missing(actual[1]) or shapely.is_empty(actual[1])


def test_clip_rectangle_keep_geom_type_area_scatter_compacts_rows_on_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        box(-1.0, -1.0, 2.0, 2.0),
        box(20.0, 20.0, 21.0, 21.0),
    ]
    owned = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    partition = vibespatial.GeoDataFrame(
        {"geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")},
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    vibespatial.clear_dispatch_events()
    result_values = clip_module._clip_polygon_partition_with_rectangle_mask(
        partition,
        (0.0, 0.0, 10.0, 10.0),
        keep_geom_type_only=True,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)
    assert not any("keep-geometry-type" in reason for reason in runtime_reasons)
    assert any(
        event.implementation == "polygon_rectangle_keep_geom_type_rowset_gpu"
        for event in vibespatial.get_dispatch_events(clear=True)
    )

    actual = np.asarray(result_values, dtype=object)
    expected = shapely.intersection(source_geoms[0], box(0.0, 0.0, 10.0, 10.0))
    assert shapely.equals(actual[0], expected)
    assert shapely.is_missing(actual[1]) or shapely.is_empty(actual[1])


def test_clip_rectangle_keep_geom_type_rescue_declines_before_host_residual_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        MultiPolygon(
            [
                box(-1.0, -1.0, 2.0, 2.0),
                box(20.0, 20.0, 21.0, 21.0),
            ]
        ),
        MultiPolygon(
            [
                box(30.0, 30.0, 31.0, 31.0),
                box(40.0, 40.0, 41.0, 41.0),
            ]
        ),
    ]
    owned = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    partition = vibespatial.GeoDataFrame(
        {"geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")},
        crs="EPSG:3857",
    )
    calls: list[tuple[str, int]] = []

    def _rescue(left_owned, rectangle_bounds):
        assert left_owned.residency is Residency.DEVICE
        calls.append(("rescue", left_owned.row_count))
        return None

    def _generic_area(left_owned, rectangle_bounds):
        assert left_owned.residency is Residency.DEVICE
        calls.append(("area", left_owned.row_count))
        return from_shapely_geometries(
            [None] * left_owned.row_count,
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        clip_module,
        "_clip_multipolygon_rectangle_keep_geom_type_owned",
        _rescue,
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_rectangle_area_intersection_owned",
        _generic_area,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    with pytest.raises(
        StrictNativeFallbackError,
        match="declined before host residual reconstruction",
    ):
        clip_module._clip_polygon_partition_with_rectangle_mask(
            partition,
            (0.0, 0.0, 10.0, 10.0),
            keep_geom_type_only=True,
        )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert calls == [("rescue", 2)]
    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)
    assert not any("keep-geometry-type" in reason for reason in runtime_reasons)


def test_clip_scalar_rectangle_device_multipolygon_uses_canonical_device_path(
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

    assert calls == []
    assert len(result) == 1
    assert result.geom_type.iloc[0] in POLYGON_GEOM_TYPES
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    _assert_native_geometry_device_resident(result.geometry.values)
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
    from vibespatial.api._native_state import get_native_state

    native_state = get_native_state(result)
    assert native_state is not None
    composition = native_state.geometry.composition
    assert composition is not None
    assert all(
        part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
        for part in composition.parts
    )

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
    cp = pytest.importorskip("cupy")

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
    exact_capacity_validity = []

    def wrapped(left_owned, mask_owned, **kwargs):
        exact_capacity_validity.append(cp.asarray(left_owned.device_state.validity, dtype=cp.bool_))
        return original(left_owned, mask_owned, **kwargs)

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        wrapped,
    )

    result = clip(gdf, mask, sort=False)

    assert len(exact_capacity_validity) == 1
    assert cp.asnumpy(exact_capacity_validity[0]).tolist() == [True, True, False, False]
    assert list(result["row"]) == [0, 1, 3]


def test_clip_polygon_mask_predicate_split_derives_exact_from_smaller_rowsets() -> None:
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert list(result["row"]) == [0, 1, 3]
    assert "binary predicate fused predicate-results host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_outside_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_outside_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces


def test_clip_polygon_mask_split_derives_inside_from_sparse_outside_rowset() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2, 3, 4]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.8, 0.8),
                box(0.1, 2.0, 0.8, 2.8),
                box(0.5, 0.5, 1.5, 1.5),
                box(2.0, 2.0, 2.5, 2.5),
                box(2.5, 0.2, 3.5, 0.8),
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert list(result["row"]) == [0, 1, 2, 4]
    assert "binary predicate fused predicate-results host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_outside_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_outside_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces


def test_clip_polygon_mask_all_hit_derives_exact_from_inside_rowset() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.1, 0.1, 0.8, 0.8),
                box(-0.5, 0.2, 0.5, 0.8),
                box(0.2, 3.5, 0.8, 4.5),
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert set(result["row"]) == {0, 1, 2}
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces


def test_clip_polygon_mask_partition_reuses_candidate_capacity_executor() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.8, 0.8),
            box(-0.5, 0.2, 0.5, 0.8),
            box(0.2, 3.5, 0.8, 4.5),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2], "geometry": DeviceGeometryArray._from_owned(owned)},
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

    result = clip_module._clip_polygon_partition_with_polygon_mask(gdf, mask)

    assert isinstance(result, GeometryNativeResult)
    assert result.row_count == len(gdf)
    assert result.composition is not None


def test_clip_polygon_mask_partition_uses_direct_topology_without_exact_row_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.api._native_rowset import NativeRowSet

    owned = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.8, 0.8),
            box(-0.5, 0.2, 0.5, 0.8),
            box(0.2, 3.5, 0.8, 4.5),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2], "geometry": DeviceGeometryArray._from_owned(owned)},
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
    original_to_host_positions = NativeRowSet.to_host_positions
    exact_host_exports: list[object] = []

    def _record_to_host_positions(
        self,
        *,
        surface="vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed=True,
    ):
        if surface == "vibespatial.api.tools.clip.polygon_mask_exact_rows":
            exact_host_exports.append(self.positions)
        return original_to_host_positions(
            self,
            surface=surface,
            strict_disallowed=strict_disallowed,
        )

    monkeypatch.setattr(NativeRowSet, "to_host_positions", _record_to_host_positions)
    vibespatial.clear_dispatch_events()

    result = clip_module._clip_polygon_partition_with_polygon_mask(gdf, mask)
    events = vibespatial.get_dispatch_events(clear=True)

    assert isinstance(result, GeometryNativeResult)
    assert result.row_count == 3
    assert exact_host_exports == []
    assert any(
        event.operation == "intersection"
        and event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_clip_owned_nonempty_polygon_rows_sparse_exports_rows_not_full_mask() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), *(Polygon() for _ in range(9))],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip_module._owned_nonempty_polygon_rows(owned)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.mask.tolist() == [True, *([False] * 9)]
    assert result.rows.tolist() == [0]
    assert cp.asnumpy(result.device_rows).tolist() == [0]
    assert "clip keep-geometry-type polygonal positive-area terminal rows export" in runtime_reasons
    assert (
        "clip keep-geometry-type polygonal positive-area terminal mask export"
        not in runtime_reasons
    )


def test_clip_owned_nonempty_polygon_mask_sparse_exports_rows_not_full_mask() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), *(Polygon() for _ in range(9))],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    mask = clip_module._owned_nonempty_polygon_mask(owned)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert mask.tolist() == [True, *([False] * 9)]
    assert "clip keep-geometry-type polygonal positive-area terminal rows export" in runtime_reasons
    assert (
        "clip keep-geometry-type polygonal positive-area terminal mask export"
        not in runtime_reasons
    )


def test_clip_owned_nonempty_polygon_rows_all_positive_skips_host_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip_module._owned_nonempty_polygon_rows(owned)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.mask.tolist() == [True, True, True]
    assert result.rows.tolist() == [0, 1, 2]
    assert cp.asnumpy(result.device_rows).tolist() == [0, 1, 2]
    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)


def test_clip_owned_nonempty_polygon_mask_all_positive_skips_host_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    mask = clip_module._owned_nonempty_polygon_mask(owned)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert mask.tolist() == [True, True, True]
    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)


def test_clip_polygon_rectangle_area_sparse_positive_rows_stay_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(4.0, 4.0, 5.0, 5.0),
            box(0.5, 0.5, 1.5, 1.5),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip_module._clip_polygon_rectangle_area_intersection_owned(
        owned,
        (0.0, 0.0, 1.0, 1.0),
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.residency is Residency.DEVICE
    assert cp.asnumpy(result._ensure_device_state().validity).tolist() == [
        True,
        False,
        True,
    ]
    assert not any("polygonal positive-area" in reason for reason in runtime_reasons)


def test_clip_concave_mask_rectangular_cells_use_validated_device_carrier() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")

    cells = [
        box(float(x), float(y), float(x) + 0.9, float(y) + 0.9)
        for x in range(36)
        for y in range(36)
    ]
    concave_mask = Polygon(
        [
            (2.0, 2.0),
            (16.0, 2.0),
            (16.0, 6.0),
            (7.0, 6.0),
            (7.0, 16.0),
            (2.0, 16.0),
            (2.0, 2.0),
        ]
    )
    left_owned = from_shapely_geometries(cells, residency=Residency.DEVICE)
    mask_owned = from_shapely_geometries([concave_mask], residency=Residency.DEVICE)

    vibespatial.clear_dispatch_events()
    result = clip_module._clip_polygon_area_intersection_owned(
        left_owned,
        mask_owned,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert result.residency is Residency.DEVICE
    assert result.row_count == left_owned.row_count
    assert any(
        event.operation == "clip"
        and event.implementation == "polygon_rect_cell_mask_split_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.implementation == "polygon_mask_broadcast_right_capacity_gpu" for event in events
    )


def test_clip_source_nonmissing_compatibility_identity_skips_host_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    rowset = clip_module._clip_source_nonmissing_rowset(
        values,
        source_token="clip-source-identity-test",
        prefer_device=True,
    )
    rows = clip_module._clip_source_nonmissing_rows_for_compatibility(
        rowset,
        surface="test.clip.identity_compatibility_rows",
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert rows.tolist() == [0, 1]
    assert not any("identity_compatibility_rows" in reason for reason in runtime_reasons)
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)


def test_clip_source_nonmissing_compatibility_exports_sparse_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    geometries = [box(float(i), 0.0, float(i + 1), 1.0) for i in range(10)]
    geometries[4] = Polygon()
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    rowset = clip_module._clip_source_nonmissing_rowset(
        values,
        source_token="clip-source-sparse-test",
        prefer_device=True,
    )
    rows = clip_module._clip_source_nonmissing_rows_for_compatibility(
        rowset,
        surface="test.clip.sparse_compatibility_rows",
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert rows.tolist() == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    assert "test.clip.sparse_compatibility_rows::rowset_to_host" in runtime_reasons
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)


def test_clip_source_nonmissing_rowset_host_contract() -> None:
    owned = from_shapely_geometries([None, Polygon(), box(0, 0, 1, 1)])
    values = GeometryArray.from_owned(owned)

    rowset = clip_module._clip_source_nonmissing_rowset(
        values,
        source_token="clip-source-test",
        prefer_device=False,
    )

    assert rowset.source_token == "clip-source-test"
    assert rowset.source_row_count == 3
    assert rowset.trusted_all_valid_rows is True
    assert rowset.positions.tolist() == [2]


def test_clip_source_nonmissing_rowset_device_stays_resident() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), Polygon(), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    rowset = clip_module._clip_source_nonmissing_rowset(
        values,
        source_token="clip-source-device-test",
        prefer_device=True,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert rowset.is_device
    assert rowset.safe_capacity_positions(fill_value=-1).get().tolist() == [0, 2, -1]
    assert rowset.logical_count.get().tolist() == [2]
    assert not any("clip source missing" in reason for reason in runtime_reasons)
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)


def test_polygon_mask_clip_uses_nonmissing_rowset_before_host_compatibility_tail() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    function_start = source.index("def _clip_polygon_partition_with_polygon_mask(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]
    host_tail = function_source.index(
        "nonmissing_rows = _clip_source_nonmissing_rows_for_compatibility"
    )

    assert "_clip_source_nonmissing_rowset(" in function_source[:host_tail]
    assert "_clip_source_missing_mask(" not in function_source


def test_polygon_mask_device_partition_has_one_capacity_executor() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    function_start = source.index("def _clip_polygon_partition_with_polygon_mask(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    host_tail = function_source.index(
        "nonmissing_rows = _clip_source_nonmissing_rows_for_compatibility"
    )
    device_branch = function_source[:host_tail]
    assert "_clip_homogeneous_polygon_device_candidates_native(" in device_branch
    assert "native_result.capacity_result.geometry" in device_branch
    assert "StrictNativeFallbackError(" in device_branch
    assert "to_host_positions(" not in device_branch
    assert "_exact_polygon_clip_boundary_rows(" not in device_branch
    assert "from_shapely_geometries(" in device_branch


def test_device_polygon_candidates_do_not_enter_host_partition_reconstruction() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    function_start = source.index("def _clip_homogeneous_polygon_device_candidates_native(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    assert "polygon_device_candidate_aligned_exact_gpu" in function_source
    assert "broadcast_right_polygon_intersection_capacity_gpu(" in function_source
    assert "_clip_polygon_partition_with_polygon_mask(" not in function_source
    assert "_geometry_series_from_values(" not in function_source
    assert "to_host_positions(" not in function_source
    exact_tail = function_source.split(
        "# Fused predicate admission is an optimization",
        1,
    )[1]
    assert "NativeDeviceSelection.from_mask(" in exact_tail
    assert "NativeTabularSelection(" in exact_tail
    assert "result_owned.device_take(" not in exact_tail
    assert "_owned_valid_nonempty_device_rows(" not in exact_tail
    assert "materialize_broadcast(" not in function_source


def test_device_polygon_candidate_predicates_keep_indexed_rows_at_capacity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clip_source = (repo_root / "src/vibespatial/api/tools/clip.py").read_text()
    predicate_start = clip_source.index(
        "def _clip_polygon_single_mask_candidate_predicates_device("
    )
    predicate_end = clip_source.index("\ndef ", predicate_start + 1)
    predicate_source = clip_source[predicate_start:predicate_end]

    binary_source = (repo_root / "src/vibespatial/predicates/binary.py").read_text()
    carrier_start = binary_source.index("def _polygonal_single_right_candidate_predicates_device(")
    carrier_end = binary_source.index("\ndef ", carrier_start + 1)
    carrier_source = binary_source[carrier_start:carrier_end]
    refine_start = binary_source.index("def _fused_polygonal_single_right_predicates_device(")
    refine_end = binary_source.index("\ndef ", refine_start + 1)
    refine_source = binary_source[refine_start:refine_end]
    relation_start = binary_source.index("def _binary_predicate_relation_pair_values_device(")
    relation_end = binary_source.index("\ndef ", relation_start + 1)
    relation_source = binary_source[relation_start:relation_end]

    assert "d_candidate_active=d_active & ~d_rectangle_active" in predicate_source
    assert "_polygonal_single_right_candidate_predicates_device(" in predicate_source
    assert "cp.flatnonzero" not in predicate_source
    assert ".device_take" not in predicate_source
    assert "int(mask_tags[0])" not in predicate_source
    assert "bool(mask_validity[0])" not in predicate_source
    assert "d_candidate_output_rows=cp.arange(" in carrier_source
    assert "NativeDeviceSelection.from_mask(" in refine_source
    assert "d_candidate_output_rows" in refine_source
    assert "d_pair_count=" in refine_source
    assert "cp.flatnonzero" not in refine_source
    assert "NativeDeviceSelection.from_mask(" in relation_source
    assert "d_pair_count=d_sub_count" in relation_source
    assert "cp.flatnonzero" not in relation_source


def test_device_point_candidates_keep_dynamic_rows_at_capacity() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    function_start = source.index("def _clip_homogeneous_point_device_candidates_native(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    assert "equal_to_selection(True)" in function_source
    assert "NativeTabularSelection(" in function_source
    assert "_clip_point_owned_with_polygon_mask_device(" not in function_source
    assert "d_hit_source_rows" not in function_source


def test_device_line_rectangle_candidates_keep_dynamic_rows_at_capacity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clip_source = (repo_root / "src/vibespatial/api/tools/clip.py").read_text()
    function_start = clip_source.index(
        "def _clip_homogeneous_line_rectangle_device_candidates_native("
    )
    function_end = clip_source.index("\ndef ", function_start + 1)
    function_source = clip_source[function_start:function_end]

    rect_source = (repo_root / "src/vibespatial/constructive/clip_rect.py").read_text()
    producer_start = rect_source.index("def _clip_line_rows_device_capacity_path(")
    producer_end = rect_source.index("\ndef ", producer_start + 1)
    producer_source = rect_source[producer_start:producer_end]

    assert "owned_result_is_row_capacity" in function_source
    assert "device_state.validity" in function_source
    assert "_owned_valid_nonempty_device_rows(result_owned)" not in function_source
    assert "NativeDeviceSelection.from_mask(" in function_source
    assert "NativeTabularSelection(" in function_source
    assert "cp.flatnonzero(d_has_output" not in producer_source
    assert "count_scatter_totals(" not in producer_source
    assert "row_count=row_count" in producer_source
    line_dispatch_start = rect_source.index("def _clip_all_lines_gpu(")
    line_dispatch_end = rect_source.index("\ndef ", line_dispatch_start + 1)
    line_dispatch_source = rect_source[line_dispatch_start:line_dispatch_end]
    assert "_clip_line_rows_device_capacity_path(" in line_dispatch_source
    assert "_extract_segments_vectorized(" not in rect_source
    assert "_build_line_clip_device_result(" not in rect_source


def test_polygon_rectangle_boundary_split_keeps_dynamic_rows_at_capacity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    kernel_source = (
        repo_root
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "polygon_rect_intersection.py"
    ).read_text()
    split_start = kernel_source.index("def polygon_rect_split_boundary_component_replacements(")
    split_end = kernel_source.index("\ndef ", split_start + 1)
    split_source = kernel_source[split_start:split_end]
    wrapper_start = kernel_source.index("def polygon_rect_split_boundary_components(")
    wrapper_end = kernel_source.index("\ndef ", wrapper_start + 1)
    wrapper_source = kernel_source[wrapper_start:wrapper_end]
    bounds_start = kernel_source.index("def polygon_rect_intersection_from_bounds(")
    bounds_end = kernel_source.index("\ndef ", bounds_start + 1)
    bounds_source = kernel_source[bounds_start:bounds_end]
    contact_start = kernel_source.index("def polygon_rect_boundary_contacts_from_bounds(")
    contact_end = kernel_source.index("\ndef ", contact_start + 1)
    contact_source = kernel_source[contact_start:contact_end]

    clip_source = (repo_root / "src/vibespatial/api/tools/clip.py").read_text()
    clip_start = clip_source.index("def _clip_polygon_rect_cell_mask_intersection_owned(")
    clip_end = clip_source.index("\ndef ", clip_start + 1)
    clip_function = clip_source[clip_start:clip_end]
    constructive_source = (
        repo_root / "src/vibespatial/constructive/binary_constructive.py"
    ).read_text()
    repair_start = constructive_source.index("    def _repair_boundary_split_rows(")
    repair_end = constructive_source.index("\n    def ", repair_start + 1)
    repair_source = constructive_source[repair_start:repair_end]

    assert "d_split_mask = d_component_counts >= 2" in split_source
    assert "component_capacity = row_count * _MAX_BOUNDARY_SPLIT_COMPONENTS" in split_source
    assert "return split_owned, d_split_mask" in split_source
    assert "cp.flatnonzero" not in split_source
    assert "count_scatter_total(" not in split_source
    assert "device_select_owned_capacity_partitions(" in wrapper_source
    assert "_bounded_polygon_rect_vertex_capacity(" in bounds_source
    assert "count_scatter_total(" not in bounds_source
    assert "allocation fence" not in bounds_source
    assert "d_output_mask = d_counts > 0" in contact_source
    assert "multiline_part_capacity = n * segment_capacity" in contact_source
    assert "MAX_CONTACT_LINES" not in kernel_source
    assert "row_count=n" in contact_source
    assert "cp.flatnonzero" not in contact_source
    assert "output_row_count" not in contact_source
    assert "device_select_owned_capacity_partitions(" in clip_function
    assert "cp.flatnonzero" not in clip_function
    assert ".device_take(" not in clip_function
    assert "device_select_owned_capacity_partitions(" in repair_source
    assert "cp.flatnonzero" not in repair_source
    assert ".device_take(" not in repair_source

    capacity_boundary_start = clip_source.index(
        "def _clip_polygon_boundary_intersection_device_capacity_parts("
    )
    capacity_boundary_end = clip_source.index(
        "\ndef ",
        capacity_boundary_start + 1,
    )
    capacity_boundary_source = clip_source[capacity_boundary_start:capacity_boundary_end]
    assert "_explode_lineal_rows_to_line_capacity_gpu" in capacity_boundary_source
    assert "_explode_point_rows_to_point_capacity_gpu" in capacity_boundary_source
    assert "_clip_pack_line_boundary_part_capacity_device" in capacity_boundary_source
    assert "_clip_pack_point_boundary_part_capacity_device" in capacity_boundary_source
    assert "_clip_partition_degenerate_line_part_capacity_device" in capacity_boundary_source
    assert "_clip_concat_point_part_capacities_device" in capacity_boundary_source
    assert "device_mask_owned_capacity(" in capacity_boundary_source
    assert "intersection_native.composition.parts" in capacity_boundary_source
    assert "return tuple((*specialized_parts, *packed_parts))" in capacity_boundary_source
    assert "device_select_owned_capacity_partitions(" not in capacity_boundary_source
    assert "cp.flatnonzero" not in capacity_boundary_source
    assert "physicalize_device_rows" not in capacity_boundary_source
    assert "materialize_broadcast(" not in capacity_boundary_source
    assert "_clip_regroup_boundary_intersections_device_owned" not in clip_source
    assert "_clip_combine_packed_boundary_parts_device" not in clip_source
    assert "_clip_pack_line_boundary_parts_device" not in clip_source
    assert "_clip_pack_point_boundary_parts_device" not in clip_source

    grouped_source = (
        repo_root / "src" / "vibespatial" / "constructive" / "grouped_mixed_union.py"
    ).read_text()
    sorted_start = grouped_source.index("def sorted_part_capacity_plan(")
    sorted_end = grouped_source.index("\ndef ", sorted_start + 1)
    sorted_source = grouped_source[sorted_start:sorted_end]
    assert "_device_indexed_take" in sorted_source
    assert "physicalize_device_rows" not in sorted_source
    line_pack_start = grouped_source.index("def pack_line_part_capacity_device(")
    line_pack_end = grouped_source.index("\ndef ", line_pack_start + 1)
    line_pack_source = grouped_source[line_pack_start:line_pack_end]
    assert "_device_gather_xy_offset_slices" in line_pack_source
    assert "coord_capacity" in line_pack_source
    assert "preserve_indexed_view=True" in line_pack_source

    candidate_start = clip_source.index("def _clip_homogeneous_polygon_device_candidates_native(")
    candidate_end = clip_source.index("\ndef ", candidate_start + 1)
    candidate_source = clip_source[candidate_start:candidate_end]
    assert "_clip_polygon_boundary_intersection_device_capacity_parts(" in candidate_source
    assert "_clip_point_contact_sliver_polygons_device(" in candidate_source
    assert "point_contact_source_row_parts" not in candidate_source
    assert "_clip_geometry_composition_at_capacity(" in candidate_source
    assert "NativeTabularSelection(" in candidate_source
    assert "_clip_exact_terminal_geodataframe_materializer(" not in candidate_source
    assert "inside_selection = NativeDeviceSelection.from_mask(" in candidate_source
    assert "exact_selection = NativeDeviceSelection.from_mask(" in candidate_source
    assert "positive_selection = NativeDeviceSelection.from_mask(" in candidate_source
    assert "device_take_owned_capacity_selection(" in candidate_source
    assert "assume_unique_indices=True" in candidate_source
    assert "_clip_device_mask_to_rows(" not in candidate_source
    assert "_owned_nonempty_polygon_device_rows(" not in candidate_source
    assert "cp.flatnonzero(" not in candidate_source
    assert "cp.unique(" not in candidate_source
    assert "materialize_broadcast(" not in candidate_source

    sliver_start = clip_source.index("def _clip_point_contact_sliver_polygons_device(")
    sliver_end = clip_source.index("\ndef ", sliver_start + 1)
    sliver_source = clip_source[sliver_start:sliver_end]
    assert "device_single_ring_polygon_mask(" in sliver_source
    assert "device_mask_owned_capacity(" in sliver_source
    assert "row_count=row_count" in sliver_source
    assert "cp.flatnonzero" not in sliver_source
    assert "d_out_count" not in sliver_source
    assert ".device_take(" not in sliver_source

    sliver_kernel_start = clip_source.index("_CLIP_POINT_CONTACT_SLIVER_KERNEL_SOURCE =")
    sliver_kernel_end = clip_source.index(
        "request_nvrtc_warmup(",
        sliver_kernel_start,
    )
    sliver_kernel_source = clip_source[sliver_kernel_start:sliver_kernel_end]
    assert "const long long base = i * 4LL" in sliver_kernel_source
    assert "out_valid[i] = 1u" in sliver_kernel_source
    assert "atomicAdd" not in sliver_kernel_source

    native_results_source = (repo_root / "src/vibespatial/api/_native_results.py").read_text()
    capacity_composition_start = native_results_source.index(
        "def _geometry_composition_from_owned_parts_at_capacity("
    )
    capacity_composition_end = native_results_source.index(
        "\ndef ",
        capacity_composition_start + 1,
    )
    capacity_composition_source = native_results_source[
        capacity_composition_start:capacity_composition_end
    ]
    assert "row_count=int(row_count)" in capacity_composition_source
    assert "cp.unique" not in capacity_composition_source
    assert "cp.searchsorted" not in capacity_composition_source


def test_multipolygon_rectangle_keep_type_uses_part_capacity_pack() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clip_source = (repo_root / "src/vibespatial/api/tools/clip.py").read_text()
    function_start = clip_source.index("def _clip_multipolygon_rectangle_keep_geom_type_owned(")
    function_end = clip_source.index("\ndef ", function_start + 1)
    function_source = clip_source[function_start:function_end]
    rectangle_start = clip_source.index("def _clip_polygon_partition_with_rectangle_mask(")
    rectangle_end = clip_source.index("\ndef ", rectangle_start + 1)
    rectangle_source = clip_source[rectangle_start:rectangle_end]

    constructive_source = (
        repo_root / "src/vibespatial/constructive/binary_constructive.py"
    ).read_text()
    pack_start = constructive_source.index(
        "def _pack_disjoint_multipart_intersection_capacity_gpu("
    )
    pack_end = constructive_source.index("\ndef ", pack_start + 1)
    pack_source = constructive_source[pack_start:pack_end]
    assembler_start = constructive_source.index("def _assemble_sorted_polygon_part_capacity_gpu(")
    assembler_end = constructive_source.index("\ndef ", assembler_start + 1)
    assembler_source = constructive_source[assembler_start:assembler_end]

    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in function_source
    assert "_clip_polygon_rectangle_area_intersection_owned(" in function_source
    assert "preserve_polygon_capacity" not in clip_source
    assert "d_valid_rows_mask=d_positive" in function_source
    assert "assume_disjoint=True" in function_source
    assert "_explode_multipolygon_rows_to_polygons_gpu" not in function_source
    assert "_owned_nonempty_polygon_device_rows" not in function_source
    assert "_regroup_intersection_parts_with_grouped_union_gpu" not in function_source
    assert "physicalize_device_rows" not in function_source
    device_branch = rectangle_source.split("    assembled =", 1)[0]
    assert "_clip_homogeneous_polygon_device_candidates_native(" in device_branch
    assert "mask_owned = _device_rectangle_owned_from_bounds(" in device_branch
    assert "clipping_by_rectangle=False" in device_branch
    assert "native_result.capacity_result.geometry" in device_branch
    assert "_clip_polygonal_rectangle_keep_geom_type_device_owned(" in device_branch
    assert "to_host_positions(" not in device_branch
    assert "np.asarray(source_values.bounds" not in device_branch
    assert "NativeGroupedSelection" in pack_source
    assert "NativeDeviceSelection.from_mask" in pack_source
    assert "_assemble_sorted_polygon_part_capacity_gpu" in pack_source
    assert "_build_device_resident_polygon_output" in assembler_source
    assert "d_valid_empty_rows=d_valid_empty_rows" in assembler_source
    assert "preserve_indexed_view=True" in assembler_source
    assert "d_part_family_rows" in assembler_source
    assert "d_sorted_output_ids" in assembler_source
    assert "d_sorted_output_edge_counts=d_sorted_output_edge_counts" in assembler_source
    assert "d_ring_offsets[1:] - d_ring_offsets[:-1]" not in assembler_source
    assert "cp.flatnonzero" not in pack_source
    assert "physicalize_device_rows" not in pack_source


def test_mixed_rectangle_candidates_select_capacity_partitions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clip_source = (repo_root / "src/vibespatial/api/tools/clip.py").read_text()
    function_start = clip_source.index("def _clip_mixed_device_candidates_native(")
    function_end = clip_source.index("\ndef ", function_start + 1)
    function_source = clip_source[function_start:function_end]

    assert "dynamic=True" in function_source
    assert "NativeDeviceSelection.from_mask(" in function_source
    assert "NativeTabularSelection(" in function_source
    assert "owned_result_is_row_capacity" in function_source
    assert "cp.flatnonzero" not in function_source
    assert "_append_rect_clip_rows" not in function_source
    assert "candidate_owned.device_take" not in function_source


def test_clip_semantic_cleanup_keeps_dynamic_rows_at_capacity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/api/_native_results.py").read_text()
    owned_start = source.index("def _clip_owned_geometry_native_result(")
    owned_end = source.index("\ndef ", owned_start + 1)
    owned_source = source[owned_start:owned_end]
    assembly_start = source.index("def _clip_constructive_parts_to_native_tabular_result(")
    assembly_end = source.index("\ndef ", assembly_start + 1)
    assembly_source = source[assembly_start:assembly_end]

    assert "def _valid_nonempty_device_mask(" in owned_source
    assert "def _device_keep_geom_type_cleanup_mask(" in owned_source
    assert "return owned, rowset, repaired_mask, d_keep" in owned_source
    assert "d_keep_rows = cp.flatnonzero(d_keep)" not in owned_source
    assert "d_degenerate_rows = cp.flatnonzero" not in owned_source
    assert "degenerate_line_centroids_owned_capacity(" in owned_source
    assert "device_select_owned_capacity_partitions(" in owned_source
    assert "degenerate_owned = owned.device_take" not in owned_source
    assert "device_concat_owned_scatter(" not in owned_source
    assert "NativeTabularSelection(" in assembly_source
    assert "NativeDeviceSelection.from_mask(" in assembly_source


def test_device_mask_cover_passthrough_keeps_dynamic_rows_at_capacity() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    rowset_start = source.index("def _native_state_valid_nonempty_rowset(")
    rowset_end = source.index("\ndef ", rowset_start + 1)
    rowset_source = source[rowset_start:rowset_end]
    take_start = source.index("def _native_state_passthrough_take(")
    take_end = source.index("\ndef ", take_start + 1)
    take_source = source[take_start:take_end]

    assert "NativeDeviceSelection.from_mask(" in rowset_source
    assert "cp.flatnonzero(d_keep)" not in rowset_source
    assert "isinstance(keep_mask, NativeDeviceSelection)" in take_source
    assert "NativeTabularSelection(" in take_source


def test_clip_polygon_single_mask_all_exact_nonmissing_stays_in_direct_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.api._native_rowset import NativeRowSet

    owned = from_shapely_geometries(
        [
            box(0.5, 0.5, 1.5, 1.5),
            Polygon(),
            box(2.5, 0.5, 3.5, 1.5),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2], "geometry": DeviceGeometryArray._from_owned(owned)},
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
    original_to_host_positions = NativeRowSet.to_host_positions

    def _guard_to_host_positions(
        self,
        *,
        surface="vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed=True,
    ):
        if surface == "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows":
            pytest.fail("all-exact nonmissing rows should stay in the device rowset")
        return original_to_host_positions(
            self,
            surface=surface,
            strict_disallowed=strict_disallowed,
        )

    monkeypatch.setattr(NativeRowSet, "to_host_positions", _guard_to_host_positions)
    vibespatial.clear_dispatch_events()

    result = clip_module._clip_polygon_partition_with_polygon_mask(gdf, mask)
    events = vibespatial.get_dispatch_events(clear=True)

    assert isinstance(result, GeometryNativeResult)
    assert result.row_count == 3
    assert result.composition is not None
    assert any(
        event.operation == "intersection"
        and event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_clip_sparse_owned_output_reuses_positive_device_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import OwnedGeometryArray

    left_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i + 1), 1.0) for i in range(4)],
        residency=Residency.DEVICE,
    )
    exact_area_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            Polygon(),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    positive_local_rows = np.asarray([0, 2], dtype=np.intp)
    positive_rows = np.asarray([1, 3], dtype=np.intp)
    d_positive_local_rows = cp.asarray(positive_local_rows, dtype=cp.int64)
    d_positive_rows = cp.asarray(positive_rows, dtype=cp.int64)
    calls: list[dict[str, object]] = []
    original_device_take = OwnedGeometryArray.device_take

    def _record_device_take(
        self,
        indices,
        *,
        host_indices_for_sizing=None,
        **device_take_kwargs,
    ):
        if self is exact_area_owned:
            calls.append(
                {
                    "indices": cp.asnumpy(indices).tolist()
                    if hasattr(indices, "__cuda_array_interface__")
                    else np.asarray(indices).tolist(),
                    "reused_positive_rows": indices is d_positive_local_rows,
                    "sizing": None
                    if host_indices_for_sizing is None
                    else np.asarray(host_indices_for_sizing).tolist(),
                }
            )
        return original_device_take(
            self,
            indices,
            host_indices_for_sizing=host_indices_for_sizing,
            **device_take_kwargs,
        )

    monkeypatch.setattr(OwnedGeometryArray, "device_take", _record_device_take)

    result = clip_module._build_sparse_owned_clip_output(
        partition_crs="EPSG:3857",
        left_owned=left_owned,
        inside_rows=np.empty(0, dtype=np.intp),
        exact_area_owned=exact_area_owned,
        positive_local_rows=positive_local_rows,
        positive_local_rows_device=d_positive_local_rows,
        positive_rows=positive_rows,
        positive_rows_device=d_positive_rows,
    )

    assert result.local_rows.tolist() == [1, 3]
    assert hasattr(result.local_rows_device, "__cuda_array_interface__")
    assert cp.asnumpy(result.local_rows_device).tolist() == [1, 3]
    assert any(
        call
        == {
            "indices": [0, 2],
            "reused_positive_rows": True,
            "sizing": [0, 2],
        }
        for call in calls
    )


def test_clip_sparse_owned_output_reorder_stays_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import OwnedGeometryArray

    left_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i + 1), 1.0) for i in range(4)],
        residency=Residency.DEVICE,
    )
    exact_area_owned = from_shapely_geometries(
        [box(1.0, 0.0, 2.0, 1.0)],
        residency=Residency.DEVICE,
    )
    d_inside_rows = cp.asarray([3], dtype=cp.int64)
    d_positive_local_rows = cp.asarray([0], dtype=cp.int64)
    d_positive_rows = cp.asarray([1], dtype=cp.int64)
    calls: list[dict[str, object]] = []
    original_device_take = OwnedGeometryArray.device_take

    def _record_device_take(
        self,
        indices,
        *,
        host_indices_for_sizing=None,
        **device_take_kwargs,
    ):
        calls.append(
            {
                "row_count": self.row_count,
                "indices": cp.asnumpy(indices).tolist()
                if hasattr(indices, "__cuda_array_interface__")
                else np.asarray(indices).tolist(),
                "device": hasattr(indices, "__cuda_array_interface__"),
                "sizing": None
                if host_indices_for_sizing is None
                else np.asarray(host_indices_for_sizing).tolist(),
            }
        )
        return original_device_take(
            self,
            indices,
            host_indices_for_sizing=host_indices_for_sizing,
            **device_take_kwargs,
        )

    monkeypatch.setattr(OwnedGeometryArray, "device_take", _record_device_take)

    result = clip_module._build_sparse_owned_clip_output(
        partition_crs="EPSG:3857",
        left_owned=left_owned,
        inside_rows=np.asarray([3], dtype=np.intp),
        inside_rows_device=d_inside_rows,
        exact_area_owned=exact_area_owned,
        positive_local_rows=np.asarray([0], dtype=np.intp),
        positive_local_rows_device=d_positive_local_rows,
        positive_rows=np.asarray([1], dtype=np.intp),
        positive_rows_device=d_positive_rows,
    )

    assert result.local_rows.tolist() == [1, 3]
    assert hasattr(result.local_rows_device, "__cuda_array_interface__")
    assert cp.asnumpy(result.local_rows_device).tolist() == [1, 3]
    assert any(
        call
        == {
            "row_count": 2,
            "indices": [1, 0],
            "device": True,
            "sizing": [1, 0],
        }
        for call in calls
    )


def test_clip_sparse_owned_output_extra_parts_keep_device_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    left_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i + 1), 1.0) for i in range(4)],
        residency=Residency.DEVICE,
    )
    exact_area_owned = from_shapely_geometries(
        [box(1.0, 0.0, 2.0, 1.0)],
        residency=Residency.DEVICE,
    )
    boundary_owned = from_shapely_geometries(
        [LineString([(3.0, 0.0), (3.0, 1.0)])],
        residency=Residency.DEVICE,
    )
    d_positive_local_rows = cp.asarray([0], dtype=cp.int64)
    d_positive_rows = cp.asarray([1], dtype=cp.int64)
    d_boundary_rows = cp.asarray([3], dtype=cp.int64)

    result = clip_module._build_sparse_owned_clip_output(
        partition_crs="EPSG:3857",
        left_owned=left_owned,
        inside_rows=np.empty(0, dtype=np.intp),
        exact_area_owned=exact_area_owned,
        positive_local_rows=np.asarray([0], dtype=np.intp),
        positive_local_rows_device=d_positive_local_rows,
        positive_rows=np.asarray([1], dtype=np.intp),
        positive_rows_device=d_positive_rows,
        extra_owned_parts=((np.asarray([3], dtype=np.intp), boundary_owned, d_boundary_rows),),
    )

    assert result.local_rows.tolist() == [1, 3]
    assert hasattr(result.local_rows_device, "__cuda_array_interface__")
    assert cp.asnumpy(result.local_rows_device).tolist() == [1, 3]


def test_clip_partition_output_threads_device_local_rows_to_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i + 1), 1.0) for i in range(4)],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {"geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")},
        crs="EPSG:3857",
    )
    result_owned = owned.device_take(
        cp.asarray([1, 3], dtype=cp.int64),
        host_indices_for_sizing=np.asarray([1, 3], dtype=np.int64),
    )
    d_local_rows = cp.asarray([1, 3], dtype=cp.int64)
    d_row_positions = cp.asarray([0, 1, 2, 3], dtype=cp.int64)
    output = clip_module._ClipPartitionOutput(
        geometry_values=DeviceGeometryArray._from_owned(result_owned, crs=source.crs),
        local_rows=np.asarray([1, 3], dtype=np.intp),
        local_rows_device=d_local_rows,
    )
    seen: dict[str, object] = {}

    def _record_source_rowset(
        source_arg,
        row_positions,
        local_rows=None,
        *,
        device_row_positions=None,
        device_local_rows=None,
    ):
        seen["row_positions"] = np.asarray(row_positions).tolist()
        seen["local_rows"] = None if local_rows is None else np.asarray(local_rows).tolist()
        seen["device_row_positions"] = device_row_positions
        seen["device_local_rows"] = device_local_rows
        return None

    monkeypatch.setattr(
        clip_module,
        "_clip_source_rowset_for_positions",
        _record_source_rowset,
    )

    part = clip_module._build_clip_partition_result(
        source,
        np.asarray([0, 1, 2, 3], dtype=np.intp),
        output,
        row_positions_device=d_row_positions,
    )

    assert part.row_positions.tolist() == [1, 3]
    assert seen["row_positions"] == [0, 1, 2, 3]
    assert seen["local_rows"] == [1, 3]
    assert seen["device_row_positions"] is d_row_positions
    assert seen["device_local_rows"] is d_local_rows


def test_clip_source_rowset_identity_local_rows_skip_host_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import NativeFrameState, attach_native_state

    owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i + 1), 1.0) for i in range(3)],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(native_payload)
    attach_native_state(source, state)

    d_row_positions = cp.asarray([2, 0], dtype=cp.int64)
    host_uploads: list[list[int]] = []
    original_asarray = cp.asarray

    def _record_asarray(values, *args, **kwargs):
        if isinstance(values, np.ndarray):
            host_uploads.append(np.asarray(values).tolist())
        return original_asarray(values, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", _record_asarray)

    rowset = clip_module._clip_source_rowset_for_positions(
        source,
        np.asarray([2, 0], dtype=np.intp),
        local_rows=np.asarray([0, 1], dtype=np.intp),
        device_row_positions=d_row_positions,
    )

    assert rowset is not None
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [2, 0]
    assert host_uploads == []


def test_clip_point_polygon_partition_preserves_device_hit_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [Point(0.25, 0.25), Point(2.0, 2.0), Point(0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {"geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")},
        crs="EPSG:3857",
    )

    result = clip_module._clip_point_partition_with_polygon_mask(
        partition,
        box(0.0, 0.0, 1.0, 1.0),
    )

    assert isinstance(result.geometry_values, DeviceGeometryArray)
    assert len(result.geometry_values) == 2
    assert result.local_rows.tolist() == [0, 2]
    assert hasattr(result.local_rows_device, "__cuda_array_interface__")
    assert cp.asnumpy(result.local_rows_device).tolist() == [0, 2]


def test_clip_homogeneous_point_candidates_native_uses_device_rowset() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import NativeFrameState, attach_native_state

    owned = from_shapely_geometries(
        [Point(0.25, 0.25), Point(2.0, 2.0), Point(0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(native_payload)
    attach_native_state(source, state)
    d_candidate_rows = cp.asarray([2, 1, 0], dtype=cp.int64)

    result = clip_module._clip_homogeneous_point_candidates_native(
        source,
        box(0.0, 0.0, 1.0, 1.0),
        np.asarray([2, 1, 0], dtype=np.intp),
        candidate_device_rows=d_candidate_rows,
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert result is not None
    assert isinstance(result, NativeTabularSelection)
    assert result.capacity_result.attributes.device_table is not None
    assert result.capacity_result.index_plan is not None
    assert result.capacity_result.index_plan.kind == "device-labels"
    assert result.provenance is not None
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [2, 1, 0]
    assert cp.asnumpy(result.selection.positions[:2]).tolist() == [0, 2]


def test_clip_homogeneous_point_candidates_native_skips_hit_row_host_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import NativeFrameState, attach_native_state
    from vibespatial.geometry.owned import OwnedGeometryArray

    owned = from_shapely_geometries(
        [Point(0.25, 0.25), Point(2.0, 2.0), Point(0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(native_payload)
    attach_native_state(source, state)
    original_to_host_positions = NativeRowSet.to_host_positions
    original_device_take = OwnedGeometryArray.device_take
    device_take_calls: list[dict[str, object]] = []

    def _guard_to_host_positions(
        self,
        *,
        surface="vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed=True,
    ):
        if surface == "vibespatial.api.tools.clip.point_polygon_mask_rows":
            pytest.fail("point polygon hit rows should stay as a device rowset")
        return original_to_host_positions(
            self,
            surface=surface,
            strict_disallowed=strict_disallowed,
        )

    def _record_device_take(
        self,
        indices,
        *,
        host_indices_for_sizing=None,
        **device_take_kwargs,
    ):
        device_take_calls.append(
            {
                "device": hasattr(indices, "__cuda_array_interface__"),
                "sizing": host_indices_for_sizing,
                "row_count": self.row_count,
            }
        )
        return original_device_take(
            self,
            indices,
            host_indices_for_sizing=host_indices_for_sizing,
            **device_take_kwargs,
        )

    monkeypatch.setattr(NativeRowSet, "to_host_positions", _guard_to_host_positions)
    monkeypatch.setattr(OwnedGeometryArray, "device_take", _record_device_take)

    result = clip_module._clip_homogeneous_point_candidates_native(
        source,
        box(0.0, 0.0, 1.0, 1.0),
        np.asarray([0, 1, 2], dtype=np.intp),
        candidate_device_rows=cp.asarray([0, 1, 2], dtype=cp.int64),
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert result is not None
    assert isinstance(result, NativeTabularSelection)
    assert result.capacity_result.attributes.device_table is not None
    assert result.provenance is not None
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [0, 1, 2]
    assert cp.asnumpy(result.selection.positions[:2]).tolist() == [0, 2]
    assert device_take_calls == []


def test_clip_homogeneous_point_device_candidates_skip_candidate_rows_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [Point(0.25, 0.25), Point(2.0, 2.0), Point(0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = Polygon(
        [
            (0.0, 0.0),
            (1.5, 0.0),
            (1.5, 0.5),
            (0.5, 0.5),
            (0.5, 1.5),
            (0.0, 1.5),
            (0.0, 0.0),
        ]
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert isinstance(native_result, NativeTabularSelection)
    assert native_result.capacity_result.attributes.device_table is not None
    assert native_result.capacity_result.index_plan is not None
    assert native_result.capacity_result.index_plan.kind == "device-labels"
    assert native_result.provenance is not None
    assert native_result.provenance.is_device
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.point_polygon_mask_rows::rowset_to_host" not in runtime_reasons
    )
    assert get_materialization_events(clear=True) == []

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10]


def test_clip_homogeneous_point_rectangle_device_candidates_skip_candidate_rows_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [Point(0.25, 0.25), Point(2.0, 2.0), Point(0.75, 0.75)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        (0.0, 0.0, 1.0, 1.0),
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    assert native_result.provenance is not None
    assert native_result.provenance.is_device
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.point_polygon_mask_rows::rowset_to_host" not in runtime_reasons
    )
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert not any("candidate" in surface for surface in materialization_surfaces)
    assert "vibespatial.api.tools.clip.point_polygon_mask_rows" not in materialization_surfaces

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 30]


def test_clip_homogeneous_line_rectangle_device_candidates_skip_candidate_rows_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            LineString([(-1.0, 0.5), (2.0, 0.5)]),
            LineString([(2.0, 2.0), (3.0, 3.0)]),
            LineString([(0.2, 0.2), (0.8, 0.8)]),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        (0.0, 0.0, 1.0, 1.0),
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    if capacity_result.geometry.owned is not None:
        assert capacity_result.geometry.owned.residency is Residency.DEVICE
    else:
        assert capacity_result.geometry.composition is not None
        assert all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in capacity_result.geometry.composition.parts
        )
    assert native_result.provenance is not None
    assert native_result.provenance.is_device
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip-rect combined row-map host export" not in runtime_reasons
    assert "clip-rect line row-map host export" not in runtime_reasons
    assert get_materialization_events(clear=True) == []

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 30]
    assert exported.geom_type.isin(LINE_GEOM_TYPES).all()


def test_clip_all_point_polygon_mask_uses_direct_native_point_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [
                Point(0.25, 0.25),
                Point(2.0, 2.0),
                Point(0.75, 0.75),
            ],
        },
        crs="EPSG:3857",
    )
    mask = Polygon(
        [
            (0.0, 0.0),
            (1.5, 0.0),
            (1.5, 0.5),
            (0.5, 0.5),
            (0.5, 1.5),
            (0.0, 1.5),
            (0.0, 0.0),
        ]
    )
    monkeypatch.setattr(
        clip_module,
        "_clip_point_partition_with_polygon_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "all-point polygon clip should not build a candidate frame partition"
        ),
    )

    result = clip(gdf, mask, sort=False)

    assert result["value"].tolist() == [10]
    assert isinstance(result.geometry.values, DeviceGeometryArray)


def test_clip_mixed_candidate_partitions_preserve_device_candidate_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import NativeFrameState, attach_native_state
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    geometries = [
        Point(0.25, 0.25),
        LineString([(0.25, 0.25), (1.75, 1.75)]),
        box(1.0, 0.1, 1.4, 0.4),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(native_payload)
    attach_native_state(source, state)
    mask = Polygon(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 0.75),
            (0.75, 0.75),
            (0.75, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = clip_module.evaluate_geopandas_clip_native(source, mask)

    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert isinstance(result, NativeTabularSelection)
    capacity_result = result.capacity_result
    assert capacity_result.geometry.residency is Residency.DEVICE
    assert capacity_result.geometry.row_count == 3
    assert capacity_result.attributes.device_table is not None
    assert hasattr(result.provenance.source_rows, "__cuda_array_interface__")
    assert "clip mixed-shape candidate partition rows host boundary" not in runtime_reasons


def test_clip_polygon_mask_all_exact_rows_skip_exact_rowset_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2]},
        geometry=vibespatial.GeoSeries(
            [
                box(-0.5, 0.2, 0.5, 0.8),
                box(2.5, 0.2, 3.5, 0.8),
                box(0.2, 2.5, 0.8, 3.5),
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert set(result["row"]) == {0, 1, 2}
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces


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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert set(result["row"]) == {0, 1, 2}
    assert "binary predicate covered-by single-mask result host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows"
        not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows" not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows"
        not in materialization_surfaces
    )


def test_clip_polygon_single_mask_keeps_all_candidate_classes_in_direct_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.api._native_rowset import NativeRowSet

    owned = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.8, 0.8),
            box(0.5, 0.5, 1.5, 1.5),
            box(2.0, 2.0, 2.5, 2.5),
        ],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2], "geometry": DeviceGeometryArray._from_owned(owned)},
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
    original_to_host_positions = NativeRowSet.to_host_positions

    def _record_to_host_positions(
        self,
        *,
        surface="vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed=True,
    ):
        if surface in {
            "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows",
            "vibespatial.api.tools.clip.polygon_single_mask_inside_rows",
        }:
            pytest.fail("direct polygon-mask topology must not export partition rowsets")
        return original_to_host_positions(
            self,
            surface=surface,
            strict_disallowed=strict_disallowed,
        )

    monkeypatch.setattr(NativeRowSet, "to_host_positions", _record_to_host_positions)
    vibespatial.clear_dispatch_events()

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)
    events = vibespatial.get_dispatch_events(clear=True)

    assert isinstance(result, GeometryNativeResult)
    assert result.row_count == 3
    assert result.composition is not None
    assert any(
        event.operation == "intersection"
        and event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_clip_polygon_single_mask_empty_inside_rowset_stays_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1, 2]},
        geometry=vibespatial.GeoSeries(
            [
                box(0.5, 0.5, 1.5, 1.5),
                box(2.5, 0.5, 3.5, 1.5),
                box(0.5, 2.5, 1.5, 3.5),
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert set(result["row"]) == {0, 1, 2}
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows" not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows"
        not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows"
        not in materialization_surfaces
    )


def test_clip_polygon_single_mask_no_hit_returns_sparse_without_empty_exports() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gdf = vibespatial.GeoDataFrame(
        {"row": [0, 1]},
        geometry=vibespatial.GeoSeries(
            [
                box(2.0, 2.0, 2.4, 2.4),
                box(2.6, 2.6, 3.0, 3.0),
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
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert result.empty
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows::rowset_to_host"
        not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_inside_rows" not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_remaining_rows"
        not in materialization_surfaces
    )
    assert (
        "vibespatial.api.tools.clip.polygon_single_mask_exact_local_rows"
        not in materialization_surfaces
    )


def test_clip_polygon_single_mask_device_partition_skips_compatibility_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    monkeypatch.setattr(
        clip_module,
        "_clip_source_nonmissing_rowset",
        lambda *_args, **_kwargs: pytest.fail(
            "device polygon partitions must not enter the compatibility rowset"
        ),
    )
    partition_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    partition = vibespatial.GeoDataFrame(
        {
            "geometry": vibespatial.GeoSeries(
                DeviceGeometryArray._from_owned(partition_owned, crs="EPSG:3857")
            )
        },
        crs="EPSG:3857",
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(
        partition,
        box(-1.0, -1.0, 3.0, 3.0),
    )

    assert isinstance(result, GeometryNativeResult)
    assert result.row_count == 2


def test_clip_declines_host_line_make_valid_cleanup(
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
    _assert_native_geometry_device_resident(result.geometry.values)
    assert shapely.equals(result.geometry.iloc[0], Point(0.0, 0.0))
    assert not any(
        event.surface == "geopandas.clip" and event.pipeline == "clip.to_spatial"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_polygon_rectangle_mask_donut_matches_exact_intersection() -> None:
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
    expected = shapely.intersection(
        multi_poly.geometry.iloc[0],
        mask.geometry.iloc[0],
    )

    assert result.geom_type.iloc[0] == expected.geom_type
    assert shapely.equals(result.geometry.iloc[0], expected)


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
        if (
            dtype is bool
            and getattr(arr, "ndim", 0) == 1
            and getattr(arr, "size", -1) == len(source)
        ):
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
    _assert_native_geometry_device_resident(result.geometry.values)
    assert any(
        event.operation == "intersection"
        and event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )

    expected = clip(
        vibespatial.GeoDataFrame(
            {
                "geometry": [
                    box(2.0, 2.0, 4.0, 4.0),
                    box(4.0, 4.0, 6.0, 6.0),
                    box(6.0, 6.0, 8.0, 8.0),
                    box(8.0, 8.0, 10.0, 10.0),
                ]
            },
            crs="EPSG:3857",
        ),
        mask,
    )
    assert set(result.geometry.to_wkt().tolist()) == set(expected.geometry.to_wkt().tolist())


def test_clip_declines_host_semantic_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for strict-native cleanup promotion")
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
    _assert_native_geometry_device_resident(result.geometry.values)
    assert shapely.equals(result.geometry.iloc[0], box(0.5, 0.5, 1.5, 1.5))
    assert not any(
        event.surface == "geopandas.clip" and event.pipeline == "clip.to_spatial"
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


def test_owned_rectangle_batch_rejects_degenerate_orthogonal_ring_without_shapely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = GeometryArray.from_owned(
        from_shapely_geometries(
            [Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 0.0), (0.0, 0.0)])],
            residency=Residency.HOST,
        )
    )

    monkeypatch.setattr(
        clip_module,
        "_is_axis_aligned_rectangle_polygon",
        lambda *_args, **_kwargs: pytest.fail(
            "authoritative owned structure should reject the degenerate ring"
        ),
    )

    assert clip_module._all_axis_aligned_rectangle_polygons(values) is False


def test_clip_polygon_mask_boundary_filter_stays_on_gpu_for_device_backing() -> None:
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
    assert not any(event.selected.value == "cpu" for event in events)
    assert any(
        event.surface == "geopandas.clip"
        and event.implementation == "polygon_device_candidate_direct_rowset_assembly_gpu"
        and event.selected.value == "gpu"
        for event in events
    )


def test_clip_polygon_rect_cell_mask_split_carrier_preserves_sparse_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

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
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    monkeypatch.setattr(
        validity_module,
        "validity_expression_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle-cell mask split should trust the boundary-split carrier "
            "instead of revalidating every output row"
        ),
    )

    vibespatial.clear_dispatch_events()
    vibespatial.clear_fallback_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result_owned = clip_module._clip_polygon_rect_cell_mask_intersection_owned(
        left_owned,
        mask_owned,
        None,
    )
    events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert result_owned is not None

    measurement_module = importlib.import_module("vibespatial.constructive.measurement")
    monkeypatch.setattr(
        measurement_module,
        "_area_gpu_device_fp64",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle-cell split positive rows should use validity metadata "
            "instead of recomputing fp64 area"
        ),
    )
    positive_rows = clip_module._owned_nonempty_polygon_device_rows(
        result_owned,
        keep_pointlike_zero_area=True,
    )

    assert cp.asnumpy(positive_rows).tolist() == [0, 1]
    actual = np.asarray(result_owned.to_shapely(), dtype=object)
    expected = np.asarray(
        shapely.intersection(cells, np.full(cells.size, mask, dtype=object)),
        dtype=object,
    )

    assert [geom.geom_type for geom in actual] == [geom.geom_type for geom in expected]
    assert all(shapely.is_valid(actual))
    assert all(
        shapely.equals(shapely.normalize(actual_geom), shapely.normalize(expected_geom))
        for actual_geom, expected_geom in zip(actual, expected, strict=True)
    )
    assert any(
        event.surface == "geopandas.clip"
        and event.operation == "clip"
        and event.implementation == "polygon_rect_cell_mask_split_gpu"
        and event.selected.value == "gpu"
        for event in events
    )
    assert any(
        event.surface == "geopandas.clip"
        and event.operation == "clip"
        and event.implementation == "polygon_rect_cell_mask_split_gpu"
        and "row_indirected_polygon_mask_device_rectangle_bounds" in event.detail
        for event in events
    )
    assert "polygon-rectangle dense single-ring scalar fence" not in runtime_reasons
    assert "polygon-rectangle empty-mask scalar fence" not in runtime_reasons
    assert "polygon-rectangle max-input-vertices scalar fence" not in runtime_reasons
    assert "polygon-rectangle intersection vertex allocation fence" not in runtime_reasons
    assert vibespatial.get_fallback_events(clear=True) == []


def test_polygon_rect_boundary_split_consumes_indexed_rows_without_take_fence() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
        polygon_rect_split_boundary_components,
    )

    mask = _benchmark_admin_star_mask()
    cells = [
        box(470.0, 410.0, 480.0, 420.0),
        box(620.0, 620.0, 630.0, 630.0),
    ]
    mask_owned = from_shapely_geometries([mask], residency=Residency.DEVICE)
    rect_owned = from_shapely_geometries(cells, residency=Residency.DEVICE)
    subject = materialize_broadcast(tile_single_row(mask_owned, len(cells)))
    clipped = polygon_rect_intersection(subject, rect_owned)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    repaired = polygon_rect_split_boundary_components(clipped, rect_owned)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert repaired is not None
    assert repaired.row_count == len(cells)
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    actual = np.asarray(repaired.to_shapely(), dtype=object)
    expected = shapely.intersection(
        np.asarray([mask] * len(cells), dtype=object),
        np.asarray(cells, dtype=object),
    )
    assert actual[0].geom_type == "MultiPolygon"
    for got, want in zip(actual, expected, strict=True):
        if shapely.is_empty(want):
            assert got is None or shapely.is_empty(got)
            continue
        assert shapely.equals(shapely.normalize(got), shapely.normalize(want))


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

    vibespatial.clear_fallback_events()
    result = clip(gdf, mask)
    fallback_reasons = [event.reason for event in vibespatial.get_fallback_events(clear=True)]
    result_by_id = {
        int(row.building_id): shapely.to_wkt(shapely.normalize(row.geometry), rounding_precision=6)
        for row in result.itertuples(index=False)
    }

    assert result_by_id == expected_by_id
    assert (
        "clip polygon point-contact boundary rows require exact compatibility materialization"
        not in fallback_reasons
    )


def test_clip_polygon_device_candidates_non_rectangle_mask_do_not_use_bbox_inside_shortcut() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device candidate polygon clip")
    cp = pytest.importorskip("cupy")

    mask = _benchmark_admin_star_mask()
    source_geom = box(470.0, 420.0, 480.0, 430.0)
    expected = shapely.intersection(source_geom, mask)
    owned = from_shapely_geometries([source_geom], residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "parcel_id": [4247],
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:4326",
            ),
        },
        crs="EPSG:4326",
    )

    vibespatial.clear_fallback_events()
    native_result = clip_module._clip_homogeneous_polygon_device_candidates_native(
        gdf,
        mask,
        cp.asarray([0], dtype=cp.int64),
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert native_result is not None
    result = native_result.to_geodataframe()
    actual = result.geometry.iloc[0]

    assert result["parcel_id"].tolist() == [4247]
    assert actual.geom_type == expected.geom_type
    assert shapely.area(actual) < shapely.area(source_geom)
    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected))
    assert vibespatial.get_fallback_events(clear=True) == []


def test_clip_polygon_device_rectangle_candidates_use_bounds_predicate_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device candidate polygon clip")
    cp = pytest.importorskip("cupy")

    import vibespatial.predicates.polygon as polygon_predicates

    mask = Polygon(
        [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
            (0.0, 0.0),
        ],
    )
    geometries = [
        box(0.2, 0.2, 0.8, 0.8),
        box(1.5, 1.5, 2.0, 2.0),
        box(0.8, 0.8, 1.2, 1.2),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "parcel_id": [1, 2, 3],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )

    bounds_predicate_capacities: list[int] = []
    original_bounds_predicates = polygon_predicates.compute_rect_bounds_polygon_mask_predicates_gpu

    def _record_bounds_predicate_capacity(mask_owned, rect_bounds, **kwargs):
        bounds_predicate_capacities.append(int(rect_bounds.shape[0]))
        return original_bounds_predicates(mask_owned, rect_bounds, **kwargs)

    monkeypatch.setattr(
        polygon_predicates,
        "compute_rect_bounds_polygon_mask_predicates_gpu",
        _record_bounds_predicate_capacity,
    )

    native_result = clip_module._clip_homogeneous_polygon_device_candidates_native(
        gdf,
        mask,
        cp.asarray([0, 1, 2], dtype=cp.int64),
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert native_result is not None
    result = native_result.to_geodataframe()
    expected = {
        parcel_id: shapely.intersection(geometry, mask)
        for parcel_id, geometry in zip([1, 2, 3], geometries, strict=True)
        if not shapely.is_empty(shapely.intersection(geometry, mask))
    }

    assert result["parcel_id"].tolist() == [1, 3]
    assert bounds_predicate_capacities == [len(geometries)]
    assert set(result["parcel_id"]) == set(expected)
    for row in result.itertuples():
        want = expected[row.parcel_id]
        assert shapely.equals(shapely.normalize(row.geometry), shapely.normalize(want))


def test_clip_polygon_device_candidates_fuse_predicate_masks_before_rowsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device candidate polygon clip")
    cp = pytest.importorskip("cupy")

    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    mask = box(0.0, 0.0, 2.0, 2.0)
    geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(1.5, 0.1, 2.5, 0.9),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "parcel_id": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    rowset_combine_calls: list[str] = []

    def _record_rowset_combine(self, other, operation):
        rowset_combine_calls.append(operation)
        raise AssertionError("predicate masks should be fused before rowset set ops")

    monkeypatch.setattr(NativeRowSet, "_combine", _record_rowset_combine)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    native_result = clip_module._clip_homogeneous_polygon_device_candidates_native(
        gdf,
        mask,
        cp.asarray([0, 1], dtype=cp.int64),
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert native_result is not None
    result = native_result.to_geodataframe()
    expected = shapely.intersection(
        np.asarray(geometries, dtype=object),
        np.asarray([mask, mask], dtype=object),
    )

    assert rowset_combine_calls == []
    assert "clip rowset count scalar fence" not in runtime_reasons
    assert result["parcel_id"].tolist() == [1, 2]
    assert all(
        shapely.equals(shapely.normalize(actual), shapely.normalize(want))
        for actual, want in zip(result.geometry, expected, strict=True)
    )


def test_clip_polygon_mask_avoids_correction_boundary_materialization(
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

    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    vibespatial.clear_fallback_events()
    vibespatial.clear_dispatch_events()
    clear_materialization_events()
    result = clip(gdf, mask)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    events = get_materialization_events(clear=True)
    fallback_reasons = [event.reason for event in vibespatial.get_fallback_events(clear=True)]

    assert len(result) == 3
    assert materialized_rows == []
    assert (
        "clip polygon point-contact boundary rows require exact compatibility materialization"
        not in fallback_reasons
    )
    assert any(
        event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        for event in dispatch_events
    )
    assert not any("point-contact" in event.reason for event in events)
    assert not any(event.operation == "clip_selected_geometry_rows_to_shapely" for event in events)


def test_clip_polygon_mask_crossings_do_not_become_point_remnants() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for polygon boundary crossing canary")

    mask = _benchmark_admin_star_mask()
    source_geometries = [
        box(400.0, 320.0, 410.0, 330.0),
        box(410.0, 330.0, 420.0, 340.0),
        box(430.0, 360.0, 440.0, 370.0),
    ]
    source = vibespatial.GeoDataFrame(
        {
            "value": np.arange(len(source_geometries), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    source_geometries,
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:4326",
            ),
        },
        crs="EPSG:4326",
    )

    result = clip(source, mask, keep_geom_type=False)
    expected = shapely.intersection(
        np.asarray(source_geometries, dtype=object),
        np.full(len(source_geometries), mask, dtype=object),
    )
    actual = np.asarray(result.geometry.values._data, dtype=object)

    assert result.geom_type.tolist() == ["Polygon", "Polygon", "Polygon"]
    assert all(shapely.equals(got, want) for got, want in zip(actual, expected, strict=True))


def test_clip_polygon_mask_collinear_edge_uses_native_line_contact() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for polygon boundary line contact")

    mask = Polygon([(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)])
    source_geometry = box(0.5, -1.0, 1.5, 0.0)
    source = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [source_geometry],
                    residency=Residency.DEVICE,
                )
            ),
        }
    )

    vibespatial.clear_dispatch_events()
    result = clip(source, mask, keep_geom_type=False)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == "LineString"
    assert shapely.equals(
        result.geometry.iloc[0],
        shapely.intersection(source_geometry, mask),
    )
    assert any(
        event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        for event in events
    )


def test_clip_polygon_mask_keeps_disjoint_collinear_contacts_separate() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for polygon boundary line contacts")

    mask = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (3.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]
    )
    source_geometry = box(0.0, -1.0, 3.0, 0.0)
    source = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [source_geometry],
                    residency=Residency.DEVICE,
                )
            ),
        }
    )

    vibespatial.clear_dispatch_events()
    result = clip(source, mask, keep_geom_type=False)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == "MultiLineString"
    assert shapely.equals(
        result.geometry.iloc[0],
        shapely.intersection(source_geometry, mask),
    )
    assert any(
        event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        for event in events
    )


def test_clip_point_contact_sliver_probe_keeps_vertex_presence_on_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for point-contact sliver probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    candidate_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    mask_owned = from_shapely_geometries(
        [box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE,
    )
    boundary_owned = from_shapely_geometries(
        [Point(0.5, 0.0)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = clip_module._clip_point_contact_sliver_polygons_device(
        candidate_owned,
        mask_owned,
        boundary_owned,
        cp.asarray([0], dtype=cp.int64),
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    result_owned, d_result_mask = result
    assert result_owned.row_count == boundary_owned.row_count
    assert not bool(d_result_mask[0])
    assert "clip point-contact sliver vertex-presence scalar fence" not in runtime_reasons


def test_clip_rectangle_lower_dimensional_rows_stay_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-backed clip materialization canary")
    cp = pytest.importorskip("cupy")

    values = [
        MultiPolygon(
            [
                box(0.1, 0.1, 0.3, 0.3),
                box(0.6, 0.6, 0.8, 0.8),
            ],
        ),
        box(1.0, 0.2, 1.4, 0.8),
        box(1.0, 1.0, 1.4, 1.4),
    ]
    mask = box(0.0, 0.0, 1.0, 1.0)
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )

    import vibespatial.constructive.binary_constructive as binary_constructive_module

    original_binary_constructive = binary_constructive_module.binary_constructive_owned

    def _reject_redundant_rectangle_boundary_difference(op, *args, **kwargs):
        if op == "difference":
            raise AssertionError(
                "rectangle-pair boundary remnants must not reopen generic difference topology"
            )
        return original_binary_constructive(op, *args, **kwargs)

    monkeypatch.setattr(
        binary_constructive_module,
        "binary_constructive_owned",
        _reject_redundant_rectangle_boundary_difference,
    )

    native_result = clip_module._clip_homogeneous_polygon_device_candidates_native(
        gdf,
        mask,
        cp.asarray([0, 1, 2], dtype=cp.int64),
        clipping_by_rectangle=False,
        rectangle_bounds=mask.bounds,
        keep_geom_type=False,
    )
    assert native_result is not None
    assert native_result.terminal_geodataframe_materializer_owns_export

    materialized_rows: list[int] = []
    original_array = DeviceGeometryArray.__array__

    def _record_array(self, dtype=None, copy=None):
        materialized_rows.append(len(self))
        return original_array(self, dtype=dtype, copy=copy)

    monkeypatch.setattr(DeviceGeometryArray, "__array__", _record_array)

    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    clear_materialization_events()
    result = native_result.to_geodataframe()
    events = get_materialization_events(clear=True)

    expected = shapely.intersection(
        np.asarray(values, dtype=object),
        np.full(len(values), mask, dtype=object),
    )
    expected = expected[~(shapely.is_missing(expected) | shapely.is_empty(expected))]

    assert result["value"].tolist() == [1, 2, 3]
    assert materialized_rows == []
    assert [
        shapely.to_wkt(shapely.normalize(geom), rounding_precision=6) for geom in result.geometry
    ] == [shapely.to_wkt(shapely.normalize(geom), rounding_precision=6) for geom in expected]
    assert not any(
        event.surface == "vibespatial.api.NativeAttributeTable.to_arrow" for event in events
    )
    assert not any(event.operation == "index_plan_to_host" for event in events)


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

    assert isinstance(native_result, NativeTabularSelection)
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
    assert (
        result.geometry.iloc[0]
        .normalize()
        .equals(
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
    )


def test_clip_promoted_single_polygon_mask_strict_uses_device_bbox_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module("vibespatial.kernels.core.geometry_analysis")
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
    assert (
        result.geometry.iloc[0]
        .normalize()
        .equals(
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
    )


def test_clip_device_backed_single_polygon_mask_uses_device_bbox_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module("vibespatial.kernels.core.geometry_analysis")
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
    assert (
        result.geometry.iloc[0]
        .normalize()
        .equals(
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

    _assert_native_geometry_trusted_valid(result.geometry.values)
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

    _assert_native_geometry_trusted_valid(filtered.geometry.values)
    assert filtered["value"].tolist() == [1]


def test_clip_device_backed_single_polygon_mask_strict_uses_device_bbox_candidates_without_sindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module("vibespatial.kernels.core.geometry_analysis")
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
    assert (
        result.geometry.iloc[0]
        .normalize()
        .equals(
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
    )
    assert not any(
        event.surface == "geopandas.clip"
        and event.pipeline == "_bbox_candidate_rows_for_scalar_clip_mask"
        for event in vibespatial.get_fallback_events(clear=True)
    )


def test_clip_device_bbox_candidate_ordering_keeps_bounds_on_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [Point(2.0, 0.0), Point(0.0, 0.0), Point(1.0, 0.0)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [2, 0, 1],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    candidate_rows = clip_module._bbox_device_candidate_rows_for_scalar_clip_mask_result(
        gdf,
        box(-1.0, -1.0, 3.0, 1.0),
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert candidate_rows is not None
    assert cp.asnumpy(candidate_rows.device_rows).tolist() == [1, 2, 0]
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip scalar-mask candidate bounds host export" not in runtime_reasons
    assert "clip bbox candidate row count scalar fence" not in runtime_reasons


def test_clip_device_bbox_candidate_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module("vibespatial.kernels.core.geometry_analysis")
    owned = from_shapely_geometries(
        [Point(0.0, 0.0), Point(1.0, 0.0)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    calls = 0

    def _fail_bounds(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic device bounds failure")

    monkeypatch.setattr(
        geometry_analysis_module,
        "compute_geometry_bounds_device",
        _fail_bounds,
    )

    with pytest.raises(RuntimeError, match="synthetic device bounds failure"):
        clip_module._bbox_device_candidate_rows_for_scalar_clip_mask_result(
            gdf,
            box(-1.0, -1.0, 2.0, 1.0),
            sort=False,
        )

    assert calls == 1


def test_clip_device_bbox_admission_has_no_exception_driven_fallback() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    start = source.index("def _bbox_device_candidate_rows_for_scalar_clip_mask_result(")
    end = source.index("\ndef _bbox_candidate_rows_for_scalar_clip_mask_result(", start + 1)
    candidate_source = source[start:end]
    bounds_start = source.index("def _spatial_device_row_bounds_private(")
    bounds_end = source.index("\ndef ", bounds_start + 1)
    bounds_source = source[bounds_start:bounds_end]

    assert "except Exception" not in candidate_source
    assert "except Exception" not in bounds_source
    assert "falling back to the generic candidate query path" not in source


def test_clip_native_dispatch_has_no_exception_driven_algorithm_switches() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    tree = ast.parse(source)
    broad_handlers = [
        handler.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if handler.type is None
        or any(
            isinstance(part, ast.Name) and part.id in {"Exception", "BaseException"}
            for part in ast.walk(handler.type)
        )
    ]
    assert broad_handlers == []
    function_names = (
        "_clip_device_candidate_rows_from_native_relation_result",
        "_clip_polygon_single_mask_candidate_predicates_device",
        "_clip_polygon_single_pair_containment_owned",
        "_clip_multipolygon_rectangle_keep_geom_type_owned",
        "_clip_gdf_with_lazy_grouped_union_mask_native",
        "_clip_mask_covers_source_bounds",
    )

    for function_name in function_names:
        start = source.index(f"def {function_name}(")
        end = source.index("\ndef ", start + 1)
        assert "except Exception" not in source[start:end], function_name

    assert "continuing to exact constructive path" not in source
    assert "lazy grouped-union clip relation query failed" not in source
    assert "coverage probe failed; continuing with exact clip" not in source


def test_lazy_grouped_mask_clip_keeps_relation_and_group_capacity() -> None:
    source = Path("src/vibespatial/api/tools/clip.py").read_text()
    start = source.index("def _clip_gdf_with_lazy_grouped_union_mask_native(")
    end = source.index("\ndef _clip_lazy_grouped_union_coverage_rows_device(", start)
    function_source = source[start:end]
    grouped_start = source.index("def _clip_grouped_polygon_pair_capacity(")
    grouped_end = source.index(
        "\ndef _clip_gdf_with_lazy_grouped_union_mask_native(",
        grouped_start,
    )
    grouped_source = source[grouped_start:grouped_end]
    coverage_start = source.index("def _clip_lazy_grouped_union_coverage_rows_device(")
    coverage_end = source.index("\ndef ", coverage_start + 1)
    coverage_source = source[coverage_start:coverage_end]

    assert "relation_pair_selection = NativeDeviceSelection.from_mask(" in function_source
    assert "d_unresolved_pair_active," in function_source
    assert "clip grouped-mask physical-plan aggregate admission counts" in function_source
    assert "clip grouped-mask partition planner aggregate counts" not in function_source
    assert "relation_pair_selection.gather_capacity(" in function_source
    assert "relation_pair_selection.positions" not in function_source
    assert ")[:relation_pair_count]" not in function_source
    assert ".device_take_capacity(" in function_source
    assert "_clip_grouped_polygon_pair_capacity(" in function_source
    assert "cp.flatnonzero(" not in function_source
    assert "cp.unique(" not in function_source
    assert "d_group_offsets = cp.flatnonzero" not in function_source
    assert "NativeGroupedSelection(" in grouped_source
    assert "d_group_counts += 1" in grouped_source
    assert "build_empty_polygon_rows_device(output_row_count)" in grouped_source
    assert "device_scatter_owned_capacity_selection(" in grouped_source
    assert "segmented_union_all_device_grouped(" in grouped_source
    assert "_capacity_all_valid_noops=True" in grouped_source
    assert "return grouped_result, d_keep" in grouped_source
    assert "device_mask_owned_capacity(" not in grouped_source
    assert "cp.flatnonzero(" not in grouped_source
    assert "NativeDeviceSelection.from_mask(" in coverage_source
    assert "cp.flatnonzero(" not in coverage_source


def test_clip_admissions_do_not_materialize_rejected_row_vectors() -> None:
    source = Path("src/vibespatial/api/tools/clip.py").read_text()

    family_start = source.index("def _owned_active_family_subset(")
    family_end = source.index("\ndef ", family_start + 1)
    family_source = source[family_start:family_end]
    assert "rows=None" not in family_source
    assert "return False" in family_source
    assert "cp.asarray" not in family_source
    assert "d_unsupported_rows" not in family_source

    cache_start = source.index("def _seed_rectangle_clip_validity_cache_if_safe(")
    cache_end = source.index("\ndef ", cache_start + 1)
    cache_source = source[cache_start:cache_end]
    assert "rectangle validity-cache span admission scalar fence" in cache_source
    assert "d_invalid_span_rows" not in cache_source

    assert "_clip_validated_polygon_rect_mask_intersection_owned" not in source
    assert "_clip_polygon_area_intersection_gpu_owned" not in source
    area_start = source.index("def _clip_polygon_area_intersection_owned(")
    area_end = source.index("\ndef ", area_start + 1)
    area_source = source[area_start:area_end]
    assert "broadcast_right_polygon_intersection_capacity_gpu(" in area_source
    assert "materialize_broadcast(" not in area_source
    assert "cp.flatnonzero(" not in area_source
    assert "dispatch_unit_count()" not in area_source

    nonmissing_start = source.index("def _clip_source_nonmissing_rowset(")
    nonmissing_end = source.index("\ndef ", nonmissing_start + 1)
    nonmissing_source = source[nonmissing_start:nonmissing_end]
    assert "d_family_mask" in nonmissing_source
    assert "d_safe_local" in nonmissing_source
    assert "NativeDeviceSelection.from_mask" in nonmissing_source
    assert "cp.flatnonzero" not in nonmissing_source


def test_clip_device_candidate_ordering_uses_soa_radix_keys() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "clip.py"
    source = path.read_text()
    function_start = source.index("def _clip_spatially_order_device_rows(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    assert "_stable_radix_order_pass" in function_source
    assert "_fp64_radix_keys" in function_source
    assert "cp.lexsort" not in function_source
    assert "cp.stack" not in function_source
    assert "d_rows.astype(cp.float64" not in function_source
    assert source.count("_clip_spatially_order_device_rows(") == 3
    assert "active_mask=d_active" in source


def test_clip_device_bbox_candidate_uses_device_mask_bounds_without_total_bounds_d2h() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source = vibespatial.GeoDataFrame(
        {
            "value": [0, 1, 2],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [
                        Point(0.0, 0.0),
                        Point(2.0, 0.0),
                        Point(10.0, 0.0),
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
                    [box(-1.0, -1.0, 3.0, 1.0)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    candidate_rows = clip_module._bbox_device_candidate_rows_for_scalar_clip_mask_result(
        source,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert candidate_rows is not None
    assert cp.asnumpy(candidate_rows.device_rows).tolist() == [0, 1]
    assert "DeviceGeometryArray total-bounds device summary host boundary" not in runtime_reasons
    assert "clip bbox candidate row count scalar fence" not in runtime_reasons


def test_clip_multi_mask_candidate_relation_preserves_device_ordered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.api.sindex import SpatialIndex
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [0, 1, 2],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoSeries(
        [
            box(-0.2, -0.2, 0.2, 0.2),
            box(1.8, 0.8, 2.2, 1.2),
            box(2.8, -0.2, 3.2, 0.2),
        ],
        crs="EPSG:3857",
    )

    def _fail_public_query(*_args, **_kwargs):
        pytest.fail("clip should consume native relation candidates before public export")

    monkeypatch.setattr(SpatialIndex, "query", _fail_public_query)
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip(gdf, mask, sort=False)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result["value"].tolist() == [0, 1, 2]
    assert "clip native-relation candidate rows host export" not in runtime_reasons
    assert not any(
        "SpatialIndex._public_relation_indices_to_host" in reason for reason in runtime_reasons
    )


def test_clip_multi_mask_relation_device_candidates_feed_point_native_without_host_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            Point(0.25, 0.25),
            Point(2.25, 2.25),
            Point(5.0, 5.0),
        ],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = vibespatial.GeoSeries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
        ],
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    assert native_result.provenance is not None
    assert native_result.provenance.is_device
    assert "clip native-relation candidate rows host export" not in runtime_reasons
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.point_polygon_mask_rows::rowset_to_host" not in runtime_reasons
    )
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert not any("candidate" in surface for surface in materialization_surfaces)
    assert "vibespatial.api.tools.clip.point_polygon_mask_rows" not in materialization_surfaces

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 20]


def test_clip_multi_mask_relation_device_candidates_feed_polygon_native_without_host_export() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geometries = [
        box(0.1, 0.1, 0.8, 0.8),
        box(2.1, 2.1, 2.8, 2.8),
        box(5.0, 5.0, 5.5, 5.5),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state_from_native_tabular_result(gdf, native_payload)
    mask = vibespatial.GeoSeries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
        ],
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    native_result = clip_module.evaluate_geopandas_clip_native(
        gdf,
        mask,
        sort=False,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    assert capacity_result.attributes.device_table is not None
    assert capacity_result.index_plan is not None
    assert capacity_result.index_plan.kind == "device-labels"
    if capacity_result.geometry.owned is not None:
        assert capacity_result.geometry.owned.residency is Residency.DEVICE
    else:
        assert capacity_result.geometry.composition is not None
        assert all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in capacity_result.geometry.composition.parts
        )
    assert native_result.provenance is not None
    assert native_result.provenance.is_device
    assert "clip native-relation candidate rows host export" not in runtime_reasons
    assert "clip scalar-mask candidate rows host export" not in runtime_reasons
    assert "clip polygon device-candidate output rows host export" not in runtime_reasons
    assert (
        "vibespatial.api.tools.clip.polygon_mask_exact_rows::rowset_to_host" not in runtime_reasons
    )
    assert (
        "vibespatial.api.tools.clip.polygon_mask_inside_rows::rowset_to_host" not in runtime_reasons
    )
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}
    assert not any("candidate" in surface for surface in materialization_surfaces)
    assert "vibespatial.api.tools.clip.polygon_mask_exact_rows" not in materialization_surfaces
    assert "vibespatial.api.tools.clip.polygon_mask_inside_rows" not in materialization_surfaces

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 20]


def test_clip_consumes_lazy_grouped_union_mask_without_materializing_union() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    source_geoms = [box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(40)]
    source = vibespatial.GeoDataFrame(
        {
            "value": np.arange(len(source_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [box(float(i) + 0.2, 0.2, float(i) + 0.8, 0.8) for i in range(len(source_geoms))]
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(mask_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()
    assert getattr(
        dissolved_mask.geometry.values.cached_owned(),
        "_is_lazy_grouped_union_owned",
        False,
    )

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert result["value"].tolist() == list(range(len(source_geoms)))
    assert any(
        event.implementation == "lazy_grouped_union_mask_relation_clip_gpu"
        for event in dispatch_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in dispatch_events
    )
    assert "clip native-relation candidate rows host export" not in runtime_reasons
    assert not any("LazyGroupedUnionOwned" in surface for surface in materialization_surfaces)

    mask_union = shapely.union_all(np.asarray(mask_geoms, dtype=object))
    expected = shapely.intersection(np.asarray(source_geoms, dtype=object), mask_union)
    actual = np.asarray(result.geometry.array, dtype=object)
    assert result["value"].tolist() == list(range(len(source_geoms)))
    assert all(
        shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0)
        for got, want in zip(actual, expected, strict=True)
    )


def test_clip_consumes_tiled_collective_coverage_without_union_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.constructive import tiled_union

    monkeypatch.setattr(
        tiled_union,
        "_DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE",
        1,
    )
    source_geoms = [
        box(0.25, 0.25, 2.75, 0.75),
        box(4.25, 0.25, 6.75, 0.75),
        box(20.0, 20.0, 21.0, 21.0),
    ]
    source = vibespatial.GeoDataFrame(
        {
            "value": np.arange(len(source_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [
        box(float(i), 0.0, float(i) + 1.0, 1.0)
        for i in range(8)
    ]
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(mask_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved_mask = mask_frame.dissolve(
        by="group",
        aggfunc="first",
        method="unary",
    ).reset_index()

    vibespatial.clear_dispatch_events()
    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result["value"].tolist() == [0, 1]
    assert any(
        event.implementation == "gpu_single_group_tiled_collective_coverage"
        for event in dispatch_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in dispatch_events
    )
    expected = shapely.intersection(
        np.asarray(source_geoms[:2], dtype=object),
        shapely.union_all(np.asarray(mask_geoms, dtype=object)),
    )
    actual = np.asarray(result.geometry.array, dtype=object)
    assert all(
        shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0)
        for got, want in zip(actual, expected, strict=True)
    )


def test_clip_lazy_grouped_union_covered_source_rows_passthrough() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    source_geoms = [box(0.2, float(i) * 3.0 - 0.2, 0.8, float(i) * 3.0 + 0.2) for i in range(40)]
    source = vibespatial.GeoDataFrame(
        {
            "value": np.arange(len(source_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    lines = [
        LineString([(0.0, float(i) * 3.0), (1.0, float(i) * 3.0)]) for i in range(len(source_geoms))
    ]
    line_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_frame = line_frame.copy()
    mask_frame["geometry"] = mask_frame.geometry.buffer(0.5)
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()
    assert getattr(
        dissolved_mask.geometry.values.cached_owned(),
        "_is_lazy_grouped_union_owned",
        False,
    )

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materialization_surfaces = {event.surface for event in get_materialization_events(clear=True)}

    assert result["value"].tolist() == list(range(len(source_geoms)))
    assert [geom.wkb for geom in result.geometry] == [geom.wkb for geom in source.geometry]
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert any(
        event.implementation == "relation_pair_covered_by_no_holes_gpu"
        and event.operation == "covered_by"
        for event in dispatch_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in dispatch_events
    )
    assert not any(
        event.implementation
        in {
            "lazy_grouped_union_mask_relation_clip_gpu",
            "lazy_grouped_union_mask_source_materialized_gpu",
            "lazy_grouped_union_mask_union_physicalized_gpu",
        }
        for event in dispatch_events
    )
    assert "clip native-relation candidate rows host export" not in runtime_reasons
    assert not any(
        "owned geometry device-take nested slice-size allocation fence" in reason
        for reason in runtime_reasons
    )
    assert not any("LazyGroupedUnionOwned" in surface for surface in materialization_surfaces)


def test_clip_device_covered_partition_uses_inactive_exact_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    source_geoms = [box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(40)]
    source = vibespatial.GeoDataFrame(
        {
            "value": np.arange(len(source_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geom = Polygon(
        [(-2.0, -1.0), (41.0, -1.0), (42.0, 1.0), (41.0, 2.0), (-2.0, 2.0)]
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries([mask_geom], residency=Residency.DEVICE),
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )

    original_exact = clip_module._clip_polygon_area_intersection_owned
    exact_inputs = []

    def _capture_exact(candidate_owned, *args, **kwargs):
        exact_inputs.append(candidate_owned)
        return original_exact(candidate_owned, *args, **kwargs)

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        _capture_exact,
    )
    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()

    result = clip(source, mask, sort=False)
    events = vibespatial.get_dispatch_events(clear=True)
    transfer_reasons = [event.reason for event in get_d2h_transfer_events()]

    assert result["value"].tolist() == list(range(len(source_geoms)))
    assert [geom.wkb for geom in result.geometry] == [geom.wkb for geom in source.geometry]
    assert len(exact_inputs) == 1
    assert not bool(cp.any(cp.asarray(exact_inputs[0].device_state.validity)).get())
    assert "clip exact-topology admission count packet" not in transfer_reasons
    assert not any(
        event.implementation == "polygon_device_covered_rowset_passthrough_gpu"
        for event in events
    )


def test_clip_collective_lazy_grouped_union_mask_physicalizes_once(monkeypatch) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        box(0.0, 0.0, 5.0, 2.0),
        box(4.0, 0.0, 9.0, 2.0),
        box(20.0, 0.0, 21.0, 1.0),
    ]
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [box(float(i) * 0.2, -1.0, float(i) * 0.2 + 1.0, 1.0) for i in range(40)]
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(mask_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()
    assert getattr(
        dissolved_mask.geometry.values.cached_owned(),
        "_is_lazy_grouped_union_owned",
        False,
    )

    def _fail_pair_coverage(**_kwargs):
        raise AssertionError("union-first admission must precede exact pair coverage")

    monkeypatch.setattr(
        clip_module,
        "_clip_lazy_grouped_union_coverage_rows_device",
        _fail_pair_coverage,
    )
    monkeypatch.setattr(clip_module, "_clip_available_device_bytes", lambda: 1)

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result["value"].tolist() == [10, 20]
    assert any(
        event.implementation == "lazy_grouped_union_mask_union_plan_gpu"
        and "semantic_pair_refinement=skipped" in event.detail
        for event in dispatch_events
    )
    assert any(
        event.implementation == "lazy_grouped_union_mask_union_physicalized_gpu"
        and "selected_rows=device-resident" in event.detail
        for event in dispatch_events
    )
    assert any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in dispatch_events
    )
    assert not any(
        event.implementation == "lazy_grouped_union_mask_relation_clip_gpu"
        for event in dispatch_events
    )
    assert "clip native-relation candidate rows host export" not in runtime_reasons

    mask_union = shapely.union_all(np.asarray(mask_geoms, dtype=object))
    expected = shapely.intersection(np.asarray(source_geoms[:2], dtype=object), mask_union)
    actual = np.asarray(result.geometry.array, dtype=object)
    assert all(
        shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0)
        for got, want in zip(actual, expected, strict=True)
    )


def test_collective_grouped_mask_shape_estimate_prefers_sparse_relation() -> None:
    source_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 3.0, 1.0) for i in range(10)],
        residency=Residency.HOST,
    )
    mask_owned = from_shapely_geometries(
        [box(float(i) * 10.0, 0.0, float(i) * 10.0 + 1.0, 1.0) for i in range(100)],
        residency=Residency.HOST,
    )

    prefers_relation, relation, union = clip_module._clip_collective_grouped_mask_prefers_relation(
        source_owned,
        mask_owned,
        relation_pair_count=20,
        collective_source_count=10,
    )

    assert prefers_relation
    assert relation.relation_pair_count == 20
    assert relation.group_count == 10
    assert relation.dispatch_unit_count() < union.dispatch_unit_count()


def test_collective_grouped_mask_shape_pages_relation_over_memory_budget() -> None:
    source_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 3.0, 1.0) for i in range(10)],
        residency=Residency.HOST,
    )
    mask_owned = from_shapely_geometries(
        [box(float(i) * 10.0, 0.0, float(i) * 10.0 + 1.0, 1.0) for i in range(100)],
        residency=Residency.HOST,
    )

    prefers_relation, relation, union = clip_module._clip_collective_grouped_mask_prefers_relation(
        source_owned,
        mask_owned,
        relation_pair_count=20,
        collective_source_count=10,
        available_device_bytes=1,
    )

    assert relation.dispatch_unit_count() < union.dispatch_unit_count()
    assert not relation.is_device_memory_admissible(1)
    assert prefers_relation


def test_collective_grouped_mask_shape_estimate_prefers_dense_union() -> None:
    source_owned = from_shapely_geometries(
        [box(0.0, 0.0, 5.0, 2.0), box(4.0, 0.0, 9.0, 2.0)],
        residency=Residency.HOST,
    )
    mask_owned = from_shapely_geometries(
        [box(float(i) * 0.2, -1.0, float(i) * 0.2 + 1.0, 1.0) for i in range(40)],
        residency=Residency.HOST,
    )

    prefers_relation, relation, union = clip_module._clip_collective_grouped_mask_prefers_relation(
        source_owned,
        mask_owned,
        relation_pair_count=80,
        collective_source_count=2,
    )

    assert not prefers_relation
    assert relation.dispatch_unit_count() > union.dispatch_unit_count()


def test_clip_sparse_collective_lazy_grouped_union_uses_relation_shape() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")

    source = vibespatial.GeoDataFrame(
        {
            "value": [10],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [box(0.0, 0.0, 3.0, 1.0)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)]
    mask_geoms.extend(box(float(i) * 10.0, 0.0, float(i) * 10.0 + 1.0, 1.0) for i in range(1, 101))
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(mask_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()

    vibespatial.clear_dispatch_events()
    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result["value"].tolist() == [10]
    assert any(
        event.implementation == "lazy_grouped_union_mask_relation_plan_gpu"
        for event in dispatch_events
    )
    assert any(
        event.implementation == "lazy_grouped_union_mask_relation_clip_gpu"
        for event in dispatch_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in dispatch_events
    )
    expected = shapely.intersection(
        source.geometry.iloc[0],
        shapely.union_all(np.asarray(mask_geoms, dtype=object)),
    )
    assert shapely.equals(result.geometry.iloc[0], expected)


def test_clip_lazy_grouped_union_partitions_covered_and_collective_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    source_geoms = [
        box(0.1, -0.5, 0.5, 0.5),
        box(0.0, 0.0, 5.0, 2.0),
        box(20.0, 0.0, 21.0, 1.0),
    ]
    source_base_owned = from_shapely_geometries(
        [box(-100.0, -100.0, -90.0, -90.0), *source_geoms],
        residency=Residency.DEVICE,
    )
    source_owned = source_base_owned._device_indexed_take(
        cp.asarray([1, 2, 3], dtype=cp.int64),
        assume_unique_indices=True,
    )
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(
                source_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [box(float(i) * 0.2, -1.0, float(i) * 0.2 + 1.0, 1.0) for i in range(30)]
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    mask_geoms,
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()

    vibespatial.clear_dispatch_events()
    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert result["value"].tolist() == [10, 20]
    assert any(
        event.implementation == "lazy_grouped_union_mask_union_plan_gpu"
        and "source_rows=1" in event.detail
        for event in dispatch_events
    )
    assert any(
        event.implementation == "lazy_grouped_union_mask_union_physicalized_gpu"
        and "selected_rows=device-resident" in event.detail
        for event in dispatch_events
    )
    mask_union = shapely.union_all(np.asarray(mask_geoms, dtype=object))
    expected = shapely.intersection(np.asarray(source_geoms[:2], dtype=object), mask_union)
    assert tuple(result.geometry.total_bounds) == pytest.approx(
        tuple(shapely.total_bounds(expected))
    )
    assert np.asarray(result.geometry.area) == pytest.approx(shapely.area(expected))
    assert all(
        shapely.equals(got, want) for got, want in zip(result.geometry, expected, strict=True)
    )


def test_clip_lazy_buffered_line_mask_uses_grouped_relation_without_fanout_probe() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        box(0.0, -1.0, 8.0, 2.5),
        box(20.0, 20.0, 21.0, 21.0),
    ]
    source = vibespatial.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(source_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    lines = [LineString([(0.0, float(i) * 0.04), (8.0, float(i) * 0.04)]) for i in range(40)]
    line_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_frame = line_frame.copy()
    mask_frame["geometry"] = mask_frame.geometry.buffer(0.2)
    dissolved_mask = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved_mask = dissolved_mask.reset_index()
    assert getattr(
        dissolved_mask.geometry.values.cached_owned(),
        "_is_lazy_grouped_union_owned",
        False,
    )

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = clip(source, dissolved_mask, sort=False)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result["value"].tolist() == [1]
    assert any(
        event.implementation == "lazy_grouped_union_mask_relation_clip_gpu"
        for event in dispatch_events
    )
    assert any(
        event.implementation == "buffered_line_grouped_union_gpu" for event in dispatch_events
    )
    assert not any("physicalized" in event.implementation for event in dispatch_events)
    assert "clip lazy grouped-union relation fanout scalar fence" not in runtime_reasons
    assert "clip grouped-mask mixed-dimension plan admission scalar fence" not in runtime_reasons

    mask_union = shapely.union_all(
        np.asarray(mask_frame.geometry.array, dtype=object),
    )
    expected = shapely.intersection(np.asarray([source_geoms[0]], dtype=object), mask_union)
    actual = np.asarray(result.geometry.array, dtype=object)
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == pytest.approx(0.0)


def test_grouped_mixed_clip_reducer_is_device_native_and_topology_shaped() -> None:
    source = Path("src/vibespatial/constructive/grouped_mixed_union.py").read_text()
    kernel = Path("src/vibespatial/kernels/constructive/grouped_mixed_union.py").read_text()
    clip_source = Path("src/vibespatial/api/tools/clip.py").read_text()
    start = clip_source.index("def _clip_gdf_with_lazy_grouped_union_mask_native(")
    end = clip_source.index("\ndef _clip_lazy_grouped_union_coverage_rows_device(", start)
    grouped_clip = clip_source[start:end]

    assert "grouped_mixed_union_capacity_device(" in grouped_clip
    assert "clip grouped-mask mixed-dimension plan admission scalar fence" not in grouped_clip
    assert "lazy_grouped_union_mask_source_materialized_gpu" not in grouped_clip
    assert "build_gpu_split_events(" in source
    assert "build_gpu_atomic_edges(" in source
    assert "isolate_rows=True" in source
    assert "extract_unique_points_owned(" in source
    assert "_geometry_composition_from_owned_parts_at_capacity(" in source
    assert "copy_device_to_host" not in source
    assert "cp.asnumpy" not in source
    assert "shapely" not in source
    assert "grouped_points_on_atomic_edges" in kernel
    assert "const int point = thread >> 5" in kernel
    assert "__ballot_sync" in kernel


@pytest.mark.gpu
def test_grouped_mixed_clip_reduces_area_line_and_point_without_mask_physicalization() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geom = box(0.0, 0.0, 2.0, 2.0)
    source = vibespatial.GeoDataFrame(
        {
            "value": [1],
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries([source_geom], residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask_geoms = [
        box(0.5, 0.5, 1.5, 1.5),
        box(2.0, 0.0, 3.0, 1.0),
        box(2.0, 2.0, 3.0, 3.0),
    ]
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(len(mask_geoms), dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(mask_geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved = dissolved.reset_index()
    assert getattr(dissolved.geometry.values.cached_owned(), "_is_lazy_grouped_union_owned", False)

    vibespatial.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = clip(source, dissolved, sort=False, keep_geom_type=False)
    events = vibespatial.get_dispatch_events(clear=True)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.intersection(source_geom, shapely.union_all(mask_geoms))
    actual = result.geometry.iloc[0]
    assert shapely.equals(actual, expected)
    assert actual.geom_type == "GeometryCollection"
    assert {part.geom_type for part in actual.geoms} == {"Polygon", "LineString", "Point"}
    assert any(
        event.implementation == "lazy_grouped_union_mask_relation_clip_gpu" for event in events
    )
    assert not any("physicalized" in event.implementation for event in events)
    assert "clip grouped-mask mixed-dimension plan admission scalar fence" not in reasons


@pytest.mark.gpu
def test_grouped_mixed_clip_nodes_duplicate_line_contacts_once() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")

    source_geom = box(0.0, 0.0, 2.0, 2.0)
    source = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries([source_geom], residency=Residency.DEVICE),
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )
    contact = box(2.0, 0.0, 3.0, 2.0)
    mask_frame = vibespatial.GeoDataFrame(
        {
            "group": np.zeros(2, dtype=np.int32),
            "geometry": DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [contact, contact],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    dissolved = mask_frame.dissolve(by="group", aggfunc="first", method="unary")
    dissolved = dissolved.reset_index()
    assert getattr(dissolved.geometry.values.cached_owned(), "_is_lazy_grouped_union_owned", False)

    result = clip(source, dissolved, sort=False, keep_geom_type=False)
    actual = result.geometry.iloc[0]
    expected = shapely.intersection(source_geom, contact)

    assert actual.geom_type == "LineString"
    assert shapely.equals(actual, expected)


def test_clip_single_row_native_polygon_mask_stays_device_resident_until_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.execution_trace import execution_trace

    source_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(10.0, 10.0, 11.0, 11.0),
        ],
        residency=Residency.DEVICE,
    )
    source_owned._validity = None
    source_owned._tags = None
    source_owned._family_row_offsets = None
    mask_owned = from_shapely_geometries(
        [box(0.5, -1.0, 2.5, 2.0)],
        residency=Residency.DEVICE,
    )
    mask_owned._validity = None
    mask_owned._tags = None
    mask_owned._family_row_offsets = None
    source = vibespatial.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(
                source_owned,
                crs="EPSG:3857",
            ),
        },
        crs="EPSG:3857",
    )
    mask = vibespatial.GeoDataFrame(
        {
            "geometry": DeviceGeometryArray._from_owned(
                mask_owned,
                crs="EPSG:3857",
            )
        },
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    with execution_trace("clip_native_mask_test") as trace:
        native_result = clip_module.evaluate_geopandas_clip_native(
            source,
            mask,
            sort=False,
        )
    d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    trace_reasons = [transfer.reason for transfer in trace.transfers]

    capacity_result = (
        native_result.capacity_result
        if isinstance(native_result, NativeTabularSelection)
        else native_result
    )
    if capacity_result.geometry.owned is not None:
        assert capacity_result.geometry.owned.residency is Residency.DEVICE
    else:
        assert capacity_result.geometry.composition is not None
        assert all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in capacity_result.geometry.composition.parts
        )
    assert capacity_result.attributes.device_table is not None
    assert not any("owned geometry polygon coordinate" in reason for reason in d2h_reasons)
    assert not any("owned geometry host metadata" in reason for reason in d2h_reasons)
    assert "created owned geometry array with device residency requested" not in trace_reasons

    exported = native_result.to_geodataframe()

    assert exported["value"].tolist() == [10, 20]


def test_clip_homogeneous_polygon_candidates_thread_device_rows_to_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [box(2.0, 0.0, 3.0, 1.0), box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    gdf = vibespatial.GeoDataFrame(
        {
            "value": [2, 0],
            "geometry": DeviceGeometryArray._from_owned(owned, crs="EPSG:3857"),
        },
        crs="EPSG:3857",
    )
    rows = np.asarray([1, 0], dtype=np.intp)
    d_rows = cp.asarray(rows, dtype=cp.int64)
    seen: dict[str, object] = {}
    original = clip_module._clip_source_rowset_for_positions

    def _wrapped_source_rowset(
        source,
        row_positions,
        local_rows=None,
        *,
        device_row_positions=None,
        device_local_rows=None,
    ):
        seen["device_rows"] = (
            None if device_row_positions is None else cp.asnumpy(device_row_positions).tolist()
        )
        seen["device_local_rows"] = (
            None if device_local_rows is None else cp.asnumpy(device_local_rows).tolist()
        )
        return original(
            source,
            row_positions,
            local_rows,
            device_row_positions=device_row_positions,
            device_local_rows=device_local_rows,
        )

    monkeypatch.setattr(
        clip_module,
        "_clip_source_rowset_for_positions",
        _wrapped_source_rowset,
    )
    mask = Polygon(
        [
            (-1.0, -1.0),
            (4.0, -1.0),
            (4.0, 0.5),
            (0.5, 0.5),
            (0.5, 2.0),
            (-1.0, 2.0),
            (-1.0, -1.0),
        ]
    )

    result = clip_module._clip_homogeneous_polygon_candidates_native(
        gdf,
        mask,
        rows,
        candidate_device_rows=d_rows,
        clipping_by_rectangle=False,
        rectangle_bounds=None,
        keep_geom_type=False,
    )

    assert result is not None
    assert seen["device_rows"] == [1, 0]


def test_clip_large_scalar_rectangle_mask_promotes_supported_host_candidates_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geometry_analysis_module = importlib.import_module("vibespatial.kernels.core.geometry_analysis")
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
    _assert_native_geometry_device_resident(result.geometry.values)


def test_clip_polygon_area_intersection_declines_atomically_when_capacity_carrier_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    left_owned = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0)],
        residency=Residency.HOST,
    )
    mask_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.HOST,
    )
    monkeypatch.setattr(
        binary_module,
        "broadcast_right_polygon_intersection_capacity_gpu",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        clip_module,
        "_host_polygonal_area_intersection_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "clip polygon area helper must decline before host exact extraction"
        ),
    )

    vibespatial.clear_fallback_events()
    with pytest.raises(
        clip_module.StrictNativeFallbackError,
        match="canonical broadcast-right device-capacity carrier",
    ):
        clip_module._clip_polygon_area_intersection_owned(left_owned, mask_owned)
    assert vibespatial.get_fallback_events(clear=True) == []


def test_clip_polygon_area_intersection_uses_constructive_capacity_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    calls: list[tuple[object, object, int, ExecutionMode]] = []

    def _capacity_carrier(left, right, *, right_row, dispatch_mode):
        calls.append((left, right, right_row, dispatch_mode))
        return expected

    monkeypatch.setattr(
        binary_module,
        "broadcast_right_polygon_intersection_capacity_gpu",
        _capacity_carrier,
    )

    result = clip_module._clip_polygon_area_intersection_owned(
        left_owned,
        mask_owned,
    )

    assert result is expected
    assert calls == [(left_owned, mask_owned, 0, ExecutionMode.GPU)]


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
    calls: list[tuple[int, bool]] = []

    def _owned_helper(
        left_owned,
        mask_owned,
        *,
        preserve_lower_dimensional=False,
    ):
        calls.append((left_owned.row_count, preserve_lower_dimensional))
        return expected_owned

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        _owned_helper,
    )
    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: np.zeros(1, dtype=bool),
    )

    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert calls == [(1, True)]
    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert result_objects[0] is not None


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
    calls: list[tuple[int, int, bool]] = []

    monkeypatch.setattr(
        shapely,
        "covered_by",
        lambda *args, **kwargs: pytest.fail(
            "rectangle polygon-mask clip should avoid host covered_by"
        ),
    )

    def _capacity_exact(left, right, *, preserve_lower_dimensional=False):
        calls.append(
            (left.row_count, right.row_count, preserve_lower_dimensional)
        )
        return expected_owned

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        _capacity_exact,
    )

    vibespatial.clear_fallback_events()
    result = clip_module._clip_polygon_partition_with_polygon_mask(partition, mask)

    assert calls == [(1, 1, True)]
    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], expected_geom)
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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], expected_owned.to_shapely()[0])


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], expected_geom)


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], box(-0.5, -0.5, 0.5, 0.5))


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], box(-0.5, -0.5, 0.5, 0.5))


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(result_objects[0], mask)


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 2
    actual = shapely.normalize(result_objects)
    expected = shapely.normalize(np.asarray(expected_owned.to_shapely(), dtype=object))
    assert [geom.wkb for geom in actual] == [geom.wkb for geom in expected]


def test_clip_polygon_partition_polygon_mask_mixed_inside_and_exact_positive_skips_null_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_rowset import NativeDeviceSelection

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
    expected_exact_capacity = owned_module.device_take_owned_capacity_selection(
        expected_owned,
        NativeDeviceSelection.from_mask(
            cp.asarray([False, True], dtype=cp.bool_),
        ),
    )

    monkeypatch.setattr(
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: expected_exact_capacity,
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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 2
    actual = shapely.normalize(result_objects)
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


def test_clip_owned_native_assembly_consumes_device_rowset_without_hot_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device clip assembly canary")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import NativeFrameState, attach_native_state
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
            box(4.0, 4.0, 5.0, 5.0),
        ],
        residency=Residency.DEVICE,
    )
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
    source = vibespatial.GeoDataFrame(
        {"value": [10, 20, 30], "geometry": values},
        crs="EPSG:3857",
    )
    arrow_table = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow_table),
                index_override=source.index,
                column_override=("value",),
                schema_override=arrow_table.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(source, state)

    d_rows = cp.asarray([2, 0], dtype=cp.int64)
    part_owned = owned.device_take(
        d_rows,
        host_indices_for_sizing=np.asarray([2, 0], dtype=np.int64),
    )
    part_owned._clip_semantically_clean = True
    part = clip_module._clip_native_part(
        source,
        np.asarray([2, 0], dtype=np.intp),
        DeviceGeometryArray._from_owned(part_owned, crs=source.crs),
        rowset=NativeRowSet.from_positions(
            d_rows,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=True,
            unique=True,
        ),
    )

    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        result = clip_module._clip_constructive_parts_to_native_tabular_result(
            source=source,
            parts=(part,),
            ordered_row_positions=np.asarray([2, 0], dtype=np.intp),
            clipping_by_rectangle=False,
            has_non_point_candidates=True,
            keep_geom_type=False,
        )

    assert result.attributes.device_table is not None
    assert result.index_plan is not None
    assert result.index_plan.kind == "device-labels"
    assert result.provenance is not None
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [2, 0]
    assert get_materialization_events(clear=True) == []


def test_clip_owned_native_cleanup_returns_capacity_selection_without_hot_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device clip cleanup canary")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeTabularResult,
    )
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import NativeFrameState, attach_native_state
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), Polygon()],
        residency=Residency.DEVICE,
    )
    values = DeviceGeometryArray._from_owned(owned, crs="EPSG:3857")
    source = vibespatial.GeoDataFrame(
        {"value": [10, 20], "geometry": values},
        crs="EPSG:3857",
    )
    arrow_table = pa.table({"value": pa.array([10, 20], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow_table),
                index_override=source.index,
                column_override=("value",),
                schema_override=arrow_table.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(source, state)
    d_rows = cp.arange(owned.row_count, dtype=cp.int64)
    part = clip_module._clip_native_part(
        source,
        np.arange(owned.row_count, dtype=np.intp),
        values,
        rowset=NativeRowSet.from_positions(
            d_rows,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=True,
            unique=True,
            identity=True,
        ),
    )

    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        result = clip_module._clip_constructive_parts_to_native_tabular_result(
            source=source,
            parts=(part,),
            ordered_row_positions=np.arange(owned.row_count, dtype=np.intp),
            clipping_by_rectangle=False,
            has_non_point_candidates=True,
            keep_geom_type=False,
        )

    assert isinstance(result, NativeTabularSelection)
    assert result.capacity == owned.row_count
    assert result.capacity_result.geometry.owned is not None
    assert cp.asnumpy(result.logical_count).tolist() == [1]
    assert get_materialization_events(clear=True) == []


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

    result_objects = _clip_partition_objects(result)
    assert len(result_objects) == 1
    assert shapely.equals(
        result_objects[0],
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


def test_clip_polygon_exact_topology_failure_is_not_silently_swallowed(
    monkeypatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        return

    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
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
        clip_module,
        "_clip_polygon_area_intersection_owned",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced clip exact-topology failure")
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

    with pytest.raises(RuntimeError, match="forced clip exact-topology failure"):
        clip(buildings, mask)
