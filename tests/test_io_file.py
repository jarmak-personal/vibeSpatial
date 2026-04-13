from __future__ import annotations

import importlib.util

import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

import vibespatial.api as geopandas
import vibespatial.io.geojson as io_geojson
from vibespatial import (
    benchmark_geojson_ingest,
    benchmark_shapefile_ingest,
    plan_geojson_ingest,
    plan_shapefile_ingest,
    read_geojson_native,
    read_geojson_owned,
    read_shapefile_native,
    read_shapefile_owned,
)
from vibespatial.api._native_results import (
    NativeAttributeTable,
    NativeTabularResult,
    _spatial_to_native_tabular_result,
)
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.io.file import (
    _native_file_result_from_owned,
    _native_geojson_result_from_gpu_result,
    _pyogrio_arrow_wkb_to_native_tabular_result,
    _read_osm_pbf_pyogrio_layer_public,
    plan_vector_file_io,
    read_vector_file,
    read_vector_file_native,
    write_vector_file,
)
from vibespatial.io.support import IOOperation, IOPathKind


def _sample_frame() -> geopandas.GeoDataFrame:
    return geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )


def test_native_attribute_table_concat_harmonizes_string_widths() -> None:
    left = NativeAttributeTable.from_value(
        pa.table({"name": pa.array(["a"], type=pa.string())})
    )
    right = NativeAttributeTable.from_value(
        pa.table({"name": pa.array(["b"], type=pa.large_string())})
    )

    concatenated = NativeAttributeTable.concat([left, right])

    assert concatenated.to_pandas(copy=False)["name"].tolist() == ["a", "b"]


@pytest.mark.gpu
def test_geojson_roundtrip_uses_gpu_adapter(tmp_path) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    path = tmp_path / "sample.geojson"

    frame = _sample_frame()
    frame.to_file(path, driver="GeoJSON")
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert result["id"].tolist() == frame["id"].tolist()
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert "gpu" in events[-1].implementation.lower()
    assert not fallbacks


@pytest.mark.gpu
def test_geojson_gpu_adapter_failure_records_explicit_fallback(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.geojson"
    frame = _sample_frame()
    frame.to_file(path, driver="GeoJSON")

    monkeypatch.setattr("vibespatial.io.file._GPU_MIN_FILE_SIZE", 0)

    def _boom(*args, **kwargs):
        raise RuntimeError("geojson-gpu-boom")

    monkeypatch.setattr("vibespatial.io.geojson_gpu.read_geojson_gpu", _boom)

    geopandas.clear_fallback_events()
    result = geopandas.read_file(path)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == len(frame)
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert fallbacks
    assert fallbacks[-1].surface == "geopandas.read_file"
    assert "geojson-gpu-boom" in fallbacks[-1].reason


@pytest.mark.gpu
def test_large_public_geojson_read_uses_default_feature_boundary_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.geojson"
    frame = _sample_frame()
    frame.to_file(path, driver="GeoJSON")

    monkeypatch.setattr("vibespatial.io.file._GPU_MIN_FILE_SIZE", 0)

    calls: list[dict[str, object]] = []

    class _FakeGpuResult:
        def __init__(self) -> None:
            self.owned = from_shapely_geometries(frame.geometry.tolist())
            self.n_features = len(frame)

        def properties_loader(self):
            def _load():
                return frame.drop(columns="geometry").to_dict("records")

            return _load

    def _fake_read_geojson_gpu(*args, **kwargs):
        calls.append(kwargs)
        return _FakeGpuResult()

    monkeypatch.setattr("vibespatial.io.geojson_gpu.read_geojson_gpu", _fake_read_geojson_gpu)

    result = geopandas.read_file(path)

    assert result["id"].tolist() == frame["id"].tolist()
    assert calls
    assert "capture_feature_boundaries" not in calls[-1]


@pytest.mark.gpu
def test_large_flatgeobuf_read_prefers_pyogrio_arrow_gpu_wkb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.fgb"
    frame = _sample_frame()
    frame.to_file(path, driver="FlatGeobuf")

    monkeypatch.setattr("vibespatial.io.file._GPU_MIN_FILE_SIZE", 0)

    def _boom(*args, **kwargs):
        raise AssertionError("direct FlatGeobuf decoder should not be used by default")

    monkeypatch.setattr("vibespatial.io.file._try_fgb_gpu_read_native", _boom)

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert len(result) == len(frame)
    assert set(result["id"]) == set(frame["id"])
    assert tuple(result.total_bounds) == tuple(frame.total_bounds)
    assert events
    assert events[-1].implementation == "flatgeobuf_pyogrio_arrow_gpu_wkb"


@pytest.mark.gpu
def test_large_csv_wkt_read_prefers_pylibcudf_table_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        'id,name,geometry\n'
        '1,a,"POINT (0 0)"\n'
        '2,b,"POINT (1 1)"\n'
    )

    monkeypatch.setattr("vibespatial.io.file._GPU_MIN_FILE_SIZE", 0)

    def _boom(*args, **kwargs):
        raise AssertionError("byte-classify CSV path should not be used for large WKT CSV")

    monkeypatch.setattr("vibespatial.io.file._try_csv_byte_classify_read_native", _boom)

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert result["name"].tolist() == ["a", "b"]
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.geometry.iloc[1].equals(Point(1, 1))
    assert events
    assert events[-1].implementation == "csv_pylibcudf_table_adapter"


def test_osm_public_read_surfaces_gpu_failure_detail(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")

    def _boom(*args, **kwargs):
        raise RuntimeError("osm-public-boom")

    monkeypatch.setattr("vibespatial.io.file._try_gpu_read_file_native", _boom)

    geopandas.clear_fallback_events()
    with pytest.raises(RuntimeError, match="osm-public-boom"):
        geopandas.read_file(path)

    fallbacks = geopandas.get_fallback_events(clear=True)
    assert fallbacks
    assert fallbacks[-1].surface == "geopandas.read_file"
    assert fallbacks[-1].pipeline == "io/read_file"
    assert "osm-public-boom" in fallbacks[-1].detail


@pytest.mark.gpu
def test_osm_native_read_uses_loader_backed_lossless_tag_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import cupy as cp

    import vibespatial.io.osm_gpu as io_osm_gpu
    from vibespatial.io import file as io_file

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")

    class _FakeOsmResult:
        nodes = None
        node_ids = None
        n_nodes = 0
        node_tags = None
        ways = from_shapely_geometries([LineString([(0, 0), (1, 1)])])
        way_ids = cp.asarray([42], dtype=cp.int64)
        way_tags = [{"name": "Main", "highway": "residential", "custom": "x"}]
        n_ways = 1
        relations = None
        relation_ids = None
        relation_tags = None
        n_relations = 0

    monkeypatch.setattr(io_osm_gpu, "read_osm_pbf", lambda *args, **kwargs: _FakeOsmResult())

    payload = io_file._try_osm_pbf_gpu_read_native(path)

    assert payload.attributes.loader is not None
    assert "osm_element" in payload.attributes.columns
    assert "osm_id" in payload.attributes.columns
    assert "other_tags" in payload.attributes.columns

    frame = payload.to_geodataframe()

    assert frame["osm_element"].tolist() == ["way"]
    assert frame["osm_id"].tolist() == [42]
    assert frame["name"].tolist() == ["Main"]
    assert frame["highway"].tolist() == ["residential"]
    assert frame["other_tags"].tolist() == ['"custom"=>"x"']


@pytest.mark.gpu
def test_osm_public_read_projects_compatibility_schema_from_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import cupy as cp

    import vibespatial.io.osm_gpu as io_osm_gpu

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")

    class _FakeRuntime:
        compute_capability = (8, 9)

        def available(self) -> bool:
            return True

    class _FakeOsmResult:
        nodes = None
        node_ids = None
        n_nodes = 0
        node_tags = None
        ways = from_shapely_geometries([LineString([(0, 0), (1, 1)])])
        way_ids = cp.asarray([42], dtype=cp.int64)
        way_tags = [{"name": "Main", "highway": "residential", "custom": "x"}]
        n_ways = 1
        relations = None
        relation_ids = None
        relation_tags = None
        n_relations = 0

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(io_osm_gpu, "read_osm_pbf", lambda *args, **kwargs: _FakeOsmResult())

    result = geopandas.read_file(path, layer="all")

    assert "osm_way_id" in result.columns
    assert "osm_element" not in result.columns
    assert result["osm_way_id"].tolist() == [42]


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_other_relations_tags_false_stays_on_pyogrio_layer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"geometry": [Point(0, 0).wkb]})
    owned = from_shapely_geometries([Point(0, 0)])
    captured: dict[str, object] = {}

    class _FakeRuntime:
        compute_capability = (8, 9)

        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **kwargs: (
            captured.update(kwargs)
            or ({"geometry_name": "geometry", "crs": "EPSG:4326"}, table)
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("supported relation layers should stay on the pyogrio layer path")
        ),
    )

    payload = read_vector_file_native(
        "example.osm.pbf",
        layer="other_relations",
        tags=False,
        geometry_only=True,
    )

    assert captured["layer"] == "other_relations"
    assert captured["columns"] == []
    assert list(payload.attributes.columns) == []
    assert payload.geometry.row_count == 1


@pytest.mark.gpu
def test_osm_public_read_projects_promoted_columns_and_other_tags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")
    payload = _native_file_result_from_owned(
        from_shapely_geometries(
            [
                Point(0, 0),
                LineString([(0, 0), (1, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        ),
        crs="EPSG:4326",
        attributes=pd.DataFrame(
            {
                "osm_element": ["node", "way", "relation"],
                "osm_id": [100, 200, 300],
                "name": ["Cafe", None, None],
                "highway": [None, "primary", None],
                "landuse": [None, None, "residential"],
                "other_tags": ['"railway"=>"halt"', '"custom_way"=>"y"', '"custom_rel"=>"z"'],
            }
        ),
    )

    class _FakeRuntime:
        compute_capability = (8, 9)

        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_supported_layers_native",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default public OSM read should use supported layers helper")
        ),
    )

    result = geopandas.read_file(path)

    assert result["osm_element"].tolist() == ["node", "way", "relation"]
    assert result["osm_id"].tolist() == [100, 200, 300]
    assert result["name"].iloc[0] == "Cafe"
    assert pd.isna(result["name"].iloc[1])
    assert pd.isna(result["name"].iloc[2])
    assert pd.isna(result["highway"].iloc[0])
    assert result["highway"].iloc[1] == "primary"
    assert pd.isna(result["highway"].iloc[2])
    assert pd.isna(result["landuse"].iloc[0])
    assert pd.isna(result["landuse"].iloc[1])
    assert result["landuse"].iloc[2] == "residential"
    assert result["other_tags"].tolist() == [
        '"railway"=>"halt"',
        '"custom_way"=>"y"',
        '"custom_rel"=>"z"',
    ]
    assert "custom_way" not in result.columns
    assert "custom_rel" not in result.columns


@pytest.mark.gpu
def test_osm_read_forwards_tags_and_geometry_only_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")

    captured: dict[str, object] = {}
    payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0)]),
        crs="EPSG:4326",
        attributes=pd.DataFrame(index=pd.RangeIndex(1)),
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    def _fake_supported_layers(*args, **kwargs):
        captured.update(kwargs)
        return payload

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("vibespatial.io.file._read_osm_pbf_supported_layers_native", _fake_supported_layers)
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default geometry-only OSM read should use supported layers helper")
        ),
    )

    result = geopandas.read_file(path, tags=False, geometry_only=True)

    assert list(result.columns) == ["geometry"]
    assert captured == {"target_crs": None, "geometry_only": True, "tags": False}


@pytest.mark.gpu
def test_osm_public_read_supports_points_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")
    table = pa.table(
        {
            "geometry": [Point(0, 0).wkb],
            "osm_id": [100],
            "name": ["Cafe"],
            "other_tags": [None],
        }
    )
    owned = from_shapely_geometries([Point(0, 0)])

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    result = geopandas.read_file(path, layer="points")

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result["osm_id"].tolist() == [100]
    assert "osm_way_id" not in result.columns
    assert "osm_element" not in result.columns


@pytest.mark.gpu
def test_osm_public_read_supports_multipolygons_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(b"")
    table = pa.table(
        {
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]).wkb,
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]).wkb,
            ],
            "osm_id": [None, 300],
            "osm_way_id": [201, None],
            "landuse": ["forest", "residential"],
            "other_tags": ['"custom_way"=>"y"', None],
        }
    )
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ]
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    result = geopandas.read_file(path, layer="multipolygons")

    assert len(result) == 2
    assert sorted(result["osm_way_id"].dropna().astype(int).tolist()) == [201]
    assert sorted(result["osm_id"].dropna().astype(int).tolist()) == [300]
    assert sorted(result["landuse"].dropna().tolist()) == ["forest", "residential"]
    assert sorted(result["other_tags"].dropna().tolist()) == ['"custom_way"=>"y"']


@pytest.mark.gpu
def test_osm_native_bundle_builds_stable_partition_results() -> None:
    import cupy as cp

    from vibespatial.io.osm_bundle import build_osm_native_bundle

    class _FakeOsmResult:
        nodes = from_shapely_geometries([Point(0, 0)])
        node_ids = cp.asarray([100], dtype=cp.int64)
        n_nodes = 1
        node_tags = [{"name": "Cafe", "railway": "halt"}]
        ways = from_shapely_geometries([LineString([(0, 0), (1, 1)])])
        way_ids = cp.asarray([200], dtype=cp.int64)
        n_ways = 1
        way_tags = [{"name": "Main", "highway": "primary", "custom_way": "y"}]
        relations = from_shapely_geometries([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])])
        relation_ids = cp.asarray([300], dtype=cp.int64)
        n_relations = 1
        relation_tags = [{"type": "multipolygon", "landuse": "residential", "custom_rel": "z"}]

    bundle = build_osm_native_bundle(_FakeOsmResult(), crs="EPSG:4326", source="sample.osm.pbf")

    assert bundle.full_counts() == (1, 1, 1)

    points = bundle.points.result.to_geodataframe()
    ways = bundle.ways.result.to_geodataframe()
    relations = bundle.relations.result.to_geodataframe()

    assert points["osm_element"].tolist() == ["node"]
    assert points["osm_id"].tolist() == [100]
    assert points["name"].tolist() == ["Cafe"]
    assert points["other_tags"].tolist() == ['"railway"=>"halt"']

    assert ways["osm_element"].tolist() == ["way"]
    assert ways["osm_id"].tolist() == [200]
    assert ways["highway"].tolist() == ["primary"]
    assert ways["other_tags"].tolist() == ['"custom_way"=>"y"']

    assert relations["osm_element"].tolist() == ["relation"]
    assert relations["osm_id"].tolist() == [300]
    assert relations["landuse"].tolist() == ["residential"]
    assert relations["other_tags"].tolist() == ['"custom_rel"=>"z"']


@pytest.mark.gpu
def test_osm_native_bundle_projects_public_compatibility_views() -> None:
    import cupy as cp

    from vibespatial.io.osm_bundle import build_osm_native_bundle

    class _FakeOsmResult:
        nodes = None
        node_ids = None
        n_nodes = 0
        node_tags = None
        ways = from_shapely_geometries(
            [
                LineString([(0, 0), (1, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ]
        )
        way_ids = cp.asarray([200, 201], dtype=cp.int64)
        n_ways = 2
        way_tags = [
            {"highway": "primary"},
            {"landuse": "forest", "custom_way": "y"},
        ]
        relations = from_shapely_geometries([Polygon([(4, 4), (5, 4), (5, 5), (4, 4)])])
        relation_ids = cp.asarray([300], dtype=cp.int64)
        n_relations = 1
        relation_tags = [{"type": "multipolygon", "landuse": "residential"}]

    bundle = build_osm_native_bundle(_FakeOsmResult(), crs="EPSG:4326", source="sample.osm.pbf")

    multipolygons = bundle.to_native_tabular_result(layer="multipolygons").to_geodataframe()
    assert sorted(multipolygons["osm_way_id"].dropna().astype(int).tolist()) == [201]
    assert sorted(multipolygons["osm_id"].dropna().astype(int).tolist()) == [300]
    assert sorted(multipolygons["landuse"].dropna().tolist()) == ["forest", "residential"]
    assert sorted(multipolygons["other_tags"].dropna().tolist()) == ['"custom_way"=>"y"']

    ways_only_bundle = build_osm_native_bundle(
        type(
            "_WaysOnlyResult",
            (),
            {
                "nodes": None,
                "node_ids": None,
                "n_nodes": 0,
                "node_tags": None,
                "ways": _FakeOsmResult.ways,
                "way_ids": _FakeOsmResult.way_ids,
                "n_ways": _FakeOsmResult.n_ways,
                "way_tags": _FakeOsmResult.way_tags,
                "relations": None,
                "relation_ids": None,
                "n_relations": 0,
                "relation_tags": None,
            },
        )(),
        crs="EPSG:4326",
        source="sample.osm.pbf",
    )
    all_view = ways_only_bundle.to_native_tabular_result(layer="all").to_geodataframe()
    assert "osm_element" not in all_view.columns
    assert "osm_way_id" in all_view.columns
    assert "osm_id" not in all_view.columns


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_points_layer_prefers_pyogrio_arrow_gpu_wkb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "geometry": [Point(0, 0).wkb],
            "osm_id": [100],
            "name": ["Cafe"],
            "other_tags": [None],
        }
    )
    owned = from_shapely_geometries([Point(0, 0)])
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **kwargs: (
            captured.update(kwargs)
            or ({"geometry_name": "geometry", "crs": "EPSG:4326"}, table)
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standard OSM layers should not use the native parser")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file("example.osm.pbf", layer="points")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["osm_id"].tolist() == [100]
    assert captured["layer"] == "points"
    assert any(event.implementation.endswith("_pyogrio_arrow_gpu_wkb") for event in events)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_points_layer_geometry_only_maps_to_pyogrio_geometry_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"geometry": [Point(0, 0).wkb]})
    owned = from_shapely_geometries([Point(0, 0)])
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **kwargs: (
            captured.update(kwargs)
            or ({"geometry_name": "geometry", "crs": "EPSG:4326"}, table)
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standard OSM layers should not use the native parser")
        ),
    )

    payload = read_vector_file_native("example.osm.pbf", layer="points", geometry_only=True)

    assert captured["layer"] == "points"
    assert captured["columns"] == []
    assert list(payload.attributes.columns) == []


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_points_layer_tags_false_projects_only_osm_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"geometry": [Point(0, 0).wkb], "osm_id": [100]})
    owned = from_shapely_geometries([Point(0, 0)])
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **kwargs: (
            captured.update(kwargs)
            or ({"geometry_name": "geometry", "crs": "EPSG:4326"}, table)
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("supported OSM point layer with tags=False should stay on the pyogrio layer path")
        ),
    )

    result = geopandas.read_file("example.osm.pbf", layer="points", tags=False)

    assert captured["layer"] == "points"
    assert captured["columns"] == ["osm_id"]
    assert list(result.columns) == ["osm_id", "geometry"]
    assert result["osm_id"].tolist() == [100]


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_public_multipolygons_without_osm_way_id_stays_relation_shaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geopandas as gpd

    frame = gpd.GeoDataFrame(
        {"osm_id": [300]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
        crs="EPSG:4326",
    )

    monkeypatch.setattr("pyogrio.read_dataframe", lambda *_args, **_kwargs: frame.copy())

    result = _read_osm_pbf_pyogrio_layer_public(
        "example.osm.pbf",
        layer="multipolygons",
        include_element_column=True,
    )

    assert result["osm_element"].tolist() == ["relation"]
    assert result["osm_id"].tolist() == [300]


def test_osm_points_layer_cpu_fallback_uses_pyogrio_compatibility_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_frame = geopandas.GeoDataFrame(
        {"osm_id": [100]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    class _NoGpuRuntime:
        def available(self) -> bool:
            return False

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _NoGpuRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("vibespatial.api.io.file._read_file", lambda *_args, **_kwargs: fallback_frame)

    result = read_vector_file("example.osm.pbf", layer="points")

    assert result.equals(fallback_frame)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_osm_other_relations_layer_uses_explicit_compatibility_bridge_for_geometrycollection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

    from vibespatial.io.file import _read_osm_pbf_pyogrio_layer_native

    metadata = {"geometry_name": "geometry", "crs": "EPSG:4326"}
    table = pa.table({"geometry": pa.array([b"010100000000000000000000000000000000000000"], type=pa.binary())})
    fallback_frame = geopandas.GeoDataFrame(
        {"osm_id": [300], "osm_element": ["relation"]},
        geometry=[GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])])],
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *args, **kwargs: (metadata, table),
    )
    monkeypatch.setattr(
        "vibespatial.io.file._pyogrio_arrow_wkb_to_native_tabular_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError("unsupported geometry family: GeometryCollection")
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_pyogrio_layer_public",
        lambda *args, **kwargs: fallback_frame,
    )

    geopandas.clear_fallback_events()
    payload = _read_osm_pbf_pyogrio_layer_native("example.osm.pbf", layer="other_relations")
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.row_count == 1
    assert payload.to_geodataframe()["osm_element"].tolist() == ["relation"]
    assert fallbacks
    assert fallbacks[-1].surface == "vibespatial.io.osm_pbf"
    assert "GeometryCollection" in (fallbacks[-1].detail or "")


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_default_public_read_prefers_supported_layers_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0), LineString([(0, 0), (1, 1)])]),
        crs="EPSG:4326",
        attributes=pd.DataFrame(
            {
                "osm_element": ["node", "way"],
                "osm_id": [100, 200],
            }
        ),
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_supported_layers_native",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default OSM read should not hit the full native parser")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file("example.osm.pbf")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["osm_element"].tolist() == ["node", "way"]
    assert any(event.implementation == "osm_pbf_pyogrio_supported_layers_gpu_wkb" for event in events)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
@pytest.mark.gpu
def test_osm_default_public_read_tags_false_prefers_supported_layers_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0), LineString([(0, 0), (1, 1)])]),
        crs="EPSG:4326",
        attributes=pd.DataFrame(
            {
                "osm_element": ["node", "way"],
                "osm_id": [100, 200],
            }
        ),
    )
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    def _fake_supported_layers(*_args, **kwargs):
        captured.update(kwargs)
        return payload

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_supported_layers_native",
        _fake_supported_layers,
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_osm_pbf_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default tags=False OSM read should not hit the full native parser")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file("example.osm.pbf", tags=False)
    events = geopandas.get_dispatch_events(clear=True)

    assert captured == {"target_crs": None, "geometry_only": False, "tags": False}
    assert result["osm_element"].tolist() == ["node", "way"]
    assert any(event.implementation == "osm_pbf_pyogrio_supported_layers_gpu_wkb" for event in events)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_osm_default_read_without_gpu_uses_supported_layers_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_frame = geopandas.GeoDataFrame(
        {"osm_element": ["node"], "osm_id": [100]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    class _NoGpuRuntime:
        def available(self) -> bool:
            return False

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _NoGpuRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_supported_layers_public",
        lambda *_args, **_kwargs: fallback_frame,
    )

    result = read_vector_file("example.osm.pbf")

    assert result.equals(fallback_frame)


@pytest.mark.gpu
def test_shapefile_roundtrip_uses_gpu_adapter(tmp_path) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    path = tmp_path / "sample.shp"

    frame = _sample_frame()
    frame.to_file(path, driver="ESRI Shapefile")
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == len(frame)
    assert result.geometry.iloc[1].equals(frame.geometry.iloc[1])
    assert "gpu" in events[-1].implementation.lower()
    assert not fallbacks


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_pyogrio_shapefile_write_passes_explicit_geometry_type_for_null_rows(monkeypatch, tmp_path) -> None:
    import pyogrio

    owned = from_shapely_geometries([None, LineString([(0, 0), (1, 1)])])
    geometry = geopandas.GeoSeries(DeviceGeometryArray._from_owned(owned), crs="EPSG:4326")
    frame = geopandas.GeoDataFrame({"name": ["null", "line"], "geometry": geometry}, crs="EPSG:4326")
    path = tmp_path / "null_geom.shp"
    captured: dict[str, object] = {}
    real_write_arrow = pyogrio.write_arrow

    def capture_write_arrow(arrow_obj, filename, *args, **kwargs):
        captured["geometry_type"] = kwargs.get("geometry_type")
        return real_write_arrow(arrow_obj, filename, *args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_arrow", capture_write_arrow)

    write_vector_file(
        _spatial_to_native_tabular_result(frame),
        path,
        driver="ESRI Shapefile",
        engine="pyogrio",
    )
    result = geopandas.read_file(path, engine="pyogrio")

    assert captured["geometry_type"] == "LineString"
    assert frame.geometry.dtype.name == "device_geometry"
    assert len(result) == 2
    assert result.geometry.iloc[0] is None
    assert result.geometry.iloc[1].equals(LineString([(0, 0), (1, 1)]))


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_pyogrio_native_write_uses_write_arrow_not_write_dataframe(monkeypatch, tmp_path) -> None:
    import pyogrio

    geometry = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(from_shapely_geometries([Point(0, 0), Point(1, 1)])),
        crs="EPSG:4326",
    )
    frame = geopandas.GeoDataFrame({"value": [10, 20], "geometry": geometry}, crs="EPSG:4326")
    path = tmp_path / "native.geojson"
    payload = _spatial_to_native_tabular_result(frame)
    arrow_calls = 0
    real_write_arrow = pyogrio.write_arrow

    def fail_write_dataframe(*_args, **_kwargs):
        raise AssertionError("pyogrio file export should use write_arrow via the native boundary")

    def capture_write_arrow(arrow_obj, filename, *args, **kwargs):
        nonlocal arrow_calls
        arrow_calls += 1
        return real_write_arrow(arrow_obj, filename, *args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_dataframe", fail_write_dataframe)
    monkeypatch.setattr(pyogrio, "write_arrow", capture_write_arrow)

    write_vector_file(payload, path, driver="GeoJSON", engine="pyogrio")
    result = geopandas.read_file(path, engine="pyogrio")

    assert arrow_calls == 1
    assert result["value"].tolist() == [10, 20]
    assert result.geometry.iloc[1].equals(Point(1, 1))


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_pyogrio_to_file_uses_compat_writer(monkeypatch, tmp_path) -> None:
    import pyogrio

    frame = geopandas.GeoDataFrame(
        {"value": [10, 20], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    path = tmp_path / "public.geojson"
    dataframe_calls = 0
    real_write_dataframe = pyogrio.write_dataframe

    def fail_write_arrow(*_args, **_kwargs):
        raise AssertionError("public GeoDataFrame.to_file should stay on the compatibility writer")

    def capture_write_dataframe(*args, **kwargs):
        nonlocal dataframe_calls
        dataframe_calls += 1
        return real_write_dataframe(*args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_arrow", fail_write_arrow)
    monkeypatch.setattr(pyogrio, "write_dataframe", capture_write_dataframe)

    frame.to_file(path, driver="GeoJSON", engine="pyogrio")
    result = geopandas.read_file(path, engine="pyogrio")

    assert dataframe_calls == 1
    assert result["value"].tolist() == [10, 20]
    assert result.geometry.iloc[1].equals(Point(1, 1))


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_pyogrio_duplicate_columns_raises_upstream_message(monkeypatch, tmp_path) -> None:
    import pyogrio

    frame = geopandas.GeoDataFrame(
        data=[[1, 2, 3]],
        columns=["a", "b", "a"],
        geometry=[Point(1, 1)],
        crs="EPSG:4326",
    )
    path = tmp_path / "duplicate.geojson"

    def fail_write_arrow(*_args, **_kwargs):
        raise AssertionError("public GeoDataFrame.to_file should not route duplicate-column cases through native write_arrow")

    monkeypatch.setattr(pyogrio, "write_arrow", fail_write_arrow)

    with pytest.raises(ValueError, match="GeoDataFrame cannot contain duplicated column names"):
        frame.to_file(path, driver="GeoJSON", engine="pyogrio")


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_pyogrio_write_preserves_geometry_alignment_for_sparse_indexes(tmp_path) -> None:
    import pandas as pd

    geometry = geopandas.GeoSeries([Point(i, i) for i in range(10)], crs="EPSG:4326")
    frame = geopandas.GeoDataFrame({"geometry": geometry})
    frame["data"] = pd.array([1, None] * 5, dtype=pd.Int32Dtype())
    filtered = frame.dropna()
    path = tmp_path / "sparse-index.gmt"

    filtered.to_file(path, driver="OGR_GMT", engine="pyogrio")
    result = geopandas.read_file(path, engine="pyogrio")

    assert len(result) == len(filtered)
    assert result["data"].tolist() == [1] * len(filtered)
    for left, right in zip(filtered.geometry, result.geometry, strict=True):
        assert right.equals(left)


def test_plan_shapefile_ingest_prefers_direct_gpu_owned_path() -> None:
    plan = plan_shapefile_ingest()

    assert plan.implementation == "shapefile_shp_direct_owned"
    assert plan.selected_strategy == "shp-direct-gpu"
    assert plan.uses_pyogrio_container is False
    assert plan.uses_native_wkb_decode is False


def test_plan_shapefile_ingest_supports_arrow_wkb_fallback() -> None:
    plan = plan_shapefile_ingest(prefer="arrow-wkb")

    assert plan.implementation == "shapefile_arrow_wkb_owned"
    assert plan.selected_strategy == "arrow-wkb"
    assert plan.uses_pyogrio_container is True
    assert plan.uses_native_wkb_decode is True


def test_read_shapefile_owned_decodes_geometry_and_preserves_attributes(tmp_path) -> None:
    path = tmp_path / "owned.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [10, 20],
            "label": ["a", "b"],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    frame.to_file(path, driver="ESRI Shapefile")

    batch = read_shapefile_owned(path)

    assert batch.geometry.row_count == 2
    assert batch.geometry.to_shapely()[1].equals(Point(1, 1))
    assert batch.attributes_table.column("value").to_pylist() == [10, 20]


def test_read_shapefile_owned_points_use_raw_arrow_fast_path(monkeypatch, tmp_path) -> None:
    path = tmp_path / "owned-fast.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    frame.to_file(path, driver="ESRI Shapefile")

    import vibespatial.io.arrow as io_arrow

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("generic WKB decode should not be used for raw Arrow point fast path")

    monkeypatch.setattr(io_arrow, "decode_wkb_owned", fail_decode)

    batch = read_shapefile_owned(path)

    assert batch.geometry.to_shapely()[1].equals(Point(1, 1))


def test_read_shapefile_owned_supports_polygon_batches(tmp_path) -> None:
    path = tmp_path / "owned-polygons.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ],
        }
    )
    frame.to_file(path, driver="ESRI Shapefile")

    batch = read_shapefile_owned(path)

    materialized = batch.geometry.to_shapely()
    assert materialized[0].equals(frame.geometry.iloc[0])
    assert materialized[1].equals(frame.geometry.iloc[1])


def test_read_shapefile_owned_lines_use_raw_arrow_fast_path(monkeypatch, tmp_path) -> None:
    path = tmp_path / "owned-lines.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
            ],
        }
    )
    frame.to_file(path, driver="ESRI Shapefile")

    import vibespatial.io.arrow as io_arrow

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("generic WKB decode should not be used for raw Arrow linestring fast path")

    monkeypatch.setattr(io_arrow, "decode_wkb_owned", fail_decode)

    batch = read_shapefile_owned(path)

    assert batch.geometry.to_shapely()[0].equals(frame.geometry.iloc[0])


def test_read_shapefile_owned_polygons_use_raw_arrow_fast_path(monkeypatch, tmp_path) -> None:
    path = tmp_path / "owned-polygons-fast.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ],
        }
    )
    frame.to_file(path, driver="ESRI Shapefile")

    import vibespatial.io.arrow as io_arrow

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("generic WKB decode should not be used for raw Arrow polygon fast path")

    monkeypatch.setattr(io_arrow, "decode_wkb_owned", fail_decode)

    batch = read_shapefile_owned(path)

    assert batch.geometry.to_shapely()[1].equals(frame.geometry.iloc[1])


def test_read_shapefile_owned_prefers_direct_gpu_native_path(monkeypatch: pytest.MonkeyPatch) -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    payload = _native_file_result_from_owned(
        owned,
        crs="EPSG:4326",
        attributes=pd.DataFrame({"value": [10, 20]}),
        row_count=2,
    )

    monkeypatch.setattr(
        "vibespatial.io.file._try_shapefile_shp_direct_gpu_read_native",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct Shapefile owned read should not call pyogrio.read_arrow")
        ),
    )

    batch = read_shapefile_owned("sample.shp")

    assert batch.geometry.row_count == 2
    assert batch.attributes_table.column("value").to_pylist() == [10, 20]


def test_read_shapefile_owned_uses_arrow_wkb_fallback_for_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    table = pa.table({"value": [10], "geometry": [Point(0, 0).wkb]})
    owned = from_shapely_geometries([Point(0, 0)])

    monkeypatch.setattr(
        "vibespatial.io.file._try_shapefile_shp_direct_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bbox reads should bypass the direct Shapefile GPU path")
        ),
    )
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    batch = read_shapefile_owned("sample.shp", bbox=(0.0, 0.0, 1.0, 1.0))

    assert batch.geometry.row_count == 1
    assert batch.attributes_table.column("value").to_pylist() == [10]


def test_pyogrio_arrow_wkb_native_result_preserves_index_and_defers_public_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "value": [10, 20],
                "geometry": [Point(0, 0).wkb, Point(1, 1).wkb],
            },
            index=pd.Index(["left", "right"], name="row_id"),
        ),
        preserve_index=True,
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])

    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native file read bridge should not materialize a GeoDataFrame")
        ),
    )

    payload = _pyogrio_arrow_wkb_to_native_tabular_result(
        table,
        {"geometry_name": "geometry", "crs": "EPSG:4326"},
    )

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.owned is owned
    assert payload.geometry_name == "geometry"
    assert list(payload.attributes.columns) == ["value"]
    assert payload.attributes.index.tolist() == ["left", "right"]


def test_pyogrio_arrow_wkb_native_result_labels_target_crs_when_source_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"geometry": [Point(0, 0).wkb]})
    owned = from_shapely_geometries([Point(0, 0)])

    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    payload = _pyogrio_arrow_wkb_to_native_tabular_result(
        table,
        {"geometry_name": "geometry"},
        target_crs="EPSG:3857",
    )

    assert payload.geometry.crs == "EPSG:3857"


def test_pyogrio_arrow_wkb_native_result_normalizes_empty_geometry_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = pa.field(
        "wkb_geometry",
        pa.binary(),
        metadata={b"ARROW:extension:name": b"geoarrow.wkb"},
    )
    table = pa.Table.from_arrays(
        [pa.array([Point(0, 0).wkb], type=pa.binary())],
        schema=pa.schema([field]),
    )
    owned = from_shapely_geometries([Point(0, 0)])

    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    payload = _pyogrio_arrow_wkb_to_native_tabular_result(
        table,
        {"geometry_name": "", "crs": "EPSG:4326"},
    )

    assert payload.geometry_name == "geometry"


def test_pyogrio_arrow_wkb_native_result_normalizes_nonstandard_geometry_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = pa.field(
        "geom",
        pa.binary(),
        metadata={b"ARROW:extension:name": b"geoarrow.wkb"},
    )
    table = pa.Table.from_arrays(
        [pa.array([Point(0, 0).wkb], type=pa.binary())],
        schema=pa.schema([field]),
    )
    owned = from_shapely_geometries([Point(0, 0)])

    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    payload = _pyogrio_arrow_wkb_to_native_tabular_result(
        table,
        {"geometry_name": "geom", "crs": "EPSG:4326"},
    )

    assert payload.geometry_name == "geometry"


def test_native_file_result_from_owned_accepts_dict_attributes_without_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])

    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("internal native file helper should not materialize a GeoDataFrame")
        ),
    )

    payload = _native_file_result_from_owned(
        owned,
        crs="EPSG:4326",
        attributes={"value": [10, 20]},
    )

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.owned is owned
    assert list(payload.attributes.columns) == ["value"]
    assert payload.attributes["value"].tolist() == [10, 20]


def test_plan_vector_file_io_exposes_promoted_read_boundary_classification() -> None:
    assert plan_vector_file_io("data.geojson", operation=IOOperation.READ).selected_path is IOPathKind.HYBRID
    assert plan_vector_file_io("data.shp", operation=IOOperation.READ).selected_path is IOPathKind.HYBRID
    assert plan_vector_file_io("data.gpkg", operation=IOOperation.READ).selected_path is IOPathKind.HYBRID
    assert plan_vector_file_io("data.unknown", operation=IOOperation.READ).selected_path is IOPathKind.FALLBACK


def test_gpkg_routes_through_legacy_host_adapter(tmp_path) -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    path = tmp_path / "sample.gpkg"

    frame = _sample_frame()
    frame.to_file(path, driver="GPKG")
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == len(frame)
    # GPKG now routes through pyogrio Arrow + GPU WKB decode
    assert any("gpu" in e.implementation.lower() for e in events)
    assert not fallbacks


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_read_vector_file_native_returns_native_payload_for_pyogrio_arrow_gpu_wkb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"geometry": [Point(0, 0).wkb], "value": [10]})
    owned = from_shapely_geometries([Point(0, 0)])

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native vector read should not materialize a GeoDataFrame")
        ),
    )

    payload = read_vector_file_native("example.gpkg")

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.owned is owned
    assert payload.attributes["value"].tolist() == [10]


def test_read_vector_file_native_lowers_cpu_path_to_shared_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_frame = geopandas.GeoDataFrame(
        {"value": [10]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    monkeypatch.setattr("vibespatial.api.io.file._read_file", lambda *_args, **_kwargs: fallback_frame)
    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CPU fallback lowering should still stop at the native boundary")
        ),
    )

    payload = read_vector_file_native("example.gpkg", engine="pyogrio")

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.row_count == 1
    assert payload.geometry_name == "geometry"
    assert payload.attributes["value"].tolist() == [10]


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_gpu_arrow_wkb_read_records_explicit_fallback_before_cpu_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def available(self) -> bool:
            return True

    fallback_frame = geopandas.GeoDataFrame(
        {"value": [10]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    table = pa.table({"geometry": [Point(0, 0).wkb]})

    geopandas.clear_fallback_events()
    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr("vibespatial.api.io.file._read_file", lambda *_args, **_kwargs: fallback_frame)

    result = read_vector_file("example.gpkg")
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert result.equals(fallback_frame)
    assert fallbacks
    assert fallbacks[-1].surface == "geopandas.read_file"
    assert "GPU-dominant file read failed" in fallbacks[-1].reason


def test_read_geojson_owned_streams_feature_collection_to_owned_buffers(tmp_path) -> None:
    path = tmp_path / "sample.geojson"
    frame = geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
        }
    )
    path.write_text(frame.to_json())

    batch = read_geojson_owned(path)

    assert batch.properties[0]["value"] == 10
    assert batch.geometry.row_count == 2
    assert batch.geometry.to_shapely()[0].equals(Point(0, 0))
    assert batch.geometry.to_shapely()[1].equals(frame.geometry.iloc[1])


def test_read_geojson_native_returns_shared_native_boundary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "native.geojson"
    frame = geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    path.write_text(frame.to_json())

    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native GeoJSON read should not materialize a GeoDataFrame")
        ),
    )

    payload = read_geojson_native(path, prefer="fast-json")

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.row_count == 2
    assert payload.geometry.crs == "EPSG:4326"
    assert payload.attributes["value"].tolist() == [10, 20]


def test_read_geojson_native_without_properties_returns_geometry_only_payload(tmp_path) -> None:
    path = tmp_path / "native-geometry-only.geojson"
    frame = geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    path.write_text(frame.to_json())

    payload = read_geojson_native(path, prefer="fast-json", track_properties=False)
    materialized = payload.to_geodataframe()

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.row_count == 2
    assert tuple(payload.attributes.columns) == ()
    assert list(materialized.columns) == ["geometry"]


def test_geojson_gpu_native_boundary_keeps_properties_lazy_until_materialization() -> None:
    load_calls = 0
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])

    class _FakeGpuResult:
        def __init__(self) -> None:
            self.owned = owned
            self.n_features = 2

        def properties_loader(self):
            def _load():
                nonlocal load_calls
                load_calls += 1
                return [{"value": 10}, {"value": 20}]

            return _load

    payload = _native_geojson_result_from_gpu_result(_FakeGpuResult())

    assert isinstance(payload, NativeTabularResult)
    assert load_calls == 0
    assert payload.geometry.row_count == 2
    assert payload.take([0]).geometry.row_count == 1
    assert load_calls == 0

    frame = payload.to_geodataframe()

    assert load_calls == 1
    assert list(frame.columns) == ["value", "geometry"]
    assert frame["value"].tolist() == [10, 20]


def test_plan_geojson_ingest_auto_selects_best_available() -> None:
    from vibespatial.runtime._runtime import has_gpu_runtime

    plan = plan_geojson_ingest()

    if has_gpu_runtime():
        assert plan.selected_strategy == "gpu-byte-classify"
        assert plan.implementation == "geojson_gpu_byte_classify"
    else:
        assert plan.selected_strategy == "fast-json"
        assert plan.implementation == "geojson_fast_json_vectorized"
    assert plan.uses_stream_tokenizer is False
    assert plan.uses_native_geometry_assembly is True


def test_plan_geojson_ingest_explicit_fast_json() -> None:
    """Explicit prefer='fast-json' always selects the CPU fast-json path."""
    plan = plan_geojson_ingest(prefer="fast-json")

    assert plan.implementation == "geojson_fast_json_vectorized"
    assert plan.selected_strategy == "fast-json"
    assert plan.uses_stream_tokenizer is False
    assert plan.uses_native_geometry_assembly is True


def test_plan_geojson_ingest_supports_pylibcudf_strategy() -> None:
    plan = plan_geojson_ingest(prefer="pylibcudf")

    assert plan.implementation == "geojson_pylibcudf_native"
    assert plan.selected_strategy == "pylibcudf"
    assert plan.uses_pylibcudf is True


def test_plan_geojson_ingest_supports_experimental_array_strategy() -> None:
    plan = plan_geojson_ingest(prefer="pylibcudf-arrays")

    assert plan.implementation == "geojson_pylibcudf_arrays_experimental"
    assert plan.selected_strategy == "pylibcudf-arrays"
    assert plan.uses_pylibcudf is True


def test_plan_geojson_ingest_supports_experimental_rowized_strategy() -> None:
    plan = plan_geojson_ingest(prefer="pylibcudf-rowized")

    assert plan.implementation == "geojson_pylibcudf_rowized_experimental"
    assert plan.selected_strategy == "pylibcudf-rowized"
    assert plan.uses_pylibcudf is True


def test_read_geojson_owned_supports_stream_strategy(tmp_path) -> None:
    path = tmp_path / "stream.geojson"
    frame = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
    path.write_text(frame.to_json())

    batch = read_geojson_owned(path, prefer="stream")

    assert batch.geometry.to_shapely()[1].equals(Point(1, 1))


def test_read_shapefile_native_returns_shared_native_boundary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "native-shapefile.shp"
    frame = geopandas.GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    frame.to_file(path, driver="ESRI Shapefile")

    monkeypatch.setattr(
        NativeTabularResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native Shapefile read should not materialize a GeoDataFrame")
        ),
    )

    payload = read_shapefile_native(path)

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.row_count == 2
    assert payload.geometry_name == "geometry"
    assert payload.geometry.crs is not None
    assert payload.attributes["value"].tolist() == [10, 20]


def test_read_geojson_owned_supports_pylibcudf_strategy(tmp_path) -> None:
    if not io_geojson._has_pylibcudf_geojson_support():
        return
    path = tmp_path / "gpu.geojson"
    frame = geopandas.GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [Point(0, 0), None, Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
        }
    )
    path.write_text(frame.to_json())

    batch = read_geojson_owned(path, prefer="pylibcudf")

    assert batch.properties[0]["value"] == 10
    assert batch.properties[1]["value"] == 20
    assert batch.geometry.row_count == 3
    materialized = batch.geometry.to_shapely()
    assert materialized[0].equals(Point(0, 0))
    assert materialized[1] is None
    assert materialized[2].equals(frame.geometry.iloc[2])


def test_read_geojson_owned_pylibcudf_default_does_not_use_experimental_rowizer(
    monkeypatch, tmp_path
) -> None:
    if not io_geojson._has_pylibcudf_geojson_support():
        return
    path = tmp_path / "gpu-default.geojson"
    frame = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
    path.write_text(frame.to_json())

    def fail_rowizer(_text: str):
        raise AssertionError("default pylibcudf path should not invoke the experimental rowizer")

    monkeypatch.setattr(io_geojson, "_read_geojson_owned_pylibcudf_rowized", fail_rowizer)

    batch = read_geojson_owned(path, prefer="pylibcudf")

    assert batch.geometry.to_shapely()[1].equals(Point(1, 1))


def test_read_geojson_owned_experimental_arrays_can_bypass_host_splitter_for_points(
    monkeypatch, tmp_path
) -> None:
    if not io_geojson._has_pylibcudf_geojson_support():
        return
    path = tmp_path / "gpu-arrays.geojson"
    frame = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]})
    path.write_text(frame.to_json())

    def fail_splitter(_text: str):
        raise AssertionError("host feature splitter should not be used for wildcard-array GeoJSON")

    monkeypatch.setattr(io_geojson, "_feature_collection_spans", fail_splitter)

    batch = read_geojson_owned(path, prefer="pylibcudf-arrays")

    assert batch.geometry.to_shapely()[2].equals(Point(2, 2))


def test_read_geojson_owned_experimental_arrays_support_polygons(monkeypatch, tmp_path) -> None:
    if not io_geojson._has_pylibcudf_geojson_support():
        return
    path = tmp_path / "gpu-polygons.geojson"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ]
        }
    )
    path.write_text(frame.to_json())

    def fail_splitter(_text: str):
        raise AssertionError("host feature splitter should not be used for wildcard-array GeoJSON")

    monkeypatch.setattr(io_geojson, "_feature_collection_spans", fail_splitter)

    batch = read_geojson_owned(path, prefer="pylibcudf-arrays")

    materialized = batch.geometry.to_shapely()
    assert materialized[0].equals(frame.geometry.iloc[0])
    assert materialized[1].equals(frame.geometry.iloc[1])


def test_read_geojson_owned_experimental_rowized_can_bypass_host_splitter_for_points(
    monkeypatch, tmp_path
) -> None:
    if not io_geojson._has_pylibcudf_geojson_support():
        return
    path = tmp_path / "gpu-points.geojson"
    frame = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]})
    path.write_text(frame.to_json())

    def fail_splitter(_text: str):
        raise AssertionError("host feature splitter should not be used for homogeneous point GeoJSON")

    monkeypatch.setattr(io_geojson, "_feature_collection_spans", fail_splitter)

    batch = read_geojson_owned(path, prefer="pylibcudf-rowized")

    assert batch.geometry.to_shapely()[2].equals(Point(2, 2))


def test_read_geojson_owned_supports_tokenizer_strategy(tmp_path) -> None:
    path = tmp_path / "tokenizer.geojson"
    frame = geopandas.GeoDataFrame(
        {"geometry": [Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]}
    )
    path.write_text(frame.to_json())

    batch = read_geojson_owned(path, prefer="tokenizer")

    assert batch.geometry.row_count == 2
    assert batch.geometry.to_shapely()[0].equals(Point(0, 0))
    assert batch.geometry.to_shapely()[1].equals(frame.geometry.iloc[1])


def test_read_geojson_owned_tokenizer_handles_braces_in_strings() -> None:
    text = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {"label": "brace } text", "quote": "say \\\"hi\\\""},
          "geometry": {"type": "Point", "coordinates": [0, 0]}
        },
        {
          "type": "Feature",
          "properties": {"label": "array [ text"},
          "geometry": {"type": "Point", "coordinates": [1, 1]}
        }
      ]
    }
    """

    batch = read_geojson_owned(text, prefer="tokenizer")

    assert batch.properties[0]["label"] == "brace } text"
    assert batch.properties[0]["quote"] == 'say "hi"'
    assert batch.properties[1]["label"] == "array [ text"
    assert batch.geometry.to_shapely()[1].equals(Point(1, 1))


def test_benchmark_geojson_ingest_reports_all_candidate_paths() -> None:
    results = benchmark_geojson_ingest(geometry_type="point", rows=100, repeat=1)

    expected = {
        "pyogrio_host",
        "full_json_baseline",
        "stream_native",
        "tokenizer_native",
        "fast_json_vectorized",
        "chunked_vectorized",
    }
    if io_geojson._HAS_SIMDJSON:
        expected.add("simdjson_vectorized")
    if io_geojson._has_pylibcudf_geojson_support():
        expected.add("pylibcudf_native")
        expected.add("pylibcudf_arrays_experimental")

    assert {result.implementation for result in results} == expected


def test_benchmark_shapefile_ingest_reports_all_candidate_paths() -> None:
    results = benchmark_shapefile_ingest(geometry_type="point", rows=100, repeat=1)

    assert {result.implementation for result in results} == {
        "pyogrio_host",
        "pyogrio_arrow_container",
        "shapefile_owned_native",
        "native_wkb_decode",
    }
