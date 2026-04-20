from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box

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
    _concat_native_tabular_results,
    _spatial_to_native_tabular_result,
)
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.io.file import (
    _is_remote_named_file_source,
    _materialize_native_file_read_result,
    _native_file_result_from_owned,
    _native_geojson_result_from_gpu_result,
    _pyogrio_arrow_wkb_to_native_tabular_result,
    _read_osm_pbf_pyogrio_layer_public,
    _resolve_named_file_source_path,
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


def _sample_polygon_frame() -> geopandas.GeoDataFrame:
    return geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ],
        },
        crs="EPSG:4326",
    )


def _write_zipped_shapefile(tmp_path: Path) -> Path:
    source_dir = tmp_path / "shapefile"
    source_dir.mkdir()
    shp_path = source_dir / "sample.shp"
    _sample_frame().to_file(shp_path, driver="ESRI Shapefile")
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for child in source_dir.iterdir():
            archive.write(child, arcname=child.name)
    return zip_path


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

    def _boom(*args, **kwargs):
        raise RuntimeError("geojson-gpu-boom")

    monkeypatch.setattr("vibespatial.io.geojson_gpu.read_geojson_gpu", _boom)
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("GeoJSON GPU fallback should stay on the repo-owned fast-json path")
        ),
    )

    geopandas.clear_fallback_events()
    result = geopandas.read_file(path)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert len(result) == len(frame)
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert fallbacks
    assert fallbacks[-1].surface == "geopandas.read_file"
    assert "geojson-gpu-boom" in fallbacks[-1].detail


@pytest.mark.gpu
def test_public_geojson_read_prefers_pipeline_gpu_adapter_even_for_small_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.geojson"
    frame = _sample_frame()
    frame.to_file(path, driver="GeoJSON")

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

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == frame["id"].tolist()
    assert calls
    assert calls[-1]["capture_feature_boundaries"] is True
    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"
    assert "format=geojson" in events[-1].detail
    assert "request=default" in events[-1].detail
    assert "engine=auto" in events[-1].detail


@pytest.mark.gpu
def test_public_explicit_pyogrio_geojson_read_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.geojson"
    frame = _sample_frame()
    frame.to_file(path, driver="GeoJSON")
    payload = _native_file_result_from_owned(
        from_shapely_geometries(frame.geometry.tolist()),
        crs="EPSG:4326",
        attributes=frame.drop(columns="geometry"),
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("vibespatial.io.file.read_geojson_native", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio GeoJSON reads should stay on the native path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == frame["id"].tolist()
    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "compat_override=1" not in events[-1].detail


@pytest.mark.gpu
def test_public_explicit_pyogrio_geojson_json_string_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "value": 10},
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            },
            {
                "type": "Feature",
                "properties": {"id": 2, "value": 20},
                "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
            },
        ],
    }
    payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0), Point(1, 1)]),
        crs="EPSG:4326",
        attributes=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
    )
    seen_sources: list[object] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file.read_geojson_native",
        lambda source, *_args, **_kwargs: seen_sources.append(source) or payload,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-memory explicit pyogrio GeoJSON reads should stay on the native path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(json.dumps(feature_collection), engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert seen_sources and isinstance(seen_sources[-1], str)
    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "source=memory" in events[-1].detail


@pytest.mark.gpu
def test_public_explicit_pyogrio_geojson_bytesio_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            }
        ],
    }
    payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0)]),
        crs="EPSG:4326",
        attributes=pd.DataFrame({"id": [1]}),
    )
    seen_sources: list[object] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file.read_geojson_native",
        lambda source, *_args, **_kwargs: seen_sources.append(source) or payload,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("BytesIO explicit pyogrio GeoJSON reads should stay on the native path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(io.BytesIO(json.dumps(feature_collection).encode("utf-8")), engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1]
    assert seen_sources and isinstance(seen_sources[-1], bytes)
    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "source=memory" in events[-1].detail


@pytest.mark.gpu
def test_public_explicit_pyogrio_tempfile_single_feature_stays_compatibility_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp = tempfile.TemporaryFile()
    temp.write(
        b"""
        {
          "type": "Feature",
          "geometry": {"type": "Point", "coordinates": [0, 0]},
          "properties": {"name": "Null Island"}
        }
        """
    )
    temp.seek(0)

    frame = geopandas.GeoDataFrame({"name": ["Null Island"], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: calls.append(_kwargs) or frame.copy(),
    )
    monkeypatch.setattr(
        "vibespatial.io.file._try_gpu_read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-feature tempfile GeoJSON should stay on the compatibility path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(temp, engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)
    temp.close()

    assert calls
    assert result["name"].tolist() == ["Null Island"]
    assert events
    assert events[-1].implementation == "legacy_gdal_adapter"
    assert "format=gdal-legacy" in events[-1].detail
    assert "source=memory" in events[-1].detail


@pytest.mark.gpu
def test_public_polygon_geojson_read_prefers_pipeline_gpu_adapter(tmp_path) -> None:
    path = tmp_path / "sample-polygons.geojson"
    frame = _sample_polygon_frame()
    frame.to_file(path, driver="GeoJSON")

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == frame["id"].tolist()
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"


@pytest.mark.gpu
def test_public_strict_mixed_geojson_read_uses_gpu_adapter() -> None:
    path = Path("tests/upstream/geopandas/tests/data/overlay/strict/polys_union_False.geojson")
    expected = read_geojson_native(path, prefer="fast-json").to_geodataframe()
    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert events
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"
    assert result.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_frame_equal(
        result.drop(columns="geometry"),
        expected.drop(columns="geometry"),
        check_dtype=False,
        check_like=False,
    )
    assert result.geom_type.tolist() == expected.geom_type.tolist()
    for expected_geom, actual_geom in zip(expected.geometry, result.geometry, strict=True):
        assert expected_geom.equals_exact(actual_geom, 0.0)


@pytest.mark.gpu
def test_public_geojson_read_preserves_null_geometry_features(tmp_path) -> None:
    path = tmp_path / "null-features.geojson"
    path.write_text(
        """
        {
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": {"Name": "Null Geometry"},
              "geometry": null
            },
            {
              "type": "Feature",
              "properties": {"Name": "SF to NY"},
              "geometry": {
                "type": "LineString",
                "coordinates": [[-122.4051293283311, 37.786780113640894], [-73.859832357849271, 40.487594916296196]]
              }
            }
          ]
        }
        """
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert result["Name"].tolist() == ["Null Geometry", "SF to NY"]
    assert result.geometry.iloc[0] is None
    assert result.geometry.iloc[1].geom_type == "LineString"
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"


@pytest.mark.gpu
def test_public_geojson_read_ignores_root_crs_properties_objects_for_feature_count() -> None:
    path = Path("tests/upstream/geopandas/tests/data/null_geom.geojson")

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert result["Name"].tolist() == ["Null Geometry", "SF to NY"]
    assert result.geometry.iloc[0] is None
    assert result.geometry.iloc[1].geom_type == "LineString"
    assert events[-1].implementation == "geojson_gpu_byte_classify_adapter"


@pytest.mark.gpu
def test_large_flatgeobuf_read_prefers_direct_gpu_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.fgb"
    frame = _sample_frame()
    frame.to_file(path, driver="FlatGeobuf")

    monkeypatch.setattr(
        "vibespatial.io.file._try_fgb_gpu_read_native",
        lambda *_args, **_kwargs: _spatial_to_native_tabular_result(frame),
    )
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default FlatGeobuf read should not drop to pyogrio.read_arrow")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert len(result) == len(frame)
    assert set(result["id"]) == set(frame["id"])
    assert tuple(result.total_bounds) == tuple(frame.total_bounds)
    assert events
    assert events[-1].implementation == "flatgeobuf_gpu_direct_decode_adapter"


@pytest.mark.gpu
def test_explicit_pyogrio_flatgeobuf_read_keeps_native_arrow_wkb_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.fgb"
    frame = _sample_frame()
    frame.to_file(path, driver="FlatGeobuf")

    monkeypatch.setattr(
        "vibespatial.io.file._try_fgb_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit engine='pyogrio' should not use the direct FlatGeobuf decoder")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert len(result) == len(frame)
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

    geopandas.clear_dispatch_events()
    payload = read_vector_file_native(
        "example.osm.pbf",
        layer="other_relations",
        tags=False,
        geometry_only=True,
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert captured["layer"] == "other_relations"
    assert captured["columns"] == []
    assert list(payload.attributes.columns) == []
    assert payload.geometry.row_count == 1
    assert any(
        event.implementation.endswith("_pyogrio_arrow_gpu_wkb")
        and event.selected is geopandas.ExecutionMode.GPU
        for event in events
    )


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
def test_osm_other_relations_public_dispatch_reports_cpu_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_frame = geopandas.GeoDataFrame(
        {"osm_id": [300], "osm_element": ["relation"]},
        geometry=[GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])])],
        crs="EPSG:4326",
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("other_relations should not attempt unsupported GPU WKB decode")
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.file._read_osm_pbf_pyogrio_layer_public",
        lambda *_args, **_kwargs: fallback_frame,
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file("example.osm.pbf", layer="other_relations")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["osm_element"].tolist() == ["relation"]
    assert result.geometry.geom_type.tolist() == ["GeometryCollection"]
    assert any(
        event.implementation == "osm_pbf_pyogrio_geometrycollection_compat_bridge"
        and event.selected is geopandas.ExecutionMode.CPU
        for event in events
    )


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
    from vibespatial.io.file import _read_osm_pbf_pyogrio_layer_native

    fallback_frame = geopandas.GeoDataFrame(
        {"osm_id": [300], "osm_element": ["relation"]},
        geometry=[GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])])],
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("other_relations should not attempt unsupported GPU WKB decode")
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


def test_osm_supported_layer_concat_preserves_geometrycollection_bridge() -> None:
    device_payload = _native_file_result_from_owned(
        from_shapely_geometries([Point(0, 0), LineString([(0, 0), (1, 1)])]),
        crs="EPSG:4326",
        attributes=pd.DataFrame(
            {
                "osm_element": ["node", "way"],
                "osm_id": [100, 200],
            }
        ),
    )
    bridge_frame = geopandas.GeoDataFrame(
        {"osm_id": [300], "osm_element": ["relation"]},
        geometry=[GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])])],
        crs="EPSG:4326",
    )

    payload = _concat_native_tabular_results(
        [device_payload, _spatial_to_native_tabular_result(bridge_frame)],
        geometry_name="geometry",
        crs="EPSG:4326",
    )
    result = payload.to_geodataframe()

    assert result["osm_element"].tolist() == ["node", "way", "relation"]
    assert result.geometry.geom_type.tolist() == ["Point", "LineString", "GeometryCollection"]


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


@pytest.mark.gpu
def test_public_shapefile_read_prefers_repo_owned_native_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    frame = _sample_frame()
    frame.to_file(path, driver="ESRI Shapefile")

    payload = read_shapefile_native(path)
    calls: list[dict[str, object]] = []

    def _fake_read_shapefile_native(source, *, target_crs=None, **kwargs):
        calls.append(
            {
                "source": source,
                "target_crs": target_crs,
                "kwargs": kwargs,
            }
        )
        return payload

    monkeypatch.setattr("vibespatial.io.file.read_shapefile_native", _fake_read_shapefile_native)

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path)
    events = geopandas.get_dispatch_events(clear=True)

    assert calls
    assert result["id"].tolist() == frame["id"].tolist()
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert events
    assert events[-1].implementation == "shapefile_native_pipeline_gpu_adapter"


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_explicit_pyogrio_shapefile_read_uses_arrow_gpu_wkb_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    path.write_text("")

    table = pa.table({"geometry": [Point(0, 0).wkb, Point(1, 1).wkb], "id": [1, 2]})
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    read_arrow_calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {
                "geometry_name": "geometry",
                "crs": "EPSG:4326",
                "geometry_type": "Point",
            },
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.io.file.read_shapefile_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio Shapefile reads should use the pyogrio Arrow bridge")
        ),
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio Shapefile reads should not demote to compatibility")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["datetime_as_string"] is True
    assert events
    assert events[-1].implementation == "shapefile_pyogrio_arrow_gpu_wkb"
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "compat_override=1" not in events[-1].detail


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_explicit_pyogrio_shapefile_filters_use_arrow_gpu_wkb_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    path.write_text("")

    table = pa.table({"geometry": [Point(0, 0).wkb], "id": [1]})
    owned = from_shapely_geometries([Point(0, 0)])
    read_arrow_calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {
                "geometry_name": "geometry",
                "crs": "EPSG:4326",
                "geometry_type": "Point",
            },
            table,
        ),
    )
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio Shapefile filters should stay native")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(
        path,
        engine="pyogrio",
        bbox=box(-1, -1, 1, 1),
        columns=["id"],
        rows=slice(1, 3),
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["bbox"] == pytest.approx((-1, -1, 1, 1))
    assert read_arrow_calls[-1]["columns"] == ["id"]
    assert read_arrow_calls[-1]["skip_features"] == 1
    assert read_arrow_calls[-1]["max_features"] == 2
    assert events[-1].implementation == "shapefile_pyogrio_arrow_gpu_wkb"
    assert "request=bbox+columns+rows" in events[-1].detail


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_explicit_pyogrio_shapefile_invalid_rows_skip_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    path.write_text("")
    read_file_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "vibespatial.io.file._try_gpu_read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid pyogrio rows should not enter the GPU read attempt")
        ),
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: read_file_calls.append(_kwargs) or (_ for _ in ()).throw(
            TypeError("rows must be an int, slice, or None")
        ),
    )

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    with pytest.raises(TypeError, match="rows must be an int, slice, or None"):
        geopandas.read_file(path, engine="pyogrio", rows="not_a_slice")
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert read_file_calls
    assert read_file_calls[-1]["rows"] == "not_a_slice"
    assert not fallbacks
    assert events
    assert events[-1].implementation == "shapefile_hybrid_adapter"
    assert "compat_override=1" in events[-1].detail


def test_public_read_file_bbox_mask_invalid_request_skips_dispatch_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    path.write_text("")

    monkeypatch.setattr(
        "vibespatial.io.file._try_gpu_read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid bbox+mask requests should not enter GPU read dispatch")
        ),
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid bbox+mask requests should fail before compatibility dispatch")
        ),
    )

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    with pytest.raises(ValueError, match="mask and bbox can not be set together"):
        geopandas.read_file(
            path,
            bbox=(-1, -1, 1, 1),
            mask=box(-1, -1, 1, 1),
        )

    assert not geopandas.get_dispatch_events(clear=True)
    assert not geopandas.get_fallback_events(clear=True)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_explicit_pyogrio_shapefile_mask_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    frame = _sample_frame()
    frame.to_file(path, driver="ESRI Shapefile")

    table = pa.table({"geometry": [Point(0, 0).wkb, Point(1, 1).wkb], "id": [1, 2]})
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    read_arrow_calls: list[dict[str, object]] = []
    mask_series = geopandas.GeoSeries([box(-0.25, -0.25, 0.25, 0.25)], crs="EPSG:4326")

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {
                "geometry_name": "geometry",
                "crs": "EPSG:4326",
                "geometry_type": "Point",
            },
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("masked explicit pyogrio Shapefile reads should stay on the native path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio", mask=mask_series)
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["datetime_as_string"] is True
    assert read_arrow_calls[-1]["mask"].bounds == pytest.approx(mask_series.iloc[0].bounds)
    assert events
    assert events[-1].implementation == "shapefile_pyogrio_arrow_gpu_wkb"
    assert "request=mask" in events[-1].detail
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "compat_override=1" not in events[-1].detail


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
def test_public_device_pyogrio_to_file_uses_native_arrow_sink(monkeypatch, tmp_path) -> None:
    import pyogrio

    geometry = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(from_shapely_geometries([Point(0, 0), Point(1, 1)])),
        crs="EPSG:4326",
    )
    frame = geopandas.GeoDataFrame({"value": [10, 20], "geometry": geometry}, crs="EPSG:4326")
    path = tmp_path / "public-device.geojson"
    arrow_calls = 0
    real_write_arrow = pyogrio.write_arrow

    def fail_write_dataframe(*_args, **_kwargs):
        raise AssertionError("device-backed public to_file should use the native Arrow sink")

    def capture_write_arrow(arrow_obj, filename, *args, **kwargs):
        nonlocal arrow_calls
        arrow_calls += 1
        return real_write_arrow(arrow_obj, filename, *args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_dataframe", fail_write_dataframe)
    monkeypatch.setattr(pyogrio, "write_arrow", capture_write_arrow)

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    frame.to_file(path, driver="GeoJSON", engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)
    result = geopandas.read_file(path, engine="pyogrio")

    write_events = [event for event in events if event.operation == "to_file"]
    assert arrow_calls == 1
    assert write_events
    assert write_events[-1].selected.value == "gpu"
    assert "native_arrow_sink=1" in write_events[-1].detail
    assert not fallbacks
    assert result["value"].tolist() == [10, 20]
    assert result.geometry.iloc[1].equals(Point(1, 1))


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_device_geopackage_to_file_uses_native_arrow_sink(monkeypatch, tmp_path) -> None:
    import pyogrio

    geometry = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(from_shapely_geometries([Point(0, 0), Point(1, 1)])),
        crs="EPSG:4326",
    )
    frame = geopandas.GeoDataFrame({"value": [10, 20], "geometry": geometry}, crs="EPSG:4326")
    path = tmp_path / "public-device.gpkg"
    captured: dict[str, object] = {}
    real_write_arrow = pyogrio.write_arrow

    def fail_write_dataframe(*_args, **_kwargs):
        raise AssertionError("device-backed public GPKG to_file should use the native Arrow sink")

    def capture_write_arrow(arrow_obj, filename, *args, **kwargs):
        captured.update(kwargs)
        return real_write_arrow(arrow_obj, filename, *args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_dataframe", fail_write_dataframe)
    monkeypatch.setattr(pyogrio, "write_arrow", capture_write_arrow)

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    frame.to_file(
        path,
        driver="GPKG",
        engine="pyogrio",
        layer="device_layer",
        metadata={"source": "vibespatial-test"},
    )
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)
    result = geopandas.read_file(path, engine="pyogrio", layer="device_layer")

    write_events = [event for event in events if event.operation == "to_file"]
    assert captured["layer"] == "device_layer"
    assert captured["metadata"] == {"source": "vibespatial-test"}
    assert write_events
    assert write_events[-1].selected.value == "gpu"
    assert "format=geopackage" in write_events[-1].detail
    assert not fallbacks
    assert result["value"].tolist() == [10, 20]
    assert result.geometry.iloc[1].equals(Point(1, 1))


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_device_flatgeobuf_to_file_uses_native_arrow_sink(monkeypatch, tmp_path) -> None:
    import pyogrio

    geometry = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(from_shapely_geometries([Point(0, 0), Point(1, 1)])),
        crs="EPSG:4326",
    )
    frame = geopandas.GeoDataFrame({"value": [10, 20], "geometry": geometry}, crs="EPSG:4326")
    path = tmp_path / "public-device.fgb"
    arrow_calls = 0
    real_write_arrow = pyogrio.write_arrow

    def fail_write_dataframe(*_args, **_kwargs):
        raise AssertionError("device-backed public FlatGeobuf to_file should use the native Arrow sink")

    def capture_write_arrow(arrow_obj, filename, *args, **kwargs):
        nonlocal arrow_calls
        arrow_calls += 1
        return real_write_arrow(arrow_obj, filename, *args, **kwargs)

    monkeypatch.setattr(pyogrio, "write_dataframe", fail_write_dataframe)
    monkeypatch.setattr(pyogrio, "write_arrow", capture_write_arrow)

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    frame.to_file(path, driver="FlatGeobuf", engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)
    result = geopandas.read_file(path)

    write_events = [event for event in events if event.operation == "to_file"]
    assert arrow_calls == 1
    assert write_events
    assert write_events[-1].selected.value == "gpu"
    assert "format=flatgeobuf" in write_events[-1].detail
    assert "native_arrow_sink=1" in write_events[-1].detail
    assert not fallbacks
    by_value = {value: geom for value, geom in zip(result["value"], result.geometry, strict=True)}
    assert sorted(by_value) == [10, 20]
    assert by_value[10].equals(Point(0, 0))
    assert by_value[20].equals(Point(1, 1))


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


def test_read_shapefile_owned_multipart_polygon_records_bypass_direct_shp_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (
        Path(__file__).resolve().parent
        / "upstream/geopandas/tests/data/overlay/nybb_qgis/qgis-difference.shp"
    )

    monkeypatch.setattr(
        "vibespatial.io.file._try_shapefile_shp_direct_gpu_read_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("multipart polygon shapefiles should stay on the Arrow/WKB path")
        ),
    )

    batch = read_shapefile_owned(path)
    materialized = batch.geometry.to_shapely()

    assert len(materialized) == 5
    assert all(geom.is_valid for geom in materialized)
    assert all(geom.geom_type == "MultiPolygon" for geom in materialized)


def test_read_shapefile_owned_prefers_direct_gpu_native_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.shp"
    _sample_frame().to_file(path, driver="ESRI Shapefile")

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

    batch = read_shapefile_owned(path)

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
    geojsonseq = plan_vector_file_io("data.geojsonseq", operation=IOOperation.READ)
    assert geojsonseq.selected_path is IOPathKind.HYBRID
    assert geojsonseq.implementation == "geojsonseq_hybrid_adapter"
    assert "GPU GeoJSON parser" in geojsonseq.reason
    assert plan_vector_file_io("data.unknown", operation=IOOperation.READ).selected_path is IOPathKind.FALLBACK


def test_resolve_named_file_source_path_normalizes_local_archive_uris(tmp_path: Path) -> None:
    zip_path = _write_zipped_shapefile(tmp_path)
    expected = zip_path

    assert _resolve_named_file_source_path(f"zip://{zip_path}") == expected
    assert _resolve_named_file_source_path(f"/vsizip/{zip_path}") == expected
    assert _resolve_named_file_source_path(f"file+file://{zip_path}") == expected
    assert _resolve_named_file_source_path(f"{zip_path}!sample.shp") == expected


def test_resolve_named_file_source_path_preserves_remote_urls() -> None:
    for source in (
        "https://example.com/data.geojson",
        "http://example.com/data.geojson",
        "s3://bucket/data.geojson",
        "zip://https://example.com/data.zip",
        "/vsizip/https://example.com/data.zip",
    ):
        assert _resolve_named_file_source_path(source) is None
        assert _is_remote_named_file_source(source) is True


def test_public_read_file_remote_geojson_skips_local_native_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def available(self) -> bool:
            return True

    calls: list[dict[str, object]] = []

    def fail_native_geojson(*_args, **_kwargs):
        raise AssertionError("remote GeoJSON URLs must not enter the local native parser")

    def fake_read_file(_filename, **kwargs):
        calls.append(kwargs)
        return _sample_frame()

    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("vibespatial.io.file.read_geojson_native", fail_native_geojson)
    monkeypatch.setattr("vibespatial.api.io.file._read_file", fake_read_file)

    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    result = read_vector_file("https://example.com/data.geojson")
    events = geopandas.get_dispatch_events(clear=True)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert calls
    assert calls[-1]["engine"] == "pyogrio"
    assert events
    assert events[-1].selected.value == "cpu"
    assert "source=remote" in events[-1].detail
    assert not fallbacks


def test_plan_vector_file_io_classifies_local_archive_uris_as_shapefile(tmp_path: Path) -> None:
    zip_path = _write_zipped_shapefile(tmp_path)

    for source in (
        f"zip://{zip_path}",
        f"/vsizip/{zip_path}",
        f"file+file://{zip_path}",
        f"{zip_path}!sample.shp",
    ):
        plan = plan_vector_file_io(source, operation=IOOperation.READ)
        assert plan.format.value == "shapefile"
        assert plan.selected_path is IOPathKind.HYBRID


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


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_geojsonseq_gpu_failure_reports_cpu_fallback_to_arrow_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.geojsonseq"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"id": 7},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
        )
    )
    table = pa.table({"geometry": [Point(0, 0).wkb], "id": [7]})
    owned = from_shapely_geometries([Point(0, 0)])

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "vibespatial.io.file._read_geojsonseq_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("geojsonseq gpu boom")),
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

    geopandas.clear_fallback_events()
    payload = read_vector_file_native(path)
    fallbacks = geopandas.get_fallback_events(clear=True)

    assert isinstance(payload, NativeTabularResult)
    assert payload.attributes["id"].tolist() == [7]
    assert any(
        "GPU GeoJSONSeq ingest failed" in event.reason
        and event.selected is geopandas.ExecutionMode.CPU
        for event in fallbacks
    )


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
def test_read_vector_file_native_explicit_pyogrio_keeps_promoted_container_on_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "example.gpkg"
    path.write_text("")

    table = pa.table({"geometry": [Point(0, 0).wkb], "value": [10]})
    owned = from_shapely_geometries([Point(0, 0)])
    read_arrow_calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio on promoted vector containers should stay on the native path")
        ),
    )

    payload = read_vector_file_native(path, engine="pyogrio")

    assert read_arrow_calls
    assert read_arrow_calls[-1]["datetime_as_string"] is False
    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.owned is owned
    assert payload.attributes["value"].tolist() == [10]


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_read_vector_file_native_masked_geopackage_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "example.gpkg"
    path.write_text("")

    table = pa.table({"geometry": [Point(0, 0).wkb], "value": [10]})
    owned = from_shapely_geometries([Point(0, 0)])
    read_arrow_calls: list[dict[str, object]] = []
    mask_frame = geopandas.GeoDataFrame(
        geometry=[box(-0.25, -0.25, 0.25, 0.25)],
        crs="EPSG:4326",
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("masked promoted native reads should not demote to compatibility")
        ),
    )

    payload = read_vector_file_native(path, mask=mask_frame)

    assert isinstance(payload, NativeTabularResult)
    assert payload.geometry.owned is owned
    assert payload.attributes["value"].tolist() == [10]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["mask"].bounds == pytest.approx(
        mask_frame.geometry.iloc[0].bounds
    )


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_explicit_pyogrio_geopackage_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.gpkg"
    path.write_text("")
    table = pa.table({"geometry": [Point(0, 0).wkb, Point(1, 1).wkb], "id": [1, 2]})
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    read_arrow_calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit pyogrio GeoPackage reads should stay on the native path")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio")
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["datetime_as_string"] is True
    assert events
    assert events[-1].implementation == "geopackage_pyogrio_arrow_gpu_wkb"
    assert "explicit_engine=pyogrio" in events[-1].detail
    assert "compat_override=1" not in events[-1].detail


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_explicit_pyogrio_geopackage_mask_keeps_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.gpkg"
    path.write_text("")
    table = pa.table({"geometry": [Point(0, 0).wkb, Point(1, 1).wkb], "id": [1, 2]})
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    read_arrow_calls: list[dict[str, object]] = []
    mask_series = geopandas.GeoSeries([box(-0.25, -0.25, 0.25, 0.25)], crs="EPSG:4326")

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("masked explicit pyogrio GeoPackage reads should stay native")
        ),
    )

    geopandas.clear_dispatch_events()
    result = geopandas.read_file(path, engine="pyogrio", mask=mask_series)
    events = geopandas.get_dispatch_events(clear=True)

    assert result["id"].tolist() == [1, 2]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["datetime_as_string"] is True
    assert read_arrow_calls[-1]["mask"].bounds == pytest.approx(mask_series.iloc[0].bounds)
    assert events
    assert events[-1].implementation == "geopackage_pyogrio_arrow_gpu_wkb"
    assert "request=mask" in events[-1].detail
    assert "explicit_engine=pyogrio" in events[-1].detail


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_explicit_pyogrio_geopackage_layer_filters_keep_native_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.gpkg"
    path.write_text("")
    table = pa.table({"geometry": [Point(0, 0).wkb], "id": [1]})
    owned = from_shapely_geometries([Point(0, 0)])
    read_arrow_calls: list[dict[str, object]] = []
    filter_frame = geopandas.GeoDataFrame(
        geometry=[box(-0.25, -0.25, 0.25, 0.25)],
        crs="EPSG:4326",
    )

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("layer-filtered explicit pyogrio GeoPackage reads should stay native")
        ),
    )

    geopandas.clear_dispatch_events()
    bbox_result = geopandas.read_file(path, engine="pyogrio", bbox=filter_frame, layer="layer1")
    mask_result = geopandas.read_file(path, engine="pyogrio", mask=filter_frame, layer="layer1")
    events = geopandas.get_dispatch_events(clear=True)

    assert bbox_result["id"].tolist() == [1]
    assert mask_result["id"].tolist() == [1]
    assert len(read_arrow_calls) == 2
    assert read_arrow_calls[0]["layer"] == "layer1"
    assert read_arrow_calls[0]["bbox"] == pytest.approx(filter_frame.total_bounds)
    assert read_arrow_calls[1]["layer"] == "layer1"
    assert read_arrow_calls[1]["mask"].bounds == pytest.approx(filter_frame.geometry.iloc[0].bounds)
    assert events[-2].implementation == "geopackage_pyogrio_arrow_gpu_wkb"
    assert events[-1].implementation == "geopackage_pyogrio_arrow_gpu_wkb"
    assert "request=bbox+layer" in events[-2].detail
    assert "request=mask+layer" in events[-1].detail
    assert "compat_override=1" not in events[-2].detail
    assert "compat_override=1" not in events[-1].detail


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_masked_geopackage_reprojects_mask_before_native_arrow_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pytest.importorskip("pyproj")

    path = tmp_path / "sample.gpkg"
    path.write_text("")
    table = pa.table({"geometry": [Point(0, 0).wkb], "id": [1]})
    owned = from_shapely_geometries([Point(0, 0)])
    read_arrow_calls: list[dict[str, object]] = []
    mask_wgs84 = geopandas.GeoSeries([box(-0.25, -0.25, 0.25, 0.25)], crs="EPSG:4326")
    mask_mercator = mask_wgs84.to_crs("EPSG:3857")

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr("pyogrio.read_info", lambda *_args, **_kwargs: {"crs": "EPSG:4326"})
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: read_arrow_calls.append(_kwargs) or (
            {"geometry_name": "geometry", "crs": "EPSG:4326"},
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched-CRS mask should still stay on the native path")
        ),
    )

    result = geopandas.read_file(path, mask=mask_mercator)

    assert result["id"].tolist() == [1]
    assert read_arrow_calls
    assert read_arrow_calls[-1]["mask"].bounds == pytest.approx(mask_wgs84.iloc[0].bounds)


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_explicit_pyogrio_geopackage_preserves_all_null_object_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.gpkg"
    path.write_text("")
    table = pa.table(
        {
            "geometry": [Point(0, 0).wkb, Point(1, 1).wkb],
            "a": pa.array([None, None], type=pa.string()),
        }
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {
                "geometry_name": "geometry",
                "crs": "EPSG:4326",
                "fields": ["a"],
                "dtypes": ["object"],
                "geometry_type": "Point",
            },
            table,
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda _column: owned,
    )

    result = geopandas.read_file(path, engine="pyogrio")

    assert str(result["a"].dtype) == "object"
    assert result["a"].tolist() == [None, None]


@pytest.mark.skipif(importlib.util.find_spec("pyogrio") is None, reason="pyogrio not available")
def test_public_read_file_explicit_pyogrio_geopackage_unsupported_geometry_type_uses_compatibility_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "sample.gpkg"
    path.write_text("")
    frame = _sample_frame()

    class _FakeRuntime:
        def available(self) -> bool:
            return True

    monkeypatch.setattr("vibespatial.cuda._runtime.get_cuda_runtime", lambda: _FakeRuntime())
    monkeypatch.setattr("vibespatial.runtime.get_requested_mode", lambda: geopandas.ExecutionMode.AUTO)
    monkeypatch.setattr(
        "pyogrio.read_arrow",
        lambda *_args, **_kwargs: (
            {"geometry_name": "geometry", "crs": "EPSG:4326", "geometry_type": "Point Z"},
            pa.table({"geometry": [Point(0, 0).wkb]}),
        ),
    )
    monkeypatch.setattr(
        "vibespatial.io.arrow.decode_wkb_arrow_array_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported GeoPackage geometry types should not enter the native WKB decode path")
        ),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "vibespatial.api.io.file._read_file",
        lambda *_args, **_kwargs: calls.append(_kwargs) or frame.copy(),
    )

    result = geopandas.read_file(path, engine="pyogrio")

    assert calls
    assert result["id"].tolist() == frame["id"].tolist()


def test_materialize_native_file_read_result_parses_mixed_offset_strings_to_utc() -> None:
    class _FakePayload:
        def to_geodataframe(self):
            return geopandas.GeoDataFrame(
                {
                    "date": [
                        "2014-08-26 10:01:23.040001+02:00",
                        "2019-03-07 17:31:43.118999+01:00",
                    ],
                    "geometry": [Point(0, 0), Point(1, 1)],
                },
                crs="EPSG:4326",
            )

    result = _materialize_native_file_read_result(_FakePayload())

    assert str(result["date"].dtype) == "datetime64[ms, UTC]"


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
    assert plan.objective == "pipeline"
    assert plan.uses_stream_tokenizer is False
    assert plan.uses_native_geometry_assembly is True


def test_plan_geojson_ingest_standalone_auto_prefers_fast_json() -> None:
    plan = plan_geojson_ingest(objective="standalone")

    assert plan.implementation == "geojson_fast_json_vectorized"
    assert plan.selected_strategy == "fast-json"
    assert plan.objective == "standalone"
    assert plan.uses_stream_tokenizer is False
    assert plan.uses_native_geometry_assembly is True


def test_plan_geojson_ingest_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        plan_geojson_ingest(objective="mystery")


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
