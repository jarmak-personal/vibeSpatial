from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

import vibespatial.api as geopandas
import vibespatial.io.geojson as io_geojson
from vibespatial import (
    benchmark_geojson_ingest,
    benchmark_shapefile_ingest,
    plan_geojson_ingest,
    plan_shapefile_ingest,
    read_geojson_owned,
    read_shapefile_owned,
)


def _sample_frame() -> geopandas.GeoDataFrame:
    return geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )


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
    assert events[-1].implementation == "geojson_gpu_adapter"
    assert not fallbacks


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
    assert events[-1].implementation == "shapefile_gpu_adapter"
    assert not fallbacks


def test_plan_shapefile_ingest_uses_arrow_wkb_owned_path() -> None:
    plan = plan_shapefile_ingest()

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
    assert events[-2].implementation == "legacy_gdal_adapter"
    assert events[-1].implementation == "legacy_gdal_adapter"
    assert not fallbacks


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


def test_plan_geojson_ingest_prefers_stream_native_path() -> None:
    plan = plan_geojson_ingest()

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
        "shapefile_owned_batch",
        "native_wkb_decode",
    }
