"""Integration tests for GPU dispatch of WKT, CSV, KML, and OSM PBF via read_file().

Verifies the full chain: write test file -> read_file() -> GeoDataFrame
with correct geometries, correct dispatch events, and proper fallback
behavior.  Also tests fused ingest+reproject (target_crs) and fused
ingest+spatial-index (build_index) pipelines, the public gpu_spatial_index
property, and import accessibility of GPU reader functions.
"""
from __future__ import annotations

import json
import math
import struct
import zlib

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import vibespatial.api as geopandas
from vibespatial.io.file import _GPU_MIN_FILE_SIZE
from vibespatial.io.support import IOFormat
from vibespatial.runtime._runtime import has_gpu_runtime

needs_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")


# ---------------------------------------------------------------------------
# WKT tests
# ---------------------------------------------------------------------------


class TestWktGpuDispatch:
    """WKT files must always route through GPU (no pyogrio/GDAL fallback)."""

    @needs_gpu
    def test_read_wkt_point_file(self, tmp_path) -> None:
        """read_file on a .wkt file with Points returns correct GeoDataFrame."""
        wkt_path = tmp_path / "points.wkt"
        wkt_path.write_text("POINT (1 2)\nPOINT (3 4)\nPOINT (5 6)\n")

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(wkt_path)

        assert len(result) == 3
        assert result.geometry.iloc[0].equals(Point(1, 2))
        assert result.geometry.iloc[1].equals(Point(3, 4))
        assert result.geometry.iloc[2].equals(Point(5, 6))

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "wkt_gpu_byte_classify_adapter" in impl_names

    @needs_gpu
    def test_read_wkt_polygon_file(self, tmp_path) -> None:
        """read_file on a .wkt file with Polygons returns correct geometries."""
        wkt_path = tmp_path / "polygons.wkt"
        wkt_path.write_text(
            "POLYGON ((0 0, 1 0, 1 1, 0 0))\n"
            "POLYGON ((2 2, 3 2, 3 3, 2 2))\n"
        )

        result = geopandas.read_file(wkt_path)

        assert len(result) == 2
        assert result.geometry.iloc[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        assert result.geometry.iloc[1].equals(Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]))

    @needs_gpu
    def test_read_wkt_linestring_file(self, tmp_path) -> None:
        """read_file on a .wkt file with LineStrings returns correct geometries."""
        wkt_path = tmp_path / "lines.wkt"
        wkt_path.write_text(
            "LINESTRING (0 0, 1 1, 2 0)\n"
            "LINESTRING (10 10, 20 20)\n"
        )

        result = geopandas.read_file(wkt_path)

        assert len(result) == 2
        assert result.geometry.iloc[0].equals(LineString([(0, 0), (1, 1), (2, 0)]))
        assert result.geometry.iloc[1].equals(LineString([(10, 10), (20, 20)]))

    @needs_gpu
    def test_read_wkt_mixed_types(self, tmp_path) -> None:
        """read_file on a .wkt file with mixed geometry types works."""
        wkt_path = tmp_path / "mixed.wkt"
        wkt_path.write_text(
            "POINT (1 2)\n"
            "LINESTRING (0 0, 1 1)\n"
            "POLYGON ((0 0, 1 0, 1 1, 0 0))\n"
        )

        result = geopandas.read_file(wkt_path)

        assert len(result) == 3
        assert result.geometry.iloc[0].equals(Point(1, 2))
        assert result.geometry.iloc[1].equals(LineString([(0, 0), (1, 1)]))
        assert result.geometry.iloc[2].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))

    def test_read_wkt_without_gpu_raises(self, tmp_path, monkeypatch) -> None:
        """WKT files must raise a clear error when GPU is not available."""
        wkt_path = tmp_path / "nogpu.wkt"
        wkt_path.write_text("POINT (1 2)\n")

        # Force the GPU path to return None by making _try_wkt_gpu_read fail.
        from vibespatial.io import file as io_file

        monkeypatch.setattr(io_file, "_try_wkt_gpu_read", lambda *a, **kw: None)
        # Also disable the general GPU gate so it falls through.
        monkeypatch.setattr(io_file, "_try_gpu_read_file", lambda *a, **kw: None)

        with pytest.raises(RuntimeError, match="GPU runtime is required for raw WKT"):
            geopandas.read_file(wkt_path)


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------


class TestCsvGpuDispatch:
    """CSV files route to GPU for large files, fall through to CPU for small."""

    @needs_gpu
    def test_read_csv_latlon_large_file(self, tmp_path) -> None:
        """read_file on a large .csv file with lat/lon uses GPU adapter."""
        csv_path = tmp_path / "spatial.csv"
        # Generate a CSV above the GPU size threshold (10 MB).
        # Each row is ~25 bytes; need ~450,000 rows to exceed 10 MB.
        n_rows = 450_000
        lines = ["name,lat,lon\n"]
        for i in range(n_rows):
            lat = 40.0 + (i % 1000) * 0.001
            lon = -74.0 + (i // 1000) * 0.001
            lines.append(f"loc_{i},{lat},{lon}\n")
        csv_path.write_text("".join(lines))

        assert csv_path.stat().st_size > _GPU_MIN_FILE_SIZE

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(csv_path)

        assert len(result) >= n_rows
        # Verify first geometry is a Point.
        assert result.geometry.iloc[0].geom_type == "Point"

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "csv_gpu_byte_classify_adapter" in impl_names

    def test_small_csv_uses_cpu_fallback(self, tmp_path) -> None:
        """Small CSV files below the threshold should not use GPU adapter."""
        csv_path = tmp_path / "small.csv"
        csv_path.write_text("name,lat,lon\nAlice,40.7,-74.0\nBob,34.0,-118.2\n")

        assert csv_path.stat().st_size < _GPU_MIN_FILE_SIZE

        geopandas.clear_dispatch_events()
        # This should go through CPU path (pyogrio).  It may or may not
        # succeed depending on pyogrio's CSV support, but it should NOT
        # dispatch to the GPU adapter.
        try:
            geopandas.read_file(csv_path)
        except Exception:
            pass  # pyogrio may not support CSV natively; that's fine

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "csv_gpu_byte_classify_adapter" not in impl_names


# ---------------------------------------------------------------------------
# KML tests
# ---------------------------------------------------------------------------


class TestKmlGpuDispatch:
    """KML files route to GPU for large files, fall through to CPU for small."""

    @staticmethod
    def _wrap_kml(*placemarks: str) -> str:
        body = "\n".join(placemarks)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            "<Document>\n"
            f"{body}\n"
            "</Document>\n"
            "</kml>\n"
        )

    @staticmethod
    def _point_placemark(lon: float, lat: float) -> str:
        return (
            "<Placemark>"
            f"<Point><coordinates>{lon},{lat}</coordinates></Point>"
            "</Placemark>"
        )

    @needs_gpu
    def test_read_kml_large_file(self, tmp_path) -> None:
        """read_file on a large .kml file uses GPU adapter."""
        kml_path = tmp_path / "points.kml"
        # Generate a KML above the GPU size threshold (10 MB).
        # Each Placemark is ~88 bytes; need ~130,000 to exceed 10 MB.
        n_features = 130_000
        placemarks = [
            self._point_placemark(-74.0 + i * 0.001, 40.0 + i * 0.001)
            for i in range(n_features)
        ]
        kml_content = self._wrap_kml(*placemarks)
        kml_path.write_text(kml_content)

        assert kml_path.stat().st_size > _GPU_MIN_FILE_SIZE

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(kml_path)

        assert len(result) == n_features
        assert result.geometry.iloc[0].geom_type == "Point"

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "kml_gpu_byte_classify_adapter" in impl_names

    def test_small_kml_uses_cpu_fallback(self, tmp_path) -> None:
        """Small KML files below the threshold should not use GPU adapter."""
        kml_path = tmp_path / "small.kml"
        kml_content = self._wrap_kml(
            self._point_placemark(-74.0, 40.7),
            self._point_placemark(-118.2, 34.0),
        )
        kml_path.write_text(kml_content)

        assert kml_path.stat().st_size < _GPU_MIN_FILE_SIZE

        geopandas.clear_dispatch_events()
        try:
            geopandas.read_file(kml_path)
        except Exception:
            pass  # pyogrio may not support KML natively

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "kml_gpu_byte_classify_adapter" not in impl_names


# ---------------------------------------------------------------------------
# Plan / format detection tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Verify that plan_vector_file_io correctly identifies WKT/CSV/KML."""

    def test_wkt_extension_detected(self) -> None:
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.wkt", operation=IOOperation.READ)
        assert plan.format is IOFormat.WKT

    def test_csv_extension_detected(self) -> None:
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.csv", operation=IOOperation.READ)
        assert plan.format is IOFormat.CSV

    def test_tsv_extension_detected_as_csv(self) -> None:
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.tsv", operation=IOOperation.READ)
        assert plan.format is IOFormat.CSV

    def test_kml_extension_detected(self) -> None:
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.kml", operation=IOOperation.READ)
        assert plan.format is IOFormat.KML


# ---------------------------------------------------------------------------
# Existing formats remain unbroken
# ---------------------------------------------------------------------------


class TestExistingFormatsUnbroken:
    """Ensure GeoJSON and Shapefile dispatch still works after changes."""

    @needs_gpu
    def test_geojson_still_dispatches(self, tmp_path) -> None:
        path = tmp_path / "test.geojson"
        frame = geopandas.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
            crs="EPSG:4326",
        )
        frame.to_file(path, driver="GeoJSON")

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(path)

        assert len(result) == 2
        assert result.geometry.iloc[0].equals(Point(0, 0))

        events = geopandas.get_dispatch_events(clear=True)
        # Should use one of the GPU adapters (geojson_gpu_adapter or
        # geojson_gpu_byte_classify_adapter) for GPU, or the CPU adapter.
        assert any(
            "geojson" in e.implementation for e in events
        )

    @needs_gpu
    def test_shapefile_still_dispatches(self, tmp_path) -> None:
        path = tmp_path / "test.shp"
        frame = geopandas.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
        )
        frame.to_file(path, driver="ESRI Shapefile")

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(path)

        assert len(result) == 2
        events = geopandas.get_dispatch_events(clear=True)
        assert any(
            "shapefile" in e.implementation for e in events
        )


# ---------------------------------------------------------------------------
# Fused ingest+reproject (target_crs) tests
# ---------------------------------------------------------------------------

# Reference values for WGS84 -> Web Mercator reprojection.
_SEMI_CIRC = 20037508.342789244


def _wgs84_to_mercator_ref(lon: float, lat: float) -> tuple[float, float]:
    """Pure-Python reference WGS84 -> Web Mercator."""
    x = lon * _SEMI_CIRC / 180.0
    lat_rad = lat * math.pi / 180.0
    y = math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) * _SEMI_CIRC / math.pi
    return x, y


def _write_geojson(path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc), encoding="utf-8")


def _point_feature(lon: float, lat: float) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


class TestFusedReprojectReadFile:
    """Test target_crs parameter through the public read_file() API."""

    @needs_gpu
    def test_geojson_target_crs_reprojects(self, tmp_path) -> None:
        """read_file with target_crs on GeoJSON produces reprojected coords."""
        path = tmp_path / "reproject.geojson"
        _write_geojson(path, [
            _point_feature(0.0, 0.0),
            _point_feature(-74.006, 40.7128),
        ])

        result = geopandas.read_file(path, target_crs="EPSG:3857")

        # CRS should be set to target.
        assert result.crs is not None
        assert result.crs.to_epsg() == 3857

        # Origin maps to ~(0, 0) in Mercator.
        pt0 = result.geometry.iloc[0]
        np.testing.assert_allclose(pt0.x, 0.0, atol=1.0)
        np.testing.assert_allclose(pt0.y, 0.0, atol=1.0)

        # NYC: verify against reference.
        ref_x, ref_y = _wgs84_to_mercator_ref(-74.006, 40.7128)
        pt1 = result.geometry.iloc[1]
        np.testing.assert_allclose(pt1.x, ref_x, rtol=1e-6)
        np.testing.assert_allclose(pt1.y, ref_y, rtol=1e-6)

    @needs_gpu
    def test_geojson_target_crs_same_is_noop(self, tmp_path) -> None:
        """read_file with target_crs=EPSG:4326 on GeoJSON is a no-op."""
        path = tmp_path / "noop.geojson"
        _write_geojson(path, [_point_feature(10.0, 20.0)])

        result = geopandas.read_file(path, target_crs="EPSG:4326")

        assert result.crs is not None
        assert result.crs.to_epsg() == 4326
        pt = result.geometry.iloc[0]
        np.testing.assert_allclose(pt.x, 10.0, atol=1e-9)
        np.testing.assert_allclose(pt.y, 20.0, atol=1e-9)

    @needs_gpu
    def test_wkt_target_crs_sets_crs_label(self, tmp_path) -> None:
        """read_file on WKT with target_crs sets CRS on output.

        WKT has no embedded CRS, so target_crs sets the CRS label
        without reprojecting coordinates.
        """
        wkt_path = tmp_path / "points.wkt"
        wkt_path.write_text("POINT (1 2)\nPOINT (3 4)\n")

        result = geopandas.read_file(wkt_path, target_crs="EPSG:3857")

        assert result.crs is not None
        assert result.crs.to_epsg() == 3857
        # Coordinates should be unchanged (no reprojection).
        assert result.geometry.iloc[0].equals(Point(1, 2))
        assert result.geometry.iloc[1].equals(Point(3, 4))


# ---------------------------------------------------------------------------
# Fused ingest+spatial-index (build_index) tests
# ---------------------------------------------------------------------------


class TestFusedBuildIndex:
    """Test build_index parameter through the public read_file() API."""

    @needs_gpu
    def test_geojson_build_index(self, tmp_path) -> None:
        """read_file with build_index=True attaches a GPU spatial index."""
        path = tmp_path / "indexed.geojson"
        _write_geojson(path, [
            _point_feature(0.0, 0.0),
            _point_feature(1.0, 1.0),
            _point_feature(2.0, 2.0),
        ])

        result = geopandas.read_file(path, build_index=True)

        assert len(result) == 3
        # Verify that the GPU spatial index was attached.
        assert hasattr(result, "_gpu_spatial_index")
        gpu_index = result._gpu_spatial_index
        assert gpu_index is not None
        assert gpu_index.n_features == 3

    @needs_gpu
    def test_geojson_no_build_index_default(self, tmp_path) -> None:
        """read_file without build_index does not attach a GPU spatial index."""
        path = tmp_path / "no_index.geojson"
        _write_geojson(path, [_point_feature(0.0, 0.0)])

        result = geopandas.read_file(path)

        assert not hasattr(result, "_gpu_spatial_index") or result._gpu_spatial_index is None

    @needs_gpu
    def test_wkt_build_index(self, tmp_path) -> None:
        """read_file on WKT with build_index=True attaches a GPU spatial index."""
        wkt_path = tmp_path / "indexed.wkt"
        wkt_path.write_text("POINT (1 2)\nPOINT (3 4)\nPOINT (5 6)\n")

        result = geopandas.read_file(wkt_path, build_index=True)

        assert len(result) == 3
        assert hasattr(result, "_gpu_spatial_index")
        gpu_index = result._gpu_spatial_index
        assert gpu_index is not None
        assert gpu_index.n_features == 3

    @needs_gpu
    def test_build_index_with_target_crs(self, tmp_path) -> None:
        """read_file with both target_crs and build_index works together."""
        path = tmp_path / "both.geojson"
        _write_geojson(path, [
            _point_feature(0.0, 0.0),
            _point_feature(1.0, 1.0),
        ])

        result = geopandas.read_file(
            path, target_crs="EPSG:3857", build_index=True,
        )

        assert result.crs is not None
        assert result.crs.to_epsg() == 3857
        assert hasattr(result, "_gpu_spatial_index")
        assert result._gpu_spatial_index is not None
        assert result._gpu_spatial_index.n_features == 2


# ---------------------------------------------------------------------------
# OSM PBF helpers (inline to avoid modifying test_osm_gpu.py)
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _encode_zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 63) if value >= 0 else ((-value - 1) << 1) | 1


def _encode_sint64(value: int) -> bytes:
    return _encode_varint(_encode_zigzag(value))


def _encode_field_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
    return _encode_field_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_varint_field(field_number: int, value: int) -> bytes:
    return _encode_field_tag(field_number, 0) + _encode_varint(value)


def _encode_packed_sint64(values: list[int]) -> bytes:
    result = bytearray()
    for v in values:
        result.extend(_encode_sint64(v))
    return bytes(result)


def _build_dense_nodes(id_deltas, lat_deltas, lon_deltas) -> bytes:
    return (
        _encode_length_delimited(1, _encode_packed_sint64(id_deltas))
        + _encode_length_delimited(8, _encode_packed_sint64(lat_deltas))
        + _encode_length_delimited(9, _encode_packed_sint64(lon_deltas))
    )


def _build_test_pbf(
    id_deltas: list[int],
    lat_deltas: list[int],
    lon_deltas: list[int],
    granularity: int = 100,
) -> bytes:
    dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
    group = _encode_length_delimited(2, dense)  # PrimitiveGroup.dense
    stringtable = _encode_length_delimited(1, b"")
    pblock = (
        stringtable
        + _encode_length_delimited(2, group)
        + _encode_varint_field(17, granularity)
    )
    # Build Blob (zlib compressed)
    compressed = zlib.compress(pblock)
    blob = (
        _encode_varint_field(2, len(pblock))
        + _encode_length_delimited(3, compressed)
    )
    # Build BlobHeader for OSMData
    blob_header = (
        _encode_length_delimited(1, b"OSMData")
        + _encode_varint_field(3, len(blob))
    )
    data_block = struct.pack(">I", len(blob_header)) + blob_header + blob

    # Build OSMHeader block
    header_payload = _encode_length_delimited(4, b"OsmSchema-V0.6")
    h_compressed = zlib.compress(header_payload)
    h_blob = (
        _encode_varint_field(2, len(header_payload))
        + _encode_length_delimited(3, h_compressed)
    )
    h_blob_header = (
        _encode_length_delimited(1, b"OSMHeader")
        + _encode_varint_field(3, len(h_blob))
    )
    header_block = struct.pack(">I", len(h_blob_header)) + h_blob_header + h_blob

    return header_block + data_block


# ---------------------------------------------------------------------------
# OSM PBF tests
# ---------------------------------------------------------------------------


class TestOsmPbfGpuDispatch:
    """Default OSM PBF reads stay native; standard layers may use pyogrio compatibility."""

    @needs_gpu
    def test_read_pbf_point_file(self, tmp_path) -> None:
        """read_file on a .pbf file with three nodes returns correct GeoDataFrame."""
        pbf_path = tmp_path / "nodes.pbf"
        # Three nodes: id=100,101,102 at lat=40.0,40.1,40.3 lon=-74.0,-74.01,-74.03
        pbf_data = _build_test_pbf(
            id_deltas=[100, 1, 1],
            lat_deltas=[400000000, 1000000, 2000000],
            lon_deltas=[-740000000, -100000, -200000],
        )
        pbf_path.write_bytes(pbf_data)

        geopandas.clear_dispatch_events()
        result = geopandas.read_file(pbf_path)

        assert len(result) == 3
        assert result.geometry.iloc[0].geom_type == "Point"

        # Verify coordinates
        np.testing.assert_allclose(result.geometry.iloc[0].y, 40.0, rtol=1e-9)
        np.testing.assert_allclose(result.geometry.iloc[0].x, -74.0, rtol=1e-9)

        # Verify node IDs column
        assert "osm_node_id" in result.columns
        np.testing.assert_array_equal(result["osm_node_id"].values, [100, 101, 102])

        events = geopandas.get_dispatch_events(clear=True)
        impl_names = [e.implementation for e in events]
        assert "osm_pbf_gpu_hybrid_adapter" in impl_names

    @needs_gpu
    def test_read_osm_pbf_extension(self, tmp_path) -> None:
        """read_file on a .osm.pbf file routes correctly."""
        pbf_path = tmp_path / "data.osm.pbf"
        pbf_data = _build_test_pbf(
            id_deltas=[1],
            lat_deltas=[515000000],
            lon_deltas=[-1000000],
        )
        pbf_path.write_bytes(pbf_data)

        result = geopandas.read_file(pbf_path)

        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "Point"
        np.testing.assert_allclose(result.geometry.iloc[0].y, 51.5, rtol=1e-9)
        np.testing.assert_allclose(result.geometry.iloc[0].x, -0.1, rtol=1e-9)

    def test_pbf_extension_detected(self) -> None:
        """plan_vector_file_io correctly identifies .pbf as OSM_PBF."""
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.pbf", operation=IOOperation.READ)
        assert plan.format is IOFormat.OSM_PBF

    def test_osm_pbf_extension_detected(self) -> None:
        """plan_vector_file_io correctly identifies .osm.pbf as OSM_PBF."""
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOOperation

        plan = plan_vector_file_io("data.osm.pbf", operation=IOOperation.READ)
        assert plan.format is IOFormat.OSM_PBF

    def test_read_pbf_without_gpu_raises(self, tmp_path, monkeypatch) -> None:
        """Full-data PBF reads still require the native GPU path."""
        pbf_path = tmp_path / "nogpu.pbf"
        pbf_path.write_bytes(_build_test_pbf(
            id_deltas=[1], lat_deltas=[0], lon_deltas=[0],
        ))

        from vibespatial.io import file as io_file

        monkeypatch.setattr(io_file, "_try_gpu_read_file", lambda *a, **kw: None)

        with pytest.raises(RuntimeError, match="GPU runtime is required"):
            geopandas.read_file(pbf_path, layer="all")


# ---------------------------------------------------------------------------
# gpu_spatial_index property tests
# ---------------------------------------------------------------------------


class TestGpuSpatialIndexProperty:
    """Test the public GeoDataFrame.gpu_spatial_index property."""

    def test_no_index_returns_none(self) -> None:
        """gpu_spatial_index returns None when build_index not used."""
        gdf = geopandas.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)]},
        )
        assert gdf.gpu_spatial_index is None

    @needs_gpu
    def test_index_present_after_build(self, tmp_path) -> None:
        """gpu_spatial_index returns GpuSpatialIndex when build_index=True."""
        path = tmp_path / "index_test.geojson"
        _write_geojson(path, [
            _point_feature(0.0, 0.0),
            _point_feature(1.0, 1.0),
            _point_feature(2.0, 2.0),
        ])

        result = geopandas.read_file(path, build_index=True)

        idx = result.gpu_spatial_index
        assert idx is not None
        assert idx.n_features == 3

    @needs_gpu
    def test_property_matches_private_attr(self, tmp_path) -> None:
        """gpu_spatial_index property returns the same object as _gpu_spatial_index."""
        path = tmp_path / "prop_test.geojson"
        _write_geojson(path, [_point_feature(0.0, 0.0)])

        result = geopandas.read_file(path, build_index=True)

        assert result.gpu_spatial_index is result._gpu_spatial_index


# ---------------------------------------------------------------------------
# Import accessibility tests
# ---------------------------------------------------------------------------


class TestImportAccessibility:
    """Verify that GPU readers are importable from vibespatial.io."""

    def test_import_read_wkt_gpu(self) -> None:
        from vibespatial.io import read_wkt_gpu

        assert callable(read_wkt_gpu)

    def test_import_read_csv_gpu(self) -> None:
        from vibespatial.io import read_csv_gpu

        assert callable(read_csv_gpu)

    def test_import_read_kml_gpu(self) -> None:
        from vibespatial.io import read_kml_gpu

        assert callable(read_kml_gpu)

    def test_import_read_geojson_gpu(self) -> None:
        from vibespatial.io import read_geojson_gpu

        assert callable(read_geojson_gpu)

    def test_import_read_dbf_gpu(self) -> None:
        from vibespatial.io import read_dbf_gpu

        assert callable(read_dbf_gpu)

    def test_import_read_osm_pbf_nodes(self) -> None:
        from vibespatial.io import read_osm_pbf_nodes

        assert callable(read_osm_pbf_nodes)

    def test_import_build_spatial_index(self) -> None:
        from vibespatial.io import build_spatial_index

        assert callable(build_spatial_index)

    def test_import_gpu_spatial_index_class(self) -> None:
        from vibespatial.io import GpuSpatialIndex

        assert isinstance(GpuSpatialIndex, type)
