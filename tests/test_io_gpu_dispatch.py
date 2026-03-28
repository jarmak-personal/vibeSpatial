"""Integration tests for GPU dispatch of WKT, CSV, and KML via read_file().

Verifies the full chain: write test file -> read_file() -> GeoDataFrame
with correct geometries, correct dispatch events, and proper fallback
behavior.
"""
from __future__ import annotations

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
