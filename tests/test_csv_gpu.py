"""Tests for GPU CSV reader: structural analysis and spatial extraction."""
from __future__ import annotations

import numpy as np
import pytest

from vibespatial.geometry.buffers import GeometryFamily

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device_bytes(text: str | bytes) -> cp.ndarray:
    """Encode text as device-resident uint8 array."""
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return cp.frombuffer(raw, dtype=cp.uint8).copy()


def _get_device_coords(owned, family: GeometryFamily):
    """Extract host x, y coordinate arrays from an OwnedGeometryArray.

    Checks device_state first (GPU-resident), falling back to host
    family buffers.
    """
    if owned.device_state is not None and family in owned.device_state.families:
        dev_buf = owned.device_state.families[family]
        return cp.asnumpy(dev_buf.x), cp.asnumpy(dev_buf.y)
    buf = owned.families[family]
    return buf.x, buf.y


# ===================================================================
# csv_structural_analysis tests
# ===================================================================


class TestCsvStructuralBasic:
    """Basic CSV structural analysis."""

    @needs_gpu
    def test_simple_csv(self):
        """Simple 3-column CSV with header."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"name,lat,lon\nAlice,40.7,-74.0\nBob,34.0,-118.2\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 2
        assert result.n_columns == 3
        assert result.column_names == ["name", "lat", "lon"]
        assert result.spatial_columns == {"lat": 1, "lon": 2}

    @needs_gpu
    def test_single_column(self):
        """Single column, no delimiters."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"value\n10\n20\n30\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 3
        assert result.n_columns == 1
        assert result.column_names == ["value"]
        assert result.spatial_columns == {}

    @needs_gpu
    def test_tab_delimiter(self):
        """Tab-delimited file."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes, delimiter="\t")

        assert result.n_rows == 2
        assert result.n_columns == 3
        assert result.column_names == ["a", "b", "c"]

    @needs_gpu
    def test_pipe_delimiter(self):
        """Pipe-delimited file."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"x|y|z\n1|2|3\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes, delimiter="|")

        assert result.n_rows == 1
        assert result.n_columns == 3

    @needs_gpu
    def test_no_trailing_newline(self):
        """File not ending with newline."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n3,4"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 2
        assert result.n_columns == 2

    @needs_gpu
    def test_empty_input(self):
        """Empty byte array."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        d_bytes = cp.empty(0, dtype=cp.uint8)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 0
        assert result.n_columns == 0
        assert result.column_names == []
        assert result.spatial_columns == {}

    @needs_gpu
    def test_header_only(self):
        """File with only a header row (no data)."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b,c\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 0
        assert result.n_columns == 3
        assert result.column_names == ["a", "b", "c"]

    @needs_gpu
    def test_no_header(self):
        """File without header -- synthetic column names."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"1,2,3\n4,5,6\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes, has_header=False)

        assert result.n_rows == 2
        assert result.n_columns == 3
        assert result.column_names == ["col_0", "col_1", "col_2"]

    @needs_gpu
    def test_crlf_line_endings(self):
        """Windows-style \\r\\n line endings."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\r\n1,2\r\n3,4\r\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 2
        assert result.n_columns == 2
        # Column names should not contain \r
        assert result.column_names == ["a", "b"]


class TestCsvQuoting:
    """CSV quoting and escaping tests (RFC 4180)."""

    @needs_gpu
    def test_quoted_field_with_delimiter(self):
        """Delimiter inside quoted field is not a column boundary."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b'a,b\n"hello, world",42\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 1
        assert result.n_columns == 2

    @needs_gpu
    def test_quoted_field_with_newline(self):
        """Newline inside quoted field is not a row boundary."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b'a,b\n"line1\nline2",42\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 1
        assert result.n_columns == 2

    @needs_gpu
    def test_doubled_quote_escaping(self):
        """Doubled quotes ("") inside quoted field preserve parity."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        # Field value is: he said "hello"
        # CSV encoding: "he said ""hello"""
        csv = b'a,b\n"he said ""hello""",42\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 1
        assert result.n_columns == 2

    @needs_gpu
    def test_quoted_header_field(self):
        """Header field name enclosed in quotes."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b'"name","latitude","longitude"\nAlice,40.7,-74.0\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.n_rows == 1
        assert result.n_columns == 3
        assert result.column_names == ["name", "latitude", "longitude"]
        assert result.spatial_columns == {"lat": 1, "lon": 2}


class TestCsvSpatialDetection:
    """Spatial column detection heuristics."""

    @needs_gpu
    def test_lat_lon_detection(self):
        """Detect latitude/longitude column pair."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"id,latitude,longitude\n1,40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.spatial_columns == {"lat": 1, "lon": 2}

    @needs_gpu
    def test_geometry_column_detection(self):
        """Detect geometry/WKT column."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"id,geometry\n1,POINT(0 0)\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.spatial_columns == {"geom": 1}

    @needs_gpu
    def test_geometry_takes_precedence(self):
        """Geometry column takes precedence over lat/lon."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"lat,lon,geometry\n40.7,-74.0,POINT(0 0)\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        # geometry detected -> lat/lon dropped
        assert result.spatial_columns == {"geom": 2}

    @needs_gpu
    def test_partial_latlon_ignored(self):
        """Only lat without lon is ignored."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"id,lat,value\n1,40.7,100\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.spatial_columns == {}

    @needs_gpu
    def test_xy_column_names(self):
        """Detect x/y as lon/lat."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"id,x,y\n1,-74.0,40.7\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert result.spatial_columns == {"lat": 2, "lon": 1}


class TestCsvDeviceArrays:
    """Verify device array properties in the result."""

    @needs_gpu
    def test_row_ends_are_device_int64(self):
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n3,4\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert isinstance(result.d_row_ends, cp.ndarray)
        assert result.d_row_ends.dtype == cp.int64

    @needs_gpu
    def test_delimiters_are_device_int64(self):
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n3,4\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert isinstance(result.d_delimiters, cp.ndarray)
        assert result.d_delimiters.dtype == cp.int64

    @needs_gpu
    def test_quote_parity_is_device_uint8(self):
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        assert isinstance(result.d_quote_parity, cp.ndarray)
        assert result.d_quote_parity.dtype == cp.uint8
        assert result.d_quote_parity.shape[0] == len(csv)

    @needs_gpu
    def test_row_end_positions_are_correct(self):
        """Row end positions point to actual \\n bytes."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n3,4\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        row_ends = cp.asnumpy(result.d_row_ends)
        # Positions of \n in the byte stream
        expected = [i for i, b in enumerate(csv) if b == ord("\n")]
        np.testing.assert_array_equal(row_ends, expected)

    @needs_gpu
    def test_delimiter_positions_are_correct(self):
        """Delimiter positions point to actual comma bytes outside quotes."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b'a,b\n"x,y",2\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        delim_positions = cp.asnumpy(result.d_delimiters)
        # Only commas outside quotes: position 1 (header) and 9 (data row)
        # csv:  a , b \n " x , y " , 2 \n
        # idx:  0 1 2  3 4 5 6 7 8 9 10 11
        # Comma at index 1: outside quotes -> delimiter
        # Comma at index 6: inside quotes -> NOT delimiter
        # Comma at index 9: outside quotes -> delimiter
        assert 1 in delim_positions
        assert 9 in delim_positions
        assert 6 not in delim_positions


class TestCsvValidation:
    """Error handling and validation."""

    @needs_gpu
    def test_invalid_delimiter_length(self):
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="single character"):
            csv_structural_analysis(d_bytes, delimiter=",,")

    @needs_gpu
    def test_inconsistent_columns(self):
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b,c\n1,2\n3,4,5\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="Inconsistent column count"):
            csv_structural_analysis(d_bytes)


class TestCsvQuoteParity:
    """Direct tests for the CSV quote parity kernel."""

    @needs_gpu
    def test_no_quotes(self):
        """No quotes -> all parity 0."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        csv = b"a,b\n1,2\n"
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        parity = cp.asnumpy(result.d_quote_parity)
        assert (parity == 0).all()

    @needs_gpu
    def test_quote_parity_toggle(self):
        """Quote parity toggles at " characters."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        # Byte: a , " x " , b \n
        # Idx:  0 1 2 3 4 5 6 7
        # Par:  0 0 0 1 0 0 0 0
        csv = b'a,"x",b\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        parity = cp.asnumpy(result.d_quote_parity)
        # Position 3 (x) should be inside quotes (parity 1)
        assert parity[3] == 1
        # Position 0 (a) and 6 (b) should be outside quotes
        assert parity[0] == 0
        assert parity[6] == 0

    @needs_gpu
    def test_doubled_quote_parity(self):
        """Doubled quotes cancel in parity."""
        from vibespatial.io.csv_gpu import csv_structural_analysis

        # CSV: "he""llo"  -> field value is: he"llo
        # Byte: " h e " " l l o " \n
        # Idx:  0 1 2 3 4 5 6 7 8 9
        # Tog:  1 0 0 1 1 0 0 0 1 0
        # Cum:  1 1 1 2 3 3 3 3 4 4
        # Par:  1 1 1 0 1 1 1 1 0 0
        # The "" at positions 3-4 toggles twice, restoring inside-quote state.
        csv = b'"he""llo"\n'
        d_bytes = _to_device_bytes(csv)
        result = csv_structural_analysis(d_bytes)

        parity = cp.asnumpy(result.d_quote_parity)
        # 'h' at index 1 is inside quotes
        assert parity[1] == 1
        # 'l' at index 5 is inside quotes (after doubled quote cancels)
        assert parity[5] == 1
        # \n at index 9 is outside quotes
        assert parity[9] == 0


# ===================================================================
# read_csv_gpu tests: lat/lon mode
# ===================================================================


class TestReadCsvGpuLatLon:
    """Coordinate extraction via lat/lon columns."""

    @needs_gpu
    def test_simple_lat_lon(self):
        """Simple CSV with lat,lon columns produces Point OGA with x=lon, y=lat."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"name,lat,lon\nAlice,40.7128,-74.0060\nBob,34.0522,-118.2437\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 2
        assert result.geometry.row_count == 2
        assert GeometryFamily.POINT in result.geometry.families

        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        # x = longitude, y = latitude (GIS convention)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)
        np.testing.assert_allclose(x[1], -118.2437, atol=1e-10)
        np.testing.assert_allclose(y[1], 34.0522, atol=1e-10)

    @needs_gpu
    def test_latitude_longitude_column_names(self):
        """Columns named 'latitude'/'longitude' are detected."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,latitude,longitude\n1,40.7128,-74.0060\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    @needs_gpu
    def test_xy_column_names(self):
        """Columns named 'y'/'x' are detected as lat/lon."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,x,y\n1,-74.0060,40.7128\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        # x column is longitude, y column is latitude
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    @needs_gpu
    def test_negative_coordinates(self):
        """Negative lat/lon values parse correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n-33.8688,151.2093\n-22.9068,-43.1729\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 2
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 151.2093, atol=1e-10)
        np.testing.assert_allclose(y[0], -33.8688, atol=1e-10)
        np.testing.assert_allclose(x[1], -43.1729, atol=1e-10)
        np.testing.assert_allclose(y[1], -22.9068, atol=1e-10)

    @needs_gpu
    def test_high_precision_coordinates(self):
        """High precision coordinates preserved within 1e-10 tolerance."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.71283456789012,-74.00601234567890\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.00601234567890, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.71283456789012, atol=1e-10)

    @needs_gpu
    def test_quoted_numeric_fields(self):
        """Quoted numeric fields like '"40.7128"' parse correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b'name,lat,lon\n"Alice","40.7128","-74.0060"\n'
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    @needs_gpu
    def test_tab_separated_lat_lon(self):
        """Tab-separated file with lat/lon columns."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"name\tlat\tlon\nAlice\t40.7128\t-74.0060\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes, delimiter="\t")

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    @needs_gpu
    def test_pipe_separated_lat_lon(self):
        """Pipe-separated file with lat/lon columns."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"name|lat|lon\nAlice|40.7128|-74.0060\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes, delimiter="|")

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)


# ===================================================================
# read_csv_gpu tests: WKT mode
# ===================================================================


class TestReadCsvGpuWkt:
    """Geometry extraction via WKT column."""

    @needs_gpu
    def test_geometry_column_with_point_wkt(self):
        """CSV with 'geometry' column containing WKT POINT values."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,geometry\n1,POINT(1.5 -2.3)\n2,POINT(3.0 4.0)\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 2
        assert GeometryFamily.POINT in result.geometry.families
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 1.5, atol=1e-10)
        np.testing.assert_allclose(y[0], -2.3, atol=1e-10)
        np.testing.assert_allclose(x[1], 3.0, atol=1e-10)
        np.testing.assert_allclose(y[1], 4.0, atol=1e-10)

    @needs_gpu
    def test_wkt_column_with_linestring(self):
        """CSV with 'wkt' column containing quoted WKT LINESTRING values.

        LINESTRING WKT contains commas, so the field MUST be quoted in CSV.
        """
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b'id,wkt\n1,"LINESTRING(0 0, 1 1, 2 2)"\n'
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        assert GeometryFamily.LINESTRING in result.geometry.families
        x, y = _get_device_coords(result.geometry, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 1.0, 2.0], atol=1e-10)

    @needs_gpu
    def test_quoted_wkt_with_commas(self):
        """Quoted WKT fields containing commas are handled correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        # WKT contains commas, so it must be quoted in CSV
        csv = b'id,geometry\n1,"LINESTRING(0 0, 1 1, 2 2)"\n'
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        assert GeometryFamily.LINESTRING in result.geometry.families
        x, y = _get_device_coords(result.geometry, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 1.0, 2.0], atol=1e-10)

    @needs_gpu
    def test_quoted_wkt_polygon(self):
        """Quoted WKT POLYGON field with commas parses correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b'id,geometry\n1,"POLYGON((0 0, 1 0, 1 1, 0 0))"\n'
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        assert GeometryFamily.POLYGON in result.geometry.families
        x, y = _get_device_coords(result.geometry, GeometryFamily.POLYGON)
        assert len(x) == 4
        np.testing.assert_allclose(x, [0.0, 1.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 0.0, 1.0, 0.0], atol=1e-10)


# ===================================================================
# read_csv_gpu tests: auto-detection and user overrides
# ===================================================================


class TestReadCsvGpuAutoDetection:
    """Auto-detection of spatial columns and user overrides."""

    @needs_gpu
    def test_auto_detects_lat_lon(self):
        """Auto-detects lat/lon columns by name without user hints."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,lon,value\n1,40.7,-74.0,100\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7, atol=1e-10)

    @needs_gpu
    def test_auto_detects_geometry_column(self):
        """Auto-detects geometry column by name."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,geometry\n1,POINT(1 2)\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 2.0, atol=1e-10)

    @needs_gpu
    def test_user_override_lat_lon_columns(self):
        """User override: lat_col='custom_lat', lon_col='custom_lon'."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,custom_lat,custom_lon\n1,40.7128,-74.0060\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes, lat_col="custom_lat", lon_col="custom_lon")

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0060, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    @needs_gpu
    def test_user_override_geom_column(self):
        """User override: geom_col='my_wkt'."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,my_wkt\n1,POINT(5 10)\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes, geom_col="my_wkt")

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 5.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 10.0, atol=1e-10)

    @needs_gpu
    def test_geometry_column_takes_precedence_over_lat_lon(self):
        """When both geometry and lat/lon columns exist, geometry wins."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon,geometry\n40.7,-74.0,POINT(99 88)\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        # Should use geometry column, not lat/lon
        np.testing.assert_allclose(x[0], 99.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 88.0, atol=1e-10)


# ===================================================================
# read_csv_gpu tests: edge cases
# ===================================================================


class TestReadCsvGpuEdgeCases:
    """Edge cases for the full CSV reader."""

    @needs_gpu
    def test_no_trailing_newline(self):
        """File not ending with newline parses correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7, atol=1e-10)

    @needs_gpu
    def test_crlf_line_endings(self):
        """Windows-style CRLF line endings parse correctly."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\r\n40.7,-74.0\r\n34.0,-118.2\r\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 2
        x, y = _get_device_coords(result.geometry, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -74.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7, atol=1e-10)
        np.testing.assert_allclose(x[1], -118.2, atol=1e-10)
        np.testing.assert_allclose(y[1], 34.0, atol=1e-10)

    @needs_gpu
    def test_single_data_row(self):
        """CSV with exactly one data row."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 1
        assert result.geometry.row_count == 1

    @needs_gpu
    def test_empty_data_rows(self):
        """CSV with header only (no data rows) returns empty geometry."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 0
        assert result.geometry.row_count == 0


# ===================================================================
# read_csv_gpu tests: error cases
# ===================================================================


class TestReadCsvGpuErrors:
    """Error handling for the full CSV reader."""

    @needs_gpu
    def test_no_spatial_columns_found(self):
        """No spatial columns detected raises clear error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"name,value\nAlice,100\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="No spatial columns"):
            read_csv_gpu(d_bytes)

    @needs_gpu
    def test_only_lat_without_lon(self):
        """Only lat column without lon results in no spatial columns."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,value\n1,40.7,100\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="No spatial columns"):
            read_csv_gpu(d_bytes)

    @needs_gpu
    def test_lat_col_not_found(self):
        """Specified lat_col not in header raises error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,lon\n1,40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="not found"):
            read_csv_gpu(d_bytes, lat_col="missing_lat", lon_col="lon")

    @needs_gpu
    def test_lon_col_not_found(self):
        """Specified lon_col not in header raises error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,lon\n1,40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="not found"):
            read_csv_gpu(d_bytes, lat_col="lat", lon_col="missing_lon")

    @needs_gpu
    def test_geom_col_not_found(self):
        """Specified geom_col not in header raises error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,geometry\n1,POINT(0 0)\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="not found"):
            read_csv_gpu(d_bytes, geom_col="missing_geom")

    @needs_gpu
    def test_lat_col_without_lon_col_raises(self):
        """Specifying lat_col without lon_col raises error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,lon\n1,40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="Both lat_col and lon_col"):
            read_csv_gpu(d_bytes, lat_col="lat")

    @needs_gpu
    def test_lon_col_without_lat_col_raises(self):
        """Specifying lon_col without lat_col raises error."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"id,lat,lon\n1,40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)

        with pytest.raises(ValueError, match="Both lat_col and lon_col"):
            read_csv_gpu(d_bytes, lon_col="lon")


# ===================================================================
# read_csv_gpu tests: OGA structure verification
# ===================================================================


class TestReadCsvGpuOgaStructure:
    """Verify OwnedGeometryArray structure from CSV reader."""

    @needs_gpu
    def test_point_family_in_output(self):
        """Lat/lon mode produces Point family in the OGA."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0\n34.0,-118.2\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert GeometryFamily.POINT in result.geometry.families
        assert len(result.geometry.families) == 1
        buf = result.geometry.families[GeometryFamily.POINT]
        assert buf.row_count == 2

    @needs_gpu
    def test_row_count_matches_input(self):
        """Output row count matches number of input data rows."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.n_rows == 5
        assert result.geometry.row_count == 5

    @needs_gpu
    def test_device_state_populated(self):
        """Verify device state is populated after GPU CSV parse."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        assert result.geometry.device_state is not None

    @needs_gpu
    def test_validity_mask_all_valid(self):
        """All rows have valid geometry in lat/lon mode."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0\n34.0,-118.2\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        validity = result.geometry.validity
        assert all(validity)

    @needs_gpu
    def test_tags_all_point(self):
        """All rows have Point family tag in lat/lon mode."""
        from vibespatial.io.csv_gpu import read_csv_gpu

        csv = b"lat,lon\n40.7,-74.0\n34.0,-118.2\n"
        d_bytes = _to_device_bytes(csv)
        result = read_csv_gpu(d_bytes)

        tags = result.geometry.tags
        # GeometryFamily.POINT ordinal is 0
        assert all(t == 0 for t in tags)


# ===================================================================
# IO dispatch wiring tests
# ===================================================================


class TestCsvIODispatchWiring:
    """Verify CSV is registered in the IO support matrix and file dispatch."""

    def test_csv_in_io_format_enum(self):
        from vibespatial.io.support import IOFormat

        assert hasattr(IOFormat, "CSV")
        assert IOFormat.CSV == "csv"

    def test_csv_support_matrix_read_is_hybrid(self):
        from vibespatial.io.support import (
            IOFormat,
            IOOperation,
            IOPathKind,
            plan_io_support,
        )

        plan = plan_io_support(IOFormat.CSV, IOOperation.READ)
        assert plan.selected_path == IOPathKind.HYBRID

    def test_csv_support_matrix_write_is_fallback(self):
        from vibespatial.io.support import (
            IOFormat,
            IOOperation,
            IOPathKind,
            plan_io_support,
        )

        plan = plan_io_support(IOFormat.CSV, IOOperation.WRITE)
        assert plan.selected_path == IOPathKind.FALLBACK

    def test_csv_not_canonical_gpu(self):
        from vibespatial.io.support import IOFormat, IOOperation, plan_io_support

        plan = plan_io_support(IOFormat.CSV, IOOperation.READ)
        assert not plan.canonical_gpu

    def test_csv_file_extension_routing(self):
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("test.csv", operation=IOOperation.READ)
        assert plan.format == IOFormat.CSV
        assert plan.driver == "CSV"
        assert plan.implementation == "csv_gpu_hybrid_adapter"

    def test_tsv_file_extension_routing(self):
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("data.tsv", operation=IOOperation.READ)
        assert plan.format == IOFormat.CSV
        assert plan.driver == "CSV"
        assert plan.implementation == "csv_gpu_hybrid_adapter"

    def test_csv_file_extension_routing_uppercase(self):
        """File extension detection is case-insensitive via Path.suffix.lower()."""
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("data.CSV", operation=IOOperation.READ)
        assert plan.format == IOFormat.CSV
