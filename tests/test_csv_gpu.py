"""Tests for GPU CSV reader: structural analysis stage."""
from __future__ import annotations

import numpy as np
import pytest

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
