"""Tests for GPU DBF reader: fixed-width record parsing on GPU.

Tests cover:
- Header parsing (CPU, unit tests)
- Numeric field extraction (GPU NVRTC kernel)
- Date field extraction (GPU NVRTC kernel)
- Logical field extraction (GPU CuPy)
- Character field extraction (GPU gather + CPU decode)
- Deletion flag handling
- Column subsetting
- Integration with real .dbf files from Shapefiles
"""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


# ---------------------------------------------------------------------------
# DBF file builder helpers
# ---------------------------------------------------------------------------


def _build_dbf_bytes(
    fields: list[tuple[str, str, int, int]],
    records: list[list[bytes]],
    *,
    deleted: list[bool] | None = None,
) -> bytes:
    """Build a minimal dBASE III DBF file in memory.

    Parameters
    ----------
    fields : list of (name, type, length, decimal_count)
        Field descriptors. name is up to 11 chars, type is one of
        'C', 'N', 'F', 'D', 'L'.
    records : list of list of bytes
        Each inner list has one bytes value per field, right-padded
        to the declared field length.
    deleted : list of bool, optional
        If provided, marks records as deleted (True = deleted).

    Returns
    -------
    bytes
        Complete DBF file content.
    """
    n_records = len(records)
    n_fields = len(fields)

    # Record length = 1 (deletion flag) + sum of field lengths
    record_length = 1 + sum(f[2] for f in fields)
    # Header length = 32 (main header) + 32 * n_fields + 1 (terminator)
    header_length = 32 + 32 * n_fields + 1

    # -- Main header (32 bytes) --
    header = bytearray(32)
    header[0] = 0x03  # dBASE III version
    header[1] = 26    # year (2026 - 1900 = 126, but we use 26 for simplicity)
    header[2] = 3     # month
    header[3] = 26    # day
    struct.pack_into("<I", header, 4, n_records)
    struct.pack_into("<H", header, 8, header_length)
    struct.pack_into("<H", header, 10, record_length)

    # -- Field descriptors (32 bytes each) --
    field_descriptors = bytearray()
    for name, ftype, flength, fdecimal in fields:
        desc = bytearray(32)
        name_bytes = name.encode("ascii")[:11]
        desc[0 : len(name_bytes)] = name_bytes
        desc[11] = ord(ftype)
        desc[16] = flength
        desc[17] = fdecimal
        field_descriptors.extend(desc)

    # -- Terminator --
    terminator = b"\r"

    # -- Records --
    record_bytes = bytearray()
    for i, rec in enumerate(records):
        is_deleted = deleted[i] if deleted else False
        flag = b"*" if is_deleted else b" "
        record_bytes.extend(flag)
        for j, field_val in enumerate(rec):
            field_length = fields[j][2]
            padded = field_val.ljust(field_length, b" ")[:field_length]
            record_bytes.extend(padded)

    return bytes(header) + bytes(field_descriptors) + terminator + bytes(record_bytes)


def _write_dbf_file(
    path: Path,
    fields: list[tuple[str, str, int, int]],
    records: list[list[bytes]],
    **kwargs,
) -> Path:
    """Write a DBF file to disk."""
    data = _build_dbf_bytes(fields, records, **kwargs)
    path.write_bytes(data)
    return path


# ===================================================================
# Header parsing tests (CPU)
# ===================================================================


class TestDbfHeaderParsing:
    """Unit tests for CPU-side header parsing."""

    def test_basic_header(self):
        """Parse a simple 2-field header."""
        from vibespatial.io.dbf_gpu import _parse_header

        fields = [
            ("NAME", "C", 20, 0),
            ("VALUE", "N", 10, 2),
        ]
        raw = _build_dbf_bytes(fields, [])
        header = _parse_header(raw)

        assert header.version == 0x03
        assert header.record_count == 0
        assert len(header.fields) == 2
        assert header.fields[0].name == "NAME"
        assert header.fields[0].field_type == "C"
        assert header.fields[0].length == 20
        assert header.fields[1].name == "VALUE"
        assert header.fields[1].field_type == "N"
        assert header.fields[1].length == 10
        assert header.fields[1].decimal_count == 2

    def test_field_offsets_are_cumulative(self):
        """Field offsets should be cumulative byte positions within the record."""
        from vibespatial.io.dbf_gpu import _parse_header

        fields = [
            ("A", "C", 5, 0),
            ("B", "N", 10, 0),
            ("C", "C", 8, 0),
        ]
        raw = _build_dbf_bytes(fields, [])
        header = _parse_header(raw)

        assert header.fields[0].offset == 0
        assert header.fields[1].offset == 5
        assert header.fields[2].offset == 15

    def test_header_too_short_raises(self):
        """Header shorter than 32 bytes should raise ValueError."""
        from vibespatial.io.dbf_gpu import _parse_header

        with pytest.raises(ValueError, match="too short"):
            _parse_header(b"\x00" * 20)

    def test_record_length_matches(self):
        """record_length should be 1 (flag) + sum(field lengths)."""
        from vibespatial.io.dbf_gpu import _parse_header

        fields = [
            ("X", "N", 12, 4),
            ("Y", "N", 12, 4),
        ]
        records = [[b"   1.2345   ", b"  -6.7890   "]]
        raw = _build_dbf_bytes(fields, records)
        header = _parse_header(raw)

        assert header.record_length == 1 + 12 + 12
        assert header.record_count == 1


# ===================================================================
# Numeric field extraction (GPU)
# ===================================================================


class TestDbfNumericExtraction:
    """Test GPU NVRTC kernel for numeric (N/F) field parsing."""

    @needs_gpu
    def test_simple_integers(self):
        """Parse integer values from numeric fields."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("COUNT", "N", 10, 0)]
        records = [
            [b"       100"],
            [b"        42"],
            [b"         0"],
            [b"       -17"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        assert result.n_records == 4
        col = result.columns["COUNT"]
        assert col.dtype == "float64"
        values = cp.asnumpy(col.data)
        np.testing.assert_allclose(values, [100.0, 42.0, 0.0, -17.0])

    @needs_gpu
    def test_decimal_values(self):
        """Parse decimal float values from numeric fields."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("AREA", "N", 15, 4)]
        records = [
            [b"      123.4567"],
            [b"       -0.0010"],
            [b"    99999.9999"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        values = cp.asnumpy(result.columns["AREA"].data)
        np.testing.assert_allclose(values, [123.4567, -0.001, 99999.9999], rtol=1e-10)

    @needs_gpu
    def test_float_type_field(self):
        """Parse 'F' (float) type fields -- same kernel as 'N'."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("TEMP", "F", 10, 2)]
        records = [
            [b"     98.60"],
            [b"    -40.00"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        values = cp.asnumpy(result.columns["TEMP"].data)
        np.testing.assert_allclose(values, [98.6, -40.0])

    @needs_gpu
    def test_null_numeric_as_nan(self):
        """Empty or space-only numeric fields should produce NaN."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("VAL", "N", 10, 2)]
        records = [
            [b"      1.50"],
            [b"          "],  # all spaces = null
            [b"      3.75"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        values = cp.asnumpy(result.columns["VAL"].data)
        null_mask = cp.asnumpy(result.columns["VAL"].null_mask)

        assert values[0] == pytest.approx(1.5)
        assert np.isnan(values[1])
        assert values[2] == pytest.approx(3.75)
        assert not null_mask[0]
        assert null_mask[1]
        assert not null_mask[2]

    @needs_gpu
    def test_star_fill_numeric(self):
        """'*' fill pattern (overflow indicator) should produce NaN."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("X", "N", 8, 2)]
        records = [
            [b"   1.00 "],
            [b"********"],  # overflow fill
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        values = cp.asnumpy(result.columns["X"].data)
        assert values[0] == pytest.approx(1.0)
        assert np.isnan(values[1])


# ===================================================================
# Date field extraction (GPU)
# ===================================================================


class TestDbfDateExtraction:
    """Test GPU NVRTC kernel for date (D) field parsing."""

    @needs_gpu
    def test_simple_dates(self):
        """Parse YYYYMMDD date fields to int32."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("CREATED", "D", 8, 0)]
        records = [
            [b"20260326"],
            [b"19991231"],
            [b"20000101"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        col = result.columns["CREATED"]
        assert col.dtype == "int32"
        values = cp.asnumpy(col.data)
        np.testing.assert_array_equal(values, [20260326, 19991231, 20000101])

    @needs_gpu
    def test_null_date(self):
        """Empty date field should produce 0 with null_mask=True."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("DT", "D", 8, 0)]
        records = [
            [b"20260101"],
            [b"        "],  # null date
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        values = cp.asnumpy(result.columns["DT"].data)
        null_mask = cp.asnumpy(result.columns["DT"].null_mask)
        assert values[0] == 20260101
        assert values[1] == 0
        assert not null_mask[0]
        assert null_mask[1]


# ===================================================================
# Logical field extraction (GPU CuPy)
# ===================================================================


class TestDbfLogicalExtraction:
    """Test CuPy element-wise logical (L) field parsing."""

    @needs_gpu
    def test_logical_values(self):
        """Parse T/F/Y/N/? logical values."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("FLAG", "L", 1, 0)]
        records = [
            [b"T"],
            [b"F"],
            [b"Y"],
            [b"N"],
            [b"?"],
            [b"t"],
            [b"f"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        col = result.columns["FLAG"]
        assert col.dtype == "bool"
        values = cp.asnumpy(col.data)
        null_mask = cp.asnumpy(col.null_mask)

        # T, Y, t are True; F, N, f are False; ? is null
        assert values[0] == 1  # T
        assert values[1] == 0  # F
        assert values[2] == 1  # Y
        assert values[3] == 0  # N
        assert values[4] == 0  # ? (value is 0, but null_mask is True)
        assert values[5] == 1  # t
        assert values[6] == 0  # f

        assert not null_mask[0]
        assert not null_mask[1]
        assert not null_mask[2]
        assert not null_mask[3]
        assert null_mask[4]      # ? is null
        assert not null_mask[5]
        assert not null_mask[6]


# ===================================================================
# Character field extraction
# ===================================================================


class TestDbfCharacterExtraction:
    """Test character (C) field extraction with trailing space stripping."""

    @needs_gpu
    def test_simple_strings(self):
        """Parse character fields, stripping trailing spaces."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("NAME", "C", 15, 0)]
        records = [
            [b"Alice          "],
            [b"Bob            "],
            [b"Charlie Brown  "],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        col = result.columns["NAME"]
        assert col.dtype == "string"
        assert list(col.data) == ["Alice", "Bob", "Charlie Brown"]

    @needs_gpu
    def test_empty_string(self):
        """All-spaces character field should be empty string with null_mask."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("LABEL", "C", 10, 0)]
        records = [
            [b"Hello     "],
            [b"          "],  # empty
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        col = result.columns["LABEL"]
        assert col.data[0] == "Hello"
        assert col.data[1] == ""
        assert not col.null_mask[0]
        assert col.null_mask[1]


# ===================================================================
# Deletion flag handling
# ===================================================================


class TestDbfDeletionFlags:
    """Test deletion flag extraction and filtering."""

    @needs_gpu
    def test_active_mask(self):
        """Active records have mask=1, deleted records have mask=0."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("X", "N", 5, 0)]
        records = [
            [b"    1"],
            [b"    2"],
            [b"    3"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records, deleted=[False, True, False])
            result = read_dbf_gpu(path)

        active = cp.asnumpy(result.active_mask)
        np.testing.assert_array_equal(active, [1, 0, 1])

    @needs_gpu
    def test_dataframe_filters_deleted(self):
        """dbf_result_to_dataframe should exclude deleted records by default."""
        from vibespatial.io.dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu

        fields = [("VAL", "N", 5, 0)]
        records = [
            [b"   10"],
            [b"   20"],
            [b"   30"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records, deleted=[False, True, False])
            result = read_dbf_gpu(path)
            df = dbf_result_to_dataframe(result)

        assert len(df) == 2
        np.testing.assert_allclose(df["VAL"].values, [10.0, 30.0])

    @needs_gpu
    def test_dataframe_include_deleted(self):
        """dbf_result_to_dataframe with include_deleted=True keeps all."""
        from vibespatial.io.dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu

        fields = [("VAL", "N", 5, 0)]
        records = [
            [b"   10"],
            [b"   20"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records, deleted=[False, True])
            result = read_dbf_gpu(path)
            df = dbf_result_to_dataframe(result, include_deleted=True)

        assert len(df) == 2


# ===================================================================
# Column subsetting
# ===================================================================


class TestDbfColumnSubsetting:
    """Test reading only a subset of columns."""

    @needs_gpu
    def test_select_columns(self):
        """Only requested columns should appear in result."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [
            ("NAME", "C", 10, 0),
            ("X", "N", 10, 4),
            ("Y", "N", 10, 4),
            ("FLAG", "L", 1, 0),
        ]
        records = [
            [b"Point A   ", b"   1.0000 ", b"   2.0000 ", b"T"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path, columns=["X", "Y"])

        assert set(result.columns.keys()) == {"X", "Y"}
        x_val = cp.asnumpy(result.columns["X"].data)
        np.testing.assert_allclose(x_val, [1.0], rtol=1e-4)


# ===================================================================
# Mixed field types
# ===================================================================


class TestDbfMixedFields:
    """Test DBF files with multiple field types together."""

    @needs_gpu
    def test_mixed_types(self):
        """Parse a file with character, numeric, date, and logical fields."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [
            ("NAME", "C", 15, 0),
            ("AREA", "N", 12, 4),
            ("CREATED", "D", 8, 0),
            ("ACTIVE", "L", 1, 0),
        ]
        records = [
            [b"Region A       ", b"   1234.5678", b"20260101", b"T"],
            [b"Region B       ", b"    567.8900", b"20250615", b"F"],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        assert result.n_records == 2
        assert set(result.columns.keys()) == {"NAME", "AREA", "CREATED", "ACTIVE"}

        # Check types
        assert result.columns["NAME"].dtype == "string"
        assert result.columns["AREA"].dtype == "float64"
        assert result.columns["CREATED"].dtype == "int32"
        assert result.columns["ACTIVE"].dtype == "bool"

        # Check values
        assert list(result.columns["NAME"].data) == ["Region A", "Region B"]
        np.testing.assert_allclose(
            cp.asnumpy(result.columns["AREA"].data),
            [1234.5678, 567.89],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.columns["CREATED"].data),
            [20260101, 20250615],
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.columns["ACTIVE"].data),
            [1, 0],
        )


# ===================================================================
# Edge cases
# ===================================================================


class TestDbfEdgeCases:
    """Edge case and boundary condition tests."""

    @needs_gpu
    def test_empty_file(self):
        """File with zero records."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("X", "N", 10, 0)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, [])
            result = read_dbf_gpu(path)

        assert result.n_records == 0
        assert len(result.columns) == 0

    @needs_gpu
    def test_single_record(self):
        """File with exactly one record."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        fields = [("VAL", "N", 5, 0)]
        records = [[b"   42"]]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        assert result.n_records == 1
        values = cp.asnumpy(result.columns["VAL"].data)
        np.testing.assert_allclose(values, [42.0])

    @needs_gpu
    def test_many_records(self):
        """File with 10,000 records -- exercises GPU parallelism."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        n = 10_000
        fields = [("IDX", "N", 10, 0)]
        records = [[f"{i:10d}".encode()] for i in range(n)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        assert result.n_records == n
        values = cp.asnumpy(result.columns["IDX"].data)
        np.testing.assert_allclose(values, np.arange(n, dtype=np.float64))

    @needs_gpu
    def test_many_columns(self):
        """File with 20 columns."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        n_cols = 20
        fields = [(f"C{i:02d}", "N", 8, 2) for i in range(n_cols)]
        records = [[f"  {i + j:.2f}  ".encode()[:8] for j in range(n_cols)] for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            _write_dbf_file(path, fields, records)
            result = read_dbf_gpu(path)

        assert result.n_records == 5
        assert len(result.columns) == n_cols

    @needs_gpu
    def test_file_too_small_raises(self):
        """File smaller than 32 bytes should raise ValueError."""
        from vibespatial.io.dbf_gpu import read_dbf_gpu

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.dbf"
            path.write_bytes(b"\x00" * 20)
            with pytest.raises(ValueError, match="too small"):
                read_dbf_gpu(path)


# ===================================================================
# Integration with real Shapefile DBF
# ===================================================================


class TestDbfShapefileIntegration:
    """Test reading DBF files generated by Shapefile writers."""

    @needs_gpu
    def test_fiona_generated_dbf(self):
        """Read a DBF sidecar from a Shapefile written by fiona/pyogrio."""
        try:
            from shapely.geometry import Point

            import geopandas as gpd
        except ImportError:
            pytest.skip("geopandas/shapely not available")

        # Create a simple GeoDataFrame and write to Shapefile
        gdf = gpd.GeoDataFrame(
            {
                "name": ["A", "B", "C"],
                "value": [1.5, 2.7, 3.9],
                "count": [10, 20, 30],
            },
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        )

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "test.shp"
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            # Read the .dbf sidecar directly
            dbf_path = Path(td) / "test.dbf"
            assert dbf_path.exists()

            from vibespatial.io.dbf_gpu import read_dbf_gpu

            result = read_dbf_gpu(dbf_path)

        assert result.n_records == 3

        # Check that column names match (Shapefile truncates to 10 chars)
        assert "name" in result.columns
        assert "value" in result.columns
        assert "count" in result.columns

        # Verify numeric values
        values = cp.asnumpy(result.columns["value"].data)
        np.testing.assert_allclose(values, [1.5, 2.7, 3.9], rtol=1e-6)

        # Verify string values
        names = list(result.columns["name"].data)
        assert names == ["A", "B", "C"]
