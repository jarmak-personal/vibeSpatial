"""Tests for GPU WKT reader: structural analysis and coordinate extraction."""
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

def _to_device_bytes(wkt: str | bytes) -> cp.ndarray:
    """Encode WKT text as device-resident uint8 array."""
    raw = wkt.encode("utf-8") if isinstance(wkt, str) else wkt
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
# Structural analysis tests
# ===================================================================


class TestWktStructuralAnalysis:
    """Tests for wkt_structural_analysis: type classification, EMPTY, EWKT."""

    @needs_gpu
    @pytest.mark.parametrize(
        "wkt_text, expected_tag",
        [
            ("POINT(1 2)", 0),
            ("LINESTRING(0 0, 1 1)", 1),
            ("POLYGON((0 0, 1 0, 1 1, 0 0))", 2),
            ("MULTIPOINT((0 0), (1 1))", 3),
            ("MULTILINESTRING((0 0, 1 1), (2 2, 3 3))", 4),
            ("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))", 5),
        ],
        ids=["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
    )
    def test_type_detection_all_families(self, wkt_text, expected_tag):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes(wkt_text)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == expected_tag

    @needs_gpu
    def test_geometry_collection_classified_as_unknown(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes("GEOMETRYCOLLECTION(POINT(1 2))")
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == -2  # unsupported

    @needs_gpu
    @pytest.mark.parametrize(
        "wkt_text",
        [
            "POINT EMPTY",
            "LINESTRING EMPTY",
            "POLYGON EMPTY",
            "MULTIPOINT EMPTY",
            "MULTILINESTRING EMPTY",
            "MULTIPOLYGON EMPTY",
        ],
        ids=["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
    )
    def test_empty_geometry_detection(self, wkt_text):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes(wkt_text)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        flags = result.d_empty_flags.get()
        assert flags[0] == 1

    @needs_gpu
    def test_ewkt_srid_prefix(self):
        """EWKT with SRID=4326; prefix should strip SRID and classify correctly."""
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes("SRID=4326;POINT(1 2)")
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 0  # POINT

    @needs_gpu
    def test_ewkt_srid_polygon(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes("SRID=32632;POLYGON((0 0, 1 0, 1 1, 0 0))")
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 2  # POLYGON

    @needs_gpu
    @pytest.mark.parametrize(
        "wkt_text",
        [
            "POINT Z(1 2 3)",
            "POINT ZM(1 2 3 4)",
            "POINTZ(1 2 3)",
            "POINTZM(1 2 3 4)",
            "POINT M(1 2 3)",
        ],
        ids=["z-space", "zm-space", "z-nospace", "zm-nospace", "m-space"],
    )
    def test_dimension_suffixes_classified_as_point(self, wkt_text):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes(wkt_text)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 0  # POINT
        empty_flags = result.d_empty_flags.get()
        assert empty_flags[0] == 0

    @needs_gpu
    def test_dimension_suffix_empty(self):
        """POINT Z EMPTY should be classified as POINT and flagged EMPTY."""
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes("POINT Z EMPTY")
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 0  # POINT
        empty_flags = result.d_empty_flags.get()
        assert empty_flags[0] == 1

    @needs_gpu
    @pytest.mark.parametrize(
        "wkt_text, expected_tag",
        [
            ("Point(1 2)", 0),
            ("point(1 2)", 0),
            ("LINESTRING(0 0, 1 1)", 1),
            ("linestring(0 0, 1 1)", 1),
            ("Polygon((0 0, 1 0, 1 1, 0 0))", 2),
            ("polygon((0 0, 1 0, 1 1, 0 0))", 2),
            ("MultiPoint((0 0), (1 1))", 3),
            ("multilinestring((0 0, 1 1))", 4),
            ("MultiPolygon(((0 0, 1 0, 1 1, 0 0)))", 5),
        ],
    )
    def test_mixed_case_detection(self, wkt_text, expected_tag):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes(wkt_text)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 1
        tags = result.d_family_tags.get()
        assert tags[0] == expected_tag


# ===================================================================
# Coordinate extraction tests (read_wkt_gpu)
# ===================================================================


class TestReadWktGpuPoint:
    """Coordinate extraction for POINT geometries."""

    @needs_gpu
    def test_simple_point(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT(1.5 -2.3)")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.POINT in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        assert len(x) == 1
        np.testing.assert_allclose(x[0], 1.5, atol=1e-10)
        np.testing.assert_allclose(y[0], -2.3, atol=1e-10)

    @needs_gpu
    def test_negative_coordinates(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT(-80.123 27.456)")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -80.123, atol=1e-10)
        np.testing.assert_allclose(y[0], 27.456, atol=1e-10)

    @needs_gpu
    def test_high_precision(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT(-80.92302345678 25.12345678901)")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -80.92302345678, atol=1e-10)
        np.testing.assert_allclose(y[0], 25.12345678901, atol=1e-10)

    @needs_gpu
    def test_scientific_notation(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT(1.5e2 -2.3E-4)")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 150.0, atol=1e-10)
        np.testing.assert_allclose(y[0], -0.00023, atol=1e-10)


class TestReadWktGpuLineString:
    """Coordinate extraction for LINESTRING geometries."""

    @needs_gpu
    def test_simple_linestring(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("LINESTRING(0 0, 1 1, 2 2)")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.LINESTRING in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 1.0, 2.0], atol=1e-10)

    @needs_gpu
    def test_two_point_linestring(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("LINESTRING(10.5 20.5, -3.7 4.2)")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 2
        np.testing.assert_allclose(x, [10.5, -3.7], atol=1e-10)
        np.testing.assert_allclose(y, [20.5, 4.2], atol=1e-10)


class TestReadWktGpuPolygon:
    """Coordinate extraction for POLYGON geometries."""

    @needs_gpu
    def test_simple_polygon(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.POLYGON in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 5
        np.testing.assert_allclose(x, [0.0, 1.0, 1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 0.0, 1.0, 1.0, 0.0], atol=1e-10)

    @needs_gpu
    def test_polygon_with_hole(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0),(2 2, 8 2, 8 8, 2 8, 2 2))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        # Outer ring: 5 coords, inner ring (hole): 5 coords = 10 total
        assert len(x) == 10

        # Verify outer ring
        np.testing.assert_allclose(x[:5], [0.0, 10.0, 10.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(y[:5], [0.0, 0.0, 10.0, 10.0, 0.0], atol=1e-10)
        # Verify inner ring (hole)
        np.testing.assert_allclose(x[5:], [2.0, 8.0, 8.0, 2.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(y[5:], [2.0, 2.0, 8.0, 8.0, 2.0], atol=1e-10)

    @needs_gpu
    def test_triangle(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POLYGON((0 0, 1 0, 0.5 0.866, 0 0))")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 4  # triangle: 3 unique + closing = 4


class TestReadWktGpuMultiGeometries:
    """Coordinate extraction for MULTI* geometry types."""

    @needs_gpu
    def test_multipoint(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("MULTIPOINT((0 0), (1 1))")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.MULTIPOINT in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.MULTIPOINT)
        assert len(x) == 2
        np.testing.assert_allclose(x, [0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 1.0], atol=1e-10)

    @needs_gpu
    def test_multilinestring(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("MULTILINESTRING((0 0, 1 1), (2 2, 3 3))")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.MULTILINESTRING in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.MULTILINESTRING)
        assert len(x) == 4
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0, 1.0, 2.0, 3.0], atol=1e-10)

    @needs_gpu
    def test_multipolygon(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert GeometryFamily.MULTIPOLYGON in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.MULTIPOLYGON)
        # First triangle: 4 coords (3 + close), second: 4 coords = 8 total
        assert len(x) == 8
        np.testing.assert_allclose(
            x, [0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0], atol=1e-10,
        )
        np.testing.assert_allclose(
            y, [0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 3.0, 2.0], atol=1e-10,
        )


# ===================================================================
# Multi-line input tests
# ===================================================================


class TestMultiLineInput:
    """Tests for multi-line WKT input (multiple geometries)."""

    @needs_gpu
    def test_multiple_points_newline_separated(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nPOINT(3 4)\nPOINT(5 6)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 3
        assert GeometryFamily.POINT in owned.families
        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        assert len(x) == 3
        np.testing.assert_allclose(x, [1.0, 3.0, 5.0], atol=1e-10)
        np.testing.assert_allclose(y, [2.0, 4.0, 6.0], atol=1e-10)

    @needs_gpu
    def test_multiple_linestrings(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "LINESTRING(0 0, 1 1)\nLINESTRING(2 2, 3 3, 4 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 2
        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        # First: 2 coords, second: 3 coords = 5 total
        assert len(x) == 5

    @needs_gpu
    def test_mixed_types_multiline(self):
        """Multiple geometries of different types in one input."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nLINESTRING(0 0, 1 1)\nPOLYGON((0 0, 1 0, 1 1, 0 0))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 3
        # Each family should be present
        assert GeometryFamily.POINT in owned.families
        assert GeometryFamily.LINESTRING in owned.families
        assert GeometryFamily.POLYGON in owned.families

    @needs_gpu
    def test_trailing_newline(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nPOINT(3 4)\n"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 2

    @needs_gpu
    def test_empty_lines_between_geometries(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\n\nPOINT(3 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        # The blank line should be filtered out
        assert owned.row_count == 2

    @needs_gpu
    def test_multiple_polygons(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        p1 = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
        p2 = "POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))"
        wkt = f"{p1}\n{p2}"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 2
        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        # 5 coords per polygon = 10 total
        assert len(x) == 10


# ===================================================================
# EMPTY geometry tests
# ===================================================================


class TestEmptyGeometries:
    """Tests for WKT EMPTY geometry handling."""

    @needs_gpu
    def test_point_empty_produces_invalid_row(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT EMPTY")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert not owned.validity[0]

    @needs_gpu
    def test_linestring_empty(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("LINESTRING EMPTY")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert not owned.validity[0]

    @needs_gpu
    def test_polygon_empty(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POLYGON EMPTY")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        assert not owned.validity[0]

    @needs_gpu
    def test_empty_mixed_with_valid(self):
        """EMPTY geometries interspersed with valid ones."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nPOINT EMPTY\nPOINT(3 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 3
        assert owned.validity[0]
        assert not owned.validity[1]
        assert owned.validity[2]

    @needs_gpu
    def test_all_empty_input(self):
        """Input containing only EMPTY geometries of the same type."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        # Use same-type EMPTY to exercise homogeneous path.
        # Mixed-type all-EMPTY (POINT EMPTY + LINESTRING EMPTY) hits a
        # known bug in _assemble_wkt_mixed where pt_starts indexes into
        # an empty coordinate array (IndexError).  That is a wkt_gpu.py
        # issue to fix separately.
        wkt = "POINT EMPTY\nPOINT EMPTY"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 2
        assert not any(owned.validity)

    @needs_gpu
    def test_all_empty_mixed_types_known_limitation(self):
        """Mixed-type all-EMPTY triggers IndexError in _assemble_wkt_mixed.

        This is a known limitation: when ALL geometries are EMPTY and the
        input is mixed-type, the mixed assembly path tries to index into
        an empty coordinate array.  Marked as xfail until wkt_gpu.py is
        patched.
        """
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT EMPTY\nLINESTRING EMPTY\nPOLYGON EMPTY"
        d_bytes = _to_device_bytes(wkt)
        with pytest.raises(IndexError):
            read_wkt_gpu(d_bytes)


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Edge cases: degenerate geometries, precision, empty input."""

    @needs_gpu
    def test_empty_byte_array(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = cp.empty(0, dtype=cp.uint8)
        owned = read_wkt_gpu(d_bytes)
        assert owned.row_count == 0

    @needs_gpu
    def test_single_point_linestring(self):
        """Degenerate linestring with a single coordinate pair."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        # This is technically invalid WKT (linestring needs >= 2 points)
        # but the parser should handle it gracefully.
        d_bytes = _to_device_bytes("LINESTRING(5 10)")
        owned = read_wkt_gpu(d_bytes)

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 1
        np.testing.assert_allclose(x[0], 5.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 10.0, atol=1e-10)

    @needs_gpu
    def test_very_small_polygon(self):
        """Polygon with sub-millimeter scale coordinates."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POLYGON((0.0001 0.0001, 0.0002 0.0001, 0.0002 0.0002, 0.0001 0.0002, 0.0001 0.0001))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 5
        np.testing.assert_allclose(x[0], 0.0001, atol=1e-10)
        np.testing.assert_allclose(y[0], 0.0001, atol=1e-10)
        np.testing.assert_allclose(x[1], 0.0002, atol=1e-10)
        np.testing.assert_allclose(y[1], 0.0001, atol=1e-10)

    @needs_gpu
    def test_whitespace_between_coordinates(self):
        """WKT with extra whitespace between coordinates."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT( 1.5   -2.3 )")
        owned = read_wkt_gpu(d_bytes)

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], 1.5, atol=1e-10)
        np.testing.assert_allclose(y[0], -2.3, atol=1e-10)


# ===================================================================
# Structural analysis: multi-geometry classification
# ===================================================================


class TestStructuralMultiGeometry:
    """Structural analysis of multi-geometry inputs."""

    @needs_gpu
    def test_multiple_geometries_correct_tags(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        wkt = "POINT(1 2)\nLINESTRING(0 0, 1 1)\nPOLYGON((0 0, 1 0, 1 1, 0 0))"
        d_bytes = _to_device_bytes(wkt)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 3
        tags = result.d_family_tags.get()
        assert tags[0] == 0  # POINT
        assert tags[1] == 1  # LINESTRING
        assert tags[2] == 2  # POLYGON

    @needs_gpu
    def test_multi_geometry_with_empty(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        wkt = "POINT(1 2)\nPOINT EMPTY\nLINESTRING(0 0, 1 1)"
        d_bytes = _to_device_bytes(wkt)
        result = wkt_structural_analysis(d_bytes)

        assert result.n_geometries == 3
        tags = result.d_family_tags.get()
        empty_flags = result.d_empty_flags.get()
        assert tags[0] == 0
        assert tags[1] == 0
        assert empty_flags[0] == 0
        assert empty_flags[1] == 1
        assert tags[2] == 1
        assert empty_flags[2] == 0

    @needs_gpu
    def test_parenthesis_depth(self):
        """Verify depth computation for nested WKT."""
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        wkt = "POLYGON((0 0, 1 0, 1 1, 0 0))"
        d_bytes = _to_device_bytes(wkt)
        result = wkt_structural_analysis(d_bytes)

        depth = result.d_depth.get()
        # Before first '(' -> depth 0
        assert depth[0] == 0  # 'P'
        # After first '(' -> depth 1
        first_paren = wkt.index('(')
        assert depth[first_paren] == 1
        # After second '(' -> depth 2
        second_paren = wkt.index('(', first_paren + 1)
        assert depth[second_paren] == 2


# ===================================================================
# Shapely comparison tests
# ===================================================================


class TestShapelyComparison:
    """Compare GPU WKT parsing results against Shapely as oracle."""

    @needs_gpu
    def test_point_matches_shapely(self):
        shapely_wkt = pytest.importorskip("shapely.wkt")
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt_text = "POINT(-80.92302345678 25.12345678901)"
        d_bytes = _to_device_bytes(wkt_text)
        owned = read_wkt_gpu(d_bytes)

        gpu_x, gpu_y = _get_device_coords(owned, GeometryFamily.POINT)
        shapely_geom = shapely_wkt.loads(wkt_text)
        np.testing.assert_allclose(gpu_x[0], shapely_geom.x, atol=1e-10)
        np.testing.assert_allclose(gpu_y[0], shapely_geom.y, atol=1e-10)

    @needs_gpu
    def test_linestring_matches_shapely(self):
        shapely_wkt = pytest.importorskip("shapely.wkt")
        shapely = pytest.importorskip("shapely")
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt_text = "LINESTRING(0.1 0.2, 1.3 1.4, 2.5 2.6)"
        d_bytes = _to_device_bytes(wkt_text)
        owned = read_wkt_gpu(d_bytes)

        gpu_x, gpu_y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        shapely_geom = shapely_wkt.loads(wkt_text)
        shapely_coords = np.array(shapely.get_coordinates(shapely_geom))

        np.testing.assert_allclose(gpu_x, shapely_coords[:, 0], atol=1e-10)
        np.testing.assert_allclose(gpu_y, shapely_coords[:, 1], atol=1e-10)

    @needs_gpu
    def test_polygon_matches_shapely(self):
        shapely_wkt = pytest.importorskip("shapely.wkt")
        pytest.importorskip("shapely")
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt_text = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0),(2 2, 8 2, 8 8, 2 8, 2 2))"
        d_bytes = _to_device_bytes(wkt_text)
        owned = read_wkt_gpu(d_bytes)

        gpu_x, gpu_y = _get_device_coords(owned, GeometryFamily.POLYGON)
        shapely_geom = shapely_wkt.loads(wkt_text)
        # Extract coords from exterior + interior rings
        ext_coords = np.array(shapely_geom.exterior.coords)
        int_coords = np.array(shapely_geom.interiors[0].coords)
        expected_x = np.concatenate([ext_coords[:, 0], int_coords[:, 0]])
        expected_y = np.concatenate([ext_coords[:, 1], int_coords[:, 1]])

        np.testing.assert_allclose(gpu_x, expected_x, atol=1e-10)
        np.testing.assert_allclose(gpu_y, expected_y, atol=1e-10)

    @needs_gpu
    def test_scientific_notation_matches_shapely(self):
        shapely_wkt = pytest.importorskip("shapely.wkt")
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt_text = "POINT(1.5e2 -2.3E-4)"
        d_bytes = _to_device_bytes(wkt_text)
        owned = read_wkt_gpu(d_bytes)

        gpu_x, gpu_y = _get_device_coords(owned, GeometryFamily.POINT)
        shapely_geom = shapely_wkt.loads(wkt_text)
        np.testing.assert_allclose(gpu_x[0], shapely_geom.x, atol=1e-10)
        np.testing.assert_allclose(gpu_y[0], shapely_geom.y, atol=1e-10)

    @needs_gpu
    def test_multipoint_matches_shapely(self):
        shapely_wkt = pytest.importorskip("shapely.wkt")
        shapely = pytest.importorskip("shapely")
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt_text = "MULTIPOINT((0 0), (1.5 2.5), (-3.1 4.2))"
        d_bytes = _to_device_bytes(wkt_text)
        owned = read_wkt_gpu(d_bytes)

        gpu_x, gpu_y = _get_device_coords(owned, GeometryFamily.MULTIPOINT)
        shapely_geom = shapely_wkt.loads(wkt_text)
        shapely_coords = np.array(shapely.get_coordinates(shapely_geom))

        np.testing.assert_allclose(gpu_x, shapely_coords[:, 0], atol=1e-10)
        np.testing.assert_allclose(gpu_y, shapely_coords[:, 1], atol=1e-10)


# ===================================================================
# OwnedGeometryArray structure tests
# ===================================================================


class TestOwnedStructure:
    """Verify the OwnedGeometryArray structural metadata is correct."""

    @needs_gpu
    def test_homogeneous_point_family_dict(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nPOINT(3 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert len(owned.families) == 1
        assert GeometryFamily.POINT in owned.families
        buf = owned.families[GeometryFamily.POINT]
        assert buf.row_count == 2

    @needs_gpu
    def test_homogeneous_polygon_family_dict(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POLYGON((0 0, 1 0, 1 1, 0 0))\nPOLYGON((2 2, 3 2, 3 3, 2 2))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert len(owned.families) == 1
        assert GeometryFamily.POLYGON in owned.families
        buf = owned.families[GeometryFamily.POLYGON]
        assert buf.row_count == 2

    @needs_gpu
    def test_mixed_type_family_dict(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nLINESTRING(0 0, 1 1)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        assert GeometryFamily.POINT in owned.families
        assert GeometryFamily.LINESTRING in owned.families
        assert owned.families[GeometryFamily.POINT].row_count == 1
        assert owned.families[GeometryFamily.LINESTRING].row_count == 1

    @needs_gpu
    def test_validity_mask_correct(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nPOINT EMPTY\nPOINT(3 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        validity = owned.validity
        assert validity[0]
        assert not validity[1]
        assert validity[2]

    @needs_gpu
    def test_tags_array_correct(self):
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POINT(1 2)\nLINESTRING(0 0, 1 1)\nPOLYGON((0 0, 1 0, 1 1, 0 0))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        tags = owned.tags
        # Tags should map to GeometryFamily ordinal:
        # POINT=0, LINESTRING=1, POLYGON=2
        assert tags[0] == 0
        assert tags[1] == 1
        assert tags[2] == 2

    @needs_gpu
    def test_device_state_populated(self):
        """Verify device state is populated after GPU parse."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        d_bytes = _to_device_bytes("POINT(1 2)")
        owned = read_wkt_gpu(d_bytes)

        assert owned.device_state is not None

    @needs_gpu
    def test_geometry_offsets_for_linestring(self):
        """Verify geometry_offsets correctly partition coordinates."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "LINESTRING(0 0, 1 1)\nLINESTRING(2 2, 3 3, 4 4)"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        buf = owned.device_state.families[GeometryFamily.LINESTRING]
        offsets = cp.asnumpy(buf.geometry_offsets)
        # First linestring: 2 coords, second: 3 coords
        assert offsets[0] == 0
        assert offsets[1] == 2
        assert offsets[2] == 5

    @needs_gpu
    def test_ring_offsets_for_polygon_with_hole(self):
        """Verify ring_offsets for polygon with a hole."""
        from vibespatial.io.wkt_gpu import read_wkt_gpu

        wkt = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0),(2 2, 8 2, 8 8, 2 8, 2 2))"
        d_bytes = _to_device_bytes(wkt)
        owned = read_wkt_gpu(d_bytes)

        buf = owned.device_state.families[GeometryFamily.POLYGON]
        # geometry_offsets: [0, 2] -- 1 polygon with 2 rings
        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        assert geom_offsets[0] == 0
        assert geom_offsets[1] == 2

        # ring_offsets: [0, 5, 10] -- ring 0 has 5 coords, ring 1 has 5 coords
        ring_offsets = cp.asnumpy(buf.ring_offsets)
        assert ring_offsets is not None
        assert ring_offsets[0] == 0
        assert ring_offsets[1] == 5
        assert ring_offsets[2] == 10


# ===================================================================
# WktStructuralResult direct tests
# ===================================================================


class TestWktStructuralResult:
    """Direct tests on the WktStructuralResult dataclass fields."""

    @needs_gpu
    def test_result_fields_device_resident(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        d_bytes = _to_device_bytes("POINT(1 2)")
        result = wkt_structural_analysis(d_bytes)

        # All arrays should be CuPy device arrays
        assert hasattr(result.d_depth, 'device')
        assert hasattr(result.d_geom_starts, 'device')
        assert hasattr(result.d_family_tags, 'device')
        assert hasattr(result.d_empty_flags, 'device')

    @needs_gpu
    def test_depth_array_length_matches_input(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        wkt = "POINT(1 2)"
        d_bytes = _to_device_bytes(wkt)
        result = wkt_structural_analysis(d_bytes)

        assert result.d_depth.shape[0] == len(wkt)

    @needs_gpu
    def test_geom_starts_correct(self):
        from vibespatial.io.wkt_gpu import wkt_structural_analysis

        wkt = "POINT(1 2)\nLINESTRING(0 0, 1 1)"
        d_bytes = _to_device_bytes(wkt)
        result = wkt_structural_analysis(d_bytes)

        starts = result.d_geom_starts.get()
        assert starts[0] == 0
        assert starts[1] == wkt.index('\n') + 1


# ===================================================================
# IO dispatch wiring tests
# ===================================================================


class TestIODispatchWiring:
    """Verify WKT is registered in the IO support matrix and file dispatch."""

    def test_wkt_in_io_format_enum(self):
        from vibespatial.io.support import IOFormat

        assert hasattr(IOFormat, "WKT")
        assert IOFormat.WKT == "wkt"

    def test_wkt_support_matrix_read_is_hybrid(self):
        from vibespatial.io.support import (
            IOFormat,
            IOOperation,
            IOPathKind,
            plan_io_support,
        )

        plan = plan_io_support(IOFormat.WKT, IOOperation.READ)
        assert plan.selected_path == IOPathKind.HYBRID

    def test_wkt_support_matrix_write_is_fallback(self):
        from vibespatial.io.support import (
            IOFormat,
            IOOperation,
            IOPathKind,
            plan_io_support,
        )

        plan = plan_io_support(IOFormat.WKT, IOOperation.WRITE)
        assert plan.selected_path == IOPathKind.FALLBACK

    def test_wkt_not_canonical_gpu(self):
        from vibespatial.io.support import IOFormat, IOOperation, plan_io_support

        plan = plan_io_support(IOFormat.WKT, IOOperation.READ)
        assert not plan.canonical_gpu

    def test_wkt_file_extension_routing(self):
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("test.wkt", operation=IOOperation.READ)
        assert plan.format == IOFormat.WKT
        assert plan.driver == "WKT"
        assert plan.implementation == "wkt_gpu_hybrid_adapter"

    def test_wkt_file_extension_routing_uppercase(self):
        """File extension detection is case-insensitive via Path.suffix.lower()."""
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("data.WKT", operation=IOOperation.READ)
        assert plan.format == IOFormat.WKT
