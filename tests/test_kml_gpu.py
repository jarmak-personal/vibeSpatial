"""Tests for GPU KML reader: structural analysis, coordinate extraction, and assembly."""
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

def _to_device_bytes(kml: str | bytes) -> cp.ndarray:
    """Encode KML text as device-resident uint8 array."""
    raw = kml.encode("utf-8") if isinstance(kml, str) else kml
    return cp.frombuffer(raw, dtype=cp.uint8).copy()


def _get_device_coords(owned, family: GeometryFamily):
    """Extract host x, y coordinate arrays from an OwnedGeometryArray."""
    if owned.device_state is not None and family in owned.device_state.families:
        dev_buf = owned.device_state.families[family]
        return cp.asnumpy(dev_buf.x), cp.asnumpy(dev_buf.y)
    buf = owned.families[family]
    return buf.x, buf.y


def _wrap_kml(*placemarks: str) -> str:
    """Wrap Placemark XML fragments in a minimal KML document."""
    body = "\n".join(placemarks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "<Document>\n"
        f"{body}\n"
        "</Document>\n"
        "</kml>\n"
    )


def _point_placemark(lon: float, lat: float, alt: float | None = None) -> str:
    """Build a KML Point Placemark."""
    if alt is not None:
        coords = f"{lon},{lat},{alt}"
    else:
        coords = f"{lon},{lat}"
    return (
        "<Placemark>"
        f"<Point><coordinates>{coords}</coordinates></Point>"
        "</Placemark>"
    )


def _linestring_placemark(coords_list: list[tuple[float, ...]]) -> str:
    """Build a KML LineString Placemark."""
    parts = []
    for c in coords_list:
        parts.append(",".join(str(v) for v in c))
    coords_str = " ".join(parts)
    return (
        "<Placemark>"
        f"<LineString><coordinates>{coords_str}</coordinates></LineString>"
        "</Placemark>"
    )


def _polygon_placemark(
    outer: list[tuple[float, ...]],
    inners: list[list[tuple[float, ...]]] | None = None,
) -> str:
    """Build a KML Polygon Placemark with optional holes."""
    def _ring_coords(ring):
        return " ".join(",".join(str(v) for v in c) for c in ring)

    outer_str = (
        "<outerBoundaryIs><LinearRing>"
        f"<coordinates>{_ring_coords(outer)}</coordinates>"
        "</LinearRing></outerBoundaryIs>"
    )
    inner_strs = ""
    if inners:
        for inner in inners:
            inner_strs += (
                "<innerBoundaryIs><LinearRing>"
                f"<coordinates>{_ring_coords(inner)}</coordinates>"
                "</LinearRing></innerBoundaryIs>"
            )
    return (
        "<Placemark>"
        f"<Polygon>{outer_str}{inner_strs}</Polygon>"
        "</Placemark>"
    )


# ===================================================================
# Structural analysis tests
# ===================================================================


class TestKmlStructuralAnalysis:
    """Tests for kml_structural_analysis: tag detection, Placemark pairing."""

    @needs_gpu
    def test_single_point_placemark(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml(_point_placemark(-122.08, 37.42, 0))
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 0  # Point

    @needs_gpu
    def test_multiple_placemarks_detected(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml(
            _point_placemark(-122.08, 37.42, 0),
            _point_placemark(-121.0, 36.0, 0),
            _point_placemark(-120.0, 35.0, 0),
        )
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 3

    @needs_gpu
    def test_linestring_classification(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml(
            _linestring_placemark([(-122.0, 37.0, 0), (-121.0, 36.0, 0)])
        )
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 1  # LineString

    @needs_gpu
    def test_polygon_classification(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        outer = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
        kml = _wrap_kml(_polygon_placemark(outer))
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 1
        tags = result.d_family_tags.get()
        assert tags[0] == 2  # Polygon

    @needs_gpu
    def test_coordinate_regions_detected(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml(
            _point_placemark(-122.08, 37.42),
            _point_placemark(-121.0, 36.0),
        )
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.d_coord_starts.shape[0] == 2
        assert result.d_coord_ends.shape[0] == 2

    @needs_gpu
    def test_namespace_prefixed_tags(self):
        """Tags like <kml:Placemark> and <kml:Point> should be handled."""
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = (
            '<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2">\n'
            "<kml:Document>\n"
            "<kml:Placemark>\n"
            "<kml:Point><kml:coordinates>-122.08,37.42</kml:coordinates></kml:Point>\n"
            "</kml:Placemark>\n"
            "</kml:Document>\n"
            "</kml:kml>\n"
        )
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 1
        assert result.d_coord_starts.shape[0] == 1

    @needs_gpu
    def test_xml_comments_masked(self):
        """Tags inside XML comments should be ignored."""
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml(
            "<!-- <Placemark><Point><coordinates>0,0</coordinates></Point></Placemark> -->",
            _point_placemark(-122.08, 37.42),
        )
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        # Only the real Placemark should be detected, not the one in the comment
        assert result.n_placemarks == 1

    @needs_gpu
    def test_empty_document(self):
        from vibespatial.io.kml_gpu import kml_structural_analysis

        kml = _wrap_kml()
        d_bytes = _to_device_bytes(kml)
        result = kml_structural_analysis(d_bytes)

        assert result.n_placemarks == 0


# ===================================================================
# Coordinate extraction and assembly tests
# ===================================================================


class TestReadKmlGpu:
    """Tests for read_kml_gpu: full parsing pipeline."""

    @needs_gpu
    def test_point_2d(self):
        """Point Placemark with 2D coordinates (lon,lat)."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(_point_placemark(-122.08, 37.42))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        assert GeometryFamily.POINT in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -122.08, rtol=1e-10)
        np.testing.assert_allclose(y[0], 37.42, rtol=1e-10)

    @needs_gpu
    def test_point_3d_altitude_dropped(self):
        """Point Placemark with 3D coordinates (lon,lat,alt) -- altitude ignored."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(_point_placemark(-122.08, 37.42, 100.5))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        # x = longitude, y = latitude (KML convention: lon first)
        np.testing.assert_allclose(x[0], -122.08, rtol=1e-10)
        np.testing.assert_allclose(y[0], 37.42, rtol=1e-10)

    @needs_gpu
    def test_linestring(self):
        """LineString Placemark with multiple coordinate tuples."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        coords = [(-122.0, 37.0, 0), (-121.0, 36.0, 0), (-120.0, 35.0, 0)]
        kml = _wrap_kml(_linestring_placemark(coords))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        assert GeometryFamily.LINESTRING in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [-122.0, -121.0, -120.0], rtol=1e-10)
        np.testing.assert_allclose(y, [37.0, 36.0, 35.0], rtol=1e-10)

    @needs_gpu
    def test_polygon_outer_ring_only(self):
        """Polygon Placemark with outer ring only."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        outer = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
        kml = _wrap_kml(_polygon_placemark(outer))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        assert GeometryFamily.POLYGON in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 4
        np.testing.assert_allclose(x, [0, 1, 1, 0], rtol=1e-10)
        np.testing.assert_allclose(y, [0, 0, 1, 0], rtol=1e-10)

        # Check geometry_offsets: 1 geometry with 1 ring
        if owned.device_state and GeometryFamily.POLYGON in owned.device_state.families:
            dev_buf = owned.device_state.families[GeometryFamily.POLYGON]
            geom_offs = cp.asnumpy(dev_buf.geometry_offsets)
            assert geom_offs.tolist() == [0, 1]  # 1 ring
            ring_offs = cp.asnumpy(dev_buf.ring_offsets)
            assert ring_offs.tolist() == [0, 4]  # 4 coordinate pairs

    @needs_gpu
    def test_polygon_with_hole(self):
        """Polygon Placemark with outer ring and one inner ring (hole)."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        outer = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 0)]
        inner = [(2, 2, 0), (8, 2, 0), (8, 8, 0), (2, 8, 0), (2, 2, 0)]
        kml = _wrap_kml(_polygon_placemark(outer, inners=[inner]))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        assert GeometryFamily.POLYGON in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 10  # 5 outer + 5 inner

        # Verify outer ring coordinates
        np.testing.assert_allclose(x[:5], [0, 10, 10, 0, 0], rtol=1e-10)
        np.testing.assert_allclose(y[:5], [0, 0, 10, 10, 0], rtol=1e-10)

        # Verify inner ring coordinates
        np.testing.assert_allclose(x[5:], [2, 8, 8, 2, 2], rtol=1e-10)
        np.testing.assert_allclose(y[5:], [2, 2, 8, 8, 2], rtol=1e-10)

        # Check ring structure: 1 geometry with 2 rings
        if owned.device_state and GeometryFamily.POLYGON in owned.device_state.families:
            dev_buf = owned.device_state.families[GeometryFamily.POLYGON]
            geom_offs = cp.asnumpy(dev_buf.geometry_offsets)
            assert geom_offs.tolist() == [0, 2]  # 2 rings
            ring_offs = cp.asnumpy(dev_buf.ring_offsets)
            assert ring_offs.tolist() == [0, 5, 10]  # 5 coords each ring

    @needs_gpu
    def test_multiple_points(self):
        """Multiple Point Placemarks of the same type."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            _point_placemark(-122.08, 37.42, 0),
            _point_placemark(-121.0, 36.0, 0),
            _point_placemark(-120.0, 35.0, 0),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 3
        assert GeometryFamily.POINT in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        assert len(x) == 3
        np.testing.assert_allclose(x, [-122.08, -121.0, -120.0], rtol=1e-10)
        np.testing.assert_allclose(y, [37.42, 36.0, 35.0], rtol=1e-10)

    @needs_gpu
    def test_mixed_types(self):
        """Mixed Placemark types: Point + LineString."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            _point_placemark(-122.08, 37.42, 0),
            _linestring_placemark([(-121.0, 36.0, 0), (-120.0, 35.0, 0)]),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 2
        assert GeometryFamily.POINT in owned.families
        assert GeometryFamily.LINESTRING in owned.families

        # Verify Point
        pt_x, pt_y = _get_device_coords(owned, GeometryFamily.POINT)
        assert len(pt_x) == 1
        np.testing.assert_allclose(pt_x[0], -122.08, rtol=1e-10)
        np.testing.assert_allclose(pt_y[0], 37.42, rtol=1e-10)

        # Verify LineString
        ls_x, ls_y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(ls_x) == 2
        np.testing.assert_allclose(ls_x, [-121.0, -120.0], rtol=1e-10)
        np.testing.assert_allclose(ls_y, [36.0, 35.0], rtol=1e-10)

    @needs_gpu
    def test_kml_with_xml_comments(self):
        """KML with XML comments -- commented Placemarks should be ignored."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<!-- This is a comment -->",
            _point_placemark(-122.08, 37.42, 0),
            "<!-- <Placemark><Point><coordinates>0,0,0</coordinates></Point></Placemark> -->",
            _point_placemark(-121.0, 36.0, 0),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        # Only 2 real Placemarks, not 3
        assert owned.row_count == 2

    @needs_gpu
    def test_kml_with_namespace_prefixes(self):
        """KML with kml: namespace prefixes on all tags."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = (
            '<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2">\n'
            "<kml:Document>\n"
            "<kml:Placemark>\n"
            "<kml:Point><kml:coordinates>-122.08,37.42</kml:coordinates></kml:Point>\n"
            "</kml:Placemark>\n"
            "</kml:Document>\n"
            "</kml:kml>\n"
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        np.testing.assert_allclose(x[0], -122.08, rtol=1e-10)
        np.testing.assert_allclose(y[0], 37.42, rtol=1e-10)

    @needs_gpu
    def test_coordinate_precision(self):
        """Verify that coordinate values are parsed with full fp64 precision."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        # High-precision coordinates typical of survey data
        lon = -122.41941550000001
        lat = 37.77493400000001
        kml = _wrap_kml(_point_placemark(lon, lat))
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        x, y = _get_device_coords(owned, GeometryFamily.POINT)
        # parse_ascii_floats has limited precision for very long decimals,
        # but should be within 1e-8 for typical coordinate values.
        np.testing.assert_allclose(x[0], lon, rtol=1e-8)
        np.testing.assert_allclose(y[0], lat, rtol=1e-8)

    @needs_gpu
    def test_2d_coordinates_no_altitude(self):
        """KML with 2D coordinates (lon,lat only, no altitude)."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<LineString><coordinates>"
            "-122.0,37.0 -121.0,36.0 -120.0,35.0"
            "</coordinates></LineString>"
            "</Placemark>"
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [-122.0, -121.0, -120.0], rtol=1e-10)
        np.testing.assert_allclose(y, [37.0, 36.0, 35.0], rtol=1e-10)

    @needs_gpu
    def test_coordinates_with_newline_separators(self):
        """KML coordinates separated by newlines instead of spaces."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<LineString><coordinates>\n"
            "-122.0,37.0,0\n"
            "-121.0,36.0,0\n"
            "-120.0,35.0,0\n"
            "</coordinates></LineString>"
            "</Placemark>"
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 1
        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 3
        np.testing.assert_allclose(x, [-122.0, -121.0, -120.0], rtol=1e-10)

    @needs_gpu
    def test_empty_document_returns_empty_owned(self):
        """Empty KML document returns empty OwnedGeometryArray."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml()
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 0

    @needs_gpu
    def test_multiple_linestrings(self):
        """Multiple LineString Placemarks."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            _linestring_placemark([(-122.0, 37.0, 0), (-121.0, 36.0, 0)]),
            _linestring_placemark([(-120.0, 35.0, 0), (-119.0, 34.0, 0), (-118.0, 33.0, 0)]),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 2
        assert GeometryFamily.LINESTRING in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.LINESTRING)
        assert len(x) == 5  # 2 + 3 total coords

        # Check geometry offsets
        if owned.device_state and GeometryFamily.LINESTRING in owned.device_state.families:
            dev_buf = owned.device_state.families[GeometryFamily.LINESTRING]
            geom_offs = cp.asnumpy(dev_buf.geometry_offsets)
            assert geom_offs.tolist() == [0, 2, 5]

    @needs_gpu
    def test_multiple_polygons(self):
        """Multiple Polygon Placemarks."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        outer1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
        outer2 = [(2, 2, 0), (3, 2, 0), (3, 3, 0), (2, 2, 0)]
        kml = _wrap_kml(
            _polygon_placemark(outer1),
            _polygon_placemark(outer2),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 2
        assert GeometryFamily.POLYGON in owned.families

        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)
        assert len(x) == 8  # 4 + 4

    @needs_gpu
    def test_mixed_point_linestring_polygon(self):
        """Mixed geometry types: Point + LineString + Polygon."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        outer = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
        kml = _wrap_kml(
            _point_placemark(-122.08, 37.42, 0),
            _linestring_placemark([(-121.0, 36.0, 0), (-120.0, 35.0, 0)]),
            _polygon_placemark(outer),
        )
        d_bytes = _to_device_bytes(kml)
        kml_result = read_kml_gpu(d_bytes)
        owned = kml_result.geometry

        assert owned.row_count == 3
        assert GeometryFamily.POINT in owned.families
        assert GeometryFamily.LINESTRING in owned.families
        assert GeometryFamily.POLYGON in owned.families


# ===================================================================
# KML attribute extraction tests
# ===================================================================


class TestKmlAttributeExtraction:
    """Tests for Placemark name/description extraction."""

    @needs_gpu
    def test_placemark_with_name(self):
        """Placemark with <name> tag extracts name column."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<name>City Hall</name>"
            "<Point><coordinates>-122.08,37.42,0</coordinates></Point>"
            "</Placemark>"
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 1
        assert result.attributes is not None
        assert "name" in result.attributes
        assert result.attributes["name"] == ["City Hall"]

    @needs_gpu
    def test_placemark_with_description(self):
        """Placemark with <description> tag extracts description column."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<description>A notable building</description>"
            "<Point><coordinates>-122.08,37.42,0</coordinates></Point>"
            "</Placemark>"
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 1
        assert result.attributes is not None
        assert "description" in result.attributes
        assert result.attributes["description"] == ["A notable building"]

    @needs_gpu
    def test_placemark_with_name_and_description(self):
        """Placemark with both name and description."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<name>City Hall</name>"
            "<description>The main government building</description>"
            "<Point><coordinates>-122.08,37.42,0</coordinates></Point>"
            "</Placemark>"
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 1
        assert result.attributes is not None
        assert result.attributes["name"] == ["City Hall"]
        assert result.attributes["description"] == ["The main government building"]

    @needs_gpu
    def test_placemark_without_name_returns_none(self):
        """Placemark without <name> produces None for that row."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<name>Has Name</name>"
            "<Point><coordinates>-122.08,37.42,0</coordinates></Point>"
            "</Placemark>",
            "<Placemark>"
            "<Point><coordinates>-121.0,36.0,0</coordinates></Point>"
            "</Placemark>",
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 2
        assert result.attributes is not None
        assert result.attributes["name"] == ["Has Name", None]

    @needs_gpu
    def test_no_attributes_returns_none(self):
        """Placemarks with no name or description produce None attributes."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(_point_placemark(-122.08, 37.42, 0))
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 1
        # No name or description found anywhere => attributes is None
        assert result.attributes is None

    @needs_gpu
    def test_namespace_prefixed_name(self):
        """Namespace-prefixed <kml:name> is extracted."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = (
            '<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2">\n'
            "<kml:Document>\n"
            "<kml:Placemark>\n"
            "<kml:name>NS Point</kml:name>\n"
            "<kml:Point><kml:coordinates>-122.08,37.42</kml:coordinates></kml:Point>\n"
            "</kml:Placemark>\n"
            "</kml:Document>\n"
            "</kml:kml>\n"
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 1
        assert result.attributes is not None
        assert result.attributes["name"] == ["NS Point"]

    @needs_gpu
    def test_multiple_placemarks_mixed_attributes(self):
        """Multiple Placemarks with varying attributes."""
        from vibespatial.io.kml_gpu import read_kml_gpu

        kml = _wrap_kml(
            "<Placemark>"
            "<name>First</name>"
            "<description>Desc 1</description>"
            "<Point><coordinates>-122.0,37.0,0</coordinates></Point>"
            "</Placemark>",
            "<Placemark>"
            "<name>Second</name>"
            "<Point><coordinates>-121.0,36.0,0</coordinates></Point>"
            "</Placemark>",
            "<Placemark>"
            "<description>Only desc</description>"
            "<Point><coordinates>-120.0,35.0,0</coordinates></Point>"
            "</Placemark>",
        )
        d_bytes = _to_device_bytes(kml)
        result = read_kml_gpu(d_bytes)

        assert result.n_placemarks == 3
        assert result.attributes is not None
        assert result.attributes["name"] == ["First", "Second", None]
        assert result.attributes["description"] == ["Desc 1", None, "Only desc"]


# ===================================================================
# IO dispatch wiring tests
# ===================================================================


class TestKmlIOWiring:
    """Tests for KML format in IO dispatch system."""

    def test_kml_in_io_format_enum(self):
        from vibespatial.io.support import IOFormat

        assert IOFormat.KML == "kml"

    def test_kml_in_support_matrix(self):
        from vibespatial.io.support import IO_SUPPORT_MATRIX, IOFormat

        assert IOFormat.KML in IO_SUPPORT_MATRIX

    def test_kml_file_routing(self):
        from vibespatial.io.file import _normalize_driver

        assert _normalize_driver("test.kml") == "KML"

    def test_kml_plan(self):
        from vibespatial.io.file import plan_vector_file_io
        from vibespatial.io.support import IOFormat, IOOperation

        plan = plan_vector_file_io("test.kml", operation=IOOperation.READ)
        assert plan.format == IOFormat.KML
        assert plan.implementation == "kml_gpu_hybrid_adapter"


# ===================================================================
# Dimensionality detection tests
# ===================================================================


class TestDimensionalityDetection:
    """Tests for 2D vs 3D coordinate detection."""

    @needs_gpu
    def test_detect_3d(self):
        from vibespatial.io.kml_gpu import _detect_dimensionality

        # 3D: lon,lat,alt separated by spaces
        kml_bytes = b"-122.0,37.0,100 -121.0,36.0,200"
        d_bytes = cp.frombuffer(kml_bytes, dtype=cp.uint8).copy()
        d_starts = cp.array([0], dtype=cp.int64)
        d_ends = cp.array([len(kml_bytes)], dtype=cp.int64)

        dim = _detect_dimensionality(d_bytes, d_starts, d_ends)
        assert dim == 3

    @needs_gpu
    def test_detect_2d(self):
        from vibespatial.io.kml_gpu import _detect_dimensionality

        # 2D: lon,lat separated by spaces
        kml_bytes = b"-122.0,37.0 -121.0,36.0"
        d_bytes = cp.frombuffer(kml_bytes, dtype=cp.uint8).copy()
        d_starts = cp.array([0], dtype=cp.int64)
        d_ends = cp.array([len(kml_bytes)], dtype=cp.int64)

        dim = _detect_dimensionality(d_bytes, d_starts, d_ends)
        assert dim == 2

    @needs_gpu
    def test_single_3d_point(self):
        from vibespatial.io.kml_gpu import _detect_dimensionality

        # Single 3D point
        kml_bytes = b"-122.08,37.42,0"
        d_bytes = cp.frombuffer(kml_bytes, dtype=cp.uint8).copy()
        d_starts = cp.array([0], dtype=cp.int64)
        d_ends = cp.array([len(kml_bytes)], dtype=cp.int64)

        dim = _detect_dimensionality(d_bytes, d_starts, d_ends)
        assert dim == 3

    @needs_gpu
    def test_single_2d_point(self):
        from vibespatial.io.kml_gpu import _detect_dimensionality

        kml_bytes = b"-122.08,37.42"
        d_bytes = cp.frombuffer(kml_bytes, dtype=cp.uint8).copy()
        d_starts = cp.array([0], dtype=cp.int64)
        d_ends = cp.array([len(kml_bytes)], dtype=cp.int64)

        dim = _detect_dimensionality(d_bytes, d_starts, d_ends)
        assert dim == 2
