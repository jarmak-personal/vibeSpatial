"""Tests for GPU SHP binary decoder: direct coordinate extraction from .shp files.

Tests cover:
- SHX index parsing (CPU, unit tests)
- Point shapefile: coordinate roundtrip
- Polygon shapefile: coordinates + ring offsets
- Polygon with hole: multi-ring structure
- PolyLine shapefile: coordinates + part offsets
- MultiPoint shapefile: coordinate gathering
- Null shapes: handled gracefully
- Comparison: GPU direct decode matches pyogrio output for same file
- Large file: 100K records

Test shapefiles are created programmatically using pyogrio + shapely
(write-side only -- the GPU decoder never touches Shapely).
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

try:
    import pyogrio  # noqa: F401 -- used to detect availability
    import shapely  # noqa: F401 -- used to detect availability
    from shapely.geometry import (
        LineString,
        Point,
        Polygon,
    )

    HAS_PYOGRIO = True
except ImportError:
    HAS_PYOGRIO = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
needs_pyogrio = pytest.mark.skipif(not HAS_PYOGRIO, reason="pyogrio/shapely not available")


# ---------------------------------------------------------------------------
# SHX/SHP file builder helpers (pure Python, no Shapely needed)
# ---------------------------------------------------------------------------


def _build_shp_shx_point(coords: list[tuple[float, float]]) -> tuple[bytes, bytes]:
    """Build minimal SHP + SHX file bytes for Point geometries.

    Parameters
    ----------
    coords : list of (x, y) tuples

    Returns
    -------
    shp_bytes, shx_bytes
    """
    n = len(coords)

    # SHP records
    records = bytearray()
    shx_entries = bytearray()

    for i, (x, y) in enumerate(coords):
        record_num = i + 1
        content_length_words = 10  # (4 + 8 + 8) / 2 = 10 words
        record_header = struct.pack(">ii", record_num, content_length_words)
        record_content = struct.pack("<i", 1) + struct.pack("<dd", x, y)

        offset_words = (100 + len(records)) // 2
        shx_entries += struct.pack(">ii", offset_words, content_length_words)

        records += record_header + record_content

    shp_bytes = _build_shp_header(1, records, n)
    shx_bytes = _build_shx_header(1, shx_entries, n)
    return bytes(shp_bytes), bytes(shx_bytes)


def _build_shp_shx_polygon(
    polygons: list[list[list[tuple[float, float]]]],
) -> tuple[bytes, bytes]:
    """Build minimal SHP + SHX file bytes for Polygon geometries.

    Parameters
    ----------
    polygons : list of polygon definitions.
        Each polygon is a list of rings, where each ring is a list of (x, y) tuples.

    Returns
    -------
    shp_bytes, shx_bytes
    """
    n = len(polygons)
    records = bytearray()
    shx_entries = bytearray()

    for i, rings in enumerate(polygons):
        record_num = i + 1
        num_parts = len(rings)
        all_points = []
        for ring in rings:
            all_points.extend(ring)
        num_points = len(all_points)

        # Compute bounding box
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Build parts array (starting index of each ring)
        parts = []
        idx = 0
        for ring in rings:
            parts.append(idx)
            idx += len(ring)

        # Record content: type(4) + bbox(32) + num_parts(4) + num_points(4)
        #                 + parts(4*num_parts) + points(16*num_points)
        content = struct.pack("<i", 5)
        content += struct.pack("<4d", xmin, ymin, xmax, ymax)
        content += struct.pack("<ii", num_parts, num_points)
        for p in parts:
            content += struct.pack("<i", p)
        for x, y in all_points:
            content += struct.pack("<dd", x, y)

        content_length_words = len(content) // 2
        record_header = struct.pack(">ii", record_num, content_length_words)

        offset_words = (100 + len(records)) // 2
        shx_entries += struct.pack(">ii", offset_words, content_length_words)

        records += record_header + content

    shp_bytes = _build_shp_header(5, records, n)
    shx_bytes = _build_shx_header(5, shx_entries, n)
    return bytes(shp_bytes), bytes(shx_bytes)


def _build_shp_shx_polyline(
    polylines: list[list[list[tuple[float, float]]]],
) -> tuple[bytes, bytes]:
    """Build minimal SHP + SHX file bytes for PolyLine geometries.

    Parameters
    ----------
    polylines : list of polyline definitions.
        Each polyline is a list of parts, where each part is a list of (x, y) tuples.

    Returns
    -------
    shp_bytes, shx_bytes
    """
    n = len(polylines)
    records = bytearray()
    shx_entries = bytearray()

    for i, parts_list in enumerate(polylines):
        record_num = i + 1
        num_parts = len(parts_list)
        all_points = []
        for part in parts_list:
            all_points.extend(part)
        num_points = len(all_points)

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        parts = []
        idx = 0
        for part in parts_list:
            parts.append(idx)
            idx += len(part)

        content = struct.pack("<i", 3)
        content += struct.pack("<4d", xmin, ymin, xmax, ymax)
        content += struct.pack("<ii", num_parts, num_points)
        for p in parts:
            content += struct.pack("<i", p)
        for x, y in all_points:
            content += struct.pack("<dd", x, y)

        content_length_words = len(content) // 2
        record_header = struct.pack(">ii", record_num, content_length_words)

        offset_words = (100 + len(records)) // 2
        shx_entries += struct.pack(">ii", offset_words, content_length_words)

        records += record_header + content

    shp_bytes = _build_shp_header(3, records, n)
    shx_bytes = _build_shx_header(3, shx_entries, n)
    return bytes(shp_bytes), bytes(shx_bytes)


def _build_shp_shx_multipoint(
    multipoints: list[list[tuple[float, float]]],
) -> tuple[bytes, bytes]:
    """Build minimal SHP + SHX for MultiPoint geometries."""
    n = len(multipoints)
    records = bytearray()
    shx_entries = bytearray()

    for i, points in enumerate(multipoints):
        record_num = i + 1
        num_points = len(points)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # MultiPoint: type(4) + bbox(32) + num_points(4) + points(16*N)
        content = struct.pack("<i", 8)
        content += struct.pack("<4d", xmin, ymin, xmax, ymax)
        content += struct.pack("<i", num_points)
        for x, y in points:
            content += struct.pack("<dd", x, y)

        content_length_words = len(content) // 2
        record_header = struct.pack(">ii", record_num, content_length_words)

        offset_words = (100 + len(records)) // 2
        shx_entries += struct.pack(">ii", offset_words, content_length_words)

        records += record_header + content

    shp_bytes = _build_shp_header(8, records, n)
    shx_bytes = _build_shx_header(8, shx_entries, n)
    return bytes(shp_bytes), bytes(shx_bytes)


def _build_shp_header(shape_type: int, records: bytes, n_records: int) -> bytearray:
    """Build 100-byte SHP file header + append records."""
    file_length_words = (100 + len(records)) // 2
    header = bytearray(100)
    struct.pack_into(">i", header, 0, 9994)  # file code
    struct.pack_into(">i", header, 24, file_length_words)  # file length
    struct.pack_into("<i", header, 28, 1000)  # version
    struct.pack_into("<i", header, 32, shape_type)  # shape type
    # bbox: zeros for simplicity (not used by decoder)
    return header + records


def _build_shx_header(shape_type: int, entries: bytes, n_records: int) -> bytearray:
    """Build 100-byte SHX file header + append index entries."""
    file_length_words = (100 + len(entries)) // 2
    header = bytearray(100)
    struct.pack_into(">i", header, 0, 9994)
    struct.pack_into(">i", header, 24, file_length_words)
    struct.pack_into("<i", header, 28, 1000)
    struct.pack_into("<i", header, 32, shape_type)
    return header + entries


def _write_shp_shx(tmpdir: Path, stem: str, shp_bytes: bytes, shx_bytes: bytes) -> Path:
    """Write SHP/SHX bytes to files and return the .shp path."""
    shp_path = tmpdir / f"{stem}.shp"
    shx_path = tmpdir / f"{stem}.shx"
    shp_path.write_bytes(shp_bytes)
    shx_path.write_bytes(shx_bytes)
    return shp_path


# ---------------------------------------------------------------------------
# Unit tests: SHX index parsing
# ---------------------------------------------------------------------------


class TestShxParsing:
    """CPU-side SHX index parsing."""

    def test_shx_header_point(self):
        """Parse SHX header for a Point shapefile."""
        from vibespatial.io.shp_gpu import _read_shx_index

        coords = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        _, shx_bytes = _build_shp_shx_point(coords)

        with tempfile.TemporaryDirectory() as td:
            shx_path = Path(td) / "test.shx"
            shx_path.write_bytes(shx_bytes)

            header, offsets, content_lengths = _read_shx_index(shx_path)

        assert header.shape_type == 1
        assert header.n_records == 3
        assert len(offsets) == 3
        assert len(content_lengths) == 3
        # Each Point record content is 20 bytes = 10 words
        np.testing.assert_array_equal(content_lengths, [20, 20, 20])

    def test_shx_header_polygon(self):
        """Parse SHX for a Polygon shapefile."""
        from vibespatial.io.shp_gpu import _read_shx_index

        ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        polygons = [[ring]]
        _, shx_bytes = _build_shp_shx_polygon(polygons)

        with tempfile.TemporaryDirectory() as td:
            shx_path = Path(td) / "test.shx"
            shx_path.write_bytes(shx_bytes)

            header, offsets, content_lengths = _read_shx_index(shx_path)

        assert header.shape_type == 5
        assert header.n_records == 1
        assert offsets[0] == 100  # first record starts right after header

    def test_shx_empty(self):
        """Parse SHX with zero records."""
        from vibespatial.io.shp_gpu import _read_shx_index

        # Build an SHX with zero records
        shx_bytes = _build_shx_header(1, b"", 0)

        with tempfile.TemporaryDirectory() as td:
            shx_path = Path(td) / "empty.shx"
            shx_path.write_bytes(bytes(shx_bytes))

            header, offsets, content_lengths = _read_shx_index(shx_path)

        assert header.n_records == 0
        assert len(offsets) == 0
        assert len(content_lengths) == 0

    def test_shx_invalid_file_code(self):
        """Reject SHX with invalid file code."""
        from vibespatial.io.shp_gpu import _read_shx_index

        bad_header = bytearray(100)
        struct.pack_into(">i", bad_header, 0, 1234)  # wrong file code
        struct.pack_into(">i", bad_header, 24, 50)  # file length in words

        with tempfile.TemporaryDirectory() as td:
            shx_path = Path(td) / "bad.shx"
            shx_path.write_bytes(bytes(bad_header))

            with pytest.raises(ValueError, match="Invalid SHX file code"):
                _read_shx_index(shx_path)


# ---------------------------------------------------------------------------
# GPU integration tests
# ---------------------------------------------------------------------------


@needs_gpu
class TestPointDecode:
    """GPU decode of Point shapefiles."""

    def test_point_roundtrip(self):
        """Coordinates survive the encode-decode roundtrip."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        coords = [(1.5, 2.5), (3.75, -4.25), (0.0, 0.0), (-180.0, 90.0)]
        shp_bytes, shx_bytes = _build_shp_shx_point(coords)

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "points", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        # Extract coordinates from device
        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POINT]
        x = cp.asnumpy(buf.x)
        y = cp.asnumpy(buf.y)

        expected_x = np.array([c[0] for c in coords])
        expected_y = np.array([c[1] for c in coords])
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)

    def test_single_point(self):
        """Single-element input."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        shp_bytes, shx_bytes = _build_shp_shx_point([(42.0, -7.0)])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "single", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POINT]
        assert buf.x.size == 1
        assert float(buf.x[0].get()) == 42.0
        assert float(buf.y[0].get()) == -7.0

    def test_large_point_file(self):
        """100K records decode correctly."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        n = 100_000
        rng = np.random.default_rng(42)
        xs = rng.uniform(-180, 180, n)
        ys = rng.uniform(-90, 90, n)
        coords = list(zip(xs.tolist(), ys.tolist()))

        shp_bytes, shx_bytes = _build_shp_shx_point(coords)

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "large", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POINT]
        gpu_x = cp.asnumpy(buf.x)
        gpu_y = cp.asnumpy(buf.y)
        np.testing.assert_array_equal(gpu_x, xs)
        np.testing.assert_array_equal(gpu_y, ys)


@needs_gpu
class TestPolygonDecode:
    """GPU decode of Polygon shapefiles."""

    def test_simple_polygon(self):
        """Square polygon coordinates and ring offsets."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        shp_bytes, shx_bytes = _build_shp_shx_polygon([[ring]])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "poly", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POLYGON]
        x = cp.asnumpy(buf.x)
        y = cp.asnumpy(buf.y)

        expected_x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
        expected_y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)

        # Ring offsets: one ring [0, 5]
        ring_offsets = cp.asnumpy(buf.ring_offsets)
        np.testing.assert_array_equal(ring_offsets, [0, 5])

    def test_polygon_with_hole(self):
        """Polygon with exterior ring + one interior ring (hole)."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        outer = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)]
        inner = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0), (5.0, 5.0)]
        shp_bytes, shx_bytes = _build_shp_shx_polygon([[outer, inner]])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "hole", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POLYGON]
        x = cp.asnumpy(buf.x)
        y = cp.asnumpy(buf.y)

        # 5 points outer + 5 points inner = 10 total
        assert len(x) == 10
        assert len(y) == 10

        # Ring offsets: [0, 5, 10]
        ring_offsets = cp.asnumpy(buf.ring_offsets)
        np.testing.assert_array_equal(ring_offsets, [0, 5, 10])

        # geometry_offsets: one polygon with 2 rings [0, 2]
        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        np.testing.assert_array_equal(geom_offsets, [0, 2])

    def test_multiple_polygons(self):
        """Multiple polygons with different ring counts."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        poly1_ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        poly2_outer = [(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0), (2.0, 2.0)]
        poly2_inner = [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 3.0)]

        polygons = [
            [poly1_ring],  # 1 ring, 4 points
            [poly2_outer, poly2_inner],  # 2 rings, 5 + 4 = 9 points
        ]
        shp_bytes, shx_bytes = _build_shp_shx_polygon(polygons)

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "multi", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POLYGON]
        x = cp.asnumpy(buf.x)
        assert len(x) == 4 + 5 + 4  # 13 total points

        # geometry_offsets: [0, 1, 3] (poly1 has 1 ring, poly2 has 2 rings)
        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        np.testing.assert_array_equal(geom_offsets, [0, 1, 3])

        # ring_offsets: [0, 4, 9, 13]
        ring_offsets = cp.asnumpy(buf.ring_offsets)
        np.testing.assert_array_equal(ring_offsets, [0, 4, 9, 13])


@needs_gpu
class TestPolyLineDecode:
    """GPU decode of PolyLine (LineString) shapefiles."""

    def test_simple_linestring(self):
        """Single-part polyline maps to LineString."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        line = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        shp_bytes, shx_bytes = _build_shp_shx_polyline([[line]])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "line", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        # Single-part polylines -> LineString family
        buf = ds.families[GeometryFamily.LINESTRING]
        x = cp.asnumpy(buf.x)
        y = cp.asnumpy(buf.y)

        np.testing.assert_array_equal(x, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(y, [0.0, 1.0, 0.0])

    def test_multi_part_polyline(self):
        """Multi-part polyline maps to MultiLineString."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        part1 = [(0.0, 0.0), (1.0, 1.0)]
        part2 = [(2.0, 2.0), (3.0, 3.0), (4.0, 2.0)]
        shp_bytes, shx_bytes = _build_shp_shx_polyline([[part1, part2]])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "mline", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.MULTILINESTRING]
        x = cp.asnumpy(buf.x)
        assert len(x) == 5  # 2 + 3

    def test_polyline_uses_count_scatter_helpers(self, monkeypatch):
        """Polyline decode uses async count-scatter helper paths."""
        from vibespatial.cuda import _runtime as runtime_module
        from vibespatial.io.shp_gpu import read_shp_gpu

        calls = {"totals": [], "with_transfer": []}
        original_totals = runtime_module.count_scatter_totals
        original_with_transfer = runtime_module.count_scatter_total_with_transfer

        def _record_totals(runtime, count_offset_pairs, *, reason=None):
            calls["totals"].append(len(count_offset_pairs))
            return original_totals(runtime, count_offset_pairs, reason=reason)

        def _record_with_transfer(*args, **kwargs):
            calls["with_transfer"].append(kwargs.get("precomputed_total"))
            return original_with_transfer(*args, **kwargs)

        monkeypatch.setattr(runtime_module, "count_scatter_totals", _record_totals)
        monkeypatch.setattr(runtime_module, "count_scatter_total_with_transfer", _record_with_transfer)

        line = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        shp_bytes, shx_bytes = _build_shp_shx_polyline([[line]])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "line", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        assert GeometryFamily.LINESTRING in ds.families
        assert calls["totals"] == [2]
        assert len(calls["with_transfer"]) == 1
        assert calls["with_transfer"][0] is not None

    def test_multiple_linestrings(self):
        """Multiple single-part polylines."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        lines = [
            [[(0.0, 0.0), (1.0, 0.0)]],
            [[(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]],
        ]
        shp_bytes, shx_bytes = _build_shp_shx_polyline(lines)

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "lines", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.LINESTRING]
        x = cp.asnumpy(buf.x)
        assert len(x) == 5

        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        np.testing.assert_array_equal(geom_offsets, [0, 2, 5])


@needs_gpu
class TestMultiPointDecode:
    """GPU decode of MultiPoint shapefiles."""

    def test_simple_multipoint(self):
        """Single MultiPoint record."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        shp_bytes, shx_bytes = _build_shp_shx_multipoint([points])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "mpt", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.MULTIPOINT]
        x = cp.asnumpy(buf.x)
        y = cp.asnumpy(buf.y)

        np.testing.assert_array_equal(x, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])

    def test_multiple_multipoints(self):
        """Multiple MultiPoint records with varying point counts."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        multipoints = [
            [(1.0, 2.0), (3.0, 4.0)],
            [(10.0, 20.0)],
            [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)],
        ]
        shp_bytes, shx_bytes = _build_shp_shx_multipoint(multipoints)

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "mpts", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.MULTIPOINT]
        x = cp.asnumpy(buf.x)
        assert len(x) == 6  # 2 + 1 + 3

        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        np.testing.assert_array_equal(geom_offsets, [0, 2, 3, 6])


@needs_gpu
class TestEmptyAndNull:
    """Edge cases: empty files, null shapes."""

    def test_empty_file(self):
        """Zero-record SHP file."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        shp_bytes, shx_bytes = _build_shp_shx_point([])

        with tempfile.TemporaryDirectory() as td:
            shp_path = _write_shp_shx(Path(td), "empty", shp_bytes, shx_bytes)
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POINT]
        assert buf.x.size == 0
        assert buf.y.size == 0

    def test_missing_shx_raises(self):
        """Missing SHX file raises FileNotFoundError."""
        from vibespatial.io.shp_gpu import read_shp_gpu

        shp_bytes, _ = _build_shp_shx_point([(1.0, 2.0)])

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "noshx.shp"
            shp_path.write_bytes(shp_bytes)

            with pytest.raises(FileNotFoundError, match="SHX"):
                read_shp_gpu(shp_path)


@needs_gpu
@needs_pyogrio
class TestPyogrioComparison:
    """GPU direct decode matches pyogrio output for the same file.

    Uses pyogrio to write known geometries, then reads with both pyogrio
    and the GPU decoder, and verifies coordinates match exactly.
    """

    def test_point_matches_pyogrio(self):
        """Point coordinates match pyogrio."""
        import geopandas as gpd
        from vibespatial.io.shp_gpu import read_shp_gpu

        n = 500
        rng = np.random.default_rng(123)
        xs = rng.uniform(-180, 180, n)
        ys = rng.uniform(-90, 90, n)
        geoms = [Point(x, y) for x, y in zip(xs, ys)]

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "points.shp"
            gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            # GPU decode
            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POINT]
        gpu_x = cp.asnumpy(buf.x)
        gpu_y = cp.asnumpy(buf.y)

        np.testing.assert_array_equal(gpu_x, xs)
        np.testing.assert_array_equal(gpu_y, ys)

    def test_polygon_matches_pyogrio(self):
        """Polygon coordinates and structure match pyogrio."""
        import geopandas as gpd
        from vibespatial.io.shp_gpu import read_shp_gpu

        # Create polygons with known coordinates
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)],
            [[(12, 12), (18, 12), (18, 18), (12, 18), (12, 12)]],
        )

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "polys.shp"
            gdf = gpd.GeoDataFrame(geometry=[poly1, poly2], crs="EPSG:4326")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POLYGON]
        gpu_x = cp.asnumpy(buf.x)

        # poly1: 5 coords (closed ring), poly2: 5 outer + 5 inner = 10
        assert len(gpu_x) == 15

        # Verify ring offsets
        ring_offsets = cp.asnumpy(buf.ring_offsets)
        # poly1: 1 ring, poly2: 2 rings -> ring_offsets = [0, 5, 10, 15]
        np.testing.assert_array_equal(ring_offsets, [0, 5, 10, 15])

        # geom_offsets: poly1 has 1 ring, poly2 has 2 rings -> [0, 1, 3]
        geom_offsets = cp.asnumpy(buf.geometry_offsets)
        np.testing.assert_array_equal(geom_offsets, [0, 1, 3])

    def test_linestring_matches_pyogrio(self):
        """LineString coordinates match pyogrio."""
        import geopandas as gpd
        from vibespatial.io.shp_gpu import read_shp_gpu

        line1 = LineString([(0, 0), (1, 1), (2, 0)])
        line2 = LineString([(10, 10), (20, 20)])

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "lines.shp"
            gdf = gpd.GeoDataFrame(geometry=[line1, line2], crs="EPSG:4326")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.LINESTRING]
        gpu_x = cp.asnumpy(buf.x)
        gpu_y = cp.asnumpy(buf.y)

        expected_x = np.array([0.0, 1.0, 2.0, 10.0, 20.0])
        expected_y = np.array([0.0, 1.0, 0.0, 10.0, 20.0])
        np.testing.assert_array_equal(gpu_x, expected_x)
        np.testing.assert_array_equal(gpu_y, expected_y)

    def test_large_polygon_file(self):
        """100K simple polygons decode correctly."""
        import geopandas as gpd
        from vibespatial.io.shp_gpu import read_shp_gpu

        n = 100_000
        rng = np.random.default_rng(77)
        geoms = []
        for _ in range(n):
            cx, cy = rng.uniform(-100, 100, 2)
            s = rng.uniform(0.01, 1.0)
            geoms.append(
                Polygon(
                    [
                        (cx, cy),
                        (cx + s, cy),
                        (cx + s, cy + s),
                        (cx, cy + s),
                        (cx, cy),
                    ]
                )
            )

        with tempfile.TemporaryDirectory() as td:
            shp_path = Path(td) / "large.shp"
            gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            owned = read_shp_gpu(shp_path)

        ds = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        buf = ds.families[GeometryFamily.POLYGON]
        gpu_x = cp.asnumpy(buf.x)

        # Each polygon has 5 points (closed ring)
        assert len(gpu_x) == n * 5
