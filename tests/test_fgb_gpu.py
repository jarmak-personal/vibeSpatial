"""Tests for GPU FlatGeobuf (.fgb) direct binary decoder.

Tests cover:
- Header parsing (CPU unit tests -- no GPU needed)
- Point coordinate roundtrip (GPU NVRTC kernel)
- LineString coordinate + structure roundtrip (GPU NVRTC kernel)
- Polygon coordinate + ring structure roundtrip (GPU NVRTC kernel)
- MultiPoint, MultiLineString, MultiPolygon roundtrip
- Attribute extraction: string, int, float columns
- Comparison: GPU decode matches pyogrio output
- Non-indexed FGB files (sequential scan)
- Empty geometry handling
- CRS extraction
"""

from __future__ import annotations

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
    import importlib.util

    HAS_PYOGRIO = importlib.util.find_spec("pyogrio") is not None
except (ImportError, ModuleNotFoundError):
    HAS_PYOGRIO = False

try:
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

    HAS_SHAPELY = True
except (ImportError, ModuleNotFoundError):
    HAS_SHAPELY = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
needs_pyogrio = pytest.mark.skipif(not HAS_PYOGRIO, reason="pyogrio not available")
needs_shapely = pytest.mark.skipif(not HAS_SHAPELY, reason="shapely/geopandas not available")


# ---------------------------------------------------------------------------
# FGB file builder helpers
# ---------------------------------------------------------------------------


def _write_test_fgb(geometries, *, properties=None, crs="EPSG:4326") -> Path:
    """Write geometries to a temporary FGB file using geopandas.

    Parameters
    ----------
    geometries : list of shapely geometries
    properties : dict of column_name -> list of values, optional
    crs : str

    Returns
    -------
    Path to the temporary .fgb file
    """
    import geopandas as gpd

    data = {}
    if properties:
        data.update(properties)

    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=crs)
    tmp = tempfile.NamedTemporaryFile(suffix=".fgb", delete=False)
    tmp.close()
    gdf.to_file(tmp.name, driver="FlatGeobuf")
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Phase 1: Header parsing tests (CPU only)
# ---------------------------------------------------------------------------


@needs_shapely
@needs_pyogrio
class TestFgbHeaderParsing:
    """Test CPU-side FlatGeobuf header parsing."""

    def test_point_header(self):
        """Parse header from a Point FGB file."""
        from vibespatial.io.fgb_gpu import FGB_GEOM_POINT, _parse_fgb_header

        path = _write_test_fgb([Point(1, 2), Point(3, 4)])
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        assert header.geometry_type == FGB_GEOM_POINT
        assert header.features_count == 2
        assert not header.has_z
        assert not header.has_m

    def test_polygon_header(self):
        """Parse header from a Polygon FGB file."""
        from vibespatial.io.fgb_gpu import FGB_GEOM_POLYGON, _parse_fgb_header

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        path = _write_test_fgb([poly])
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        assert header.geometry_type == FGB_GEOM_POLYGON
        assert header.features_count == 1

    def test_header_with_columns(self):
        """Parse header with attribute columns."""
        from vibespatial.io.fgb_gpu import _parse_fgb_header

        path = _write_test_fgb(
            [Point(1, 2), Point(3, 4)],
            properties={"name": ["a", "b"], "value": [1.5, 2.5]},
        )
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        assert header.features_count == 2
        col_names = [c.name for c in header.columns]
        assert "name" in col_names
        assert "value" in col_names

    def test_header_crs(self):
        """Parse CRS from header."""
        from vibespatial.io.fgb_gpu import _parse_fgb_header

        path = _write_test_fgb([Point(1, 2)], crs="EPSG:4326")
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        # CRS should be present (either as WKT or org:code)
        assert header.crs_wkt is not None

    def test_linestring_header(self):
        """Parse header from a LineString FGB file."""
        from vibespatial.io.fgb_gpu import FGB_GEOM_LINESTRING, _parse_fgb_header

        ls = LineString([(0, 0), (1, 1), (2, 0)])
        path = _write_test_fgb([ls])
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        assert header.geometry_type == FGB_GEOM_LINESTRING
        assert header.features_count == 1

    def test_invalid_magic(self):
        """Reject non-FGB files."""
        from vibespatial.io.fgb_gpu import _parse_fgb_header

        with pytest.raises(ValueError, match="Not a FlatGeobuf"):
            _parse_fgb_header(b"not a flatgeobuf file at all")

    def test_truncated_file(self):
        """Reject truncated files."""
        from vibespatial.io.fgb_gpu import _parse_fgb_header

        with pytest.raises(ValueError, match="too small"):
            _parse_fgb_header(b"short")


# ---------------------------------------------------------------------------
# Phase 2: Feature offset scanning tests
# ---------------------------------------------------------------------------


@needs_shapely
@needs_pyogrio
class TestFeatureOffsetScan:
    """Test feature offset scanning for non-indexed and indexed FGB files."""

    def test_scan_offsets_sequential(self):
        """Sequential scan produces correct feature offsets."""
        from vibespatial.io.fgb_gpu import _parse_fgb_header, _scan_feature_offsets

        path = _write_test_fgb([Point(1, 2), Point(3, 4), Point(5, 6)])
        with open(path, "rb") as f:
            data = f.read()

        header = _parse_fgb_header(data)
        # For non-indexed path, scan from data_section_start
        # We need to calculate where the data section starts
        data_start = header.header_size
        if header.index_node_size > 0:
            from vibespatial.io.fgb_gpu import _calc_index_size

            data_start += _calc_index_size(header.features_count, header.index_node_size)

        offsets = _scan_feature_offsets(data, data_start, header.features_count)
        assert len(offsets) == 3
        # Offsets should be monotonically increasing
        assert np.all(np.diff(offsets) > 0)


# ---------------------------------------------------------------------------
# Phase 4: GPU geometry decode tests
# ---------------------------------------------------------------------------


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestPointDecodeGpu:
    """Test GPU Point decode."""

    def test_point_roundtrip(self):
        """Point coordinates survive FGB -> GPU -> host roundtrip."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        coords = [(1.5, 2.5), (3.25, 4.75), (-10.0, 20.0)]
        path = _write_test_fgb([Point(*c) for c in coords])
        result = read_fgb_gpu(path)

        assert result.n_features == 3
        owned = result.geometry
        device_state = owned._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POINT]
        host_x = cp.asnumpy(dbuf.x)
        host_y = cp.asnumpy(dbuf.y)

        # FGB Hilbert index reorders features, so compare as sets
        gpu_coords = set(zip(host_x.tolist(), host_y.tolist()))
        ref_coords = set(coords)
        assert gpu_coords == ref_coords

    def test_many_points(self):
        """Decode many points."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        n = 1000
        xs = np.random.default_rng(42).uniform(-180, 180, n)
        ys = np.random.default_rng(42).uniform(-90, 90, n)
        points = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
        path = _write_test_fgb(points)
        result = read_fgb_gpu(path)

        assert result.n_features == n
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POINT]
        host_x = cp.asnumpy(dbuf.x)
        host_y = cp.asnumpy(dbuf.y)

        # FGB index may reorder features (Hilbert sort), so compare sorted
        gpu_coords = set(zip(np.round(host_x, 10), np.round(host_y, 10)))
        ref_coords = set(zip(np.round(xs, 10), np.round(ys, 10)))
        assert gpu_coords == ref_coords


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestLineStringDecodeGpu:
    """Test GPU LineString decode."""

    def test_linestring_roundtrip(self):
        """LineString coordinates and structure survive roundtrip."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        ls1 = LineString([(0, 0), (1, 1), (2, 0)])
        ls2 = LineString([(10, 10), (20, 20)])
        path = _write_test_fgb([ls1, ls2])
        result = read_fgb_gpu(path)

        assert result.n_features == 2
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.LINESTRING]
        host_x = cp.asnumpy(dbuf.x)
        host_y = cp.asnumpy(dbuf.y)
        host_geom_off = cp.asnumpy(dbuf.geometry_offsets)

        # Total coords = 3 + 2 = 5
        assert len(host_x) == 5
        assert len(host_y) == 5
        # geometry_offsets should have 3 entries (0, 3, 5) -- but Hilbert
        # index may reorder, so just check shape
        assert len(host_geom_off) == 3


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestPolygonDecodeGpu:
    """Test GPU Polygon decode."""

    def test_simple_polygon_roundtrip(self):
        """Simple polygon coordinates survive roundtrip."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        path = _write_test_fgb([poly])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POLYGON]
        host_x = cp.asnumpy(dbuf.x)
        host_y = cp.asnumpy(dbuf.y)

        # Polygon has 5 coords (closed ring)
        assert len(host_x) == 5
        assert len(host_y) == 5

    def test_polygon_with_hole(self):
        """Polygon with hole preserves ring structure."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        exterior = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)]
        poly = Polygon(exterior, [hole])
        path = _write_test_fgb([poly])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POLYGON]
        host_x = cp.asnumpy(dbuf.x)

        # 5 exterior + 5 hole = 10 coordinates
        assert len(host_x) == 10

        # Should have ring_offsets
        assert dbuf.ring_offsets is not None
        host_ring_off = cp.asnumpy(dbuf.ring_offsets)
        # 2 rings -> 3 offset entries
        assert len(host_ring_off) == 3

    def test_polygon_decode_uses_count_scatter_total_helper(self, monkeypatch):
        """Complex FGB decode uses async count-scatter total helpers."""
        from vibespatial.cuda import _runtime as runtime_module
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        calls = {"totals": []}
        original_totals = runtime_module.count_scatter_totals

        def _record_totals(runtime, count_offset_pairs):
            calls["totals"].append(len(count_offset_pairs))
            return original_totals(runtime, count_offset_pairs)

        monkeypatch.setattr(runtime_module, "count_scatter_totals", _record_totals)

        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        path = _write_test_fgb([poly])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        assert calls["totals"] == [3]

    def test_multiple_polygons(self):
        """Multiple polygons decode correctly."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]),
            Polygon([(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]),
        ]
        path = _write_test_fgb(polys)
        result = read_fgb_gpu(path)

        assert result.n_features == 3
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POLYGON]
        host_x = cp.asnumpy(dbuf.x)

        # 3 polygons * 5 coords each = 15
        assert len(host_x) == 15


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestMultiGeometryDecodeGpu:
    """Test GPU Multi* geometry decode."""

    def test_multipoint(self):
        """MultiPoint decode."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        mp = MultiPoint([(1, 2), (3, 4), (5, 6)])
        path = _write_test_fgb([mp])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.MULTIPOINT]
        host_x = cp.asnumpy(dbuf.x)
        assert len(host_x) == 3

    def test_multilinestring(self):
        """MultiLineString decode."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        mls = MultiLineString(
            [
                [(0, 0), (1, 1)],
                [(10, 10), (20, 20), (30, 10)],
            ]
        )
        path = _write_test_fgb([mls])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.MULTILINESTRING]
        host_x = cp.asnumpy(dbuf.x)
        # 2 + 3 = 5 coords
        assert len(host_x) == 5

    def test_multipolygon(self):
        """MultiPolygon decode."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        mp = MultiPolygon(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]),
            ]
        )
        path = _write_test_fgb([mp])
        result = read_fgb_gpu(path)

        assert result.n_features == 1
        device_state = result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.MULTIPOLYGON]
        host_x = cp.asnumpy(dbuf.x)
        # 2 polygons * 5 coords = 10
        assert len(host_x) == 10


# ---------------------------------------------------------------------------
# Attribute extraction tests
# ---------------------------------------------------------------------------


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestAttributeExtraction:
    """Test CPU-side attribute extraction from FGB properties."""

    def test_string_attributes(self):
        """String columns extracted correctly."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb(
            [Point(1, 2), Point(3, 4)],
            properties={"name": ["hello", "world"]},
        )
        result = read_fgb_gpu(path)

        assert result.attributes is not None
        assert "name" in result.attributes
        names = result.attributes["name"]
        assert set(n for n in names if n is not None) == {"hello", "world"}

    def test_numeric_attributes(self):
        """Numeric columns extracted correctly."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb(
            [Point(1, 2), Point(3, 4), Point(5, 6)],
            properties={"value": [1.5, 2.5, 3.5], "count": [10, 20, 30]},
        )
        result = read_fgb_gpu(path)

        assert result.attributes is not None
        assert "value" in result.attributes
        assert "count" in result.attributes

    def test_mixed_attributes(self):
        """Mixed string + numeric columns."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb(
            [Point(1, 2), Point(3, 4)],
            properties={
                "name": ["a", "b"],
                "val": [1.0, 2.0],
                "flag": [True, False],
            },
        )
        result = read_fgb_gpu(path)

        assert result.attributes is not None
        col_names = set(result.attributes.keys())
        assert "name" in col_names
        assert "val" in col_names


# ---------------------------------------------------------------------------
# CRS tests
# ---------------------------------------------------------------------------


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestFgbCrs:
    """Test CRS extraction from FGB files."""

    def test_crs_present(self):
        """CRS is extracted from FGB header."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb([Point(1, 2)], crs="EPSG:4326")
        result = read_fgb_gpu(path)
        assert result.crs is not None

    def test_crs_custom(self):
        """Custom CRS (non-4326) is extracted."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb([Point(500000, 5500000)], crs="EPSG:32632")
        result = read_fgb_gpu(path)
        assert result.crs is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestFgbEdgeCases:
    """Test edge cases."""

    def test_single_point(self):
        """Single point file."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb([Point(42.0, -73.0)])
        result = read_fgb_gpu(path)
        assert result.n_features == 1

    def test_no_attributes(self):
        """File with no attribute columns."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        path = _write_test_fgb([Point(1, 2), Point(3, 4)])
        result = read_fgb_gpu(path)
        # attributes may be None or empty dict
        if result.attributes is not None:
            # All values should be None if no real columns
            pass  # geopandas may add index column


# ---------------------------------------------------------------------------
# Comparison with pyogrio reference
# ---------------------------------------------------------------------------


@needs_gpu
@needs_shapely
@needs_pyogrio
class TestFgbVsPyogrio:
    """Compare GPU FGB decode against pyogrio as oracle."""

    def test_point_matches_pyogrio(self):
        """Point coordinates match pyogrio output."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        coords = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        path = _write_test_fgb([Point(*c) for c in coords])

        # GPU decode
        gpu_result = read_fgb_gpu(path)
        device_state = gpu_result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POINT]
        gpu_x = cp.asnumpy(dbuf.x)
        gpu_y = cp.asnumpy(dbuf.y)

        # pyogrio reference
        import geopandas as gpd

        ref_gdf = gpd.read_file(str(path))
        ref_x = np.array([g.x for g in ref_gdf.geometry])
        ref_y = np.array([g.y for g in ref_gdf.geometry])

        # FGB Hilbert index may reorder features, so compare sets
        gpu_set = set(zip(np.round(gpu_x, 12), np.round(gpu_y, 12)))
        ref_set = set(zip(np.round(ref_x, 12), np.round(ref_y, 12)))
        assert gpu_set == ref_set

    def test_polygon_coord_count_matches(self):
        """Polygon total coordinate count matches pyogrio."""
        from vibespatial.io.fgb_gpu import read_fgb_gpu

        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]),
        ]
        path = _write_test_fgb(polys)

        gpu_result = read_fgb_gpu(path)
        device_state = gpu_result.geometry._ensure_device_state()
        from vibespatial.geometry.buffers import GeometryFamily

        dbuf = device_state.families[GeometryFamily.POLYGON]
        gpu_total_coords = int(dbuf.x.size)

        # Reference
        import geopandas as gpd

        ref_gdf = gpd.read_file(str(path))
        ref_total_coords = sum(len(g.exterior.coords) for g in ref_gdf.geometry)

        assert gpu_total_coords == ref_total_coords
