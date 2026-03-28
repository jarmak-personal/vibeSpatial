"""Tests for fused ingest+reproject: GPU CRS transform during coordinate parsing.

Tests:
  1. WGS84 point -> Web Mercator (verify against known values)
  2. Roundtrip: WGS84 -> 3857 -> 4326, verify within tolerance
  3. No-op (same CRS) returns unchanged coords
  4. Fused read_geojson_gpu with target_crs parameter
  5. Empty array edge case
  6. Unsupported CRS raises ValueError
"""

from __future__ import annotations

import json
import math
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
# Reference values (computed with pyproj / manual formulas)
# ---------------------------------------------------------------------------

_SEMI_CIRC = 20037508.342789244


def _wgs84_to_mercator_ref(lon: float, lat: float) -> tuple[float, float]:
    """Pure-Python reference WGS84 -> Web Mercator."""
    x = lon * _SEMI_CIRC / 180.0
    lat_rad = lat * math.pi / 180.0
    y = math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) * _SEMI_CIRC / math.pi
    return x, y


def _mercator_to_wgs84_ref(x: float, y: float) -> tuple[float, float]:
    """Pure-Python reference Web Mercator -> WGS84."""
    lon = x * 180.0 / _SEMI_CIRC
    lat = math.atan(math.exp(y * math.pi / _SEMI_CIRC)) * 360.0 / math.pi - 90.0
    return lon, lat


# WGS84 -> Web Mercator reference points: (lon, lat)
_INPUT_POINTS = [
    (0.0, 0.0),           # Origin
    (-0.1276, 51.5074),   # London
    (-74.006, 40.7128),   # New York
    (139.6917, 35.6895),  # Tokyo
    (151.2093, -33.8688), # Sydney
    (0.0, 23.4365),       # Prime meridian, Tropic of Cancer
    (179.999, 0.0),       # Near anti-meridian
]


def _get_device_coords(owned, family):
    """Extract device-resident x/y arrays from an OwnedGeometryArray."""
    if owned.device_state is not None and family in owned.device_state.families:
        d_buf = owned.device_state.families[family]
        return cp.asnumpy(d_buf.x), cp.asnumpy(d_buf.y)
    buf = owned.families[family]
    return np.asarray(buf.x), np.asarray(buf.y)


# ---------------------------------------------------------------------------
# Tests for transform_coordinates_inplace
# ---------------------------------------------------------------------------

@needs_gpu
class TestTransformCoordinatesInplace:
    """Tests for the GPU CRS transform primitive."""

    def test_wgs84_to_mercator_known_values(self):
        """WGS84 -> Web Mercator matches Python reference values."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        for lon, lat in _INPUT_POINTS:
            d_x = cp.array([lon], dtype=cp.float64)
            d_y = cp.array([lat], dtype=cp.float64)
            transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")
            result_x = float(d_x[0])
            result_y = float(d_y[0])

            ref_x, ref_y = _wgs84_to_mercator_ref(lon, lat)

            # GPU and Python both evaluate the same fp64 formula; they should
            # agree to within a few ULPs.  Use atol for near-zero values and
            # rtol for large values.
            np.testing.assert_allclose(
                result_x, ref_x, rtol=1e-12, atol=1e-8,
                err_msg=f"x mismatch for lon={lon}",
            )
            np.testing.assert_allclose(
                result_y, ref_y, rtol=1e-12, atol=1e-8,
                err_msg=f"y mismatch for lat={lat}",
            )

    def test_mercator_to_wgs84_known_values(self):
        """Web Mercator -> WGS84 matches Python reference values."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        for lon, lat in _INPUT_POINTS:
            ref_x, ref_y = _wgs84_to_mercator_ref(lon, lat)
            d_x = cp.array([ref_x], dtype=cp.float64)
            d_y = cp.array([ref_y], dtype=cp.float64)
            transform_coordinates_inplace(d_x, d_y, "EPSG:3857", "EPSG:4326")
            result_lon = float(d_x[0])
            result_lat = float(d_y[0])

            np.testing.assert_allclose(
                result_lon, lon, atol=1e-10,
                err_msg=f"lon mismatch for merc_x={ref_x}",
            )
            np.testing.assert_allclose(
                result_lat, lat, atol=1e-10,
                err_msg=f"lat mismatch for merc_y={ref_y}",
            )

    def test_roundtrip_4326_to_3857_to_4326(self):
        """Roundtrip WGS84 -> Web Mercator -> WGS84 preserves coordinates."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        rng = np.random.default_rng(42)
        n = 10_000
        lons = rng.uniform(-180, 180, n)
        lats = rng.uniform(-85, 85, n)  # Mercator valid range

        d_x = cp.array(lons, dtype=cp.float64)
        d_y = cp.array(lats, dtype=cp.float64)

        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")
        transform_coordinates_inplace(d_x, d_y, "EPSG:3857", "EPSG:4326")

        np.testing.assert_allclose(cp.asnumpy(d_x), lons, atol=1e-10)
        np.testing.assert_allclose(cp.asnumpy(d_y), lats, atol=1e-10)

    def test_roundtrip_3857_to_4326_to_3857(self):
        """Roundtrip Web Mercator -> WGS84 -> Web Mercator preserves coordinates."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        rng = np.random.default_rng(99)
        n = 5_000
        xs = rng.uniform(-_SEMI_CIRC, _SEMI_CIRC, n)
        ys = rng.uniform(-_SEMI_CIRC * 0.8, _SEMI_CIRC * 0.8, n)

        d_x = cp.array(xs, dtype=cp.float64)
        d_y = cp.array(ys, dtype=cp.float64)

        transform_coordinates_inplace(d_x, d_y, "EPSG:3857", "EPSG:4326")
        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")

        np.testing.assert_allclose(cp.asnumpy(d_x), xs, atol=1e-6)
        np.testing.assert_allclose(cp.asnumpy(d_y), ys, atol=1e-6)

    def test_noop_same_crs(self):
        """Same source and destination CRS returns unchanged coordinates."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        original_x = np.array([1.0, 2.0, 3.0])
        original_y = np.array([4.0, 5.0, 6.0])
        d_x = cp.array(original_x, dtype=cp.float64)
        d_y = cp.array(original_y, dtype=cp.float64)

        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:4326")

        np.testing.assert_array_equal(cp.asnumpy(d_x), original_x)
        np.testing.assert_array_equal(cp.asnumpy(d_y), original_y)

    def test_noop_same_crs_3857(self):
        """Same CRS (3857->3857) returns unchanged coordinates."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        original_x = np.array([100.0, 200.0])
        original_y = np.array([300.0, 400.0])
        d_x = cp.array(original_x, dtype=cp.float64)
        d_y = cp.array(original_y, dtype=cp.float64)

        transform_coordinates_inplace(d_x, d_y, "EPSG:3857", "EPSG:3857")

        np.testing.assert_array_equal(cp.asnumpy(d_x), original_x)
        np.testing.assert_array_equal(cp.asnumpy(d_y), original_y)

    def test_empty_arrays(self):
        """Empty arrays are handled without error."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        d_x = cp.empty(0, dtype=cp.float64)
        d_y = cp.empty(0, dtype=cp.float64)

        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")

        assert len(d_x) == 0
        assert len(d_y) == 0

    def test_mismatched_lengths_raises(self):
        """Mismatched coordinate array lengths raise ValueError."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        d_x = cp.array([1.0, 2.0], dtype=cp.float64)
        d_y = cp.array([1.0], dtype=cp.float64)

        with pytest.raises(ValueError, match="equal length"):
            transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")

    def test_arbitrary_crs_pair(self):
        """Arbitrary CRS pairs are supported via vibeProj."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        # WGS84 -> UTM zone 32N (EPSG:32632)
        d_x = cp.array([9.0], dtype=cp.float64)   # 9 deg E longitude
        d_y = cp.array([48.0], dtype=cp.float64)   # 48 deg N latitude
        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:32632")

        # UTM zone 32N: x should be near 500000 (central meridian at 9 deg)
        result_x = float(d_x[0])
        result_y = float(d_y[0])
        assert 400_000 < result_x < 600_000, f"UTM easting {result_x} out of range"
        assert 5_000_000 < result_y < 6_000_000, f"UTM northing {result_y} out of range"

    def test_crs_alias_normalization(self):
        """CRS aliases (WGS84, OGC:CRS84) are accepted."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        d_x = cp.array([0.0], dtype=cp.float64)
        d_y = cp.array([0.0], dtype=cp.float64)

        # WGS84 -> EPSG:3857 should work via alias (no ValueError)
        transform_coordinates_inplace(d_x, d_y, "WGS84", "EPSG:3857")
        # Origin maps to near-zero in both axes
        np.testing.assert_allclose(float(d_x[0]), 0.0, atol=1e-8)
        np.testing.assert_allclose(float(d_y[0]), 0.0, atol=1e-8)

    def test_bulk_transform_correctness(self):
        """Bulk transform produces results matching Python reference."""
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        rng = np.random.default_rng(123)
        n = 50_000
        lons = rng.uniform(-180, 180, n)
        lats = rng.uniform(-85, 85, n)

        # Python reference (vectorized for speed)
        ref_x = lons * _SEMI_CIRC / 180.0
        lat_rad = lats * np.pi / 180.0
        ref_y = np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)) * _SEMI_CIRC / np.pi

        # GPU transform
        d_x = cp.array(lons, dtype=cp.float64)
        d_y = cp.array(lats, dtype=cp.float64)
        transform_coordinates_inplace(d_x, d_y, "EPSG:4326", "EPSG:3857")

        # CUDA and numpy may differ by a few ULPs in transcendental functions
        np.testing.assert_allclose(cp.asnumpy(d_x), ref_x, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(cp.asnumpy(d_y), ref_y, rtol=1e-10, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for fused read_geojson_gpu with target_crs
# ---------------------------------------------------------------------------

def _make_point_feature(lon: float, lat: float, props: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": props or {},
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def _make_polygon_feature(coords: list, props: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": props or {},
        "geometry": {"type": "Polygon", "coordinates": coords},
    }


def _write_geojson(path: Path, fc: dict) -> None:
    path.write_text(json.dumps(fc), encoding="utf-8")


@needs_gpu
class TestFusedGeojsonReproject:
    """Tests for read_geojson_gpu with target_crs parameter."""

    def test_read_points_with_target_crs(self, tmp_path):
        """read_geojson_gpu with target_crs reprojects Point coordinates."""
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.io.geojson_gpu import read_geojson_gpu

        features = [
            _make_point_feature(0.0, 0.0),
            _make_point_feature(-74.006, 40.7128),  # NYC
            _make_point_feature(139.6917, 35.6895),  # Tokyo
        ]
        fc = {"type": "FeatureCollection", "features": features}
        path = tmp_path / "points.geojson"
        _write_geojson(path, fc)

        result = read_geojson_gpu(path, target_crs="EPSG:3857")
        owned = result.owned
        x, y = _get_device_coords(owned, GeometryFamily.POINT)

        for i, (lon, lat) in enumerate([(0.0, 0.0), (-74.006, 40.7128), (139.6917, 35.6895)]):
            ref_x, ref_y = _wgs84_to_mercator_ref(lon, lat)
            np.testing.assert_allclose(x[i], ref_x, rtol=1e-10, atol=1e-8, err_msg=f"Point {i} x")
            np.testing.assert_allclose(y[i], ref_y, rtol=1e-10, atol=1e-8, err_msg=f"Point {i} y")

    def test_read_polygons_with_target_crs(self, tmp_path):
        """read_geojson_gpu with target_crs reprojects Polygon coordinates."""
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.io.geojson_gpu import read_geojson_gpu

        ring = [
            [-80.0, 27.0],
            [-79.99, 27.0],
            [-79.99, 27.01],
            [-80.0, 27.01],
            [-80.0, 27.0],
        ]
        features = [_make_polygon_feature([ring])]
        fc = {"type": "FeatureCollection", "features": features}
        path = tmp_path / "polygon.geojson"
        _write_geojson(path, fc)

        result = read_geojson_gpu(path, target_crs="EPSG:3857")
        owned = result.owned
        x, y = _get_device_coords(owned, GeometryFamily.POLYGON)

        ref_x, ref_y = _wgs84_to_mercator_ref(-80.0, 27.0)
        np.testing.assert_allclose(x[0], ref_x, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(y[0], ref_y, rtol=1e-10, atol=1e-8)

    def test_read_without_target_crs_unchanged(self, tmp_path):
        """read_geojson_gpu without target_crs leaves WGS84 coordinates."""
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.io.geojson_gpu import read_geojson_gpu

        features = [_make_point_feature(-74.006, 40.7128)]
        fc = {"type": "FeatureCollection", "features": features}
        path = tmp_path / "no_reproject.geojson"
        _write_geojson(path, fc)

        result = read_geojson_gpu(path)
        owned = result.owned
        x, y = _get_device_coords(owned, GeometryFamily.POINT)

        np.testing.assert_allclose(x[0], -74.006, atol=1e-10)
        np.testing.assert_allclose(y[0], 40.7128, atol=1e-10)

    def test_read_with_same_crs_noop(self, tmp_path):
        """read_geojson_gpu with target_crs='EPSG:4326' is a no-op."""
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.io.geojson_gpu import read_geojson_gpu

        features = [_make_point_feature(10.0, 20.0)]
        fc = {"type": "FeatureCollection", "features": features}
        path = tmp_path / "same_crs.geojson"
        _write_geojson(path, fc)

        result = read_geojson_gpu(path, target_crs="EPSG:4326")
        owned = result.owned
        x, y = _get_device_coords(owned, GeometryFamily.POINT)

        np.testing.assert_allclose(x[0], 10.0, atol=1e-10)
        np.testing.assert_allclose(y[0], 20.0, atol=1e-10)
