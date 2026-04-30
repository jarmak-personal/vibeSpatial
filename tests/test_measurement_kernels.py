"""Tests for GPU-accelerated area and length measurement kernels.

Covers CPU fallback, GPU acceleration, dispatch thresholds, precision,
zero-copy device-resident path, and end-to-end integration through
DeviceGeometryArray and GeoSeries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial.constructive.measurement import (
    area_owned,
    length_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.testing import build_owned as _make_owned


def _has_gpu():
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime

        return get_cuda_runtime().available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")
def _shapely_area(geometries: list) -> np.ndarray:
    return shapely.area(np.array(geometries, dtype=object))


def _shapely_length(geometries: list) -> np.ndarray:
    return shapely.length(np.array(geometries, dtype=object))


# =====================================================================
# CPU area tests
# =====================================================================


class TestAreaCPU:
    def test_unit_square(self):
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        owned = _make_owned([sq])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_triangle(self):
        tri = Polygon([(0, 0), (4, 0), (0, 3)])
        owned = _make_owned([tri])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [6.0], atol=1e-12)

    def test_polygon_with_hole(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly = Polygon(exterior, [hole])
        owned = _make_owned([poly])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = 100.0 - 36.0  # 64.0
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        mp = MultiPolygon([p1, p2])
        owned = _make_owned([mp])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [1.0 + 4.0], atol=1e-12)

    def test_linestring_area_is_zero(self):
        ls = LineString([(0, 0), (1, 1), (2, 0)])
        owned = _make_owned([ls])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [0.0], atol=1e-15)

    def test_point_area_is_zero(self):
        pt = Point(1, 2)
        owned = _make_owned([pt])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [0.0], atol=1e-15)

    def test_matches_shapely_batch(self):
        """Batch comparison with Shapely oracle."""
        rng = np.random.default_rng(42)
        geoms = []
        for _ in range(50):
            n_verts = rng.integers(3, 8)
            angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
            radii = rng.uniform(1, 10, n_verts)
            cx, cy = rng.uniform(-100, 100, 2)
            coords = [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles)]
            geoms.append(Polygon(coords))
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_empty_array(self):
        owned = _make_owned([])
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        assert len(result) == 0

    def test_mixed_geometry_types(self):
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# =====================================================================
# CPU length tests
# =====================================================================


class TestLengthCPU:
    def test_linestring(self):
        ls = LineString([(0, 0), (3, 0), (3, 4)])
        owned = _make_owned([ls])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [3.0 + 4.0], atol=1e-12)

    def test_polygon_perimeter(self):
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        owned = _make_owned([sq])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [4.0], atol=1e-12)

    def test_polygon_with_hole_all_rings(self):
        """Polygon length includes both exterior and hole perimeters."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly = Polygon(exterior, [hole])
        owned = _make_owned([poly])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_length([poly])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_multilinestring(self):
        mls = MultiLineString([
            [(0, 0), (1, 0)],
            [(-1, 0), (1, 0)],
        ])
        owned = _make_owned([mls])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [1.0 + 2.0], atol=1e-12)

    def test_multipolygon_length(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        mp = MultiPolygon([p1, p2])
        owned = _make_owned([mp])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_length([mp])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_point_length_is_zero(self):
        pt = Point(1, 2)
        owned = _make_owned([pt])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [0.0], atol=1e-15)

    def test_multipoint_length_is_zero(self):
        mp = MultiPoint([(0, 0), (1, 1)])
        owned = _make_owned([mp])
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(result, [0.0], atol=1e-15)

    def test_matches_shapely_batch(self):
        rng = np.random.default_rng(123)
        geoms = []
        for _ in range(50):
            n = rng.integers(2, 10)
            coords = rng.uniform(-100, 100, (n, 2)).tolist()
            geoms.append(LineString(coords))
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_length(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_mixed_geometry_types(self):
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        expected = _shapely_length(geoms)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# =====================================================================
# GPU tests
# =====================================================================


@requires_gpu
class TestAreaGPU:
    def test_matches_shapely_oracle(self):
        """GPU area matches Shapely oracle on random polygons."""
        rng = np.random.default_rng(42)
        geoms = []
        for _ in range(200):
            n_verts = rng.integers(3, 8)
            angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
            radii = rng.uniform(1, 10, n_verts)
            cx, cy = rng.uniform(-100, 100, 2)
            coords = [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles)]
            geoms.append(Polygon(coords))
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_area(geoms)
        # fp32 Kahan on consumer GPUs gives ~1e-4 precision (ADR-0002)
        np.testing.assert_allclose(result, expected, rtol=5e-3)

    def test_matches_cpu(self):
        """GPU and CPU produce identical results on simple polygons."""
        geoms = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(0, 0), (5, 0), (5, 3)]),
        ]
        owned = _make_owned(geoms)
        gpu_result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        cpu_result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=5e-3)

    def test_polygon_with_holes(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly = Polygon(exterior, [hole])
        owned = _make_owned([poly])
        result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        np.testing.assert_allclose(result, [64.0], atol=1e-10)

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        mp = MultiPolygon([p1, p2])
        owned = _make_owned([mp])
        result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        np.testing.assert_allclose(result, [5.0], atol=1e-12)

    def test_mixed_types(self):
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            MultiPolygon([
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
            ]),
        ]
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tiny_polygon_area_at_large_coordinates(self):
        base_x = 366_000.0
        base_y = 3_080_000.0
        geoms = [
            Polygon(
                [
                    (base_x, base_y),
                    (base_x + 0.001, base_y),
                    (base_x + 0.001, base_y + 0.001),
                    (base_x, base_y + 0.001),
                ]
            )
        ]
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-12)


@requires_gpu
class TestLengthGPU:
    def test_linestring_matches_shapely(self):
        rng = np.random.default_rng(99)
        geoms = []
        for _ in range(200):
            n = rng.integers(2, 10)
            coords = rng.uniform(-100, 100, (n, 2)).tolist()
            geoms.append(LineString(coords))
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_length(geoms)
        # fp32 Kahan on consumer GPUs gives ~1e-4 precision (ADR-0002)
        np.testing.assert_allclose(result, expected, rtol=5e-3)

    def test_polygon_perimeter_matches_shapely(self):
        rng = np.random.default_rng(77)
        geoms = []
        for _ in range(100):
            n = rng.integers(3, 8)
            angles = np.sort(rng.uniform(0, 2 * math.pi, n))
            radii = rng.uniform(1, 10, n)
            coords = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
            geoms.append(Polygon(coords))
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_length(geoms)
        # fp32 Kahan on consumer GPUs gives ~1e-4 precision (ADR-0002)
        np.testing.assert_allclose(result, expected, rtol=5e-3)

    def test_multilinestring_matches_shapely(self):
        geoms = [
            MultiLineString([[(0, 0), (1, 0)], [(-1, 0), (2, 0)]]),
            MultiLineString([[(0, 0), (3, 4)]]),
        ]
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_length(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_matches_cpu(self):
        geoms = [
            LineString([(0, 0), (3, 0), (3, 4)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
        owned = _make_owned(geoms)
        gpu_result = length_owned(owned, dispatch_mode=ExecutionMode.GPU)
        cpu_result = length_owned(owned, dispatch_mode=ExecutionMode.CPU)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=5e-3)

    def test_mixed_types(self):
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            MultiLineString([[(0, 0), (1, 0)], [(2, 2), (3, 3)]]),
        ]
        owned = _make_owned(geoms)
        result = length_owned(owned, dispatch_mode=ExecutionMode.GPU)
        expected = _shapely_length(geoms)
        # fp32 Kahan on consumer GPUs gives ~1e-4 precision (ADR-0002)
        np.testing.assert_allclose(result, expected, rtol=5e-3)


# =====================================================================
# Dispatch tests
# =====================================================================


class TestDispatch:
    def test_auto_small_batch_uses_cpu(self):
        """Small batch (<500 rows) dispatches to CPU in AUTO mode."""
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 10
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.AUTO)
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_explicit_cpu_mode(self):
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 100
        owned = _make_owned(geoms)
        result = area_owned(owned, dispatch_mode=ExecutionMode.CPU)
        assert len(result) == 100
        np.testing.assert_allclose(result, [1.0] * 100, atol=1e-12)


# =====================================================================
# Integration tests
# =====================================================================


class TestDGAIntegration:
    def test_dga_area_no_shapely_materialization(self):
        """DGA.area should NOT produce materialization diagnostic events."""
        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.geometry.owned import DiagnosticKind

        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 5
        owned = _make_owned(geoms)
        dga = DeviceGeometryArray._from_owned(owned)
        _ = dga.area
        mat_events = [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION]
        assert len(mat_events) == 0, f"Unexpected materialization events: {mat_events}"

    def test_dga_length_no_shapely_materialization(self):
        """DGA.length should NOT produce materialization diagnostic events."""
        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.geometry.owned import DiagnosticKind

        geoms = [LineString([(0, 0), (1, 0), (1, 1)])] * 5
        owned = _make_owned(geoms)
        dga = DeviceGeometryArray._from_owned(owned)
        _ = dga.length
        mat_events = [e for e in owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION]
        assert len(mat_events) == 0, f"Unexpected materialization events: {mat_events}"

    def test_dga_area_matches_shapely(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (5, 0), (5, 3)]),
        ]
        owned = _make_owned(geoms)
        dga = DeviceGeometryArray._from_owned(owned)
        result = dga.area
        expected = _shapely_area(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dga_length_matches_shapely(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [
            LineString([(0, 0), (3, 0), (3, 4)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
        owned = _make_owned(geoms)
        dga = DeviceGeometryArray._from_owned(owned)
        result = dga.length
        expected = _shapely_length(geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
