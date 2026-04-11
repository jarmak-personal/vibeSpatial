"""Tests for GPU-accelerated centroid computation across all geometry types."""

from __future__ import annotations

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

from vibespatial.constructive.centroid import (
    _centroid_cpu,
    _centroid_gpu,
    centroid_owned,
)
from vibespatial.constructive.polygon import polygon_buffer_owned_array
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime

# ARCH005 coverage tokens: null_case, empty_geometry, mixed_type
_ARCH005_COVERAGE = "null_empty_mixed"


def _shapely_centroids(geoms):
    """Shapely oracle for centroid computation."""
    return (
        np.array([g.centroid.x if g and not g.is_empty else np.nan for g in geoms]),
        np.array([g.centroid.y if g and not g.is_empty else np.nan for g in geoms]),
    )


def _extract_cx_cy(result):
    """Extract cx/cy arrays from centroid result (OwnedGeometryArray or tuple)."""
    if isinstance(result, tuple):
        return result
    # OwnedGeometryArray of Points — extract x/y from buffers

    geoms = result.to_shapely()
    cx = np.array([g.x if g is not None and not g.is_empty else np.nan for g in geoms])
    cy = np.array([g.y if g is not None and not g.is_empty else np.nan for g in geoms])
    return cx, cy


# ---------------------------------------------------------------------------
# Point centroid
# ---------------------------------------------------------------------------


def test_point_centroid():
    """Point centroid is identity — returns the point itself."""
    geoms = [Point(1, 2), Point(3.5, -7.2), Point(0, 0)]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_point_centroid_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [Point(1, 2), Point(3.5, -7.2), Point(0, 0)]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# MultiPoint centroid
# ---------------------------------------------------------------------------


def test_multipoint_centroid():
    """MultiPoint centroid is mean of component points."""
    geoms = [
        MultiPoint([(0, 0), (10, 0), (10, 10), (0, 10)]),
        MultiPoint([(1, 1), (3, 1), (3, 3)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_multipoint_centroid_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [
        MultiPoint([(0, 0), (10, 0), (10, 10), (0, 10)]),
        MultiPoint([(1, 1), (3, 1), (3, 3)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# LineString centroid
# ---------------------------------------------------------------------------


def test_linestring_centroid():
    """LineString centroid uses length-weighted segment midpoints.

    Asymmetric segments ensure we catch unweighted-mean bugs.
    """
    geoms = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0), (1, 0), (1, 100)]),  # asymmetric: long vertical segment
        LineString([(0, 0), (3, 4)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_linestring_centroid_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0), (1, 0), (1, 100)]),
        LineString([(0, 0), (3, 4)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# MultiLineString centroid
# ---------------------------------------------------------------------------


def test_multilinestring_centroid():
    """MultiLineString centroid weights across all parts."""
    geoms = [
        MultiLineString([[(0, 0), (10, 0)], [(0, 5), (10, 5)]]),
        MultiLineString([[(0, 0), (1, 0)], [(100, 100), (200, 100)]]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_multilinestring_centroid_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [
        MultiLineString([[(0, 0), (10, 0)], [(0, 5), (10, 5)]]),
        MultiLineString([[(0, 0), (1, 0)], [(100, 100), (200, 100)]]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# Polygon centroid (delegation through new path)
# ---------------------------------------------------------------------------


def test_polygon_centroid_delegation():
    """Polygons still work correctly through the new centroid_owned path."""
    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (6, 0), (3, 6)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_polygon_centroid_delegation_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (6, 0), (3, 6)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


def test_polygonal_centroid_cpu_handles_holes_and_multipolygons():
    geoms = [
        Polygon(
            [(0, 0), (8, 0), (8, 8), (0, 8)],
            holes=[[(1, 1), (1, 3), (3, 3), (3, 1)]],
        ),
        MultiPolygon(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(10, 0), (14, 0), (14, 4), (10, 4)]),
            ]
        ),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_polygonal_centroid_gpu_handles_holes_and_multipolygons():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        Polygon(
            [(0, 0), (8, 0), (8, 8), (0, 8)],
            holes=[[(1, 1), (1, 3), (3, 3), (3, 1)]],
        ),
        MultiPolygon(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(10, 0), (14, 0), (14, 4), (10, 4)]),
            ]
        ),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# Mixed geometry centroid
# ---------------------------------------------------------------------------


def test_mixed_geometry_centroid():
    """Array with Points + LineStrings + Polygons computes all centroids."""
    geoms = [
        Point(5, 5),
        LineString([(0, 0), (10, 0)]),
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Point(100, 200),
        LineString([(0, 0), (0, 10)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_mixed_geometry_centroid_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    geoms = [
        Point(5, 5),
        LineString([(0, 0), (10, 0)]),
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Point(100, 200),
        LineString([(0, 0), (0, 10)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# Null / empty geometry handling
# ---------------------------------------------------------------------------


def test_null_centroid():
    """Null geometries produce NaN centroids."""
    geoms = [Point(1, 2), None, Point(3, 4)]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(centroid_owned(owned))
    assert np.isfinite(cx[0])
    assert np.isnan(cx[1])
    assert np.isfinite(cx[2])
    assert np.isfinite(cy[0])
    assert np.isnan(cy[1])
    assert np.isfinite(cy[2])


def test_empty_centroid():
    """Empty geometries produce NaN centroids."""
    geoms = [
        Point(1, 2),
        shapely.from_wkt("LINESTRING EMPTY"),
        Point(3, 4),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(centroid_owned(owned))
    assert np.isfinite(cx[0])
    assert np.isfinite(cx[2])


# ---------------------------------------------------------------------------
# Large-coordinate centering test
# ---------------------------------------------------------------------------


def test_large_coordinates():
    """UTM-scale coordinates (~500,000) exercise coordinate centering."""
    base_x, base_y = 500_000.0, 4_500_000.0
    geoms = [
        Point(base_x + 1, base_y + 2),
        LineString([
            (base_x, base_y),
            (base_x + 100, base_y),
            (base_x + 100, base_y + 50),
        ]),
        Polygon([
            (base_x, base_y),
            (base_x + 10, base_y),
            (base_x + 10, base_y + 10),
            (base_x, base_y + 10),
        ]),
        MultiPoint([
            (base_x + 1, base_y + 1),
            (base_x + 9, base_y + 9),
        ]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _centroid_cpu(owned)
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


@pytest.mark.gpu
def test_large_coordinates_gpu():
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    base_x, base_y = 500_000.0, 4_500_000.0
    geoms = [
        Point(base_x + 1, base_y + 2),
        LineString([
            (base_x, base_y),
            (base_x + 100, base_y),
            (base_x + 100, base_y + 50),
        ]),
        Polygon([
            (base_x, base_y),
            (base_x + 10, base_y),
            (base_x + 10, base_y + 10),
            (base_x, base_y + 10),
        ]),
        MultiPoint([
            (base_x + 1, base_y + 1),
            (base_x + 9, base_y + 9),
        ]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(_centroid_gpu(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


# ---------------------------------------------------------------------------
# Public API auto dispatch
# ---------------------------------------------------------------------------


def test_centroid_owned_auto_dispatch():
    """centroid_owned() auto dispatch works for all types."""
    geoms = [
        Point(5, 5),
        LineString([(0, 0), (10, 0)]),
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
    ]
    owned = from_shapely_geometries(geoms)
    cx, cy = _extract_cx_cy(centroid_owned(owned))
    ex, ey = _shapely_centroids(geoms)
    np.testing.assert_allclose(cx, ex, atol=1e-10)
    np.testing.assert_allclose(cy, ey, atol=1e-10)


def test_centroid_owned_empty_array():
    """centroid_owned() handles zero-length arrays."""
    owned = from_shapely_geometries([])
    cx, cy = _extract_cx_cy(centroid_owned(owned))
    assert len(cx) == 0
    assert len(cy) == 0


# ---------------------------------------------------------------------------
# 10k random polygon verification (Step 1b gate)
# ---------------------------------------------------------------------------


def test_centroid_owned_10k_random_polygons():
    """centroid_owned on 10k random polygons matches Shapely within 1e-10."""
    rng = np.random.default_rng(42)
    polys = []
    for _ in range(10_000):
        cx, cy = rng.uniform(-1000, 1000, size=2)
        r = rng.uniform(1, 100)
        n_verts = rng.integers(3, 8)
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_verts))
        coords = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        polys.append(Polygon(coords))

    owned = from_shapely_geometries(polys)
    result_cx, result_cy = _extract_cx_cy(centroid_owned(owned, precision="fp64"))
    expect_cx, expect_cy = _shapely_centroids(polys)
    np.testing.assert_allclose(result_cx, expect_cx, atol=1e-10)
    np.testing.assert_allclose(result_cy, expect_cy, atol=1e-10)


@pytest.mark.gpu
def test_centroid_owned_device_resident_polygon_input_uses_device_stats() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    source = from_shapely_geometries([Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])])
    device_polys = polygon_buffer_owned_array(source, 0.25, dispatch_mode=ExecutionMode.GPU)

    assert device_polys.device_state is not None
    assert device_polys.families[GeometryFamily.POLYGON].host_materialized is False

    expected_cx, expected_cy = _shapely_centroids(
        [shapely.buffer(Polygon([(0, 0), (3, 0), (3, 2), (0, 2)]), 0.25, quad_segs=8)]
    )
    result_cx, result_cy = _extract_cx_cy(centroid_owned(device_polys, dispatch_mode=ExecutionMode.GPU))

    np.testing.assert_allclose(result_cx, expected_cx, atol=1e-10)
    np.testing.assert_allclose(result_cy, expected_cy, atol=1e-10)
