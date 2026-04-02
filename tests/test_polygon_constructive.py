from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from vibespatial.constructive.polygon import (
    _polygon_centroids_cpu,
    _polygon_centroids_gpu,
    polygon_buffer_owned_array,
    polygon_centroids_owned,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency


def _buffer_matches_shapely(gpu_geom, shapely_geom, *, tol: float = 5e-4) -> bool:
    """Check geometric equivalence using area-based comparison."""
    if gpu_geom is None or shapely_geom is None:
        return gpu_geom is shapely_geom
    sym_diff = gpu_geom.symmetric_difference(shapely_geom)
    return sym_diff.area < tol * max(gpu_geom.area, shapely_geom.area, 1e-12)


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_round_join() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_triangle_round_join() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (5, 8)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_convex_polygon_many_vertices() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # Regular hexagon
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    coords = list(zip(5 * np.cos(angles), 5 * np.sin(angles)))
    poly = Polygon(coords)
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 1.0, quad_segs=4)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_multiple_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(0, 0), (10, 0), (5, 8)]),
    ]
    polys = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, 2.0, 0.5])
    gpu = polygon_buffer_owned_array(polys, radii, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4)
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("join_style", ["round", "mitre", "bevel"])
def test_polygon_buffer_gpu_join_styles(join_style: str) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, 1.0, quad_segs=4, join_style=join_style,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(poly, 1.0, quad_segs=4, join_style=join_style)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"join={join_style}: area diff={result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_mitre_limit_exceeded() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, 1.0, quad_segs=4,
        join_style="mitre", mitre_limit=1.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    # mitre_limit=1.0 < sqrt(2) for right angle → bevel fallback
    expected = shapely.buffer(poly, 1.0, quad_segs=4, join_style="bevel")
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected, tol=0.02), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_quad_segs_1() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 1.0, quad_segs=1)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    assert gpu.residency is Residency.DEVICE
    assert gpu.device_state is not None
    from vibespatial.geometry.buffers import GeometryFamily
    assert gpu.families[GeometryFamily.POLYGON].host_materialized is False

    # Materialization on to_shapely
    _ = gpu.to_shapely()
    assert gpu.families[GeometryFamily.POLYGON].host_materialized is True


@pytest.mark.gpu
def test_polygon_centroids_device_resident_input_uses_device_stats() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    source_poly = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    source = from_shapely_geometries([source_poly])
    device_polys = polygon_buffer_owned_array(source, 0.25, dispatch_mode=ExecutionMode.GPU)

    assert device_polys.device_state is not None
    assert device_polys.families[GeometryFamily.POLYGON].host_materialized is False

    expected = shapely.buffer(source_poly, 0.25, quad_segs=8).centroid
    cx, cy = polygon_centroids_owned(device_polys, dispatch_mode=ExecutionMode.GPU)

    np.testing.assert_allclose(cx, np.asarray([expected.x]), atol=1e-10)
    np.testing.assert_allclose(cy, np.asarray([expected.y]), atol=1e-10)


@pytest.mark.gpu
def test_polygon_buffer_gpu_bevel_multiple_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
    ]
    polys = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, 2.0])
    gpu = polygon_buffer_owned_array(
        polys, radii, quad_segs=4, join_style="bevel",
        dispatch_mode=ExecutionMode.GPU,
    )

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4, join_style="bevel")
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )


# --- Phase 4.5: Negative buffer (inward shrinking) ---


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_negative_round() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, -1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, -1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_negative_mitre() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, -1.0, quad_segs=4, join_style="mitre",
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(poly, -1.0, quad_segs=4, join_style="mitre")
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_negative_bevel() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, -1.0, quad_segs=4, join_style="bevel",
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(poly, -1.0, quad_segs=4, join_style="bevel")
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_triangle_negative_round() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (5, 8)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, -0.5, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, -0.5, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_mixed_positive_negative() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(0, 0), (10, 0), (5, 8)]),
    ]
    polys = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, -1.0, 2.0, -0.5])
    gpu = polygon_buffer_owned_array(polys, radii, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4)
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )


@pytest.mark.gpu
def test_polygon_buffer_gpu_concave_negative_round() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # L-shaped concave polygon
    poly = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, -0.5, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, -0.5, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_large_negative_no_crash() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # d=-6 exceeds half-width of 10x10 square — polygon collapses
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    # Should not crash; result may be degenerate
    gpu = polygon_buffer_owned_array(polys, -6.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)
    result = gpu.to_shapely()
    assert len(result) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("join_style", ["round", "mitre", "bevel"])
def test_polygon_buffer_gpu_negative_join_styles(join_style: str) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, -1.0, quad_segs=4, join_style=join_style,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(poly, -1.0, quad_segs=4, join_style=join_style)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"join={join_style}: area diff={result.symmetric_difference(expected).area}"
    )


# --- Phase 4.75: Polygon buffer with holes ---


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_with_hole_positive() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
    poly = Polygon(exterior, holes=[hole])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_square_with_hole_negative() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
    poly = Polygon(exterior, holes=[hole])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, -1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, -1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_polygon_with_multiple_holes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]
    hole1 = [(2, 2), (8, 2), (8, 8), (2, 8)]
    hole2 = [(12, 12), (18, 12), (18, 18), (12, 18)]
    hole3 = [(22, 2), (28, 2), (28, 8), (22, 8)]
    poly = Polygon(exterior, holes=[hole1, hole2, hole3])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(polys, 0.5, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(poly, 0.5, quad_segs=4)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_polygon_buffer_gpu_hole_collapse_large_distance() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # Small hole (4x4) with large positive buffer (d=3) should collapse the hole
    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
    poly = Polygon(exterior, holes=[hole])
    polys = from_shapely_geometries([poly])
    # Should not crash; hole may collapse
    gpu = polygon_buffer_owned_array(polys, 3.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)
    result = gpu.to_shapely()
    assert len(result) == 1


@pytest.mark.gpu
def test_polygon_buffer_gpu_mixed_holes_and_simple() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # simple
        Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)],
            holes=[[(5, 5), (15, 5), (15, 15), (5, 15)]],
        ),  # with hole
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),  # simple
    ]
    polys = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, 0.5, -0.5])
    gpu = polygon_buffer_owned_array(polys, radii, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4)
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("join_style", ["round", "mitre", "bevel"])
def test_polygon_buffer_gpu_hole_join_styles(join_style: str) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
    poly = Polygon(exterior, holes=[hole])
    polys = from_shapely_geometries([poly])
    gpu = polygon_buffer_owned_array(
        polys, 1.0, quad_segs=4, join_style=join_style,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(poly, 1.0, quad_segs=4, join_style=join_style)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"join={join_style}: area diff={result.symmetric_difference(expected).area}"
    )


# ---------------------------------------------------------------------------
# Polygon centroid tests (GPU shoelace kernel)
# ---------------------------------------------------------------------------


def _shapely_centroids(polys):
    """Oracle: Shapely centroids for comparison."""
    cx = np.array([p.centroid.x for p in polys])
    cy = np.array([p.centroid.y for p in polys])
    return cx, cy


def test_polygon_centroids_cpu_square() -> None:
    """CPU centroid of axis-aligned square."""
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
    owned = from_shapely_geometries(polys)
    cx, cy = _polygon_centroids_cpu(owned)
    np.testing.assert_allclose(cx, [5.0], rtol=1e-10)
    np.testing.assert_allclose(cy, [5.0], rtol=1e-10)


def test_polygon_centroids_cpu_triangle() -> None:
    """CPU centroid of triangle matches Shapely oracle."""
    polys = [Polygon([(0, 0), (6, 0), (3, 6)])]
    owned = from_shapely_geometries(polys)
    cx, cy = _polygon_centroids_cpu(owned)
    expected_cx, expected_cy = _shapely_centroids(polys)
    np.testing.assert_allclose(cx, expected_cx, rtol=1e-10)
    np.testing.assert_allclose(cy, expected_cy, rtol=1e-10)


def test_polygon_centroids_cpu_batch() -> None:
    """CPU centroid of mixed polygons matches Shapely oracle."""
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (6, 0), (3, 6)]),
        Polygon([(1, 1), (5, 1), (5, 5), (1, 5)]),
        Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),
    ]
    owned = from_shapely_geometries(polys)
    cx, cy = _polygon_centroids_cpu(owned)
    expected_cx, expected_cy = _shapely_centroids(polys)
    np.testing.assert_allclose(cx, expected_cx, rtol=1e-10)
    np.testing.assert_allclose(cy, expected_cy, rtol=1e-10)


@pytest.mark.gpu
def test_polygon_centroids_gpu_matches_shapely_oracle() -> None:
    """GPU centroid kernel produces same results as Shapely oracle."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    rng = np.random.default_rng(42)
    polys = []
    for _ in range(1000):
        x, y = rng.uniform(0, 100, 2)
        w, h = rng.uniform(0.1, 10, 2)
        polys.append(shapely.box(x, y, x + w, y + h))
    # Add some non-trivial polygons (triangles, pentagons)
    for _ in range(200):
        cx, cy = rng.uniform(0, 100, 2)
        angles = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(3, 8)))
        r = rng.uniform(1, 10)
        coords = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        coords.append(coords[0])
        polys.append(Polygon(coords))

    owned = from_shapely_geometries(polys)
    gpu_cx, gpu_cy = _polygon_centroids_gpu(owned)
    expected_cx, expected_cy = _shapely_centroids(polys)
    np.testing.assert_allclose(gpu_cx, expected_cx, rtol=1e-8)
    np.testing.assert_allclose(gpu_cy, expected_cy, rtol=1e-8)


@pytest.mark.gpu
def test_polygon_centroids_gpu_matches_cpu() -> None:
    """GPU and CPU centroid paths produce identical results."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (6, 0), (3, 6)]),
        Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
    ]
    owned = from_shapely_geometries(polys)
    cpu_cx, cpu_cy = _polygon_centroids_cpu(owned)
    gpu_cx, gpu_cy = _polygon_centroids_gpu(owned)
    np.testing.assert_allclose(gpu_cx, cpu_cx, rtol=1e-12)
    np.testing.assert_allclose(gpu_cy, cpu_cy, rtol=1e-12)


def test_polygon_centroids_auto_dispatch() -> None:
    """Auto dispatch falls back to CPU gracefully."""
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
    owned = from_shapely_geometries(polys)
    cx, cy = polygon_centroids_owned(owned)
    assert cx is not None
    assert cy is not None
    np.testing.assert_allclose(cx, [5.0], rtol=1e-10)
    np.testing.assert_allclose(cy, [5.0], rtol=1e-10)
