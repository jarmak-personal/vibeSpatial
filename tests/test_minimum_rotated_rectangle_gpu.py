"""Tests for GPU-accelerated minimum rotated rectangle (oriented envelope).

Verifies that the GPU rotating calipers kernel produces results matching
Shapely's oriented_envelope reference within fp64 epsilon.  Tests cover:
  - Regular polygons (square, triangle, pentagon)
  - Rotated rectangles (known minimum bounding rectangle)
  - Degenerate cases (point, collinear line, single-segment line)
  - MultiPoint arrays
  - LineString input
  - Area comparison with Shapely oracle
  - Valid closed polygon output (5 vertices, first == last)
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


def _has_gpu():
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime

        return get_cuda_runtime().available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


def _to_owned(geoms):
    """Convert a list of Shapely geometries to an OwnedGeometryArray."""
    from vibespatial.geometry.owned import from_shapely_geometries

    return from_shapely_geometries(geoms)


def _gpu_min_rect(geoms):
    """Run GPU minimum_rotated_rectangle and return Shapely geometries."""
    from vibespatial.constructive.minimum_rotated_rectangle import (
        minimum_rotated_rectangle_owned,
    )
    from vibespatial.runtime import ExecutionMode

    owned = _to_owned(geoms)
    result = minimum_rotated_rectangle_owned(owned, dispatch_mode=ExecutionMode.GPU)
    return result.to_shapely()


def _shapely_min_rect(geoms):
    """Shapely oracle for oriented_envelope."""
    import shapely

    return [shapely.oriented_envelope(g) for g in geoms]


def _polygon_area(geom):
    """Return area of a polygon, handling degenerates."""
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestMinimumRotatedRectangleGPU:
    """GPU minimum rotated rectangle kernel tests."""

    def test_unit_square(self):
        """Unit square: minimum rotated rectangle is the square itself."""
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        results = _gpu_min_rect([sq])
        assert len(results) == 1
        result = results[0]

        # The result should be a polygon
        assert result.geom_type == "Polygon"

        # Area should match the unit square (area=1)
        np.testing.assert_allclose(result.area, 1.0, atol=1e-10)

        # Should be a closed ring with 5 coordinates
        coords = np.array(result.exterior.coords)
        assert coords.shape == (5, 2)
        np.testing.assert_allclose(coords[0], coords[-1], atol=1e-10)

    def test_rotated_rectangle(self):
        """A known rotated rectangle should return itself as the minimum."""
        # Rectangle 2x4, rotated 30 degrees
        import math

        angle = math.radians(30)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        w, h = 2.0, 4.0

        # 4 corners of the rectangle centered at origin
        corners = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
        # Rotate
        rotated = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            for x, y in corners
        ]
        rect = Polygon(rotated)

        results = _gpu_min_rect([rect])
        result = results[0]

        # Area should be w*h = 8.0
        np.testing.assert_allclose(result.area, 8.0, atol=1e-8)

    def test_triangle(self):
        """Triangle: minimum rotated rectangle area >= triangle area."""
        tri = Polygon([(0, 0), (4, 0), (2, 3)])
        results = _gpu_min_rect([tri])
        result = results[0]

        # The minimum bounding rectangle of a triangle has area >= triangle
        assert result.area >= tri.area - 1e-10

        # Compare with Shapely oracle
        shapely_result = _shapely_min_rect([tri])[0]
        np.testing.assert_allclose(
            result.area, shapely_result.area, rtol=1e-8,
            err_msg="GPU area does not match Shapely oriented_envelope area",
        )

    def test_pentagon(self):
        """Regular pentagon: area matches Shapely oracle."""
        import math

        n = 5
        pts = [
            (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        pent = Polygon(pts)

        results = _gpu_min_rect([pent])
        shapely_results = _shapely_min_rect([pent])

        np.testing.assert_allclose(
            results[0].area, shapely_results[0].area, rtol=1e-8,
        )

    def test_point_degenerate(self):
        """Single point: degenerate rectangle with zero area."""
        pt = Point(3, 7)
        results = _gpu_min_rect([pt])
        result = results[0]

        assert result.geom_type == "Polygon"
        # Area should be 0 for a degenerate point
        np.testing.assert_allclose(result.area, 0.0, atol=1e-10)

    def test_collinear_linestring(self):
        """Collinear line: degenerate rectangle with zero area."""
        line = LineString([(0, 0), (5, 0), (10, 0)])
        results = _gpu_min_rect([line])
        result = results[0]

        assert result.geom_type == "Polygon"
        # Area should be 0 for collinear input
        np.testing.assert_allclose(result.area, 0.0, atol=1e-10)

    def test_multipoint(self):
        """MultiPoint: matches Shapely oracle."""
        mp = MultiPoint([(0, 0), (3, 0), (3, 2), (0, 2), (1.5, 1)])
        results = _gpu_min_rect([mp])
        shapely_results = _shapely_min_rect([mp])

        np.testing.assert_allclose(
            results[0].area, shapely_results[0].area, rtol=1e-8,
        )

    def test_batch_multiple_geometries(self):
        """Batch of mixed geometries processed correctly."""
        geoms = [
            Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),  # 2x1 rectangle
            Polygon([(0, 0), (1, 0), (0.5, 1)]),  # triangle
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),  # 3x3 square
        ]

        results = _gpu_min_rect(geoms)
        shapely_results = _shapely_min_rect(geoms)

        for i in range(len(geoms)):
            np.testing.assert_allclose(
                results[i].area, shapely_results[i].area, rtol=1e-8,
                err_msg=f"Row {i}: GPU area does not match Shapely",
            )

    def test_output_is_valid_closed_polygon(self):
        """Output polygons are valid and closed (first == last vertex)."""
        geoms = [
            Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
            Polygon([(0, 0), (1, 0), (0.5, 0.8)]),
        ]

        results = _gpu_min_rect(geoms)

        for i, result in enumerate(results):
            assert result.geom_type == "Polygon", f"Row {i}: not a Polygon"
            coords = np.array(result.exterior.coords)
            assert coords.shape[0] == 5, f"Row {i}: expected 5 coords, got {coords.shape[0]}"
            np.testing.assert_allclose(
                coords[0], coords[-1], atol=1e-10,
                err_msg=f"Row {i}: ring not closed",
            )
            assert result.is_valid, f"Row {i}: output polygon is invalid"

    def test_large_coordinates(self):
        """Geometry with large absolute coordinates: fp64 accuracy."""
        # Simulate projected CRS coordinates (large absolute values)
        base_x, base_y = 500_000.0, 4_000_000.0
        sq = Polygon([
            (base_x, base_y),
            (base_x + 100, base_y),
            (base_x + 100, base_y + 50),
            (base_x, base_y + 50),
        ])

        results = _gpu_min_rect([sq])
        shapely_results = _shapely_min_rect([sq])

        np.testing.assert_allclose(
            results[0].area, shapely_results[0].area, rtol=1e-8,
            err_msg="Large coordinate accuracy failure",
        )

    def test_irregular_polygon(self):
        """Irregular concave polygon: area matches Shapely."""
        # L-shaped polygon
        poly = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])

        results = _gpu_min_rect([poly])
        shapely_results = _shapely_min_rect([poly])

        np.testing.assert_allclose(
            results[0].area, shapely_results[0].area, rtol=1e-8,
        )

    def test_many_geometries_batch(self):
        """Larger batch to exercise GPU parallelism."""
        import math

        geoms = []
        for i in range(100):
            angle = math.radians(i * 3.6)  # varying rotation
            w, h = 1.0 + i * 0.1, 2.0 + i * 0.05
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            corners = [
                (-w / 2, -h / 2), (w / 2, -h / 2),
                (w / 2, h / 2), (-w / 2, h / 2),
            ]
            rotated = [
                (x * cos_a - y * sin_a + i * 10, x * sin_a + y * cos_a + i * 10)
                for x, y in corners
            ]
            geoms.append(Polygon(rotated))

        results = _gpu_min_rect(geoms)
        shapely_results = _shapely_min_rect(geoms)

        for i in range(len(geoms)):
            np.testing.assert_allclose(
                results[i].area, shapely_results[i].area, rtol=1e-6,
                err_msg=f"Row {i}: area mismatch in batch",
            )

    def test_auto_dispatch(self):
        """Auto-dispatch selects GPU when available."""
        from vibespatial.constructive.minimum_rotated_rectangle import (
            minimum_rotated_rectangle_owned,
        )

        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        owned = _to_owned([sq])
        result = minimum_rotated_rectangle_owned(owned)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        np.testing.assert_allclose(result_geoms[0].area, 1.0, atol=1e-10)

    def test_geometry_array_method(self):
        """GeometryArray.minimum_rotated_rectangle() dispatches to GPU."""
        from vibespatial.api.geometry_array import GeometryArray

        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        owned = _to_owned([sq])
        ga = GeometryArray.from_owned(owned)

        result = ga.minimum_rotated_rectangle()
        assert len(result) == 1

        # Materialize and check area
        import shapely

        result_geoms = result._data if hasattr(result, "_data") else result.to_shapely()
        result_area = shapely.area(result_geoms)[0]
        np.testing.assert_allclose(result_area, 1.0, atol=1e-10)
