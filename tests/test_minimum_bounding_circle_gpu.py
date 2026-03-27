"""Tests for GPU-accelerated minimum_bounding_circle and minimum_bounding_radius.

Validates that:
  1. Ritter's bounding circle CONTAINS all geometry points (key invariant).
  2. Radius is within ~20% of Shapely (Ritter is approximate; Shapely uses Welzl).
  3. Both circle (Polygon) and radius (float) outputs are correct.
  4. All geometry families are supported: Point, MultiPoint, LineString,
     MultiLineString, Polygon, MultiPolygon.
  5. Degenerate cases are handled (single point, empty).
"""

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

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.minimum_bounding_circle import (
    minimum_bounding_circle_owned,
    minimum_bounding_radius_owned,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_circle_contains_all_points(circle_geom, source_geom, *, tolerance=1e-9):
    """Assert the bounding circle polygon contains all points of the source geometry."""
    if source_geom is None or source_geom.is_empty:
        return

    # Use shapely.get_coordinates to extract all coordinates regardless of type
    source_coords = shapely.get_coordinates(source_geom)
    if len(source_coords) == 0:
        return

    # Get the circle center and radius from the polygon
    circle_centroid = circle_geom.centroid
    cx, cy = circle_centroid.x, circle_centroid.y

    # The circle polygon's vertices are on the circumference;
    # use distance from center to the first vertex as radius
    circle_coords = np.array(circle_geom.exterior.coords)
    radii = np.sqrt((circle_coords[:, 0] - cx) ** 2 + (circle_coords[:, 1] - cy) ** 2)
    circle_radius = radii.max()

    # Check all source points are within the circle (with tolerance)
    dists = np.sqrt((source_coords[:, 0] - cx) ** 2 + (source_coords[:, 1] - cy) ** 2)
    assert np.all(dists <= circle_radius + tolerance), (
        f"Some points are outside the bounding circle. "
        f"Max dist={dists.max():.10f}, radius={circle_radius:.10f}"
    )


def _shapely_radius(geom):
    """Get Shapely's minimum bounding radius for comparison."""
    return shapely.minimum_bounding_radius(geom)


# ---------------------------------------------------------------------------
# Test geometries
# ---------------------------------------------------------------------------

SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
TRIANGLE = Polygon([(0, 0), (4, 0), (2, 3)])
LINE = LineString([(0, 0), (3, 4)])
MULTIPOINT = MultiPoint([(0, 0), (3, 0), (0, 4)])
MULTILINE = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
COMPLEX_POLYGON = Polygon([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)])
MULTIPOLYGON = MultiPolygon([
    ([(0, 0), (1, 0), (1, 1), (0, 1)], []),
    ([(5, 5), (6, 5), (6, 6), (5, 6)], []),
])
SINGLE_POINT = Point(3.5, 7.2)


# ---------------------------------------------------------------------------
# minimum_bounding_radius tests
# ---------------------------------------------------------------------------

@requires_gpu
def test_radius_point():
    """Point geometry should have radius 0."""
    owned = from_shapely_geometries([SINGLE_POINT])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(0.0, abs=1e-12)


@requires_gpu
def test_radius_linestring():
    """LineString bounding radius should contain all points."""
    owned = from_shapely_geometries([LINE])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    # line (0,0)-(3,4), length=5, so radius should be >= 2.5
    assert result[0] >= 2.5 - 1e-9
    # Ritter's approximation should be within 20% of Shapely
    shapely_r = _shapely_radius(LINE)
    assert result[0] <= shapely_r * 1.25, (
        f"Ritter radius {result[0]} is more than 25% larger than Shapely {shapely_r}"
    )


@requires_gpu
def test_radius_polygon():
    """Polygon bounding radius should contain all vertices."""
    owned = from_shapely_geometries([SQUARE])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    # Unit square: exact minimum bounding radius = sqrt(2)/2 ~ 0.7071
    shapely_r = _shapely_radius(SQUARE)
    assert result[0] >= shapely_r - 1e-9, "Radius too small to contain all points"
    assert result[0] <= shapely_r * 1.25


@requires_gpu
def test_radius_multipoint():
    """MultiPoint bounding radius."""
    owned = from_shapely_geometries([MULTIPOINT])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    shapely_r = _shapely_radius(MULTIPOINT)
    assert result[0] >= shapely_r - 1e-9
    assert result[0] <= shapely_r * 1.25


@requires_gpu
def test_radius_multilinestring():
    """MultiLineString bounding radius."""
    owned = from_shapely_geometries([MULTILINE])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    shapely_r = _shapely_radius(MULTILINE)
    assert result[0] >= shapely_r - 1e-9
    assert result[0] <= shapely_r * 1.25


@requires_gpu
def test_radius_multipolygon():
    """MultiPolygon bounding radius."""
    owned = from_shapely_geometries([MULTIPOLYGON])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (1,)
    shapely_r = _shapely_radius(MULTIPOLYGON)
    assert result[0] >= shapely_r - 1e-9
    assert result[0] <= shapely_r * 1.25


@requires_gpu
def test_radius_batch():
    """Multiple geometries in a batch."""
    geoms = [SQUARE, TRIANGLE, LINE, SINGLE_POINT]
    owned = from_shapely_geometries(geoms)
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (len(geoms),)

    for i, geom in enumerate(geoms):
        shapely_r = _shapely_radius(geom)
        assert result[i] >= shapely_r - 1e-9, f"Row {i}: radius too small"
        assert result[i] <= shapely_r * 1.25, f"Row {i}: radius too large"


@requires_gpu
def test_radius_empty():
    """Empty input should return empty array."""
    owned = from_shapely_geometries([])
    result = minimum_bounding_radius_owned(owned)
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# minimum_bounding_circle tests
# ---------------------------------------------------------------------------

@requires_gpu
def test_circle_point():
    """Point: circle should be degenerate (all coords at same location)."""
    owned = from_shapely_geometries([SINGLE_POINT])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    assert len(geoms) == 1
    assert geoms[0].geom_type == "Polygon"
    # For a point, all tessellated coords should be at the same location
    coords = np.array(geoms[0].exterior.coords)
    assert np.allclose(coords[:, 0], SINGLE_POINT.x, atol=1e-10)
    assert np.allclose(coords[:, 1], SINGLE_POINT.y, atol=1e-10)


@requires_gpu
def test_circle_polygon_contains_all_points():
    """Polygon: bounding circle must contain all source vertices."""
    owned = from_shapely_geometries([SQUARE])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    assert len(geoms) == 1
    assert geoms[0].geom_type == "Polygon"
    _assert_circle_contains_all_points(geoms[0], SQUARE)


@requires_gpu
def test_circle_triangle_contains_all_points():
    """Triangle: bounding circle must contain all source vertices."""
    owned = from_shapely_geometries([TRIANGLE])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    _assert_circle_contains_all_points(geoms[0], TRIANGLE)


@requires_gpu
def test_circle_linestring_contains_all_points():
    """LineString: bounding circle must contain all endpoints."""
    owned = from_shapely_geometries([LINE])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    _assert_circle_contains_all_points(geoms[0], LINE)


@requires_gpu
def test_circle_multipoint_contains_all_points():
    """MultiPoint: bounding circle must contain all points."""
    owned = from_shapely_geometries([MULTIPOINT])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    _assert_circle_contains_all_points(geoms[0], MULTIPOINT)


@requires_gpu
def test_circle_multipolygon_contains_all_points():
    """MultiPolygon: bounding circle must contain all vertices."""
    owned = from_shapely_geometries([MULTIPOLYGON])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    _assert_circle_contains_all_points(geoms[0], MULTIPOLYGON)


@requires_gpu
def test_circle_is_valid_polygon():
    """Output polygon should be valid (closed ring, correct winding)."""
    owned = from_shapely_geometries([COMPLEX_POLYGON])
    result_owned = minimum_bounding_circle_owned(owned)
    geoms = result_owned.to_shapely()
    assert geoms[0].is_valid, f"Circle polygon is not valid: {geoms[0]}"
    # Closed ring: first == last coordinate
    coords = np.array(geoms[0].exterior.coords)
    assert np.allclose(coords[0], coords[-1], atol=1e-12)


@requires_gpu
def test_circle_batch():
    """Batch of different geometry types."""
    geoms = [SQUARE, TRIANGLE, LINE, SINGLE_POINT]
    owned = from_shapely_geometries(geoms)
    result_owned = minimum_bounding_circle_owned(owned)
    result_geoms = result_owned.to_shapely()
    assert len(result_geoms) == len(geoms)

    for i, (circle, source) in enumerate(zip(result_geoms, geoms)):
        assert circle.geom_type == "Polygon", f"Row {i}: expected Polygon"
        _assert_circle_contains_all_points(circle, source)


@requires_gpu
def test_circle_empty():
    """Empty input should return empty output."""
    owned = from_shapely_geometries([])
    result_owned = minimum_bounding_circle_owned(owned)
    assert result_owned.row_count == 0


@requires_gpu
def test_radius_containment_guarantee():
    """Key invariant: circle with returned radius always contains all points.

    This is the most important test. Ritter's may overestimate the radius
    but must NEVER underestimate it.
    """
    geoms = [SQUARE, TRIANGLE, LINE, MULTIPOINT, COMPLEX_POLYGON, MULTIPOLYGON]
    for geom in geoms:
        owned = from_shapely_geometries([geom])
        radius = minimum_bounding_radius_owned(owned)

        # Get the circle center from the circle polygon
        circle_owned = minimum_bounding_circle_owned(owned)
        circle_geom = circle_owned.to_shapely()[0]
        centroid = circle_geom.centroid
        cx, cy = centroid.x, centroid.y

        # All source coordinates must be within (cx, cy, radius)
        all_coords = shapely.get_coordinates(geom)
        dists = np.sqrt((all_coords[:, 0] - cx) ** 2 + (all_coords[:, 1] - cy) ** 2)
        assert np.all(dists <= radius[0] + 1e-9), (
            f"Containment violation for {geom.geom_type}: "
            f"max_dist={dists.max():.10f}, radius={radius[0]:.10f}"
        )


# ---------------------------------------------------------------------------
# CPU fallback tests (no @requires_gpu)
# ---------------------------------------------------------------------------

def test_cpu_fallback_radius():
    """CPU path should produce correct results."""
    owned = from_shapely_geometries([SQUARE, LINE])
    # Force CPU by using the CPU function directly
    from vibespatial.constructive.minimum_bounding_circle import (
        _minimum_bounding_radius_cpu,
    )
    result = _minimum_bounding_radius_cpu(owned)
    assert result.shape == (2,)
    # Square: exact min bounding radius = sqrt(2)/2 ~ 0.7071
    shapely_r = _shapely_radius(SQUARE)
    assert result[0] >= shapely_r - 1e-9


def test_cpu_fallback_circle():
    """CPU circle path should produce valid polygons."""
    owned = from_shapely_geometries([SQUARE])
    from vibespatial.constructive.minimum_bounding_circle import (
        _minimum_bounding_circle_cpu,
    )
    result_owned = _minimum_bounding_circle_cpu(owned)
    geoms = result_owned.to_shapely()
    assert len(geoms) == 1
    assert geoms[0].geom_type == "Polygon"
    _assert_circle_contains_all_points(geoms[0], SQUARE)


# ---------------------------------------------------------------------------
# GeoSeries-level integration test
# ---------------------------------------------------------------------------

@requires_gpu
def test_geoseries_minimum_bounding_circle():
    """GeoSeries.minimum_bounding_circle() dispatches to GPU."""
    import vibespatial as gpd

    gs = gpd.GeoSeries([SQUARE, TRIANGLE, LINE])
    result = gs.minimum_bounding_circle()
    assert len(result) == 3
    for i, (circle, source) in enumerate(zip(result, [SQUARE, TRIANGLE, LINE])):
        assert circle.geom_type == "Polygon", f"Row {i}: expected Polygon"
        _assert_circle_contains_all_points(circle, source)


@requires_gpu
def test_geoseries_minimum_bounding_radius():
    """GeoSeries.minimum_bounding_radius() dispatches to GPU."""
    import vibespatial as gpd

    gs = gpd.GeoSeries([SQUARE, TRIANGLE, LINE])
    result = gs.minimum_bounding_radius()
    assert len(result) == 3
    for i, (r, source) in enumerate(zip(result, [SQUARE, TRIANGLE, LINE])):
        shapely_r = _shapely_radius(source)
        assert r >= shapely_r - 1e-9, f"Row {i}: radius too small"
        assert r <= shapely_r * 1.25, f"Row {i}: radius too large"
