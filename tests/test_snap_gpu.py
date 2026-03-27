"""Tests for GPU-accelerated snap kernel.

Validates that snap_owned produces results matching Shapely's snap() for all
geometry types.  Covers pairwise and broadcast-right modes, ring closure
preservation, deduplication, and edge projection snapping.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import has_gpu_runtime

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available",
)


def _snap_and_compare(left_geoms, right_geoms, tolerance, *, rtol=1e-10, atol=1e-12):
    """Run GPU snap and compare against Shapely oracle."""
    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.snap import snap_owned

    result = snap_owned(left, right, tolerance, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)

    result_geoms = result.to_shapely()

    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray(right_geoms, dtype=object)
    expected = shapely.snap(left_arr, right_arr, tolerance=tolerance)

    for i in range(len(left_geoms)):
        if left_geoms[i] is None:
            continue
        actual = result_geoms[i]
        exp = expected[i]

        if exp is None or exp.is_empty:
            continue

        assert actual is not None, f"Row {i}: GPU returned None, expected {exp.wkt}"
        assert not actual.is_empty, f"Row {i}: GPU returned empty, expected {exp.wkt}"

        # Compare coordinates
        actual_coords = shapely.get_coordinates(actual)
        expected_coords = shapely.get_coordinates(exp)

        np.testing.assert_allclose(
            actual_coords, expected_coords,
            rtol=rtol, atol=atol,
            err_msg=f"Row {i}: GPU={actual.wkt}, Shapely={exp.wkt}",
        )

    return result


# ---------------------------------------------------------------------------
# Point snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_point_to_point():
    """Snap a point to a nearby point within tolerance."""
    left = [Point(0.0, 0.0)]
    right = [Point(0.1, 0.0)]
    _snap_and_compare(left, right, tolerance=0.5)


@requires_gpu
def test_snap_point_no_snap():
    """Point outside tolerance should remain unchanged."""
    left = [Point(0.0, 0.0)]
    right = [Point(10.0, 10.0)]
    _snap_and_compare(left, right, tolerance=0.5)


@requires_gpu
def test_snap_point_to_linestring():
    """Snap a point to the nearest vertex on a linestring."""
    # Point(0.05, 0) is within 0.5 of vertex (0,0) -> snaps to (0,0)
    left = [Point(0.05, 0.0)]
    right = [LineString([(0, 0), (1, 0)])]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# LineString snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_linestring_to_linestring():
    """Snap linestring vertices to another linestring."""
    left = [LineString([(0, 0), (1, 0.1), (2, 0)])]
    right = [LineString([(0, 0), (1, 0), (2, 0)])]
    _snap_and_compare(left, right, tolerance=0.5)


@requires_gpu
def test_snap_linestring_vertex_to_point():
    """Snap linestring vertices to a point geometry."""
    left = [LineString([(0.0, 0.0), (1.0, 0.1), (2.0, 0.0)])]
    right = [Point(1.0, 0.0)]
    _snap_and_compare(left, right, tolerance=0.5)


@requires_gpu
def test_snap_linestring_no_snap():
    """LineString vertices outside tolerance should remain unchanged."""
    left = [LineString([(0, 0), (1, 0), (2, 0)])]
    right = [LineString([(0, 10), (1, 10), (2, 10)])]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# Polygon snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_polygon_to_polygon():
    """Snap polygon vertices to another polygon."""
    left = [Polygon([(0, 0), (1, 0.1), (1, 1), (0, 1)])]
    right = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    _snap_and_compare(left, right, tolerance=0.5)


@requires_gpu
def test_snap_polygon_ring_closure():
    """Snapping must preserve ring closure for polygons."""
    left = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    right = [Point(0.05, 0.05)]

    from vibespatial.constructive.snap import snap_owned

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)
    result = snap_owned(left_owned, right_owned, tolerance=0.5, dispatch_mode="gpu")

    result_geoms = result.to_shapely()
    for i, g in enumerate(result_geoms):
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon":
            ext = list(g.exterior.coords)
            assert ext[0] == ext[-1], (
                f"Row {i}: ring not closed: first={ext[0]}, last={ext[-1]}"
            )


@requires_gpu
def test_snap_polygon_to_linestring():
    """Snap polygon vertices to a linestring (cross-family)."""
    left = [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    right = [LineString([(0, -0.05), (2, -0.05)])]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# MultiLineString snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_multilinestring_to_point():
    """Snap multilinestring vertices to a point."""
    left = [MultiLineString([[(0, 0), (1, 0.1)], [(2, 0), (3, 0.1)]])]
    right = [Point(1, 0)]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# MultiPolygon snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_multipolygon_to_point():
    """Snap multipolygon vertices to a point."""
    left = [MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
    ])]
    right = [Point(0.05, 0.05)]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# Pairwise mode (multiple rows)
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_pairwise_multiple_rows():
    """Pairwise snap with multiple geometry rows."""
    left = [
        Point(0.1, 0.0),
        LineString([(0, 0), (1, 0.1), (2, 0)]),
        Polygon([(0, 0), (1, 0.05), (1, 1), (0, 1)]),
    ]
    right = [
        Point(0, 0),
        LineString([(0, 0), (1, 0), (2, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ]
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# Broadcast mode (N vs 1)
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_broadcast_right():
    """Broadcast-right mode: multiple geometries snapped to one target."""
    left = [
        Point(0.1, 0.0),
        Point(0.2, 0.0),
        Point(0.3, 0.0),
    ]
    right = [Point(0, 0)]

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)

    from vibespatial.constructive.snap import snap_owned

    result = snap_owned(left_owned, right_owned, tolerance=0.5, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    assert result.row_count == 3

    result_geoms = result.to_shapely()
    # All points within tolerance should snap to (0, 0)
    for i, g in enumerate(result_geoms):
        assert g is not None
        coords = shapely.get_coordinates(g)
        np.testing.assert_allclose(
            coords, [[0.0, 0.0]],
            atol=1e-12,
            err_msg=f"Row {i}: expected (0,0), got {coords}",
        )


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_null_left():
    """Null left geometry produces null result."""
    left = [None, Point(0.1, 0)]
    right = [Point(0, 0), Point(0, 0)]

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)

    from vibespatial.constructive.snap import snap_owned

    result = snap_owned(left_owned, right_owned, tolerance=0.5, dispatch_mode="gpu")
    # Row 0: null -> null (validity should be False)
    assert not result.validity[0]


@requires_gpu
def test_snap_null_right():
    """Null right geometry keeps left unchanged."""
    left = [Point(0.1, 0)]
    right = [None]

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)

    from vibespatial.constructive.snap import snap_owned

    result = snap_owned(left_owned, right_owned, tolerance=0.5, dispatch_mode="gpu")
    result_geoms = result.to_shapely()
    # Left should be unchanged since right is null
    coords = shapely.get_coordinates(result_geoms[0])
    np.testing.assert_allclose(
        coords, [[0.1, 0.0]], atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Edge projection snapping (snap to segment interior, not just vertices)
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_to_vertex_not_edge():
    """Snap matches GEOS semantics: snap to vertices of B, not edge projections."""
    # Point at (0.5, 0.05) is near the edge (0,0)-(1,0) but far from both vertices
    # (distance to (0,0) = ~0.502, distance to (1,0) = ~0.502).
    # With tolerance=0.1, it should NOT snap (no vertex within tolerance).
    left = [Point(0.5, 0.05)]
    right = [LineString([(0, 0), (1, 0)])]
    _snap_and_compare(left, right, tolerance=0.1)


# ---------------------------------------------------------------------------
# Deduplication after snapping
# ---------------------------------------------------------------------------

@requires_gpu
def test_snap_dedup_coincident_vertices():
    """Snapping should deduplicate coincident vertices."""
    # Two close vertices both snap to the same point
    left = [LineString([(0, 0), (0.01, 0), (0.02, 0), (1, 0)])]
    right = [Point(0, 0)]
    # After snapping, (0,0), (0.01,0), (0.02,0) all snap to (0,0)
    # Dedup should remove the duplicates
    _snap_and_compare(left, right, tolerance=0.5)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_snap_cpu_fallback():
    """Snap works via CPU fallback when GPU is not requested."""
    left = [Point(0.1, 0)]
    right = [Point(0, 0)]

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)

    from vibespatial.constructive.snap import snap_owned

    result = snap_owned(left_owned, right_owned, tolerance=0.5, dispatch_mode="cpu")
    result_geoms = result.to_shapely()
    coords = shapely.get_coordinates(result_geoms[0])
    np.testing.assert_allclose(coords, [[0.0, 0.0]], atol=1e-12)
