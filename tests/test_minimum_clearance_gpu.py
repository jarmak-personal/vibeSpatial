"""Tests for GPU-accelerated minimum_clearance.

Validates that minimum_clearance produces results matching Shapely within
fp64 epsilon for all geometry families.  Zero D2H transfers during
computation (only final result transfer).
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
from vibespatial.constructive.minimum_clearance import minimum_clearance_owned

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU runtime not available"
)


def _assert_clearance_matches_shapely(
    geoms: list,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-14,
):
    """Assert GPU minimum_clearance matches Shapely oracle."""
    owned = from_shapely_geometries(geoms)
    gpu_result = minimum_clearance_owned(owned)

    shapely_arr = np.asarray(geoms, dtype=object)
    cpu_result = shapely.minimum_clearance(shapely_arr)

    for i, (gpu_val, cpu_val) in enumerate(zip(gpu_result, cpu_result)):
        if np.isinf(cpu_val) and np.isinf(gpu_val):
            continue  # both infinity is correct
        if np.isnan(cpu_val) and np.isnan(gpu_val):
            continue  # both NaN for null geometries
        np.testing.assert_allclose(
            gpu_val,
            cpu_val,
            rtol=rtol,
            atol=atol,
            err_msg=f"Row {i}: GPU clearance {gpu_val} != Shapely {cpu_val} for {geoms[i]}",
        )


# ---------------------------------------------------------------------------
# Point family -- clearance is infinity
# ---------------------------------------------------------------------------


@requires_gpu
def test_point_clearance_is_infinity():
    """Single points have no segments, clearance is infinity."""
    geoms = [Point(0, 0), Point(1, 2), Point(3, 4)]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_multipoint_clearance_is_infinity():
    """MultiPoints have no segments/edges in the GIS clearance sense."""
    geoms = [
        MultiPoint([(0, 0), (1, 1)]),
        MultiPoint([(2, 3), (4, 5), (6, 7)]),
    ]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# LineString family
# ---------------------------------------------------------------------------


@requires_gpu
def test_linestring_two_vertices():
    """LineString with only 2 vertices -> 1 segment -> no non-adjacent pairs -> infinity."""
    geoms = [LineString([(0, 0), (1, 1)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_linestring_three_vertices():
    """LineString with 3 vertices -> 2 segments that are adjacent -> infinity.
    Wait, segments 0 and 1 share vertex 1, so they ARE adjacent.
    Only 2 segments total, and they are adjacent -> still need to check.
    Actually with j >= i+2, we get j=2 but nsegs=2 so j<2 is false.
    So this should be infinity."""
    geoms = [LineString([(0, 0), (1, 0), (2, 0)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_linestring_four_vertices_collinear():
    """LineString with 4 collinear vertices: seg 0 and seg 2 are non-adjacent.
    Seg 0 = (0,0)-(1,0), Seg 2 = (2,0)-(3,0).
    Distance between them = 1.0 (distance from (1,0) to (2,0) on the projection)."""
    geoms = [LineString([(0, 0), (1, 0), (2, 0), (3, 0)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_linestring_narrow_u_shape():
    """U-shaped linestring where opposite sides are close together."""
    geoms = [LineString([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_linestring_zigzag():
    """Zigzag pattern where non-adjacent segments pass close."""
    geoms = [
        LineString([
            (0, 0), (4, 0), (1, 0.1), (5, 0.1),
        ])
    ]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_linestring_closed_ring():
    """Closed linestring (ring): first and last segments share closure vertex."""
    geoms = [LineString([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# Polygon family
# ---------------------------------------------------------------------------


@requires_gpu
def test_polygon_square():
    """Square polygon: clearance is the side length (4) since non-adjacent
    edges are the opposite sides."""
    geoms = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_polygon_narrow_rectangle():
    """Narrow rectangle: clearance should be the narrow dimension."""
    geoms = [Polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_polygon_triangle():
    """Triangle: 3 edges, each pair shares a vertex except... actually,
    in a triangle all edges share vertices with each other (each edge is
    adjacent to both others).  So triangle clearance should be infinity
    or the altitude depending on Shapely's definition.

    Actually, a triangle has 3 segments: (0,1), (1,2), (2,0).
    After stripping closure vertex: 3 unique vertices, 3 segments.
    Seg 0 is adjacent to seg 1 (share vertex 1).
    Seg 1 is adjacent to seg 2 (share vertex 2).
    Seg 0 and seg 2: seg 0 uses vertices (0,1), seg 2 uses vertices (2,0).
    They share vertex 0 (the closure vertex). So they are adjacent.
    All pairs are adjacent -> clearance = infinity."""
    geoms = [Polygon([(0, 0), (4, 0), (2, 3)])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_polygon_with_hole():
    """Polygon with a hole: cross-ring segment pairs are never adjacent."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(2, 2), (8, 2), (8, 8), (2, 8)]
    geoms = [Polygon(outer, [inner])]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_polygon_concave_l_shape():
    """L-shaped concave polygon with tight clearance."""
    geoms = [Polygon([
        (0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4),
    ])]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# MultiPolygon family
# ---------------------------------------------------------------------------


@requires_gpu
def test_multipolygon_two_squares():
    """Two squares: clearance includes cross-polygon segment pairs."""
    geoms = [
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
        ])
    ]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_multipolygon_close_polygons():
    """Two polygons very close together: clearance is the gap."""
    geoms = [
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1.01, 0), (2, 0), (2, 1), (1.01, 1)]),
        ])
    ]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# MultiLineString family
# ---------------------------------------------------------------------------


@requires_gpu
def test_multilinestring_two_lines():
    """Two linestrings: cross-part segment pairs are never adjacent."""
    geoms = [
        MultiLineString([
            [(0, 0), (10, 0)],
            [(0, 0.5), (10, 0.5)],
        ])
    ]
    _assert_clearance_matches_shapely(geoms)


@requires_gpu
def test_multilinestring_single_short_line():
    """Single short linestring in a multi: should match LineString behavior."""
    geoms = [MultiLineString([[(0, 0), (1, 0)]])]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# Mixed family batch
# ---------------------------------------------------------------------------


@requires_gpu
def test_mixed_families():
    """Mixed geometry types in a single array."""
    geoms = [
        Point(0, 0),
        LineString([(0, 0), (10, 0), (10, 0.5), (0, 0.5)]),
        Polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)]),
        MultiPoint([(0, 0), (1, 1)]),
        None,  # null geometry
    ]
    owned = from_shapely_geometries(geoms)
    gpu_result = minimum_clearance_owned(owned)

    shapely_arr = np.asarray(geoms, dtype=object)
    cpu_result = shapely.minimum_clearance(shapely_arr)

    for i in range(len(geoms)):
        if geoms[i] is None:
            assert np.isnan(gpu_result[i]), f"Row {i}: expected NaN for null, got {gpu_result[i]}"
            continue
        if np.isinf(cpu_result[i]) and np.isinf(gpu_result[i]):
            continue
        np.testing.assert_allclose(
            gpu_result[i],
            cpu_result[i],
            rtol=1e-10,
            atol=1e-14,
            err_msg=f"Row {i}: GPU {gpu_result[i]} != Shapely {cpu_result[i]}",
        )


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


@requires_gpu
def test_empty_array():
    """Empty geometry array returns empty result."""
    owned = from_shapely_geometries([])
    result = minimum_clearance_owned(owned)
    assert len(result) == 0


@requires_gpu
def test_single_vertex_linestring():
    """Degenerate linestring with single vertex (if possible)."""
    # LineString requires at least 2 points; test with 2-point case
    geoms = [LineString([(0, 0), (1, 1)])]
    _assert_clearance_matches_shapely(geoms)


# ---------------------------------------------------------------------------
# Precision / larger geometries
# ---------------------------------------------------------------------------


@requires_gpu
def test_polygon_many_vertices():
    """Polygon with many vertices (stress test for O(n^2) kernel)."""
    import math

    n = 50
    coords = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    geoms = [Polygon(coords)]
    _assert_clearance_matches_shapely(geoms, rtol=1e-8)


@requires_gpu
def test_large_coordinate_values():
    """Geometries with large absolute coordinate values (UTM-like)."""
    # UTM-scale coordinates
    base_x, base_y = 500000.0, 4500000.0
    geoms = [
        Polygon([
            (base_x, base_y),
            (base_x + 10, base_y),
            (base_x + 10, base_y + 0.1),
            (base_x, base_y + 0.1),
        ])
    ]
    _assert_clearance_matches_shapely(geoms, rtol=1e-8)


@requires_gpu
def test_batch_polygons():
    """Batch of multiple polygons to verify per-geometry independence."""
    geoms = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),  # square
        Polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)]),  # narrow rect
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # unit square
    ]
    _assert_clearance_matches_shapely(geoms)
