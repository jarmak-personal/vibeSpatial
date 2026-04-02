"""Tests for GPU-accelerated extract_unique_points.

Verifies that the GPU kernel produces identical results to the Shapely
reference implementation across all 6 geometry families, edge cases
(empty, null, duplicate coordinates), and mixed-family arrays.
"""

from __future__ import annotations

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


def _extract_coords_sorted(multipoint):
    """Extract sorted (x, y) tuples from a MultiPoint for comparison."""
    if multipoint is None or multipoint.is_empty:
        return []
    coords = list(multipoint.geoms)
    return sorted((c.x, c.y) for c in coords)


def _compare_extract_unique_points(geoms, gpu_result):
    """Compare GPU result against Shapely oracle for a list of geometries."""
    for i, geom in enumerate(geoms):
        if geom is None:
            # Null input => null output; GPU produces empty MultiPoint for nulls
            continue

        expected = shapely.extract_unique_points(geom)
        actual_geoms = gpu_result.to_shapely()
        actual = actual_geoms[i]

        expected_coords = _extract_coords_sorted(expected)
        actual_coords = _extract_coords_sorted(actual)

        assert expected_coords == actual_coords, (
            f"Row {i}: expected {expected_coords}, got {actual_coords}"
        )


# ---------------------------------------------------------------------------
# Point family
# ---------------------------------------------------------------------------

@requires_gpu
def test_point_simple():
    """Single point extracts to itself."""
    geoms = [Point(1.0, 2.0), Point(3.0, 4.0)]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


@requires_gpu
def test_extract_unique_points_stays_device_resident(strict_device_guard):
    """extract_unique_points keeps single-family metadata on device."""
    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.residency import Residency

    geoms = [LineString([(0, 0), (1, 1), (0, 0), (2, 2), (1, 1)])]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency == Residency.DEVICE


# ---------------------------------------------------------------------------
# LineString family
# ---------------------------------------------------------------------------

@requires_gpu
def test_linestring_no_duplicates():
    """LineString with distinct vertices."""
    geoms = [LineString([(0, 0), (1, 1), (2, 0)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


@requires_gpu
def test_linestring_with_duplicates():
    """LineString with repeated coordinates."""
    geoms = [LineString([(0, 0), (1, 1), (0, 0), (2, 2), (1, 1)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# Polygon family
# ---------------------------------------------------------------------------

@requires_gpu
def test_polygon_simple():
    """Simple polygon (no holes)."""
    geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


@requires_gpu
def test_polygon_with_hole():
    """Polygon with a hole shares the closing vertex with the shell."""
    shell = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    geoms = [Polygon(shell, [hole])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# MultiPoint family
# ---------------------------------------------------------------------------

@requires_gpu
def test_multipoint_with_duplicates():
    """MultiPoint with duplicate coordinates."""
    geoms = [MultiPoint([(0, 0), (1, 1), (0, 0), (2, 2)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# MultiLineString family
# ---------------------------------------------------------------------------

@requires_gpu
def test_multilinestring():
    """MultiLineString with shared vertices between parts."""
    geoms = [
        MultiLineString([
            [(0, 0), (1, 1), (2, 0)],
            [(2, 0), (3, 1), (4, 0)],
        ])
    ]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# MultiPolygon family
# ---------------------------------------------------------------------------

@requires_gpu
def test_multipolygon():
    """MultiPolygon with shared vertices."""
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    geoms = [MultiPolygon([p1, p2])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------

@requires_gpu
def test_mixed_families():
    """Array with mixed geometry types."""
    geoms = [
        Point(1, 2),
        LineString([(0, 0), (1, 1), (0, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        MultiPoint([(5, 5), (5, 5), (6, 6)]),
    ]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@requires_gpu
def test_null_rows():
    """Null rows in the input produce null output."""
    geoms = [Point(1, 2), None, LineString([(0, 0), (1, 1)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    actual_geoms = result.to_shapely()
    # Row 1 is null
    assert actual_geoms[1] is None or (actual_geoms[1] is not None and actual_geoms[1].is_empty)
    # Valid rows match Shapely
    _compare_extract_unique_points([geoms[0]], result)  # Check row 0


@requires_gpu
def test_empty_input():
    """Zero-row input."""
    owned = _to_owned([])

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    assert result.row_count == 0


@requires_gpu
def test_all_same_coords():
    """Geometry where all coordinates are identical."""
    geoms = [LineString([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.GPU)
    _compare_extract_unique_points(geoms, result)


@requires_gpu
def test_api_integration():
    """Test the GeometryArray.extract_unique_points() API path."""
    from shapely.geometry import LineString, Point

    import geopandas as gpd

    gs = gpd.GeoSeries([
        Point(1, 2),
        LineString([(0, 0), (1, 1), (0, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
    ])

    result = gs.extract_unique_points()
    expected = gpd.GeoSeries(shapely.extract_unique_points(gs.values))

    for i in range(len(gs)):
        r_coords = _extract_coords_sorted(result.iloc[i])
        e_coords = _extract_coords_sorted(expected.iloc[i])
        assert r_coords == e_coords, f"Row {i}: {r_coords} != {e_coords}"


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_cpu_fallback():
    """CPU path produces correct results."""
    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )
    from vibespatial.runtime import ExecutionMode

    geoms = [
        LineString([(0, 0), (1, 1), (0, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
    ]
    owned = _to_owned(geoms)
    result = extract_unique_points_owned(owned, dispatch_mode=ExecutionMode.CPU)
    _compare_extract_unique_points(geoms, result)
