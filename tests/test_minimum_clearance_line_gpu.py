"""Tests for GPU-accelerated minimum_clearance_line kernel.

Validates that the GPU kernel produces 2-point LineStrings whose endpoints
match Shapely's minimum_clearance_line within fp64 epsilon.

Tests cover LineString, Polygon, MultiLineString, MultiPolygon, and
degenerate cases (points, too few segments).
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

from vibespatial import from_shapely_geometries
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.constructive.minimum_clearance import minimum_clearance_line_owned

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime

        return get_cuda_runtime().available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


def _assert_clearance_line_match(gpu_result, cpu_result, rtol=1e-10, atol=1e-12):
    """Assert GPU clearance line endpoints match Shapely reference.

    For each geometry, the clearance line is a 2-point LineString.
    The two endpoints may be in either order (GPU might flip a/b vs Shapely),
    so we compare the sorted coordinates.
    """
    assert len(gpu_result) == len(cpu_result)

    for i in range(len(gpu_result)):
        gpu_geom = gpu_result[i]
        cpu_geom = cpu_result[i]

        if cpu_geom is None or cpu_geom.is_empty:
            # GPU should also produce empty or None
            assert gpu_geom is None or gpu_geom.is_empty, (
                f"Row {i}: expected empty, got {gpu_geom}"
            )
            continue

        assert gpu_geom is not None and not gpu_geom.is_empty, (
            f"Row {i}: expected non-empty, got empty. CPU={cpu_geom}"
        )

        # Extract coords, sort them for order-independent comparison
        gpu_coords = sorted(gpu_geom.coords)
        cpu_coords = sorted(cpu_geom.coords)

        np.testing.assert_allclose(
            gpu_coords,
            cpu_coords,
            rtol=rtol,
            atol=atol,
            err_msg=f"Row {i}: GPU coords {gpu_coords} != CPU coords {cpu_coords}",
        )


# ---------------------------------------------------------------------------
# LineString tests
# ---------------------------------------------------------------------------


@requires_gpu
def test_linestring_simple_zigzag():
    """LineString with a narrow zigzag -- clearance line connects closest parts."""
    ls = LineString([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
    ga = GeometryArray._from_sequence([ls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([ls]))
    _assert_clearance_line_match(result._data, cpu)


@requires_gpu
def test_linestring_closed():
    """Closed linestring (ring-like) -- first/last segments are adjacent."""
    ls = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    ga = GeometryArray._from_sequence([ls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([ls]))
    _assert_clearance_line_match(result._data, cpu)


@requires_gpu
def test_linestring_two_segments():
    """Two segments only -- both are adjacent, should produce empty."""
    ls = LineString([(0, 0), (5, 0), (10, 0)])
    ga = GeometryArray._from_sequence([ls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([ls]))
    # Both should be empty since there are no non-adjacent pairs
    # (2 segments, seg 0 and seg 1 share vertex -> adjacent)
    _assert_clearance_line_match(result._data, cpu)


@requires_gpu
def test_linestring_three_segments():
    """Three segments -- seg 0 and seg 2 are non-adjacent."""
    ls = LineString([(0, 0), (5, 0), (5, 3), (0, 3)])
    ga = GeometryArray._from_sequence([ls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([ls]))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# Polygon tests
# ---------------------------------------------------------------------------


@requires_gpu
def test_polygon_simple_square():
    """Square polygon -- clearance is the side length."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    ga = GeometryArray._from_sequence([poly])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([poly]))
    _assert_clearance_line_match(result._data, cpu)


@requires_gpu
def test_polygon_with_hole():
    """Polygon with hole -- clearance may cross between exterior and hole."""
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    poly = Polygon(exterior, [hole])
    ga = GeometryArray._from_sequence([poly])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([poly]))
    _assert_clearance_line_match(result._data, cpu)


@requires_gpu
def test_polygon_narrow():
    """Narrow polygon -- clearance line crosses the narrow gap."""
    poly = Polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)])
    ga = GeometryArray._from_sequence([poly])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([poly]))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# MultiLineString tests
# ---------------------------------------------------------------------------


@requires_gpu
def test_multilinestring():
    """MultiLineString with two close-passing parts."""
    mls = MultiLineString([
        [(0, 0), (10, 0)],
        [(5, 0.5), (15, 0.5)],
    ])
    ga = GeometryArray._from_sequence([mls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([mls]))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# MultiPolygon tests
# ---------------------------------------------------------------------------


@requires_gpu
def test_multipolygon():
    """MultiPolygon with two close polygons."""
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1.5, 0), (2.5, 0), (2.5, 1), (1.5, 1)])
    mp = MultiPolygon([p1, p2])
    ga = GeometryArray._from_sequence([mp])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([mp]))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


@requires_gpu
def test_point_returns_empty():
    """Point has no segments -- should return empty LineString."""
    pt = Point(1, 2)
    ga = GeometryArray._from_sequence([pt])
    result = ga.minimum_clearance_line()
    assert result._data[0] is None or result._data[0].is_empty


@requires_gpu
def test_single_segment_linestring():
    """Single-segment linestring -- no non-adjacent pairs, empty result."""
    ls = LineString([(0, 0), (1, 1)])
    ga = GeometryArray._from_sequence([ls])
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array([ls]))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# Mixed geometry array
# ---------------------------------------------------------------------------


@requires_gpu
def test_mixed_geometries():
    """Mixed array of different geometry types."""
    geoms = [
        Point(0, 0),
        LineString([(0, 0), (10, 0), (10, 0.5), (0, 0.5)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        MultiLineString([[(0, 0), (10, 0), (10, 1)], [(5, 2), (15, 2)]]),
    ]
    ga = GeometryArray._from_sequence(geoms)
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array(geoms, dtype=object))
    _assert_clearance_line_match(result._data, cpu)


# ---------------------------------------------------------------------------
# Batch test
# ---------------------------------------------------------------------------


@requires_gpu
def test_batch_linestrings():
    """Batch of linestrings -- verify consistent results."""
    rng = np.random.default_rng(42)
    geoms = []
    for _ in range(50):
        n_pts = rng.integers(4, 10)
        coords = rng.uniform(-100, 100, size=(n_pts, 2))
        geoms.append(LineString(coords))

    ga = GeometryArray._from_sequence(geoms)
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array(geoms, dtype=object))
    _assert_clearance_line_match(result._data, cpu, rtol=1e-8, atol=1e-10)


@requires_gpu
def test_batch_polygons():
    """Batch of simple polygons."""
    rng = np.random.default_rng(123)
    geoms = []
    for _ in range(30):
        # Generate a convex polygon via convex hull of random points
        pts = rng.uniform(-50, 50, size=(8, 2))
        from shapely.geometry import MultiPoint

        hull = MultiPoint(pts).convex_hull
        if hull.geom_type == "Polygon":
            geoms.append(hull)
        else:
            # Degenerate (collinear) -- skip
            geoms.append(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    ga = GeometryArray._from_sequence(geoms)
    result = ga.minimum_clearance_line()

    cpu = shapely.minimum_clearance_line(np.array(geoms, dtype=object))
    _assert_clearance_line_match(result._data, cpu, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Endpoint distance validation
# ---------------------------------------------------------------------------


@requires_gpu
def test_clearance_line_length_matches_clearance():
    """The length of the clearance line should equal the minimum clearance."""
    ls = LineString([(0, 0), (10, 0), (10, 2), (0, 2)])
    ga = GeometryArray._from_sequence([ls])

    clearance = ga.minimum_clearance()
    line = ga.minimum_clearance_line()

    line_geom = line._data[0]
    if line_geom is not None and not line_geom.is_empty:
        line_length = line_geom.length
        np.testing.assert_allclose(line_length, clearance[0], rtol=1e-10)


@requires_gpu
def test_minimum_clearance_line_stays_device_resident(strict_device_guard):
    """GPU minimum_clearance_line keeps routing metadata on device."""
    from vibespatial.runtime.residency import Residency

    owned = from_shapely_geometries(
        [LineString([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])],
        residency=Residency.DEVICE,
    )

    result = minimum_clearance_line_owned(owned, dispatch_mode="gpu")

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
