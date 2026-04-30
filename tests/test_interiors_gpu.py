"""Tests for GPU-accelerated interior ring extraction.

Verifies that the GPU kernel produces identical results to the Shapely
reference implementation for polygons with various hole counts, edge cases
(empty polygons, polygons with no holes, null rows), non-polygon types,
and mixed-family arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import (
    LinearRing,
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


def _compare_interiors(geoms, gpu_owned):
    """Compare GPU result against Shapely oracle for a list of geometries.

    The GPU result is a MultiLineString OGA where each row's parts
    are the interior rings. We compare against shapely's .interiors
    property.
    """
    gpu_geoms = gpu_owned.to_shapely()
    for i, geom in enumerate(geoms):
        if geom is None:
            assert gpu_geoms[i] is None, (
                f"Row {i}: expected None for null input, got {gpu_geoms[i]}"
            )
            continue

        if not hasattr(geom, "interiors"):
            # Non-polygon: should be None
            assert gpu_geoms[i] is None, (
                f"Row {i}: expected None for non-polygon, got {gpu_geoms[i]}"
            )
            continue

        expected_rings = list(geom.interiors)
        actual = gpu_geoms[i]

        if len(expected_rings) == 0:
            # Should be an empty MultiLineString
            assert actual is not None, f"Row {i}: expected empty MultiLineString, got None"
            if hasattr(actual, "geoms"):
                assert len(list(actual.geoms)) == 0, (
                    f"Row {i}: expected 0 parts, got {len(list(actual.geoms))}"
                )
            else:
                # Empty geometry
                assert actual.is_empty, f"Row {i}: expected empty, got {actual}"
            continue

        # Non-empty interior rings
        assert actual is not None, f"Row {i}: expected MultiLineString, got None"
        actual_parts = list(actual.geoms)
        assert len(actual_parts) == len(expected_rings), (
            f"Row {i}: expected {len(expected_rings)} interior rings, "
            f"got {len(actual_parts)}"
        )

        for j, (expected_ring, actual_part) in enumerate(
            zip(expected_rings, actual_parts)
        ):
            expected_coords = list(expected_ring.coords)
            actual_coords = list(actual_part.coords)
            assert len(actual_coords) == len(expected_coords), (
                f"Row {i}, ring {j}: expected {len(expected_coords)} coords, "
                f"got {len(actual_coords)}"
            )
            for k, (ec, ac) in enumerate(zip(expected_coords, actual_coords)):
                assert abs(ec[0] - ac[0]) < 1e-12 and abs(ec[1] - ac[1]) < 1e-12, (
                    f"Row {i}, ring {j}, coord {k}: "
                    f"expected ({ec[0]}, {ec[1]}), got ({ac[0]}, {ac[1]})"
                )


# ---------------------------------------------------------------------------
# Polygon with holes
# ---------------------------------------------------------------------------

@requires_gpu
def test_polygon_single_hole():
    """Polygon with a single interior ring."""
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_polygon_multiple_holes():
    """Polygon with multiple interior rings."""
    poly = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
        [
            [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)],
            [(5, 5), (5, 8), (8, 8), (8, 5), (5, 5)],
            [(10, 10), (10, 15), (15, 15), (15, 10), (10, 10)],
        ],
    )
    geoms = [poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_polygon_no_holes():
    """Polygon without interior rings should produce empty MultiLineString."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    geoms = [poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_mixed_holes_and_no_holes():
    """Mix of polygons with and without holes."""
    poly_with_holes = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    poly_without_holes = Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])
    poly_with_two = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
        [
            [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)],
            [(5, 5), (5, 8), (8, 8), (8, 5), (5, 5)],
        ],
    )
    geoms = [poly_with_holes, poly_without_holes, poly_with_two]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_interiors_gpu_metadata_stays_device_resident():
    """GPU interiors keeps routing metadata on device."""
    from vibespatial.constructive.interiors import interiors_owned
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    poly = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]],
    )

    owned = from_shapely_geometries([poly], residency=Residency.DEVICE)
    with assert_no_transfers():
        result = interiors_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency == Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@requires_gpu
def test_interiors_gpu_device_only_metadata_does_not_materialize_for_dispatch():
    """GPU interiors does not host-materialize routing metadata before dispatch."""
    from vibespatial.constructive.interiors import interiors_owned
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.execution_trace import assert_no_transfers
    from vibespatial.runtime.residency import Residency

    poly = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]],
    )

    owned = from_shapely_geometries([poly], residency=Residency.DEVICE)
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    with assert_no_transfers():
        result = interiors_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency == Residency.DEVICE
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None


# ---------------------------------------------------------------------------
# Null and empty handling
# ---------------------------------------------------------------------------

@requires_gpu
def test_null_rows():
    """Null geometries should produce null output."""
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [None, poly, None]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_all_null():
    """All null input should produce all null output."""
    geoms = [None, None, None]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    result_geoms = result.to_shapely()
    for g in result_geoms:
        assert g is None


@requires_gpu
def test_empty_polygon():
    """Empty polygon should produce empty MultiLineString."""
    empty_poly = Polygon()
    normal_poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [empty_poly, normal_poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_empty_input():
    """Empty input should produce empty output."""
    geoms = []
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Non-polygon types
# ---------------------------------------------------------------------------

@requires_gpu
def test_point_returns_null():
    """Point geometry should produce null output."""
    geoms = [Point(1, 2)]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    result_geoms = result.to_shapely()
    assert result_geoms[0] is None


@requires_gpu
def test_linestring_returns_null():
    """LineString should produce null output."""
    geoms = [LineString([(0, 0), (1, 1), (2, 0)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    result_geoms = result.to_shapely()
    assert result_geoms[0] is None


@requires_gpu
def test_multipoint_returns_null():
    """MultiPoint should produce null output."""
    geoms = [MultiPoint([(0, 0), (1, 1)])]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    result_geoms = result.to_shapely()
    assert result_geoms[0] is None


# ---------------------------------------------------------------------------
# Mixed-family arrays
# ---------------------------------------------------------------------------

@requires_gpu
def test_mixed_polygon_and_point():
    """Mixed array: polygons + points. Points get null, polygons get interiors."""
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [Point(1, 2), poly, Point(3, 4)]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


@requires_gpu
def test_mixed_polygon_linestring_null():
    """Mixed array with polygon, linestring, and null."""
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [
            [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)],
            [(5, 5), (5, 8), (8, 8), (8, 5), (5, 5)],
        ],
    )
    line = LineString([(0, 0), (5, 5)])
    geoms = [poly, None, line, poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


# ---------------------------------------------------------------------------
# GeometryArray.interiors property integration
# ---------------------------------------------------------------------------

@requires_gpu
def test_geometry_array_interiors_property():
    """Test that GeometryArray.interiors returns correct numpy array."""
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.geometry.owned import from_shapely_geometries

    poly_with_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    poly_no_hole = Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])

    owned = from_shapely_geometries([poly_with_hole, poly_no_hole])
    ga = GeometryArray.from_owned(owned)

    result = ga.interiors

    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    assert len(result) == 2

    # First polygon has 1 interior ring
    assert len(result[0]) == 1
    ring = result[0][0]
    assert isinstance(ring, LinearRing)
    expected_coords = [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]
    actual_coords = list(ring.coords)
    for (ex, ey), (ax, ay) in zip(expected_coords, actual_coords):
        assert abs(ex - ax) < 1e-12 and abs(ey - ay) < 1e-12

    # Second polygon has no interior rings
    assert result[1] == [] or len(result[1]) == 0


@requires_gpu
def test_geometry_array_interiors_with_non_polygon():
    """Test GeometryArray.interiors with non-polygon types issues warning."""
    import warnings

    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.geometry.owned import from_shapely_geometries

    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [poly, Point(1, 2)]
    owned = from_shapely_geometries(geoms)
    ga = GeometryArray.from_owned(owned)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ga.interiors
        assert len(w) == 1
        assert "Only Polygon objects" in str(w[0].message)

    assert result[1] is None


# ---------------------------------------------------------------------------
# Coordinate exactness
# ---------------------------------------------------------------------------

@requires_gpu
def test_coordinate_exactness():
    """Verify interior ring coordinates are exact subsets of input."""

    # Use coordinates that test floating point precision
    hole_coords = [
        (1.123456789012345, 2.987654321098765),
        (3.111111111111111, 4.222222222222222),
        (5.333333333333333, 2.444444444444444),
        (1.123456789012345, 2.987654321098765),
    ]
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [hole_coords],
    )
    geoms = [poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    result_geoms = result.to_shapely()
    actual = result_geoms[0]
    actual_parts = list(actual.geoms)
    assert len(actual_parts) == 1

    actual_coords = list(actual_parts[0].coords)
    for (ex, ey), (ax, ay) in zip(hole_coords, actual_coords):
        # Exact equality — coordinates should be bit-identical since
        # we're doing pure index-based extraction.
        assert ex == ax, f"Expected x={ex}, got x={ax}"
        assert ey == ay, f"Expected y={ey}, got y={ay}"


# ---------------------------------------------------------------------------
# Large polygon with many holes
# ---------------------------------------------------------------------------

@requires_gpu
def test_many_holes():
    """Polygon with many interior rings to stress the offset arithmetic."""

    exterior = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    holes = []
    for i in range(10):
        for j in range(10):
            x0 = i * 10 + 1
            y0 = j * 10 + 1
            holes.append([
                (x0, y0),
                (x0 + 2, y0),
                (x0 + 2, y0 + 2),
                (x0, y0 + 2),
                (x0, y0),
            ])

    poly = Polygon(exterior, holes)
    geoms = [poly]
    owned = _to_owned(geoms)

    from vibespatial.constructive.interiors import interiors_owned

    result = interiors_owned(owned)
    _compare_interiors(geoms, result)


# ---------------------------------------------------------------------------
# CPU fallback path
# ---------------------------------------------------------------------------

def test_cpu_fallback():
    """Test that the CPU fallback produces correct results."""
    from vibespatial.constructive.interiors import _interiors_cpu

    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geoms = [poly, None, Point(1, 2)]
    owned = _to_owned(geoms)

    result = _interiors_cpu(owned)
    result_geoms = result.to_shapely()

    # Polygon with hole
    assert result_geoms[0] is not None
    parts = list(result_geoms[0].geoms)
    assert len(parts) == 1

    # Null input
    assert result_geoms[1] is None

    # Non-polygon
    assert result_geoms[2] is None
