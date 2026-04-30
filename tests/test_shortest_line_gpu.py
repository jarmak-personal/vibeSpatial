"""Tests for GPU shortest_line kernel.

Verifies the NVRTC shortest_line kernel produces results matching
Shapely's shortest_line within fp64 epsilon for all geometry family
combinations.
"""

from __future__ import annotations

import ast
import inspect

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
from vibespatial.runtime.residency import Residency

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available",
)


def test_shortest_line_gpu_path_has_no_raw_scalar_item_syncs():
    import vibespatial.constructive.shortest_line as module

    tree = ast.parse(inspect.getsource(module))
    raw_item_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert raw_item_calls == []


def _assert_shortest_line_matches(gpu_result, left_geoms, right_geoms, rtol=1e-10):
    """Assert GPU shortest_line matches Shapely for each geometry pair."""
    for i, (lg, rg) in enumerate(zip(left_geoms, right_geoms)):
        if lg is None or rg is None:
            continue
        expected = shapely.shortest_line(lg, rg)
        if expected is None or expected.is_empty:
            continue

        actual = gpu_result.to_shapely()[i]
        if actual is None:
            pytest.fail(f"Row {i}: GPU returned None but Shapely returned {expected.wkt}")
            continue

        expected_coords = np.array(expected.coords)
        actual_coords = np.array(actual.coords)

        np.testing.assert_allclose(
            actual_coords, expected_coords,
            rtol=rtol, atol=1e-12,
            err_msg=f"Row {i}: GPU={actual.wkt}, Shapely={expected.wkt}",
        )


# ---------------------------------------------------------------------------
# Point-to-Point
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_point_point():
    """GPU shortest_line: Point x Point."""
    left_geoms = [Point(0, 0), Point(1, 1), Point(5, 5)]
    right_geoms = [Point(3, 4), Point(4, 5), Point(5, 5)]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


@requires_gpu
def test_shortest_line_stays_device_resident():
    """GPU shortest_line keeps output residency and routing metadata on device."""
    from vibespatial.constructive.shortest_line import shortest_line_owned
    left = from_shapely_geometries([Point(0, 0)], residency=Residency.DEVICE)
    right = from_shapely_geometries([Point(3, 4)], residency=Residency.DEVICE)

    result = shortest_line_owned(left, right, dispatch_mode="gpu")

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@requires_gpu
def test_shortest_line_gpu_subgroup_writes_do_not_force_runtime_sync(monkeypatch):
    from vibespatial.constructive.shortest_line import shortest_line_owned
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()

    def _fail_sync():
        raise AssertionError("shortest_line subgroup writes should not force runtime.synchronize()")

    monkeypatch.setattr(runtime, "synchronize", _fail_sync)

    left = from_shapely_geometries([Point(0, 0), Point(1, 1)], residency=Residency.DEVICE)
    right = from_shapely_geometries([Point(3, 4), Point(4, 5)], residency=Residency.DEVICE)

    result = shortest_line_owned(left, right, dispatch_mode="gpu")

    assert result.device_state is not None
    assert result.row_count == 2


@requires_gpu
def test_shortest_line_output_assembly_avoids_runtime_d2h():
    from vibespatial.constructive.shortest_line import shortest_line_owned
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers, reset_d2h_transfer_count

    left = from_shapely_geometries(
        [Point(0, 0), None, Point(1, 1)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [Point(3, 4), Point(1, 1), Point(4, 5)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        result = shortest_line_owned(left, right, dispatch_mode="gpu")

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    reset_d2h_transfer_count()


@requires_gpu
def test_shortest_line_all_invalid_rows_stays_device_resident(strict_device_guard):
    """GPU shortest_line keeps the all-empty result branch device-resident."""
    from vibespatial.constructive.shortest_line import shortest_line_owned

    left = from_shapely_geometries([None], residency=Residency.DEVICE)
    right = from_shapely_geometries([Point(3, 4)], residency=Residency.DEVICE)

    result = shortest_line_owned(left, right, dispatch_mode="gpu")

    assert result.residency is Residency.DEVICE
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# Point-to-LineString
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_point_linestring():
    """GPU shortest_line: Point x LineString."""
    left_geoms = [
        Point(0, 0),
        Point(5, 5),
        Point(0, 1),
    ]
    right_geoms = [
        LineString([(3, 0), (3, 4)]),
        LineString([(0, 0), (10, 0)]),
        LineString([(2, 0), (2, 2)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# Point-to-Polygon
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_point_polygon():
    """GPU shortest_line: Point x Polygon, including containment."""
    left_geoms = [
        Point(5, 5),   # inside polygon -> distance 0
        Point(10, 0),  # outside polygon
        Point(0, 3),   # outside polygon
    ]
    right_geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# LineString-to-LineString
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_linestring_linestring():
    """GPU shortest_line: LineString x LineString."""
    left_geoms = [
        LineString([(0, 0), (2, 0)]),
        LineString([(0, 0), (0, 5)]),
        LineString([(0, 0), (1, 1), (2, 0)]),
    ]
    right_geoms = [
        LineString([(3, 0), (3, 4)]),
        LineString([(3, 3), (5, 3)]),
        LineString([(3, 0), (3, 3)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# LineString-to-Polygon
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_linestring_polygon():
    """GPU shortest_line: LineString x Polygon."""
    left_geoms = [
        LineString([(5, 5), (5, 15)]),   # crosses polygon
        LineString([(-5, 0), (-3, 0)]),  # outside polygon
    ]
    right_geoms = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# Polygon-to-Polygon
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_polygon_polygon():
    """GPU shortest_line: Polygon x Polygon."""
    left_geoms = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ]
    right_geoms = [
        Polygon([(5, 0), (7, 0), (7, 2), (5, 2)]),
        Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),  # overlapping
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# MultiLineString combinations
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_multilinestring():
    """GPU shortest_line: MultiLineString combinations."""
    left_geoms = [
        MultiLineString([[(0, 0), (1, 0)], [(0, 1), (1, 1)]]),
        LineString([(0, 0), (1, 0)]),
    ]
    right_geoms = [
        LineString([(5, 0), (5, 1)]),
        MultiLineString([[(3, 0), (3, 1)], [(4, 0), (4, 1)]]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# MultiPolygon
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_multipolygon():
    """GPU shortest_line: MultiPolygon combinations."""
    left_geoms = [
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
        ]),
    ]
    right_geoms = [
        Point(2, 0.5),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_null_geometry():
    """GPU shortest_line: null geometries produce null output."""
    left_geoms = [Point(0, 0), None, Point(1, 1)]
    right_geoms = [Point(3, 4), Point(5, 5), None]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)

    geoms = result.to_shapely()
    # Row 0: valid pair
    assert geoms[0] is not None
    # Row 1: left is null -> output is null
    assert geoms[1] is None
    # Row 2: right is null -> output is null
    assert geoms[2] is None


# ---------------------------------------------------------------------------
# Broadcast mode (N vs 1)
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_broadcast():
    """GPU shortest_line: broadcast mode (N vs 1)."""
    left_geoms = [
        Point(0, 0),
        Point(5, 5),
        Point(10, 0),
    ]
    right_geoms = [
        Polygon([(2, 2), (8, 2), (8, 8), (2, 8)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    assert result.row_count == 3

    # Verify against Shapely
    right_shapely = right_geoms[0]
    for i, lg in enumerate(left_geoms):
        expected = shapely.shortest_line(lg, right_shapely)
        actual = result.to_shapely()[i]
        expected_coords = np.array(expected.coords)
        actual_coords = np.array(actual.coords)
        np.testing.assert_allclose(
            actual_coords, expected_coords,
            rtol=1e-10, atol=1e-12,
            err_msg=f"Broadcast row {i}",
        )


# ---------------------------------------------------------------------------
# Mixed families in one array
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_mixed_families():
    """GPU shortest_line: mixed geometry families in the same array."""
    left_geoms = [
        Point(0, 0),
        LineString([(0, 0), (1, 0)]),
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
    ]
    right_geoms = [
        LineString([(3, 0), (3, 4)]),
        Polygon([(5, 0), (7, 0), (7, 2), (5, 2)]),
        Point(5, 5),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# GeometryArray surface API
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_geometry_array_api():
    """GPU shortest_line: accessible via GeometryArray.shortest_line."""
    import geopandas

    left = geopandas.GeoSeries([Point(0, 0), Point(5, 5)])
    right = geopandas.GeoSeries([Point(3, 4), LineString([(0, 0), (10, 0)])])

    result = left.shortest_line(right)
    assert len(result) == 2

    # Verify against Shapely
    for i in range(len(left)):
        expected = shapely.shortest_line(left.iloc[i], right.iloc[i])
        actual = result.iloc[i]
        expected_coords = np.array(expected.coords)
        actual_coords = np.array(actual.coords)
        np.testing.assert_allclose(
            actual_coords, expected_coords,
            rtol=1e-10, atol=1e-12,
            err_msg=f"GeometryArray API row {i}",
        )


# ---------------------------------------------------------------------------
# Polygon with holes
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_polygon_with_hole():
    """GPU shortest_line: polygon with holes."""
    # Polygon with a hole: the point is inside the hole, so the shortest
    # line should connect to the hole boundary, not the exterior.
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly_with_hole = Polygon(outer, [hole])

    left_geoms = [Point(5, 5)]  # inside the hole
    right_geoms = [poly_with_hole]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    _assert_shortest_line_matches(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# Identical geometries (distance = 0)
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_identical_geometries():
    """GPU shortest_line: identical geometries produce zero-length line."""
    left_geoms = [
        Point(5, 5),
        LineString([(0, 0), (10, 0)]),
    ]
    right_geoms = left_geoms.copy()

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)

    for i in range(len(left_geoms)):
        actual = result.to_shapely()[i]
        assert actual is not None
        coords = np.array(actual.coords)
        # Both endpoints should be the same (zero-length line)
        np.testing.assert_allclose(
            coords[0], coords[1], atol=1e-12,
            err_msg=f"Row {i}: expected zero-length shortest line for identical geometries",
        )


# ---------------------------------------------------------------------------
# Empty array
# ---------------------------------------------------------------------------

@requires_gpu
def test_shortest_line_empty():
    """GPU shortest_line: empty input arrays."""
    left = from_shapely_geometries([])
    right = from_shapely_geometries([])

    from vibespatial.constructive.shortest_line import shortest_line_owned

    result = shortest_line_owned(left, right, dispatch_mode="gpu")
    assert isinstance(result, OwnedGeometryArray)
    assert result.row_count == 0
