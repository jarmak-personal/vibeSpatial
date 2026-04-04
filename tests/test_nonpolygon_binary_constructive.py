"""Tests for non-polygon binary constructive GPU kernels.

Validates each family combination kernel against Shapely oracle for
correctness. Tests cover:
- Point-Point: intersection, difference, union, symmetric_difference
- Point-LineString: intersection, difference
- MultiPoint-Polygon: intersection, difference
- LineString-Polygon: intersection, difference
- LineString-LineString: intersection

Also tests the binary_constructive_owned dispatcher to ensure no family
pair returns None (except for exotic multi-type combinations).
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiPoint,
    Point,
    box,
)

from vibespatial.constructive.multipoint_polygon_constructive import (
    multipoint_polygon_difference,
    multipoint_polygon_intersection,
)
from vibespatial.kernels.constructive.nonpolygon_binary import (
    linestring_linestring_intersection,
    linestring_polygon_intersection,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency
from vibespatial.testing import build_owned as _make_owned

try:
    from vibespatial.cuda._runtime import has_cuda_device

    _has_gpu = has_cuda_device()
except (ImportError, ModuleNotFoundError):
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")
def _shapely_op(op_name, left_geoms, right_geoms):
    """Shapely oracle: element-wise binary constructive."""
    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms
    return getattr(shapely, op_name)(left_arr, right_arr)


def _assert_geom_close(gpu_geom, ref_geom, *, tol=1e-6, msg=""):
    """Assert two geometries are equivalent within tolerance."""
    if ref_geom is None or (hasattr(ref_geom, "is_empty") and ref_geom.is_empty):
        if gpu_geom is not None and hasattr(gpu_geom, "is_empty"):
            assert gpu_geom.is_empty, f"Expected empty but got {gpu_geom}. {msg}"
        return
    if gpu_geom is None:
        pytest.fail(f"GPU returned None but expected {ref_geom}. {msg}")
    if hasattr(gpu_geom, "is_empty") and gpu_geom.is_empty:
        pytest.fail(f"GPU returned empty but expected {ref_geom}. {msg}")
    assert shapely.equals_exact(gpu_geom, ref_geom, tol), (
        f"Mismatch: GPU={gpu_geom.wkt}, ref={ref_geom.wkt}. {msg}"
    )


# ---------------------------------------------------------------------------
# Point-Point tests
# ---------------------------------------------------------------------------

class TestPointPointIntersection:
    @requires_gpu
    def test_matching_points(self, make_owned):
        left_geoms = [Point(1, 2), Point(3, 4), Point(5, 6)]
        right_geoms = [Point(1, 2), Point(7, 8), Point(5, 6)]
        left = make_owned(left_geoms)
        right = make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()
        ref_geoms = _shapely_op("intersection", left_geoms, right_geoms)

        assert len(result_geoms) == 3
        # First point matches -> keep
        _assert_geom_close(result_geoms[0], ref_geoms[0], msg="row 0")
        # Second point differs -> empty
        assert result_geoms[1] is None or result_geoms[1].is_empty, "row 1 should be empty"
        # Third matches -> keep
        _assert_geom_close(result_geoms[2], ref_geoms[2], msg="row 2")

    @requires_gpu
    def test_all_different(self, make_owned):
        left_geoms = [Point(0, 0), Point(1, 1)]
        right_geoms = [Point(2, 2), Point(3, 3)]
        left = make_owned(left_geoms)
        right = make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        for g in result_geoms:
            assert g is None or g.is_empty

    @requires_gpu
    def test_matching_points_stay_device_resident(self, make_owned, strict_device_guard):
        left = make_owned([Point(1, 2), Point(3, 4)])
        right = make_owned([Point(1, 2), Point(7, 8)])

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


class TestPointPointDifference:
    @requires_gpu
    def test_basic_difference(self):
        left_geoms = [Point(1, 2), Point(3, 4), Point(5, 6)]
        right_geoms = [Point(1, 2), Point(7, 8), Point(5, 6)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        # Row 0: same -> empty
        assert result_geoms[0] is None or result_geoms[0].is_empty
        # Row 1: different -> keep left
        _assert_geom_close(result_geoms[1], Point(3, 4), msg="row 1")
        # Row 2: same -> empty
        assert result_geoms[2] is None or result_geoms[2].is_empty


class TestPointPolygonIntersection:
    @requires_gpu
    def test_points_inside_outside(self):
        left_geoms = [Point(1, 1), Point(5, 5)]
        right_geoms = [box(0, 0, 3, 3), box(0, 0, 3, 3)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        _assert_geom_close(result_geoms[0], Point(1, 1), msg="row 0")
        assert result_geoms[1] is None or result_geoms[1].is_empty

    @requires_gpu
    def test_points_inside_outside_stay_device_resident(self, strict_device_guard):
        left = _make_owned([Point(1, 1), Point(5, 5)])
        right = _make_owned([box(0, 0, 3, 3), box(0, 0, 3, 3)])

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


class TestPointPointUnion:
    @requires_gpu
    def test_basic_union(self):
        left_geoms = [Point(1, 2), Point(3, 4)]
        right_geoms = [Point(1, 2), Point(5, 6)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("union", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 2
        # Row 0: same point -> result is a single point (in MultiPoint wrapper)
        assert result_geoms[0] is not None and not result_geoms[0].is_empty
        # Row 1: different points -> 2-point MultiPoint
        assert result_geoms[1] is not None and not result_geoms[1].is_empty


class TestPointPointSymmetricDifference:
    @requires_gpu
    def test_basic_symm_diff(self):
        left_geoms = [Point(1, 2), Point(3, 4)]
        right_geoms = [Point(1, 2), Point(5, 6)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("symmetric_difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 2
        # Row 0: same -> empty
        assert result_geoms[0] is None or result_geoms[0].is_empty
        # Row 1: different -> 2-point MultiPoint
        assert result_geoms[1] is not None and not result_geoms[1].is_empty

    @requires_gpu
    def test_same_points_empty_result_stays_device_resident(self, strict_device_guard):
        left = _make_owned([Point(1, 2)])
        right = _make_owned([Point(1, 2)])

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned(
            "symmetric_difference",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# Point-LineString tests
# ---------------------------------------------------------------------------

class TestPointLineStringIntersection:
    @requires_gpu
    def test_point_on_line(self):
        """Point at the midpoint of a line segment -> intersection keeps it."""
        left_geoms = [Point(0.5, 0.5), Point(2, 2)]
        right_geoms = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 2
        # Point (0.5, 0.5) is on line -> keep
        _assert_geom_close(result_geoms[0], Point(0.5, 0.5), msg="on line")
        # Point (2, 2) is NOT on line -> empty
        assert result_geoms[1] is None or result_geoms[1].is_empty

    @requires_gpu
    def test_point_at_endpoint(self):
        """Point at a line endpoint -> intersection keeps it."""
        left_geoms = [Point(0, 0)]
        right_geoms = [LineString([(0, 0), (1, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_point_line_intersection_stays_device_resident(self, strict_device_guard):
        left = _make_owned([Point(0.5, 0.5)])
        right = _make_owned([LineString([(0, 0), (1, 1)])])

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


class TestPointLineStringDifference:
    @requires_gpu
    def test_point_off_line(self):
        """Point NOT on line -> difference keeps it."""
        left_geoms = [Point(2, 0)]
        right_geoms = [LineString([(0, 0), (1, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        _assert_geom_close(result_geoms[0], Point(2, 0))

    @requires_gpu
    def test_point_off_line_stays_device_resident_without_d2h(
        self,
        strict_device_guard,
    ):
        left = _make_owned([Point(2, 0)])
        right = _make_owned([LineString([(0, 0), (1, 1)])])

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        with assert_zero_d2h_transfers():
            result = binary_constructive_owned(
                "difference",
                left,
                right,
                dispatch_mode=ExecutionMode.GPU,
            )

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# LineString-Polygon tests
# ---------------------------------------------------------------------------

class TestLineStringPolygonIntersection:
    @requires_gpu
    def test_line_inside_polygon(self):
        """Line fully inside polygon -> intersection keeps entire line."""
        left_geoms = [LineString([(1, 1), (2, 2)])]
        right_geoms = [box(0, 0, 4, 4)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_line_outside_polygon(self):
        """Line fully outside polygon -> intersection is empty."""
        left_geoms = [LineString([(10, 10), (20, 20)])]
        right_geoms = [box(0, 0, 4, 4)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert result_geoms[0] is None or result_geoms[0].is_empty

    @requires_gpu
    def test_line_touching_polygon_corner_returns_point(self):
        """A touching line/polygon intersection must preserve point-collapse slivers."""
        left = _make_owned([LineString([(10, 5), (13, 5), (15, 5)])])
        right = _make_owned([box(0, 0, 10, 10)])

        result = linestring_polygon_intersection(left, right)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        _assert_geom_close(result_geoms[0], Point(10, 5), msg="touching corner")

    @requires_gpu
    def test_line_inside_polygon_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([box(0, 0, 4, 4)])
        result = linestring_polygon_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

    @requires_gpu
    def test_nonpolygon_right_empty_result_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([Point(0, 0)])
        result = linestring_polygon_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


class TestLineStringPolygonDifference:
    @requires_gpu
    def test_line_outside_polygon_kept(self):
        """Line fully outside polygon -> difference keeps entire line."""
        left_geoms = [LineString([(10, 10), (20, 20)])]
        right_geoms = [box(0, 0, 4, 4)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_crossing_line_splits_into_multiline_outside_pieces(self):
        left_geoms = [
            LineString([(2, 0), (2, 4), (6, 4)]),
            LineString([(0, 3), (6, 3)]),
        ]
        right_geoms = [
            box(1, 1, 3, 3),
            box(3, 3, 5, 5),
        ]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()
        ref_geoms = _shapely_op("difference", left_geoms, right_geoms)

        assert len(result_geoms) == 2
        _assert_geom_close(result_geoms[0], ref_geoms[0], msg="row 0 split outside fragments")
        _assert_geom_close(result_geoms[1], ref_geoms[1], msg="row 1 boundary overlap fragments")

    @requires_gpu
    def test_boundary_coincident_line_becomes_empty_geometry(self):
        left_geoms = [LineString([(0, 0), (1, 0)])]
        right_geoms = [box(0, 0, 2, 2)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()
        ref_geoms = _shapely_op("difference", left_geoms, right_geoms)

        assert len(result_geoms) == 1
        _assert_geom_close(result_geoms[0], ref_geoms[0], msg="boundary-coincident line should become LINESTRING EMPTY")

    @requires_gpu
    def test_nonpolygon_right_empty_result_stays_device_resident(self, strict_device_guard):
        from vibespatial.kernels.constructive.nonpolygon_binary import (
            linestring_polygon_difference,
        )

        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([Point(0, 0)])
        result = linestring_polygon_difference(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# MultiPoint-Polygon tests
# ---------------------------------------------------------------------------

class TestMultiPointPolygonIntersection:
    @requires_gpu
    def test_some_points_inside(self):
        """Some MultiPoint points inside polygon, some outside."""
        left_geoms = [MultiPoint([(1, 1), (5, 5), (2, 2)])]
        right_geoms = [box(0, 0, 3, 3)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()
        ref_geoms = _shapely_op("intersection", left_geoms, right_geoms)

        assert len(result_geoms) == 1
        # (1,1) and (2,2) are inside; (5,5) is outside
        result_geom = result_geoms[0]
        ref_geom = ref_geoms[0]
        assert result_geom is not None and not result_geom.is_empty
        # Check that the result has the right number of points
        if hasattr(result_geom, "geoms"):
            n_result = len(list(result_geom.geoms))
        else:
            n_result = 1  # single Point
        if hasattr(ref_geom, "geoms"):
            n_ref = len(list(ref_geom.geoms))
        else:
            n_ref = 1
        assert n_result == n_ref, f"Expected {n_ref} points, got {n_result}"

    @requires_gpu
    def test_empty_multipoint_stays_device_resident(self, strict_device_guard):
        left = _make_owned([MultiPoint([])])
        right = _make_owned([box(0, 0, 3, 3)])

        result = multipoint_polygon_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


class TestMultiPointPolygonDifference:
    @requires_gpu
    def test_some_points_outside(self):
        """Some MultiPoint points outside polygon."""
        left_geoms = [MultiPoint([(1, 1), (5, 5), (2, 2)])]
        right_geoms = [box(0, 0, 3, 3)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("difference", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        result_geom = result_geoms[0]
        # Only (5,5) should remain
        assert result_geom is not None and not result_geom.is_empty

    @requires_gpu
    def test_empty_multipoint_stays_device_resident(self, strict_device_guard):
        left = _make_owned([MultiPoint([])])
        right = _make_owned([box(0, 0, 3, 3)])

        result = multipoint_polygon_difference(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# LineString-LineString tests
# ---------------------------------------------------------------------------

class TestLineStringLineStringIntersection:
    @requires_gpu
    def test_crossing_lines(self):
        """Two crossing line segments produce an intersection point."""
        left_geoms = [LineString([(0, 0), (2, 2)])]
        right_geoms = [LineString([(0, 2), (2, 0)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert len(result_geoms) == 1
        result_geom = result_geoms[0]
        assert result_geom is not None and not result_geom.is_empty
        # Intersection should be at (1, 1)
        if result_geom.geom_type == "Point":
            np.testing.assert_allclose([result_geom.x, result_geom.y], [1.0, 1.0], atol=1e-6)
        elif result_geom.geom_type == "MultiPoint":
            pts = list(result_geom.geoms)
            assert len(pts) == 1
            np.testing.assert_allclose([pts[0].x, pts[0].y], [1.0, 1.0], atol=1e-6)

    @requires_gpu
    def test_parallel_lines(self):
        """Two parallel lines produce no intersection."""
        left_geoms = [LineString([(0, 0), (2, 0)])]
        right_geoms = [LineString([(0, 1), (2, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert result_geoms[0] is None or result_geoms[0].is_empty

    @requires_gpu
    def test_multiple_crossings(self):
        """A zigzag line crossing a straight line produces multiple intersection points."""
        left_geoms = [LineString([(0, 0), (4, 0)])]
        right_geoms = [LineString([(1, -1), (1, 1), (3, -1), (3, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = result.to_shapely()

        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_crossing_lines_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(0, 0), (2, 2)])])
        right = _make_owned([LineString([(0, 2), (2, 0)])])
        result = linestring_linestring_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

    @requires_gpu
    def test_disjoint_lines_empty_result_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(0, 0), (1, 0)])])
        right = _make_owned([LineString([(0, 1), (1, 1)])])
        result = linestring_linestring_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# Dispatcher integration: no family pair returns None
# ---------------------------------------------------------------------------

class TestDispatcherCoversAllFamilies:
    """Ensure _binary_constructive_gpu handles all common family pairs."""

    @requires_gpu
    def test_point_point_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned([Point(1, 2)])
        right = _make_owned([Point(1, 2)])
        for op in ["intersection", "difference", "union", "symmetric_difference"]:
            result = binary_constructive_owned(op, left, right, dispatch_mode=ExecutionMode.GPU)
            assert result is not None, f"Point-Point {op} returned None"

    @requires_gpu
    def test_point_linestring_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned([Point(0.5, 0.5)])
        right = _make_owned([LineString([(0, 0), (1, 1)])])
        for op in ["intersection", "difference"]:
            result = binary_constructive_owned(op, left, right, dispatch_mode=ExecutionMode.GPU)
            assert result is not None, f"Point-LineString {op} returned None"

    @requires_gpu
    def test_linestring_polygon_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([box(0, 0, 4, 4)])
        for op in ["intersection", "difference"]:
            result = binary_constructive_owned(op, left, right, dispatch_mode=ExecutionMode.GPU)
            assert result is not None, f"LineString-Polygon {op} returned None"

    @requires_gpu
    def test_linestring_linestring_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned([LineString([(0, 0), (2, 2)])])
        right = _make_owned([LineString([(0, 2), (2, 0)])])
        result = binary_constructive_owned("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        assert result is not None, "LineString-LineString intersection returned None"

    @requires_gpu
    def test_multipoint_polygon_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned([MultiPoint([(1, 1), (5, 5)])])
        right = _make_owned([box(0, 0, 3, 3)])
        for op in ["intersection", "difference"]:
            result = binary_constructive_owned(op, left, right, dispatch_mode=ExecutionMode.GPU)
            assert result is not None, f"MultiPoint-Polygon {op} returned None"

    @requires_gpu
    def test_mixed_linestring_and_polygon_intersection_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left = _make_owned(
            [
                LineString([(1, 1), (5, 5)]),
                box(2, 2, 6, 6),
            ]
        )
        right = _make_owned(
            [
                box(0, 0, 4, 4),
                box(0, 0, 5, 5),
            ]
        )

        result = binary_constructive_owned(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

        assert result is not None
        assert result.residency is Residency.DEVICE
        assert result.row_count == 2
