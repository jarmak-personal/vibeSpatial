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

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from vibespatial.constructive.multipoint_polygon_constructive import (
    multipoint_polygon_difference,
    multipoint_polygon_intersection,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.kernels.constructive.nonpolygon_binary import (
    linestring_linestring_intersection_native,
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


def test_linestring_linestring_intersection_uses_shared_native_topology() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "nonpolygon_binary.py"
    ).read_text()
    kernel_source = (
        repo_root
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "nonpolygon_binary_source.py"
    ).read_text()

    assert "linestring-linestring intersection point allocation fence" not in source
    assert "linestring_linestring_count" not in kernel_source
    assert "classify_segment_intersections(" in source
    assert "_classified_page_consumer=_retain_classified_page" in source
    assert "atomic_line_union_from_part_capacity_device(" in source
    assert "unique_points_from_part_capacity_device(" in source
    assert "_geometry_composition_from_owned_parts_at_capacity(" in source


def test_multipoint_polygon_constructive_is_device_rowset_shaped() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "src"
        / "vibespatial"
        / "constructive"
        / "multipoint_polygon_constructive.py"
    ).read_text()

    assert "copy_device_to_host" not in source
    assert "to_shapely" not in source
    assert "from_shapely" not in source
    assert "for i in range" not in source
    assert "binary_predicate_expression" in source
    assert "_explode_point_rows_to_point_capacity_gpu" in source
    assert "active_capacity_mask" in source
    assert "_selected_point_rows_to_owned" in source
    assert "_deduplicate_selected_points" in source
    assert "_explode_multipoint_rows_to_points_gpu" not in source

    binary_source = (
        repo_root / "src" / "vibespatial" / "constructive" / "binary_constructive.py"
    ).read_text()
    assert "point-polygon intersection fallback PIP mask export" not in binary_source
    assert "point-polygon difference fallback PIP mask export" not in binary_source
    assert "_build_point_polygon_result" not in binary_source
    assert "constructive indexed multipoint point-count allocation fence" not in binary_source
    assert "constructive multilinestring part-count allocation fence" not in binary_source
    assert "polygon contained-hole ring allocation fence" not in binary_source
    assert "polygon contained-hole coordinate allocation fence" not in binary_source
    assert "polygon contained-hole ring-count scalar fence" not in binary_source
    assert "polygon contained-hole ring-length scalar fence" not in binary_source
    assert "right-segment row-order scalar fence" not in binary_source
    assert "binary constructive expanded right-segment allocation fence" not in binary_source
    assert "_expand_right_segments_for_pair_rows" not in binary_source
    assert "constructive.point_polygon.intersection" in binary_source
    assert "constructive.point_polygon.difference" in binary_source


def test_point_pair_dynamic_outputs_use_physical_coordinate_capacity() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "nonpolygon_binary.py"
    ).read_text()

    union_source = source.split("def point_point_union(", 1)[1].split(
        "def point_point_symmetric_difference(", 1
    )[0]
    symmetric_difference_source = source.split(
        "def point_point_symmetric_difference(", 1
    )[1].split("# Point-LineString constructive operations", 1)[0]

    for function_source in (union_source, symmetric_difference_source):
        assert "coordinate_capacity = n * 2" in function_source
        assert "count_scatter_total(" not in function_source
        assert "d_geometry_offsets[n] = d_offsets_cp[-1] + d_counts[-1]" in function_source

    assert "point-point constructive coordinate allocation fence" not in source
    assert "point-point difference coordinate allocation fence" not in source


def test_line_polygon_constructive_uses_collective_capacity_topology() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root.joinpath(
        "src/vibespatial/kernels/constructive/nonpolygon_binary.py"
    ).read_text()
    topology = repo_root.joinpath(
        "src/vibespatial/constructive/line_polygon_difference.py"
    ).read_text()
    kernel_source = repo_root.joinpath(
        "src/vibespatial/kernels/constructive/nonpolygon_binary_source.py"
    ).read_text()

    assert "lineal_polygonal_constructive_topology_gpu" in source
    assert "_linestring_polygon_constructive" not in source
    assert "linestring-polygon constructive vertex allocation fence" not in source
    assert "linestring-polygon difference totals allocation fence" not in source
    assert "_LINESTRING_POLYGON_KERNEL_SOURCE" not in kernel_source
    assert "d_interval_active" in topology
    assert "d_event_active" in topology
    assert "NativeGeometryComposition" not in topology
    assert "_geometry_composition_from_owned_parts_at_capacity" in topology
    assert "unique_points_from_part_capacity_device" in topology

    binary_source = repo_root.joinpath(
        "src/vibespatial/constructive/binary_constructive.py"
    ).read_text()
    assert "def _device_take_polygon_family_rows(" not in binary_source
    assert "_dispatch_lineal_polygonal_constructive_gpu" in binary_source
    lineal_dispatch = binary_source.split(
        "# --- Lineal-Polygonal collective topology ---",
        1,
    )[1].split("# --- LineString-LineString ---", 1)[0]
    assert "try:" not in lineal_dispatch
    assert "except Exception" not in lineal_dispatch


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


def _native_result_geometries(result):
    series = result.to_geoseries(
        index=np.arange(result.row_count),
        name="geometry",
    )
    return series.to_numpy()


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

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = _native_result_geometries(result)
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

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = _native_result_geometries(result)

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

    @requires_gpu
    def test_boundary_empty_and_null_rows_match_constructive_semantics(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left_geoms = [Point(0, 1), Point(5, 5), Point(1, 1), None]
        right_geoms = [box(0, 0, 3, 3), box(0, 0, 3, 3), None, box(0, 0, 3, 3)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        for operation in ("intersection", "difference"):
            result = binary_constructive_owned(
                operation,
                left,
                right,
                dispatch_mode=ExecutionMode.GPU,
            )
            expected = _shapely_op(operation, left_geoms, right_geoms)
            for got, want in zip(result.to_shapely(), expected, strict=True):
                if want is None:
                    assert got is None
                else:
                    assert got is not None
                    assert got.geom_type == want.geom_type
                    assert shapely.equals(got, want)


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


@requires_gpu
@pytest.mark.parametrize("operation", ["union", "symmetric_difference"])
def test_point_pair_dynamic_output_has_no_allocation_fence(operation):
    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left = _make_owned([Point(1, 2), Point(3, 4), None])
    right = _make_owned([Point(1, 2), Point(5, 6), None])

    reset_d2h_transfer_count()
    result = binary_constructive_owned(
        operation,
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.residency is Residency.DEVICE
    assert result.device_state is not None
    assert result.device_state.families[GeometryFamily.MULTIPOINT].x.size == 6
    assert not any("point-point" in reason and "allocation fence" in reason for reason in runtime_reasons)


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

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )
        result_geoms = _native_result_geometries(result)

        assert len(result_geoms) == 1
        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_line_outside_polygon(self):
        """Line fully outside polygon -> intersection is empty."""
        left_geoms = [LineString([(10, 10), (20, 20)])]
        right_geoms = [box(0, 0, 4, 4)]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )
        result_geoms = _native_result_geometries(result)

        assert result_geoms[0] is None or result_geoms[0].is_empty

    @requires_gpu
    def test_line_touching_polygon_corner_returns_point(self):
        """A touching line/polygon intersection must preserve point-collapse slivers."""
        left = _make_owned([LineString([(10, 5), (13, 5), (15, 5)])])
        right = _make_owned([box(0, 0, 10, 10)])

        result = linestring_polygon_intersection(left, right)
        result_geoms = _native_result_geometries(result)

        assert len(result_geoms) == 1
        _assert_geom_close(result_geoms[0], Point(10, 5), msg="touching corner")

    @requires_gpu
    def test_line_crossing_polygon_hole_preserves_disconnected_intervals(self):
        left_geoms = [LineString([(-1, 5), (11, 5)])]
        right_geoms = [
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                holes=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]],
            )
        ]

        result = linestring_polygon_intersection(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )
        result_geoms = _native_result_geometries(result)
        expected = _shapely_op("intersection", left_geoms, right_geoms)

        assert result_geoms[0].geom_type == "MultiLineString"
        _assert_geom_close(result_geoms[0], expected[0], msg="polygon hole")

    @requires_gpu
    def test_multipart_polygon_emits_line_and_isolated_point_composition(self):
        left_geoms = [LineString([(-1, 0), (3, 0), (5, 0)])]
        right_geoms = [
            MultiPolygon(
                [
                    box(0, -1, 2, 1),
                    box(5, 0, 6, 1),
                ]
            )
        ]

        result = linestring_polygon_intersection(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )
        result_geoms = _native_result_geometries(result)
        expected = _shapely_op("intersection", left_geoms, right_geoms)

        assert result.composition is not None
        assert isinstance(result_geoms[0], GeometryCollection)
        _assert_geom_close(
            shapely.normalize(result_geoms[0]),
            shapely.normalize(expected[0]),
            msg="multipart line and point",
        )

    @requires_gpu
    def test_boundary_coincident_interval_is_retained(self):
        left_geoms = [LineString([(-1, 0), (3, 0)])]
        right_geoms = [box(0, 0, 2, 2)]

        result = linestring_polygon_intersection(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )
        result_geoms = _native_result_geometries(result)
        expected = _shapely_op("intersection", left_geoms, right_geoms)

        _assert_geom_close(result_geoms[0], expected[0], msg="boundary overlap")

    @requires_gpu
    def test_line_inside_polygon_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([box(0, 0, 4, 4)])
        result = linestring_polygon_intersection(left, right)

        assert result.residency is Residency.DEVICE
        assert result.composition is not None
        assert all(
            part.geometry.owned is not None
            and part.geometry.owned._validity is None
            and part.geometry.owned._tags is None
            and part.geometry.owned._family_row_offsets is None
            for part in result.composition.parts
        )

    @requires_gpu
    def test_nonpolygon_right_empty_result_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([shapely.Polygon()])
        result = linestring_polygon_intersection(left, right)

        assert result.residency is Residency.DEVICE
        concrete = (
            (result.owned,)
            if result.owned is not None
            else tuple(part.geometry.owned for part in result.composition.parts)
        )
        assert all(
            owned is not None
            and owned._validity is None
            and owned._tags is None
            and owned._family_row_offsets is None
            for owned in concrete
        )


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
    def test_translated_sloped_boundary_line_becomes_empty_geometry(self):
        edge_start = (359675.78571516427, 3080293.5870556235)
        edge_end = (359060.2560933399, 3080300.753509197)
        left_geoms = [LineString([edge_start, edge_end])]
        right_geoms = [
            Polygon(
                [
                    edge_start,
                    edge_end,
                    (359057.803519292, 3080090.539),
                    (359671.987, 3079966.54),
                    edge_start,
                ]
            )
        ]

        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        result = binary_constructive_owned(
            "difference",
            _make_owned(left_geoms),
            _make_owned(right_geoms),
            dispatch_mode=ExecutionMode.GPU,
        )
        expected = _shapely_op("difference", left_geoms, right_geoms)

        _assert_geom_close(
            result.to_shapely()[0],
            expected[0],
            msg="translated sloped boundary line",
        )

    @requires_gpu
    def test_polygon_hole_remains_in_line_difference(self):
        left_geoms = [LineString([(-1, 5), (11, 5)])]
        right_geoms = [
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                holes=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]],
            )
        ]

        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        result = binary_constructive_owned(
            "difference",
            _make_owned(left_geoms),
            _make_owned(right_geoms),
            dispatch_mode=ExecutionMode.GPU,
        )
        expected = _shapely_op("difference", left_geoms, right_geoms)

        assert isinstance(result.to_shapely()[0], MultiLineString)
        _assert_geom_close(result.to_shapely()[0], expected[0], msg="polygon hole")

    @requires_gpu
    def test_nonpolygon_right_empty_result_stays_device_resident(self, strict_device_guard):
        from vibespatial.kernels.constructive.nonpolygon_binary import (
            linestring_polygon_difference,
        )

        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([shapely.Polygon()])
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
    def test_single_empty_duplicate_and_null_rows_match_set_semantics(self):
        left_geoms = [
            MultiPoint([(1, 1), (5, 5)]),
            MultiPoint([(5, 5), (6, 6)]),
            MultiPoint([(1, 1), (1, 1), (2, 2)]),
            MultiPoint([(1, 1)]),
        ]
        right_geoms = [box(0, 0, 3, 3), box(0, 0, 3, 3), box(0, 0, 3, 3), None]
        result = multipoint_polygon_intersection(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )
        expected = _shapely_op("intersection", left_geoms, right_geoms)

        for got, want in zip(result.to_shapely(), expected, strict=True):
            if want is None:
                assert got is None
            else:
                assert got is not None
                assert got.geom_type == want.geom_type
                assert shapely.equals(got, want)

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
    def test_empty_duplicate_and_null_rows_match_set_semantics(self):
        left_geoms = [
            MultiPoint([(1, 1), (1, 1)]),
            MultiPoint([(1, 1), (5, 5), (5, 5)]),
            MultiPoint([]),
            MultiPoint([(1, 1)]),
        ]
        right_geoms = [box(0, 0, 3, 3), box(0, 0, 3, 3), box(0, 0, 3, 3), None]
        result = multipoint_polygon_difference(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )
        expected = _shapely_op("difference", left_geoms, right_geoms)

        for got, want in zip(result.to_shapely(), expected, strict=True):
            if want is None:
                assert got is None
            else:
                assert got is not None
                assert got.geom_type == want.geom_type
                assert shapely.equals(got, want)

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
    @staticmethod
    def _to_shapely(result):
        return np.asarray(
            result.to_geoseries(
                index=np.arange(result.row_count),
                name="geometry",
            ),
            dtype=object,
        )

    @requires_gpu
    def test_crossing_lines(self):
        """Two crossing line segments produce an intersection point."""
        left_geoms = [LineString([(0, 0), (2, 2)])]
        right_geoms = [LineString([(0, 2), (2, 0)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = self._to_shapely(result)

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

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = self._to_shapely(result)

        assert result_geoms[0].geom_type == "LineString"
        assert result_geoms[0].is_empty

    @requires_gpu
    def test_multiple_crossings(self):
        """A zigzag line crossing a straight line produces multiple intersection points."""
        left_geoms = [LineString([(0, 0), (4, 0)])]
        right_geoms = [LineString([(1, -1), (1, 1), (3, -1), (3, 1)])]
        left = _make_owned(left_geoms)
        right = _make_owned(right_geoms)

        from vibespatial.constructive.binary_constructive import binary_constructive_native
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
        result_geoms = self._to_shapely(result)

        assert result_geoms[0] is not None and not result_geoms[0].is_empty

    @requires_gpu
    def test_crossing_lines_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(0, 0), (2, 2)])])
        right = _make_owned([LineString([(0, 2), (2, 0)])])
        result = linestring_linestring_intersection_native(left, right)

        assert result.residency is Residency.DEVICE
        assert result.composition is not None
        assert all(
            part.geometry.residency is Residency.DEVICE
            for part in result.composition.parts
        )

    @requires_gpu
    def test_disjoint_lines_empty_result_stays_device_resident(self, strict_device_guard):
        left = _make_owned([LineString([(0, 0), (1, 0)])])
        right = _make_owned([LineString([(0, 1), (1, 1)])])
        result = linestring_linestring_intersection_native(left, right)

        assert result.residency is Residency.DEVICE
        assert result.composition is not None

    @requires_gpu
    def test_overlap_and_isolated_points_form_terminal_collection(self):
        left_geoms = [LineString([(0, 0), (3, 0), (3, 3)])]
        right_geoms = [
            LineString([(1, 0), (2, 0), (4, 1), (2, 1), (3, 2)])
        ]
        result = linestring_linestring_intersection_native(
            _make_owned(left_geoms),
            _make_owned(right_geoms),
        )

        got = self._to_shapely(result)[0]
        expected = shapely.intersection(left_geoms[0], right_geoms[0])
        assert got.geom_type == "GeometryCollection"
        assert shapely.equals(got, expected)


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
        from vibespatial.constructive.binary_constructive import (
            binary_constructive_native,
            binary_constructive_owned,
        )

        left = _make_owned([LineString([(1, 1), (2, 2)])])
        right = _make_owned([box(0, 0, 4, 4)])
        intersection = binary_constructive_native(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )
        difference = binary_constructive_owned(
            "difference",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )
        assert intersection is not None
        assert difference is not None

    @requires_gpu
    def test_linestring_linestring_dispatch(self):
        from vibespatial.constructive.binary_constructive import binary_constructive_native

        left = _make_owned([LineString([(0, 0), (2, 2)])])
        right = _make_owned([LineString([(0, 2), (2, 0)])])
        result = binary_constructive_native("intersection", left, right, dispatch_mode=ExecutionMode.GPU)
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
        from vibespatial.constructive.binary_constructive import binary_constructive_native

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

        result = binary_constructive_native(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

        assert result is not None
        assert result.residency is Residency.DEVICE
        assert result.row_count == 2
        assert result.composition is not None
