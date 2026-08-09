"""Tests for GPU polygon intersection kernel.

Validates the Sutherland-Hodgman GPU kernel against Shapely oracle
for correctness, edge cases, and precision compliance (ADR-0002).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from vibespatial.runtime import ExecutionMode
from vibespatial.testing import build_owned as _make_owned_polygons

try:
    from vibespatial.cuda._runtime import has_cuda_device

    _has_gpu = has_cuda_device()
except (ImportError, ModuleNotFoundError):
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")


def test_polygon_intersection_uses_shape_bounded_vertex_capacity() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "kernels"
        / "constructive"
        / "polygon_intersection.py"
    ).read_text()

    assert "_polygon_intersection_vertex_capacity(" in source
    assert "workspace_bound = row_count * (int(_MAX_CLIP_VERTS) + 1)" in source
    assert "not owned.is_indexed_view" in source
    assert "count_scatter_total(" not in source
    assert "polygon-polygon intersection vertex allocation fence" not in source
    assert "runtime.synchronize()" not in source


def _shapely_intersection(left_geoms, right_geoms):
    """Shapely oracle: element-wise intersection."""
    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms
    return shapely.intersection(left_arr, right_arr)


def _assert_geom_equal(gpu_geom, ref_geom, *, rtol=1e-6, msg=""):
    """Assert two geometries are equal within tolerance.

    Handles empty geometries and None values. Uses shapely.equals_exact
    for geometric comparison.
    """
    if ref_geom is None or (hasattr(ref_geom, "is_empty") and ref_geom.is_empty):
        # GPU result should also be None or empty
        if gpu_geom is not None and hasattr(gpu_geom, "is_empty"):
            assert gpu_geom.is_empty, f"Expected empty/None but got {gpu_geom}. {msg}"
        return

    if gpu_geom is None:
        pytest.fail(f"GPU returned None but expected {ref_geom}. {msg}")

    if hasattr(gpu_geom, "is_empty") and gpu_geom.is_empty:
        pytest.fail(f"GPU returned empty but expected {ref_geom}. {msg}")

    # For polygon-polygon intersection, the result should be a polygon.
    # Check area-based similarity since vertex ordering may differ.
    gpu_area = shapely.area(gpu_geom)
    ref_area = shapely.area(ref_geom)

    if ref_area < 1e-12:
        # Degenerate result (point or line intersection)
        return

    area_ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
    assert area_ratio < rtol, (
        f"Area mismatch: GPU={gpu_area}, ref={ref_area}, ratio={area_ratio}. {msg}"
    )

    # Also check that the symmetric difference is small
    sym_diff = shapely.area(shapely.symmetric_difference(gpu_geom, ref_geom))
    sym_ratio = sym_diff / max(abs(ref_area), 1e-15)
    assert sym_ratio < rtol, (
        f"Symmetric difference too large: {sym_diff} (ratio={sym_ratio}). {msg}"
    )


# ---------------------------------------------------------------------------
# Test: basic overlapping rectangles
# ---------------------------------------------------------------------------


@requires_gpu
def test_basic_rectangle_overlap(make_owned):
    """Two overlapping axis-aligned rectangles."""
    left_geoms = [box(0, 0, 4, 4)]
    right_geoms = [box(2, 2, 6, 6)]

    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="basic rectangle overlap")


# ---------------------------------------------------------------------------
# Test: fully contained polygon
# ---------------------------------------------------------------------------


@requires_gpu
def test_fully_contained(make_owned):
    """Left polygon fully inside right polygon."""
    left_geoms = [box(1, 1, 3, 3)]
    right_geoms = [box(0, 0, 10, 10)]

    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="fully contained")


# ---------------------------------------------------------------------------
# Test: no overlap (empty result)
# ---------------------------------------------------------------------------


@requires_gpu
def test_no_overlap():
    """Disjoint polygons -> empty result."""
    left_geoms = [box(0, 0, 1, 1)]
    right_geoms = [box(5, 5, 6, 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 1
    # Should be empty or None
    geom = result_geoms[0]
    assert geom is None or geom.is_empty, f"Expected empty but got {geom}"


# ---------------------------------------------------------------------------
# Test: touching edges
# ---------------------------------------------------------------------------


@requires_gpu
def test_touching_edges():
    """Polygons that share an edge but have zero-area intersection."""
    left_geoms = [box(0, 0, 2, 2)]
    right_geoms = [box(2, 0, 4, 2)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    import cupy as cp

    supported = result._polygon_intersection_sh_supported

    # Touching edge -> degenerate (line) intersection -> treated as empty polygon
    assert len(result_geoms) == 1
    geom = result_geoms[0]
    assert cp.asnumpy(supported).tolist() == [False]
    if geom is not None and not geom.is_empty:
        # If it produces a polygon, its area should be negligible
        assert shapely.area(geom) < 1e-10


# ---------------------------------------------------------------------------
# Test: multiple pairs (batched)
# ---------------------------------------------------------------------------


@requires_gpu
def test_multiple_pairs():
    """Batched intersection of multiple polygon pairs."""
    left_geoms = [
        box(0, 0, 4, 4),
        box(0, 0, 2, 2),
        box(10, 10, 20, 20),
    ]
    right_geoms = [
        box(2, 2, 6, 6),
        box(5, 5, 8, 8),
        box(15, 15, 25, 25),
    ]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 3

    # Pair 0: overlapping
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="pair 0")
    # Pair 1: disjoint
    _assert_geom_equal(result_geoms[1], ref_geoms[1], msg="pair 1")
    # Pair 2: overlapping
    _assert_geom_equal(result_geoms[2], ref_geoms[2], msg="pair 2")


@requires_gpu
def test_validated_simple_polygon_intersection_handles_concave_pair():
    """Validated simple-polygon carrier handles concave single-ring pairs."""
    import cupy as cp

    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        box(0.0, 0.0, 3.0, 3.0),
        box(10.0, 0.0, 13.0, 3.0),
        box(10.0, 10.0, 11.0, 11.0),
    ]
    right_geoms = [
        Polygon([(1, -1), (4, 1), (1, 4), (2, 1), (1, -1)]),
        Polygon([(11, -1), (14, 1), (11, 4), (12, 1), (11, -1)]),
        box(20.0, 20.0, 21.0, 21.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    result_tuple = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = get_dispatch_events(clear=True)

    assert result_tuple is not None
    result, supported = result_tuple
    assert cp.asnumpy(supported).tolist() == [True, True, True]
    assert any(
        event.implementation == "validated_simple_polygon_intersection_gpu" for event in events
    )

    got = result.to_shapely()
    expected = _shapely_intersection(left_geoms, right_geoms)
    for got_geom, expected_geom in zip(got, expected, strict=True):
        _assert_geom_equal(got_geom, expected_geom)


@requires_gpu
def test_validated_simple_polygon_intersection_is_translation_invariant() -> None:
    """Projected-coordinate nodes must match the canonical exact topology."""
    import cupy as cp

    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection,
    )

    left_geom = Polygon(
        [
            (362082.6129809683, 3075418.12889014),
            (362090.5054597832, 3076110.5708671827),
            (361474.7678884961, 3076117.605466008),
            (361466.8754096812, 3075425.1634890),
        ]
    )
    right_geom = Polygon(
        [
            (362084.19134555734, 3075556.6172384047),
            (362550.5237908291, 3078441.0256827176),
            (361683.7484846617, 3078450.9131343374),
        ]
    )
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries([right_geom], residency=Residency.DEVICE)

    result_tuple = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result_tuple is not None
    result, supported = result_tuple
    assert cp.asnumpy(supported).tolist() == [True]
    assert shapely.equals(result.to_shapely()[0], left_geom.intersection(right_geom))


@requires_gpu
def test_validated_simple_intersection_preserves_strictly_contained_fp64_sliver() -> None:
    import cupy as cp

    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection,
    )
    from vibespatial.runtime.residency import Residency

    sliver = Polygon(
        [
            (673.2050807568874, 400.0),
            (673.2050807568877, 399.9999999999999),
            (673.2050807568876, 400.0),
        ]
    )
    container = box(600.0, 300.0, 800.0, 500.0)
    left = from_shapely_geometries([sliver], residency=Residency.DEVICE)
    right = from_shapely_geometries([container], residency=Residency.DEVICE)

    result_tuple = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result_tuple is not None
    result, supported = result_tuple
    assert cp.asnumpy(supported).tolist() == [True]
    assert shapely.equals_exact(result.to_shapely()[0], sliver, tolerance=0.0)


@requires_gpu
def test_projected_near_incidence_routes_bounded_carriers_to_exact_topology() -> None:
    """Uncertain source incidences must preserve shared source-edge identity."""
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_intersection import (
        polygon_intersection_sh_eligible_mask,
    )
    from vibespatial.kernels.constructive.polygon_simple_intersection import (
        polygon_simple_intersection,
    )

    parcel = Polygon(
        [
            (360125.33218083536, 3076582.0618681083),
            (360124.6256267402, 3076520.961623712),
            (360229.9370452518, 3076519.744716444),
            (360323.09733569185, 3076643.4940596),
            (360324.21483259567, 3076740.2675098213),
            (360180.3158424565, 3076741.9304189333),
        ]
    )
    primary_building = Polygon(
        [
            (360125.33218083536, 3076582.0618681083),
            (360124.6256267402, 3076520.961623712),
            (360229.9370452518, 3076519.744716444),
            (360323.0973356905, 3076643.494059598),
            (360324.21483259427, 3076740.2675098213),
            (360180.3158424565, 3076741.9304189333),
        ]
    )
    adjacent_building = Polygon(
        [
            (360323.0973356905, 3076643.494059598),
            (360395.33196741896, 3076739.4468329023),
            (360324.21483259427, 3076740.2675098213),
        ]
    )
    buildings = [primary_building, adjacent_building]
    left = from_shapely_geometries([parcel, parcel], residency=Residency.DEVICE)
    right = from_shapely_geometries(buildings, residency=Residency.DEVICE)

    sh_eligible = polygon_intersection_sh_eligible_mask(left, right)
    simple_result = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    result = _dispatch_partitioned_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert cp.asnumpy(sh_eligible).tolist() == [False, False]
    assert simple_result is not None
    assert cp.asnumpy(cp.asarray(simple_result[1])).tolist() == [False, False]
    assert result is not None
    actual = result.to_shapely()
    expected = shapely.intersection(
        np.asarray([parcel, parcel], dtype=object),
        np.asarray(buildings, dtype=object),
    )
    assert all(
        shapely.equals(got, want)
        for got, want in zip(actual, expected, strict=True)
    )
    shared = shapely.intersection(
        shapely.boundary(actual[0]),
        shapely.boundary(actual[1]),
    )
    assert shapely.get_type_id(shared) == shapely.GeometryType.LINESTRING
    assert shapely.length(shared) > 90.0


@requires_gpu
def test_row_indirected_polygon_rows_stay_virtual_in_gpu_kernel(monkeypatch):
    """Direct SH kernel consumes device family-row offsets without resolving views."""
    import cupy as cp

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    source_left = [
        Polygon([(0, 0), (4, 0), (4, 1), (2, 3), (0, 3), (0, 0)]),
        Polygon([(10, 0), (13, 0), (14, 2), (12, 4), (10, 2), (10, 0)]),
        Polygon([(20, 0), (24, 0), (25, 3), (22, 5), (20, 3), (20, 0)]),
    ]
    source_right = [
        box(1, 1, 5, 4),
        box(11, 1, 15, 5),
        box(21, 1, 26, 4),
    ]
    left_rows = cp.asarray([0, 1, 0, 2, 1], dtype=cp.int64)
    right_rows = cp.asarray([0, 1, 0, 2, 1], dtype=cp.int64)
    left = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(source_left, residency=Residency.DEVICE),
        left_rows,
    )
    right = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(source_right, residency=Residency.DEVICE),
        right_rows,
    )

    def _fail_resolve(*_args, **_kwargs):
        raise AssertionError("polygon_intersection should consume row-indirected views")

    monkeypatch.setattr(left, "_resolve", _fail_resolve)
    monkeypatch.setattr(left, "_device_resolve", _fail_resolve)
    monkeypatch.setattr(right, "_resolve", _fail_resolve)
    monkeypatch.setattr(right, "_device_resolve", _fail_resolve)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    events = get_dispatch_events(clear=True)

    assert left.is_indexed_view
    assert right.is_indexed_view
    assert result.row_count == int(left_rows.size)
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)
    assert not any("intersection vertex allocation fence" in reason for reason in runtime_reasons)
    assert any(event.implementation == "polygon_intersection_gpu" for event in events)

    got = result.to_shapely()
    expected = _shapely_intersection(
        [source_left[index] for index in cp.asnumpy(left_rows)],
        [source_right[index] for index in cp.asnumpy(right_rows)],
    )
    for row, (got_geom, expected_geom) in enumerate(zip(got, expected, strict=True)):
        _assert_geom_equal(got_geom, expected_geom, msg=f"row-indirected pair {row}")


# ---------------------------------------------------------------------------
# Test: null input propagation
# ---------------------------------------------------------------------------


@requires_gpu
def test_null_input_propagation():
    """Null inputs should produce null outputs."""
    left_geoms = [box(0, 0, 4, 4), None, box(1, 1, 3, 3)]
    right_geoms = [None, box(2, 2, 6, 6), box(0, 0, 5, 5)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 3
    # Pair 0: right is None -> null
    assert result_geoms[0] is None or (
        hasattr(result_geoms[0], "is_empty") and result_geoms[0].is_empty
    )
    # Pair 1: left is None -> null
    assert result_geoms[1] is None or (
        hasattr(result_geoms[1], "is_empty") and result_geoms[1].is_empty
    )
    # Pair 2: both valid -> should have area
    ref = shapely.intersection(
        shapely.from_wkt("POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"),
        shapely.from_wkt("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))"),
    )
    _assert_geom_equal(result_geoms[2], ref, msg="pair 2 with valid inputs")


# ---------------------------------------------------------------------------
# Test: non-axis-aligned polygons (triangles)
# ---------------------------------------------------------------------------


@requires_gpu
def test_triangle_intersection():
    """Intersection of two overlapping triangles."""
    left_geoms = [Polygon([(0, 0), (4, 0), (2, 4), (0, 0)])]
    right_geoms = [Polygon([(1, 0), (5, 0), (3, 4), (1, 0)])]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="triangle intersection")


@requires_gpu
def test_convex_diamond_overlap_normalizes_without_duplicate_vertices():
    """Uncertain diamond vertices decline SH and complete through exact topology."""
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )

    left_geoms = [Point(0, 0).buffer(1, quad_segs=2)]
    right_geoms = [Point(1, 1).buffer(1, quad_segs=2)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    bounded = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    assert cp.asnumpy(cp.asarray(bounded._polygon_intersection_sh_supported)).tolist() == [
        False
    ]
    result = _dispatch_partitioned_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    assert result is not None
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    assert shapely.equals_exact(
        shapely.normalize(result_geoms[0]),
        shapely.normalize(ref_geoms[0]),
        tolerance=0.5 * 10 ** (-6),
    )


# ---------------------------------------------------------------------------
# Test: identical polygons
# ---------------------------------------------------------------------------


@requires_gpu
def test_identical_polygons():
    """Intersection of identical polygons should return the same polygon."""
    geom = box(0, 0, 5, 5)
    left_geoms = [geom]
    right_geoms = [geom]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], geom, msg="identical polygons")


# ---------------------------------------------------------------------------
# Test: large coordinate values (precision stress)
# ---------------------------------------------------------------------------


@requires_gpu
def test_large_coordinates():
    """Polygons with large absolute coordinate values."""
    offset = 1_000_000.0
    left_geoms = [box(offset, offset, offset + 4, offset + 4)]
    right_geoms = [box(offset + 2, offset + 2, offset + 6, offset + 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(
        result_geoms[0],
        ref_geoms[0],
        rtol=1e-5,
        msg="large coordinates",
    )


# ---------------------------------------------------------------------------
# Test: CPU fallback
# ---------------------------------------------------------------------------


def test_cpu_fallback():
    """CPU fallback produces correct results via Shapely."""
    left_geoms = [box(0, 0, 4, 4)]
    right_geoms = [box(2, 2, 6, 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.CPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="CPU fallback")


# ---------------------------------------------------------------------------
# Test: row count mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_row_count_mismatch():
    """Mismatched row counts should raise ValueError."""
    left = _make_owned_polygons([box(0, 0, 1, 1)])
    right = _make_owned_polygons([box(0, 0, 1, 1), box(2, 2, 3, 3)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    with pytest.raises(ValueError, match="row count mismatch"):
        polygon_intersection(left, right)


# ---------------------------------------------------------------------------
# Test: empty input array
# ---------------------------------------------------------------------------


def test_empty_input():
    """Empty input arrays should return empty result."""
    left = _make_owned_polygons([])
    right = _make_owned_polygons([])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right)
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Test: device-resident result (no D->H in hot path)
# ---------------------------------------------------------------------------


@requires_gpu
def test_result_is_device_resident(strict_device_guard):
    """GPU result should stay device-resident with lazy host metadata."""
    left = _make_owned_polygons([box(0, 0, 4, 4)])
    right = _make_owned_polygons([box(2, 2, 6, 6)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    assert result.residency == Residency.DEVICE
    assert result.device_state is not None
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# Test: partial overlap (L-shaped result)
# ---------------------------------------------------------------------------


@requires_gpu
def test_partial_overlap_pentagon():
    """Intersection that produces a non-rectangular polygon."""
    left_geoms = [Polygon([(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)])]
    right_geoms = [Polygon([(2, -1), (8, -1), (8, 4), (2, 4), (2, -1)])]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="partial overlap pentagon")


# ---------------------------------------------------------------------------
# Test: many pairs (stress test)
# ---------------------------------------------------------------------------


@requires_gpu
def test_many_pairs():
    """Stress test with many polygon pairs."""
    rng = np.random.default_rng(42)
    n = 500
    left_geoms = []
    right_geoms = []
    for _ in range(n):
        x, y = rng.uniform(0, 100, 2)
        w, h = rng.uniform(1, 10, 2)
        left_geoms.append(box(x, y, x + w, y + h))
        x2, y2 = rng.uniform(0, 100, 2)
        w2, h2 = rng.uniform(1, 10, 2)
        right_geoms.append(box(x2, y2, x2 + w2, y2 + h2))

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == n

    # Check each pair
    mismatches = 0
    for i in range(n):
        ref = ref_geoms[i]
        gpu = result_geoms[i]

        ref_area = shapely.area(ref) if ref is not None else 0.0
        gpu_area = shapely.area(gpu) if gpu is not None else 0.0

        if ref_area < 1e-10 and gpu_area < 1e-10:
            continue  # Both empty/degenerate

        if ref_area < 1e-10:
            if gpu_area > 1e-6:
                mismatches += 1
            continue

        ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
        if ratio > 1e-4:
            mismatches += 1

    # Allow a small fraction of mismatches (Sutherland-Hodgman is not exact
    # for concave polygons, but boxes are convex so this should be zero)
    assert mismatches == 0, f"{mismatches}/{n} pairs had area mismatches"


# ---------------------------------------------------------------------------
# Test: ADR-0002 precision plan is wired through
# ---------------------------------------------------------------------------


@requires_gpu
def test_precision_plan_wired():
    """Verify that the precision plan is computed and stays fp64 for CONSTRUCTIVE."""
    from vibespatial.runtime.dispatch import get_dispatch_events

    left = _make_owned_polygons([box(0, 0, 4, 4)])
    right = _make_owned_polygons([box(2, 2, 6, 6)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)

    # Check dispatch event was recorded
    events = get_dispatch_events()
    pi_events = [e for e in events if e.operation == "polygon_intersection"]
    assert len(pi_events) >= 1
    last_event = pi_events[-1]
    assert "precision=fp64" in last_event.detail


# ---------------------------------------------------------------------------
# Test: CW-wound clip polygon (winding direction fix)
# ---------------------------------------------------------------------------


@requires_gpu
def test_cw_clip_polygon_validity():
    """Clip polygons with CW winding must produce correct intersections.

    Regression test for the validity bitmap bug where Sutherland-Hodgman
    assumed CCW winding on the clip (right) polygon.  CW-wound polygons
    inverted the inside/outside test, producing degenerate (< 3 vertex)
    results that were marked invalid.  The fix computes the signed area
    of the clip polygon to detect winding and flips the test accordingly.
    """
    n = 200
    rng = np.random.default_rng(42)
    left_geoms = []
    right_geoms = []
    for _ in range(n):
        # Random triangles -- Polygon() from random points produces
        # CW or CCW winding depending on vertex order.
        pts_l = rng.uniform(0, 100, (3, 2))
        pts_r = pts_l + rng.uniform(-2, 2, (3, 2))
        left_geoms.append(Polygon(pts_l))
        right_geoms.append(Polygon(pts_r))

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    # GPU valid count should match Shapely (some pairs may have degenerate
    # intersections that the kernel correctly marks invalid)
    false_invalids = 0
    for i in range(n):
        ref = ref_geoms[i]
        ref_area = shapely.area(ref) if ref is not None else 0.0
        if ref_area > 1e-10 and not result.validity[i]:
            false_invalids += 1

    assert false_invalids == 0, (
        f"{false_invalids} false invalids (GPU marked invalid but Shapely "
        f"produced non-degenerate polygon)"
    )

    # Geometric correctness for valid pairs
    for i in range(n):
        if not result.validity[i]:
            continue
        ref_area = shapely.area(ref_geoms[i])
        gpu_area = shapely.area(result_geoms[i]) if result_geoms[i] is not None else 0.0
        if ref_area < 1e-10:
            continue
        ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
        assert ratio < 1e-4, f"Pair {i}: GPU area={gpu_area}, ref area={ref_area}, ratio={ratio}"


@requires_gpu
def test_mixed_winding_boxes():
    """Intersection works regardless of left/right winding direction.

    Tests all 4 combinations: CCW/CCW, CCW/CW, CW/CCW, CW/CW.
    """
    n = 10
    ccw_left = [box(i, i, i + 2, i + 2) for i in range(n)]
    ccw_right = [box(i + 1, i + 1, i + 3, i + 3) for i in range(n)]
    # Reverse coordinate order to get CW winding
    cw_left = [Polygon(list(g.exterior.coords)[::-1]) for g in ccw_left]
    cw_right = [Polygon(list(g.exterior.coords)[::-1]) for g in ccw_right]

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    for label, left_geoms, right_geoms in [
        ("CCW/CCW", ccw_left, ccw_right),
        ("CCW/CW", ccw_left, cw_right),
        ("CW/CCW", cw_left, ccw_right),
        ("CW/CW", cw_left, cw_right),
    ]:
        left = _make_owned_polygons(left_geoms)
        right = _make_owned_polygons(right_geoms)
        result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
        assert result.validity.all(), (
            f"{label}: expected all {n} valid, got {result.validity.sum()}"
        )

        # Check areas match expected (1x1 intersection box = area 1.0)
        result_geoms = result.to_shapely()
        for i in range(n):
            gpu_area = shapely.area(result_geoms[i])
            assert abs(gpu_area - 1.0) < 1e-6, (
                f"{label} pair {i}: expected area 1.0, got {gpu_area}"
            )


# ---------------------------------------------------------------------------
# Import guard for Residency
# ---------------------------------------------------------------------------

from vibespatial.runtime.residency import Residency  # noqa: E402
