"""Tests for broadcast-right and scalar-right GPU dispatch of binary predicates.

Verifies that all 10 standard binary predicates (+ equals_exact separately)
work correctly when the right operand is:
  - A scalar Shapely geometry (WorkloadShape.SCALAR_RIGHT)
  - A 1-row OwnedGeometryArray (WorkloadShape.BROADCAST_RIGHT)

Oracle: Shapely with np.broadcast_to / tiling serves as the reference.
"""
from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, box

from vibespatial import (
    ExecutionMode,
    NullBehavior,
    clear_dispatch_events,
    evaluate_binary_predicate,
    from_shapely_geometries,
    get_dispatch_events,
    has_gpu_runtime,
)

# The 10 standard predicates routed through evaluate_binary_predicate.
_STANDARD_PREDICATES = (
    "intersects",
    "within",
    "contains",
    "covers",
    "covered_by",
    "touches",
    "crosses",
    "overlaps",
    "disjoint",
    "equals",
)


def _shapely_oracle(
    predicate: str,
    left_geoms: list[object | None],
    right_geom: object | None,
) -> list[bool | None]:
    """Compute expected results by tiling right_geom and using Shapely."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray([right_geom] * len(left_geoms), dtype=object)
    raw = getattr(shapely, predicate)(left_arr, right_arr)
    out: list[bool | None] = []
    for left_val, right_val, val in zip(left_geoms, [right_geom] * len(left_geoms), raw.tolist(), strict=True):
        if left_val is None or right_val is None:
            out.append(None)
        else:
            out.append(bool(val))
    return out


# -----------------------------------------------------------------------
# 1. Oracle correctness: broadcast result == tiled-pairwise Shapely result
# -----------------------------------------------------------------------


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_broadcast_scalar_right_matches_shapely_oracle(predicate: str) -> None:
    """Scalar-right broadcast produces the same result as tiled Shapely."""
    left = [
        Point(1, 1),
        Point(0, 0),
        Point(3, 3),
        Point(0.5, 0.5),
        None,
    ]
    right_geom = box(0, 0, 2, 2)

    result = evaluate_binary_predicate(
        predicate,
        left,
        right_geom,
        null_behavior=NullBehavior.PROPAGATE,
    )
    expected = _shapely_oracle(predicate, left, right_geom)
    assert result.values.tolist() == expected, (
        f"predicate={predicate}: got {result.values.tolist()}, expected {expected}"
    )


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_broadcast_owned_right_matches_shapely_oracle(predicate: str) -> None:
    """1-row OwnedGeometryArray broadcast produces the same result as tiled Shapely."""
    left = [
        Point(1, 1),
        Point(0, 0),
        Point(3, 3),
        Point(0.5, 0.5),
    ]
    right_geom = box(0, 0, 2, 2)
    right_owned = from_shapely_geometries([right_geom])

    result = evaluate_binary_predicate(
        predicate,
        left,
        right_owned,
        null_behavior=NullBehavior.PROPAGATE,
    )
    expected = _shapely_oracle(predicate, left, right_geom)
    assert result.values.tolist() == expected, (
        f"predicate={predicate}: got {result.values.tolist()}, expected {expected}"
    )


# -----------------------------------------------------------------------
# 2. SCALAR_RIGHT vs BROADCAST_RIGHT parity
# -----------------------------------------------------------------------


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_scalar_vs_broadcast_parity(predicate: str) -> None:
    """gs.pred(geom) == gs.pred(from_shapely_geometries([geom]))."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 6, 6)]
    right_geom = Point(1, 1)

    scalar_result = evaluate_binary_predicate(
        predicate, left, right_geom,
        null_behavior=NullBehavior.PROPAGATE,
    )
    broadcast_result = evaluate_binary_predicate(
        predicate, left, from_shapely_geometries([right_geom]),
        null_behavior=NullBehavior.PROPAGATE,
    )
    assert scalar_result.values.tolist() == broadcast_result.values.tolist(), (
        f"predicate={predicate}: scalar={scalar_result.values.tolist()}, "
        f"broadcast={broadcast_result.values.tolist()}"
    )


# -----------------------------------------------------------------------
# 3. Null broadcast: single null geometry -> all-False (NullBehavior.FALSE)
# -----------------------------------------------------------------------


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_null_broadcast_all_false(predicate: str) -> None:
    """Broadcasting a null right geometry with NullBehavior.FALSE yields all False."""
    left = [Point(1, 1), box(0, 0, 2, 2), LineString([(0, 0), (1, 1)])]
    result = evaluate_binary_predicate(
        predicate, left, None,
        null_behavior=NullBehavior.FALSE,
    )
    assert all(v is False or v == False for v in result.values.tolist())  # noqa: E712


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_null_broadcast_propagate(predicate: str) -> None:
    """Broadcasting a null right geometry with NullBehavior.PROPAGATE yields all None."""
    left = [Point(1, 1), box(0, 0, 2, 2)]
    result = evaluate_binary_predicate(
        predicate, left, None,
        null_behavior=NullBehavior.PROPAGATE,
    )
    assert all(v is None for v in result.values.tolist())


# -----------------------------------------------------------------------
# 4. Empty broadcast: single empty geometry
# -----------------------------------------------------------------------


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_empty_broadcast(predicate: str) -> None:
    """Broadcasting an empty geometry matches Shapely's empty-geometry semantics."""
    left = [Point(1, 1), box(0, 0, 2, 2)]
    empty_geom = shapely.from_wkt("POLYGON EMPTY")

    result = evaluate_binary_predicate(
        predicate, left, empty_geom,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle(predicate, left, empty_geom)
    # With NullBehavior.FALSE, None -> False
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false


# -----------------------------------------------------------------------
# 5. Mixed-family broadcast: left has Points+Polygons, right is single Polygon
# -----------------------------------------------------------------------


@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_mixed_family_broadcast(predicate: str) -> None:
    """Mixed left families (Point + Polygon) with single Polygon right."""
    left = [
        Point(1, 1),
        box(0, 0, 2, 2),
        Point(5, 5),
        box(0.5, 0.5, 1.5, 1.5),
    ]
    right_geom = box(0, 0, 3, 3)

    result = evaluate_binary_predicate(
        predicate, left, right_geom,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle(predicate, left, right_geom)
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false


# -----------------------------------------------------------------------
# 6. Dispatch event: workload_shape appears in dispatch event detail
# -----------------------------------------------------------------------


def test_dispatch_event_records_workload_shape_scalar() -> None:
    """Dispatch event detail includes workload_shape for scalar-right."""
    clear_dispatch_events()
    left = [Point(1, 1), Point(0, 0)]
    evaluate_binary_predicate("intersects", left, box(0, 0, 2, 2))
    events = get_dispatch_events()
    matching = [
        e for e in events
        if e.operation == "intersects" and "workload_shape" in e.detail
    ]
    assert len(matching) >= 1
    assert any("scalar_right" in e.detail for e in matching)


def test_dispatch_event_records_workload_shape_broadcast() -> None:
    """Dispatch event detail includes workload_shape for broadcast-right."""
    clear_dispatch_events()
    left = [Point(1, 1), Point(0, 0)]
    right_owned = from_shapely_geometries([box(0, 0, 2, 2)])
    evaluate_binary_predicate("intersects", left, right_owned)
    events = get_dispatch_events()
    matching = [
        e for e in events
        if e.operation == "intersects" and "workload_shape" in e.detail
    ]
    assert len(matching) >= 1
    assert any("broadcast_right" in e.detail for e in matching)


# -----------------------------------------------------------------------
# 7. GPU dispatch verification: scalar-right dispatches to GPU, not CPU fallback
# -----------------------------------------------------------------------


@pytest.mark.gpu
def test_scalar_right_gpu_dispatch() -> None:
    """Scalar-right with explicit GPU mode dispatches to GPU."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([Point(1, 1)] * 100)
    right_geom = box(0, 0, 2, 2)

    clear_dispatch_events()
    result = evaluate_binary_predicate(
        "intersects",
        left,
        right_geom,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    assert bool(result.values[0]) is True
    assert result.runtime_selection.selected is ExecutionMode.GPU


@pytest.mark.gpu
def test_broadcast_right_gpu_dispatch() -> None:
    """Broadcast-right with explicit GPU mode dispatches to GPU."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([Point(1, 1)] * 100)
    right_owned = from_shapely_geometries([box(0, 0, 2, 2)])

    result = evaluate_binary_predicate(
        "intersects",
        left,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    assert bool(result.values[0]) is True
    assert result.runtime_selection.selected is ExecutionMode.GPU


@pytest.mark.gpu
@pytest.mark.parametrize("predicate", _STANDARD_PREDICATES)
def test_all_predicates_gpu_scalar_right(predicate: str) -> None:
    """All 10 standard predicates work with scalar-right on GPU."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([
        Point(1, 1), Point(0, 0), Point(3, 3), Point(0.5, 0.5),
    ])
    right_geom = box(0, 0, 2, 2)

    result = evaluate_binary_predicate(
        predicate,
        left,
        right_geom,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle(predicate, [Point(1, 1), Point(0, 0), Point(3, 3), Point(0.5, 0.5)], right_geom)
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false, (
        f"predicate={predicate}: got {result.values.tolist()}, expected {expected_false}"
    )


# -----------------------------------------------------------------------
# Additional edge cases
# -----------------------------------------------------------------------


def test_broadcast_polygon_vs_polygon() -> None:
    """Broadcast a single polygon against multiple polygons."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 6, 6)]
    right_geom = box(1, 1, 4, 4)

    result = evaluate_binary_predicate(
        "intersects", left, right_geom,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle("intersects", left, right_geom)
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false


def test_broadcast_multipolygon_right() -> None:
    """Broadcast a single MultiPolygon right."""
    left = [Point(1, 1), Point(5, 5), Point(11, 11)]
    right_geom = MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])

    for predicate in ("intersects", "within", "disjoint"):
        result = evaluate_binary_predicate(
            predicate, left, right_geom,
            null_behavior=NullBehavior.FALSE,
        )
        expected = _shapely_oracle(predicate, left, right_geom)
        expected_false = [False if v is None else v for v in expected]
        assert result.values.tolist() == expected_false, (
            f"predicate={predicate}: got {result.values.tolist()}, expected {expected_false}"
        )


def test_broadcast_single_row_owned_pairwise() -> None:
    """1-row owned vs 1-row left should be PAIRWISE, not broadcast."""
    left = from_shapely_geometries([Point(1, 1)])
    right = from_shapely_geometries([box(0, 0, 2, 2)])

    result = evaluate_binary_predicate(
        "intersects", left, right,
        null_behavior=NullBehavior.FALSE,
    )
    assert bool(result.values[0]) is True


def test_broadcast_with_left_nulls() -> None:
    """Broadcast right against left with some null rows."""
    left = [Point(1, 1), None, Point(3, 3), None]
    right_geom = box(0, 0, 2, 2)

    result = evaluate_binary_predicate(
        "intersects", left, right_geom,
        null_behavior=NullBehavior.PROPAGATE,
    )
    expected = _shapely_oracle("intersects", left, right_geom)
    assert result.values.tolist() == expected


def test_broadcast_numpy_array_right() -> None:
    """1-element numpy array triggers broadcast-right."""
    left = [Point(1, 1), Point(0, 0), Point(3, 3)]
    right_arr = np.asarray([box(0, 0, 2, 2)], dtype=object)

    result = evaluate_binary_predicate(
        "intersects", left, right_arr,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle("intersects", left, box(0, 0, 2, 2))
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false


def test_broadcast_list_right() -> None:
    """1-element list triggers broadcast-right."""
    left = [Point(1, 1), Point(0, 0), Point(3, 3)]
    right_list = [box(0, 0, 2, 2)]

    result = evaluate_binary_predicate(
        "intersects", left, right_list,
        null_behavior=NullBehavior.FALSE,
    )
    expected = _shapely_oracle("intersects", left, box(0, 0, 2, 2))
    expected_false = [False if v is None else v for v in expected]
    assert result.values.tolist() == expected_false
