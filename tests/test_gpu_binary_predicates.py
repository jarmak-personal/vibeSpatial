from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, box

from vibespatial import ExecutionMode, NullBehavior, evaluate_binary_predicate, from_shapely_geometries, has_gpu_runtime


def _expected(predicate: str, left: list[object | None], right: list[object | None]) -> list[bool | None]:
    exact = getattr(shapely, predicate)(np.asarray(left, dtype=object), np.asarray(right, dtype=object))
    out: list[bool | None] = []
    for left_value, right_value, value in zip(left, right, exact.tolist(), strict=True):
        if left_value is None or right_value is None:
            out.append(None)
        else:
            out.append(bool(value))
    return out


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([Point(0, 0), Point(0, 0), Point()], [Point(0, 0), Point(1, 1), Point(0, 0)]),
        ([Point(0, 0), Point(1, 0), Point(3, 0)], [LineString([(0, 0), (2, 0)]), LineString([(0, 0), (2, 0)]), LineString([(0, 0), (2, 0)])]),
        ([LineString([(0, 0), (2, 0)]), LineString([(0, 0), (2, 0)]), LineString()], [Point(0, 0), Point(1, 0), Point(0, 0)]),
        ([Point(0, 0), Point(1, 1), Point(3, 3)], [box(0, 0, 2, 2), box(0, 0, 2, 2), box(0, 0, 2, 2)]),
        ([box(0, 0, 2, 2), box(0, 0, 2, 2), box(0, 0, 2, 2)], [Point(0, 0), Point(1, 1), Point(3, 3)]),
        (
            [Point(0, 0), Point(1, 0), Point(3, 0)],
            [
                MultiLineString([[(0, 0), (2, 0)], [(10, 0), (11, 0)]]),
                MultiLineString([[(0, 0), (2, 0)], [(10, 0), (11, 0)]]),
                MultiLineString([[(0, 0), (2, 0)], [(10, 0), (11, 0)]]),
            ],
        ),
        (
            [
                MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)]),
                MultiPolygon([box(0, 0, 2, 2)]),
                MultiPolygon([]),
            ],
            [Point(1, 1), Point(3, 3), Point(0, 0)],
        ),
    ],
)
def test_gpu_binary_predicates_match_shapely_for_supported_point_pairs(left, right) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    for predicate in (
        "intersects",
        "within",
        "contains",
        "covers",
        "covered_by",
        "touches",
        "crosses",
        "overlaps",
        "disjoint",
        "contains_properly",
    ):
        result = evaluate_binary_predicate(
            predicate,
            from_shapely_geometries(left),
            from_shapely_geometries(right),
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.PROPAGATE,
        )
        assert result.values.tolist() == _expected(predicate, left, right)


@pytest.mark.gpu
def test_gpu_binary_predicates_auto_uses_gpu_for_large_supported_batches() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 2, 2)] * 20_000)
    right = from_shapely_geometries([Point(1, 1)] * 20_000)

    result = evaluate_binary_predicate(
        "contains",
        left,
        right,
        dispatch_mode=ExecutionMode.AUTO,
        null_behavior=NullBehavior.FALSE,
    )

    assert bool(result.values[0]) is True
    assert result.runtime_selection.selected is ExecutionMode.GPU
    report = left.diagnostics_report()
    assert not any("fallback" in reason.lower() for reason in report["runtime_history"])


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("left", "right"),
    [
        # MP × Point
        (
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(5, 5), (6, 6)])],
            [Point(1, 1), Point(0, 0)],
        ),
        # Point × MP
        (
            [Point(1, 1), Point(0, 0)],
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(5, 5), (6, 6)])],
        ),
        # MP × LineString
        (
            [MultiPoint([(0, 0), (1, 0)]), MultiPoint([(5, 5)])],
            [LineString([(0, 0), (2, 0)]), LineString([(0, 0), (2, 0)])],
        ),
        # MP × Polygon
        (
            [MultiPoint([(0.5, 0.5), (3, 3)]), MultiPoint([(1, 1)])],
            [box(0, 0, 2, 2), box(0, 0, 2, 2)],
        ),
        # Polygon × MP
        (
            [box(0, 0, 2, 2), box(0, 0, 2, 2)],
            [MultiPoint([(0.5, 0.5), (3, 3)]), MultiPoint([(1, 1)])],
        ),
        # MP × MultiPolygon
        (
            [MultiPoint([(1, 1), (11, 11)])],
            [MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])],
        ),
        # MP × MP
        (
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(0, 0), (1, 1)])],
            [MultiPoint([(1, 1), (2, 2)]), MultiPoint([(5, 5)])],
        ),
        # MP × MultiLineString
        (
            [MultiPoint([(0, 0), (10, 0)])],
            [MultiLineString([[(0, 0), (2, 0)], [(10, 0), (11, 0)]])],
        ),
    ],
)
def test_gpu_binary_predicates_match_shapely_for_multipoint_pairs(left, right) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    for predicate in (
        "intersects",
        "within",
        "contains",
        "covers",
        "covered_by",
        "touches",
        "disjoint",
    ):
        result = evaluate_binary_predicate(
            predicate,
            from_shapely_geometries(left),
            from_shapely_geometries(right),
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.PROPAGATE,
        )
        expected = _expected(predicate, left, right)
        assert result.values.tolist() == expected, (
            f"{predicate}: got {result.values.tolist()}, expected {expected}"
        )
