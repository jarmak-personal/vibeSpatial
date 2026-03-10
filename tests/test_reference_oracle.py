from __future__ import annotations

import itertools

import pytest
from shapely.geometry import Point, Polygon

from vibespatial import ExecutionMode
from vibespatial.testing import (
    OracleAssertionError,
    build_point_in_polygon_scenario,
    compare_with_shapely,
    point_in_polygon_reference,
)


@compare_with_shapely(reference=point_in_polygon_reference, handle_empty=True)
def test_reference_oracle_matches_mock_point_in_polygon(dispatch_mode, oracle_runner) -> None:
    scenario = build_point_in_polygon_scenario(scale=32, seed=4)
    comparison = oracle_runner(
        scenario.operation,
        *scenario.args,
        dispatch_mode=dispatch_mode,
        **scenario.kwargs,
    )

    assert comparison.selection.requested is dispatch_mode
    assert len(comparison.actual) == 32


@compare_with_shapely(reference=point_in_polygon_reference, handle_empty=True, max_mismatches=2)
def test_reference_oracle_reports_first_mismatch_with_geometry_context(oracle_runner) -> None:
    points = [Point(1, 1), Point(8, 8), Point()]
    polygons = [
        Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        Polygon([(5, 5), (9, 5), (9, 9), (5, 9)]),
        Polygon(),
    ]

    def broken_kernel(points, polygons, *, dispatch_mode=ExecutionMode.CPU):
        del dispatch_mode
        result = point_in_polygon_reference(points, polygons)
        result[1] = not result[1]
        return result

    with pytest.raises(OracleAssertionError, match=r"index 1: actual=False, expected=True") as exc_info:
        oracle_runner(broken_kernel, points, polygons)

    message = str(exc_info.value)
    assert "arg0=POINT (8 8)" in message
    assert "arg1=POLYGON ((5 5, 9 5, 9 9, 5 9, 5 5))" in message


@compare_with_shapely(reference=point_in_polygon_reference, handle_empty=True, check_determinism=True)
def test_reference_oracle_can_spot_nondeterministic_drift(oracle_runner) -> None:
    scenario = build_point_in_polygon_scenario(scale=8, seed=2)
    flip = itertools.count()

    def flaky_kernel(points, polygons, *, dispatch_mode=ExecutionMode.CPU):
        del dispatch_mode
        result = point_in_polygon_reference(points, polygons)
        if next(flip) > 0:
            result[4] = not result[4]
        return result

    with pytest.raises(OracleAssertionError, match=r"Determinism check failed on repeat 2"):
        oracle_runner(flaky_kernel, *scenario.args)


def test_reference_oracle_runner_handles_explicit_cpu_mode(oracle_runner) -> None:
    scenario = build_point_in_polygon_scenario(scale=16, seed=1)
    comparison = oracle_runner(
        scenario.operation,
        *scenario.args,
        dispatch_mode=ExecutionMode.CPU,
        config=scenario.config,
        **scenario.kwargs,
    )

    assert comparison.selection.selected is ExecutionMode.CPU
    assert comparison.mismatches == ()
