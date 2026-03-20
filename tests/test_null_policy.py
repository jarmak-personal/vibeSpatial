from __future__ import annotations

import math

from shapely.geometry import LineString, Point, Polygon

from vibespatial import GeometryPresence
from vibespatial.runtime.nulls import (
    NULL_BOUNDS,
    classify_geometry,
    measurement_result_for_geometry,
    predicate_result_for_pair,
    unary_result_for_missing_input,
)


def test_classify_geometry_distinguishes_null_empty_and_value() -> None:
    assert classify_geometry(None).presence is GeometryPresence.NULL
    assert classify_geometry(Point()).presence is GeometryPresence.EMPTY
    assert classify_geometry(LineString([(0, 0), (1, 1)])).presence is GeometryPresence.VALUE


def test_unary_nulls_propagate() -> None:
    assert unary_result_for_missing_input(None) is None


def test_empty_measurements_follow_contract() -> None:
    assert measurement_result_for_geometry(Point(), kind="area") == 0.0
    assert measurement_result_for_geometry(Point(), kind="length") == 0.0
    bounds = measurement_result_for_geometry(Polygon(), kind="bounds")
    assert bounds == NULL_BOUNDS
    assert all(math.isnan(value) for value in bounds)


def test_null_measurements_propagate() -> None:
    assert measurement_result_for_geometry(None, kind="area") is None


def test_predicates_distinguish_null_from_empty() -> None:
    valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert predicate_result_for_pair(None, valid) is None
    assert predicate_result_for_pair(Point(), valid) is False
