from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon, box

from vibespatial import ExecutionMode
from vibespatial.kernels.predicates.point_bounds import point_bounds
from vibespatial.testing import compare_with_shapely


def point_bounds_reference(points, polygons):
    results = []
    for point, polygon in zip(points, polygons, strict=True):
        if point is None or polygon is None:
            results.append(None)
            continue
        if point.is_empty or polygon.is_empty:
            results.append((float("nan"),) * 4)
            continue
        minx, miny, maxx, maxy = polygon.bounds
        results.append((minx, miny, maxx, maxy))
    return results


@compare_with_shapely(reference=point_bounds_reference, handle_empty=True)
@pytest.mark.parametrize(
    "null_case,empty_case,mixed_case",
    [
        (True, True, "point, polygon"),
    ],
)
def test_point_bounds(dispatch_mode, oracle_runner, null_case, empty_case, mixed_case) -> None:
    del null_case, empty_case, mixed_case
    points = [None, Point(), Point(1, 1), Point(8, 2)]
    polygons = [box(0, 0, 2, 2), Polygon(), None, box(5, 1, 9, 3)]

    try:
        oracle_runner(
            point_bounds,
            points,
            polygons,
            dispatch_mode=dispatch_mode,
        )
    except NotImplementedError:
        pytest.xfail("point_bounds kernel scaffold is still a placeholder")

def test_point_bounds_reports_scaffold_placeholder() -> None:
    with pytest.raises(NotImplementedError):
        point_bounds([], [], dispatch_mode=ExecutionMode.CPU)
