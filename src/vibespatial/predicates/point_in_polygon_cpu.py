from __future__ import annotations

import numpy as np

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.kernels.predicates.point_within_bounds import (
    NormalizedBoundsInput,
    _evaluate_point_within_bounds,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


def evaluate_point_in_polygon_cpu(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
) -> np.ndarray:
    coarse = _evaluate_point_within_bounds(points, right)
    candidate_rows = np.flatnonzero(coarse == True)  # noqa: E712  # object-dtype array
    if candidate_rows.size == 0:
        return coarse

    point_values = points.to_shapely()
    polygon_values = right.geometry_array.to_shapely()
    refined = np.asarray(
        [
            bool(polygon_values[row_index].covers(point_values[row_index]))
            for row_index in candidate_rows
        ],
        dtype=bool,
    )
    coarse[candidate_rows] = refined
    return coarse


@register_kernel_variant(
    "point_in_polygon",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("coarse-filter", "refine", "covers"),
)
def point_in_polygon_cpu_variant(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
) -> np.ndarray:
    return evaluate_point_in_polygon_cpu(points, right)
