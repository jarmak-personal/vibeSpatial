"""CPU shortest_line variant registration module."""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "shortest_line",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "shortest_line"),
)
def shortest_line_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """CPU shortest_line via Shapely."""
    left_shapely = np.asarray(left.to_shapely(), dtype=object)
    right_shapely = np.asarray(right.to_shapely(), dtype=object)
    return shapely.shortest_line(left_shapely, right_shapely)
