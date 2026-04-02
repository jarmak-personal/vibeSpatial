"""CPU-only binary constructive variant registration module."""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "binary_constructive",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely",),
)
def binary_constructive_cpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
) -> OwnedGeometryArray:
    """CPU-only mode: Shapely element-wise binary constructive."""
    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    kwargs = {}
    if grid_size is not None:
        kwargs["grid_size"] = grid_size

    result_arr = getattr(shapely, op)(left_arr, right_arr, **kwargs)
    return from_shapely_geometries(result_arr.tolist())
