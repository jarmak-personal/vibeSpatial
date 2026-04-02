from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "polygon_difference",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("shapely", "constructive"),
)
def polygon_difference_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """CPU fallback: Shapely element-wise polygon difference."""
    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    result_arr = shapely.difference(left_arr, right_arr)
    return from_shapely_geometries(list(result_arr))
