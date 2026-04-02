from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode


@register_kernel_variant(
    "polygon_intersection",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    tags=("shapely", "constructive", "intersection"),
)
def polygon_intersection_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """CPU fallback: Shapely element-wise polygon intersection."""
    del precision

    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    result_arr = shapely.intersection(left_arr, right_arr)
    return from_shapely_geometries(list(result_arr))
