from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "remove_repeated_points",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "multilinestring",
        "polygon", "multipolygon", "multipoint",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "remove_repeated_points"),
)
def _remove_repeated_points_cpu(owned: OwnedGeometryArray, tolerance: float):
    """CPU remove_repeated_points via Shapely."""
    geoms = owned.to_shapely()
    geom_array = np.asarray(geoms, dtype=object)
    results = shapely.remove_repeated_points(geom_array, tolerance=tolerance)
    return from_shapely_geometries(list(results))
