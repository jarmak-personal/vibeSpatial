from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "set_precision",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "set_precision"),
)
def _set_precision_cpu(owned: OwnedGeometryArray, grid_size: float, mode: str):
    """CPU set_precision via Shapely."""
    geoms = owned.to_shapely()
    geom_array = np.asarray(geoms, dtype=object)
    results = shapely.set_precision(geom_array, grid_size=grid_size, mode=mode)
    return from_shapely_geometries(list(results))
