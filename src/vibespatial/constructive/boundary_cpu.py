from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "boundary",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "polygon", "linestring",
        "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "boundary"),
)
def _boundary_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute boundary of each geometry using Shapely."""
    geoms = owned.to_shapely()
    results = [shapely.boundary(g) if g is not None else None for g in geoms]
    return from_shapely_geometries(results)
