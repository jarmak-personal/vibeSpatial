from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "exterior_ring",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("shapely", "constructive", "exterior"),
)
def _exterior_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute exterior ring using Shapely."""
    geoms = owned.to_shapely()
    results = [shapely.get_exterior_ring(g) if g is not None else None for g in geoms]
    return from_shapely_geometries(results)
