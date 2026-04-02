from __future__ import annotations

from shapely.geometry import MultiLineString as ShapelyMultiLineString

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "interior_rings",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("shapely", "constructive", "interiors"),
)
def _interiors_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute interior rings using Shapely."""
    geoms = owned.to_shapely()
    results = []
    for g in geoms:
        if g is None:
            results.append(None)
            continue
        if hasattr(g, "interiors"):
            interior_rings = list(g.interiors)
            if len(interior_rings) == 0:
                results.append(ShapelyMultiLineString())
            else:
                results.append(ShapelyMultiLineString(interior_rings))
        else:
            results.append(None)
    return from_shapely_geometries(results)
