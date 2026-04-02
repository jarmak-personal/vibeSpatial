from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "line_merge",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "multilinestring"),
    supports_mixed=True,
    tags=("shapely", "constructive", "line_merge"),
)
def _line_merge_cpu(
    owned: OwnedGeometryArray,
    *,
    directed: bool = False,
) -> OwnedGeometryArray:
    """Compute line_merge via Shapely."""
    geoms = owned.to_shapely()
    results = [
        shapely.line_merge(g, directed=directed) if g is not None else None
        for g in geoms
    ]
    return from_shapely_geometries(results)
