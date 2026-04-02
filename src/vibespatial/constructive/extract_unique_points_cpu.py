from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "extract_unique_points",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "extract_unique_points"),
)
def _extract_unique_points_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute extract_unique_points via Shapely."""
    geoms = owned.to_shapely()
    results = [
        shapely.extract_unique_points(g) if g is not None else None
        for g in geoms
    ]
    return from_shapely_geometries(results)
