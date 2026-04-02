"""CPU-only helpers for constructive properties surfaces."""

from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "get_geometry",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "polygon", "linestring",
        "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "get_geometry"),
)
def get_geometry_cpu(
    owned: OwnedGeometryArray,
    index: int | object,
) -> OwnedGeometryArray:
    """Extract i-th sub-geometry using Shapely as the reference implementation."""
    geoms = owned.to_shapely()
    results = shapely.get_geometry(geoms, index=index)
    return from_shapely_geometries(list(results))
