from __future__ import annotations

import shapely

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "geometry_simplify",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "simplify"),
)
def _simplify_cpu(owned, tolerance, preserve_topology=True):
    """CPU simplify via Shapely's optimized C implementation."""
    geoms = owned.to_shapely()
    results = shapely.simplify(geoms, tolerance, preserve_topology=preserve_topology)
    return from_shapely_geometries(list(results))
