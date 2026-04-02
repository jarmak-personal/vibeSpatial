from __future__ import annotations

import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


def _minimum_rotated_rectangle_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU minimum rotated rectangle via Shapely oriented_envelope."""
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])
    geoms = owned.to_shapely()
    result_geoms = shapely.oriented_envelope(geoms)
    return from_shapely_geometries(result_geoms)


@register_kernel_variant(
    "minimum_rotated_rectangle",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "minimum_rotated_rectangle"),
)
def _min_rect_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU minimum rotated rectangle using Shapely oriented_envelope."""
    return _minimum_rotated_rectangle_cpu(owned)
