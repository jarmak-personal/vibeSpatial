from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "normalize",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
    supports_mixed=True,
    tags=("shapely",),
)
def _normalize_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU fallback: Shapely normalize."""
    record_dispatch_event(
        surface="normalize",
        operation="normalize",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={owned.row_count}",
        selected=ExecutionMode.CPU,
    )
    geoms = owned.to_shapely()
    result = shapely.normalize(np.asarray(geoms, dtype=object))
    return from_shapely_geometries(result.tolist())
