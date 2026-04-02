from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


@register_kernel_variant(
    "snap",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "multilinestring",
        "polygon", "multipolygon", "multipoint",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "snap"),
)
def _snap_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> np.ndarray:
    """CPU snap via Shapely."""
    left_geoms = np.asarray(left.to_shapely(), dtype=object)
    right_geoms = np.asarray(right.to_shapely(), dtype=object)
    return shapely.snap(left_geoms, right_geoms, tolerance=tolerance)
