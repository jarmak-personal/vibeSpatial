"""Host-side shared_paths assembly and CPU variant registration."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, MultiLineString

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass


def empty_shared_paths_result() -> GeometryCollection:
    """Return an empty shared_paths GeometryCollection."""
    return GeometryCollection([MultiLineString(), MultiLineString()])


@register_kernel_variant(
    "shared_paths",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "multilinestring"),
    supports_mixed=True,
    tags=("shapely", "constructive", "shared_paths"),
)
def shared_paths_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """CPU shared_paths via Shapely."""
    left_shapely = np.asarray(left.to_shapely(), dtype=object)
    right_shapely = np.asarray(right.to_shapely(), dtype=object)
    results = np.empty(len(left_shapely), dtype=object)

    for i, (left_geom, right_geom) in enumerate(zip(left_shapely, right_shapely, strict=False)):
        if left_geom is None or right_geom is None:
            results[i] = None
            continue
        try:
            result = shapely.shared_paths(left_geom, right_geom)
            results[i] = result if result is not None else empty_shared_paths_result()
        except Exception:
            results[i] = empty_shared_paths_result()

    return results
