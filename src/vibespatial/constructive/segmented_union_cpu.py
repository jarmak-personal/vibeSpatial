from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

from vibespatial.geometry.host_bridge import owned_to_shapely
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

_EMPTY_POLYGON = Polygon()
_EMPTY_OWNED: OwnedGeometryArray | None = None


def get_empty_owned() -> OwnedGeometryArray:
    """Lazily create and cache the empty-polygon sentinel."""
    global _EMPTY_OWNED
    if _EMPTY_OWNED is None:
        _EMPTY_OWNED = from_shapely_geometries([Polygon()])
    return _EMPTY_OWNED


def segmented_union_pair_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Union a single pair on host for tree-reduce fallback."""
    left_geoms = owned_to_shapely(left)
    right_geoms = owned_to_shapely(right)
    left_geom = (
        left_geoms[0]
        if left_geoms.size > 0 and left_geoms[0] is not None
        else _EMPTY_POLYGON
    )
    right_geom = (
        right_geoms[0]
        if right_geoms.size > 0 and right_geoms[0] is not None
        else _EMPTY_POLYGON
    )

    merged = shapely.union(left_geom, right_geom)
    if merged is not None and not shapely.is_valid(merged):
        merged = shapely.make_valid(merged)
    return from_shapely_geometries([merged if merged is not None else _EMPTY_POLYGON])


@register_kernel_variant(
    "segmented_union_all",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("constructive", "segmented-union", "grouped"),
)
def segmented_union_cpu_variant(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """CPU variant: iterate groups and call shapely.union_all per group."""
    del dispatch_mode, precision

    group_offsets = np.asarray(group_offsets, dtype=np.int64)
    n_groups = len(group_offsets) - 1
    return segmented_union_cpu(geometries, group_offsets, n_groups=n_groups)


def segmented_union_cpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
) -> OwnedGeometryArray:
    """CPU implementation: per-group shapely.union_all."""
    all_geoms = owned_to_shapely(geometries)

    results: list[object] = []
    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        group_size = end - start

        if group_size == 0:
            results.append(_EMPTY_POLYGON)
        elif group_size == 1:
            geom = all_geoms[start]
            results.append(geom if geom is not None else _EMPTY_POLYGON)
        else:
            block = all_geoms[start:end]
            valid = block[block != np.array(None)]
            if len(valid) == 0:
                results.append(_EMPTY_POLYGON)
            elif len(valid) == 1:
                results.append(valid[0])
            else:
                merged = shapely.union_all(valid)
                if merged is not None and not shapely.is_valid(merged):
                    merged = shapely.make_valid(merged)
                results.append(merged if merged is not None else _EMPTY_POLYGON)

    return from_shapely_geometries(results)
