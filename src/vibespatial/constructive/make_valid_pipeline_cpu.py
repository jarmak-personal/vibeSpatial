"""CPU-only helpers for the make_valid pipeline."""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency


@register_kernel_variant(
    "make_valid",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
    supports_mixed=True,
    tags=("shapely", "constructive", "make_valid"),
)
def make_valid_cpu_repair(geometries, repaired_rows, *, method, keep_collapsed):
    """CPU-only mode: repair via shapely.make_valid on invalid subset."""
    result = geometries.copy()
    if repaired_rows.size:
        result[repaired_rows] = shapely.make_valid(
            geometries[repaired_rows],
            method=method,
            keep_collapsed=keep_collapsed,
        )
    return result


def make_valid_cpu_is_valid(geometries) -> np.ndarray:
    """Evaluate OGC validity via Shapely on host geometry arrays."""
    return np.asarray(shapely.is_valid(geometries), dtype=bool)


def make_valid_cpu_baseline(geometries, *, method: str, keep_collapsed: bool):
    """Run the Shapely benchmark baseline for make_valid."""
    return shapely.make_valid(geometries, method=method, keep_collapsed=keep_collapsed)


_OWNED_MAKE_VALID_TYPE_IDS = frozenset({0, 1, 3, 4, 5, 6})


def values_support_owned_make_valid(values) -> bool:
    geometries = np.asarray(values, dtype=object)
    if geometries.size == 0:
        return True
    type_ids = shapely.get_type_id(geometries)
    present_type_ids = type_ids[type_ids >= 0]
    if present_type_ids.size == 0:
        return True
    return bool(np.isin(present_type_ids, list(_OWNED_MAKE_VALID_TYPE_IDS)).all())


def build_make_valid_warmup_owned():
    """Build a representative device-resident warmup dataset."""
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    square = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    warmup_geoms = [bowtie] * 100 + [square] * 100
    return from_shapely_geometries(warmup_geoms, residency=Residency.DEVICE)
