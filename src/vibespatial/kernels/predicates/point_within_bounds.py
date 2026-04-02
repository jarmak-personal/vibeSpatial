from __future__ import annotations

from vibespatial.predicates.point_within_bounds_cpu import (
    BoundsSequence,
    NormalizedBoundsInput,
    PointSequence,
    point_within_bounds_cpu,
)
from vibespatial.predicates.point_within_bounds_cpu import (
    evaluate_point_within_bounds as _evaluate_point_within_bounds,
)
from vibespatial.predicates.point_within_bounds_cpu import (
    normalize_right_input as _normalize_right_input,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

__all__ = [
    "NormalizedBoundsInput",
    "_evaluate_point_within_bounds",
    "_normalize_right_input",
    "point_within_bounds",
]


@register_kernel_variant(
    "point_within_bounds",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
    tags=("coarse-filter", "bounds"),
)
def point_within_bounds(
    points: PointSequence,
    polygons_or_bounds: BoundsSequence | PointSequence,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> list[bool | None]:
    return point_within_bounds_cpu(
        points,
        polygons_or_bounds,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
