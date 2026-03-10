from __future__ import annotations

from vibespatial.kernel_registry import register_kernel_variant
from vibespatial.precision import KernelClass, PrecisionMode, normalize_precision_mode
from vibespatial.runtime import ExecutionMode


@register_kernel_variant(
    "point_bounds",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=('point', 'polygon'),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
)
def point_bounds(
    left,
    right,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
):
    """Stub for the point bounds kernel on point/polygon inputs."""
    del dispatch_mode
    normalize_precision_mode(precision)
    raise NotImplementedError("point_bounds kernel scaffold is not implemented yet")
