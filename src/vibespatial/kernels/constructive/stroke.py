from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vibespatial.kernel_registry import register_kernel_variant
from vibespatial.precision import KernelClass
from vibespatial.runtime import ExecutionMode
from vibespatial.stroke_kernels import BufferKernelResult, OffsetCurveKernelResult, offset_curve_owned, point_buffer_owned


StrokeInput = Sequence[object | None] | np.ndarray | object


@register_kernel_variant(
    "point_buffer",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("point",),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
    tags=("constructive", "stroke", "buffer", "owned"),
)
def point_buffer_kernel(
    values: StrokeInput,
    distance,
    *,
    quad_segs: int = 16,
) -> BufferKernelResult:
    return point_buffer_owned(values, distance, quad_segs=quad_segs)


@register_kernel_variant(
    "offset_curve",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("linestring",),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
    tags=("constructive", "stroke", "offset-curve", "owned"),
)
def offset_curve_kernel(
    values: StrokeInput,
    distance,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OffsetCurveKernelResult:
    return offset_curve_owned(
        values,
        distance,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
