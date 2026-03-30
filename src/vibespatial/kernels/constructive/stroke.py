from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np  # hygiene:ok — used for type alias and array construction, not device computation

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    from vibespatial.constructive.stroke import BufferKernelResult, OffsetCurveKernelResult

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
    from vibespatial.constructive.stroke import point_buffer_owned  # lazy

    return point_buffer_owned(values, distance, quad_segs=quad_segs)


@register_kernel_variant(
    "point_buffer",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("point",),
    execution_modes=(ExecutionMode.GPU,),
    preferred_residency=Residency.DEVICE,
    supports_mixed=False,
    tags=("constructive", "stroke", "buffer", "owned", "cuda-python"),
)
def point_buffer_kernel_gpu(
    values: StrokeInput,
    distance,
    *,
    quad_segs: int = 16,
) -> BufferKernelResult:
    # NOTE: This registry path always starts from Shapely objects and pays
    # the full Shapely -> OwnedGeometryArray -> Device transfer.  Callers
    # that already hold a device-resident OwnedGeometryArray should call
    # point_buffer_owned_array() directly to skip the H->D round-trip.
    from vibespatial.constructive.point import point_buffer_owned_array  # lazy
    from vibespatial.constructive.stroke import BufferKernelResult  # lazy
    from vibespatial.geometry.owned import from_shapely_geometries

    geometries = np.asarray(values, dtype=object)  # hygiene:ok — host Shapely objects
    owned = from_shapely_geometries(list(geometries))
    result = point_buffer_owned_array(owned, distance, quad_segs=quad_segs, dispatch_mode=ExecutionMode.GPU)
    row_count = result.row_count
    return BufferKernelResult(
        geometries=None,
        row_count=row_count,
        fast_rows=np.arange(row_count, dtype=np.int32),
        fallback_rows=np.asarray([], dtype=np.int32),  # hygiene:ok — host index array
        owned_result=result,
    )


@register_kernel_variant(
    "polygon_buffer",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon",),
    execution_modes=(ExecutionMode.GPU,),
    preferred_residency=Residency.DEVICE,
    supports_mixed=False,
    tags=("constructive", "buffer", "owned", "cuda-python", "count-scatter"),
)
def polygon_buffer_kernel_gpu(
    values: StrokeInput,
    distance,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> BufferKernelResult:
    from vibespatial.constructive.stroke import point_buffer_owned  # lazy

    # Dispatch handled by stroke_kernels.evaluate_geopandas_buffer
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
    from vibespatial.constructive.stroke import offset_curve_owned  # lazy

    return offset_curve_owned(
        values,
        distance,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
