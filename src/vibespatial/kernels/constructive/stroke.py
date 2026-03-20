from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vibespatial.constructive.stroke import (
    BufferKernelResult,
    OffsetCurveKernelResult,
    offset_curve_owned,
    point_buffer_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

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
    from vibespatial.constructive.point import point_buffer_owned_array
    from vibespatial.geometry.owned import from_shapely_geometries

    geometries = np.asarray(values, dtype=object)
    owned = from_shapely_geometries(list(geometries))
    result = point_buffer_owned_array(owned, distance, quad_segs=quad_segs, dispatch_mode=ExecutionMode.GPU)
    # Convert back to BufferKernelResult for API compatibility
    shapely_geoms = np.asarray(result.to_shapely(), dtype=object)
    return BufferKernelResult(
        geometries=shapely_geoms,
        row_count=len(shapely_geoms),
        fast_rows=np.arange(len(shapely_geoms), dtype=np.int32),
        fallback_rows=np.asarray([], dtype=np.int32),
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
    return offset_curve_owned(
        values,
        distance,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
