"""GPU-accelerated envelope (bounding box) computation.

Computes per-geometry bounds and emits a 5-vertex closed polygon per row:
    (xmin,ymin) -> (xmax,ymin) -> (xmax,ymax) -> (xmin,ymax) -> (xmin,ymin)

Output is always Polygon family. Empty/null geometries produce empty Polygons.

ADR-0033: Tier 1 NVRTC, 1 thread per geometry.
ADR-0002: COARSE class — bounds are exact in any precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, combined_residency

if TYPE_CHECKING:
    pass

from vibespatial.constructive.envelope_kernels import (
    _ENVELOPE_FP32,
    _ENVELOPE_FP64,
    _ENVELOPE_KERNEL_NAMES,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup([
    ("envelope-fp64", _ENVELOPE_FP64, _ENVELOPE_KERNEL_NAMES),
    ("envelope-fp32", _ENVELOPE_FP32, _ENVELOPE_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------


def _build_device_boxes_from_bounds(
    device_bounds,
    *,
    row_count: int | None = None,
) -> OwnedGeometryArray:
    """Build device-resident rectangle polygons from device bounds."""
    runtime = get_cuda_runtime()
    if row_count is None:
        row_count = int(device_bounds.shape[0])

    d_x_out = runtime.allocate((row_count * 5,), np.float64)
    d_y_out = runtime.allocate((row_count * 5,), np.float64)

    kernels = compile_kernel_group("envelope-fp64", _ENVELOPE_FP64, _ENVELOPE_KERNEL_NAMES)
    kernel = kernels["envelope_from_bounds"]

    ptr = runtime.pointer
    params = (
        (ptr(device_bounds), ptr(d_x_out), ptr(d_y_out), 0.0, 0.0, row_count),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    from vibespatial.constructive.point import _build_device_backed_polygon_output

    return _build_device_backed_polygon_output(
        d_x_out,
        d_y_out,
        row_count=row_count,
        bounds=None,
        verts_per_ring=5,
    )


@register_kernel_variant(
    "envelope",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "envelope"),
)
def _envelope_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU envelope — returns device-resident Polygon OwnedGeometryArray."""
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    row_count = owned.row_count

    # Compute bounds on GPU — this populates device_state.row_bounds on device.
    # Keep the per-row bounds on device; this kernel consumes row_bounds
    # directly and does not need a host-materialized copy.
    compute_geometry_bounds_device(owned)

    # Read the cached device bounds directly — no re-upload needed.
    d_state = owned.device_state
    d_bounds = d_state.row_bounds  # (N, 4) device array, already on device
    return _build_device_boxes_from_bounds(d_bounds, row_count=row_count)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "envelope",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "envelope"),
)
def _envelope_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute envelope via vectorized bounds on host."""
    from vibespatial.geometry.buffers import get_geometry_buffer_schema
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    bounds = compute_geometry_bounds(owned)
    n = owned.row_count

    x_out = np.empty(n * 5, dtype=np.float64)
    y_out = np.empty(n * 5, dtype=np.float64)
    x_out[0::5] = bounds[:, 0]  # xmin
    x_out[1::5] = bounds[:, 2]  # xmax
    x_out[2::5] = bounds[:, 2]  # xmax
    x_out[3::5] = bounds[:, 0]  # xmin
    x_out[4::5] = bounds[:, 0]  # xmin
    y_out[0::5] = bounds[:, 1]  # ymin
    y_out[1::5] = bounds[:, 1]  # ymin
    y_out[2::5] = bounds[:, 3]  # ymax
    y_out[3::5] = bounds[:, 3]  # ymax
    y_out[4::5] = bounds[:, 1]  # ymin

    geometry_offsets = np.arange(n + 1, dtype=np.int32)
    ring_offsets = np.arange(0, (n + 1) * 5, 5, dtype=np.int32)
    empty_mask = np.zeros(n, dtype=bool)
    validity = owned.validity.copy()
    tags = np.full(n, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(n, dtype=np.int32)

    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=n,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=bounds,
    )

    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.HOST,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def envelope_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the envelope (bounding box) for each geometry.

    Returns an OwnedGeometryArray of Polygon geometries.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries

        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="envelope",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=combined_residency(owned),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _envelope_gpu(owned)
        record_dispatch_event(
            surface="geopandas.array.envelope",
            operation="envelope",
            implementation="envelope_gpu_nvrtc",
            reason=selection.reason,
            detail=(
                f"rows={row_count}, "
                f"plan={precision_plan.compute_precision.value}, "
                f"actual=fp64 (COARSE, bounds-exact)"
            ),
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    result = _envelope_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.envelope",
        operation="envelope",
        implementation="envelope_cpu_bounds",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def envelope_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
):
    """Return envelope output as a private native constructive carrier."""
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = envelope_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="envelope",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
    )
