"""GPU-accelerated coordinate reversal for all geometry types.

Reverses the order of coordinates within each ring/part/geometry.
For LineStrings: reverse the coordinate sequence.
For Polygons: reverse each ring's coordinate sequence.

ADR-0033: Tier 1 NVRTC, 1 thread per coordinate.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

from .measurement import _PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: reverse coordinates within spans defined by offsets
# ---------------------------------------------------------------------------

_REVERSE_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void reverse_spans(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ span_offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double center_x, double center_y,
    int total_coords
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_coords) return;

    /* Binary search for which span this coordinate belongs to.
       span_offsets is sorted, so we find the span where
       span_offsets[s] <= i < span_offsets[s+1]. */
    x_out[i] = x_in[i];
    y_out[i] = y_in[i];
}}

extern "C" __global__ void reverse_by_offsets(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int span_count
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = offsets[span];
    const int end = offsets[span + 1];
    const int length = end - start;

    for (int j = 0; j < length; j++) {{
        x_out[start + j] = x_in[end - 1 - j];
        y_out[start + j] = y_in[end - 1 - j];
    }}
}}
"""

_REVERSE_KERNEL_NAMES = ("reverse_spans", "reverse_by_offsets")
_REVERSE_FP64 = _REVERSE_KERNEL_SOURCE.format(compute_type="double")

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("reverse-fp64", _REVERSE_FP64, _REVERSE_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _reverse_family_gpu(runtime, device_buf, family):
    """Reverse coordinates within appropriate spans for one family."""
    coord_count = int(device_buf.x.shape[0])
    if coord_count == 0:
        return device_buf.x, device_buf.y

    kernels = compile_kernel_group("reverse-fp64", _REVERSE_FP64, _REVERSE_KERNEL_NAMES)
    kernel = kernels["reverse_by_offsets"]

    d_x_out = runtime.allocate((coord_count,), np.float64)
    d_y_out = runtime.allocate((coord_count,), np.float64)

    # Determine which offsets define the spans to reverse
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        # Reverse within each ring
        d_offsets = device_buf.ring_offsets
        span_count = int(d_offsets.shape[0]) - 1
    elif family in (GeometryFamily.MULTILINESTRING,):
        # Reverse within each part
        d_offsets = device_buf.part_offsets
        span_count = int(d_offsets.shape[0]) - 1
    else:
        # Reverse within each geometry
        d_offsets = device_buf.geometry_offsets
        span_count = int(d_offsets.shape[0]) - 1

    if span_count <= 0:
        return device_buf.x, device_buf.y

    ptr = runtime.pointer
    params = (
        (ptr(device_buf.x), ptr(device_buf.y), ptr(d_offsets),
         ptr(d_x_out), ptr(d_y_out), span_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, span_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_x_out, d_y_out


@register_kernel_variant(
    "reverse",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "reverse"),
)
def _reverse_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU reverse — returns device-resident OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        d_x_out, d_y_out = _reverse_family_gpu(runtime, device_buf, family)
        new_device_families[family] = DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=device_buf.bounds,  # bounds unchanged by reverse
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=owned.tags.copy(),
        validity=owned.validity.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def reverse_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Reverse coordinate order for all geometries."""
    row_count = owned.row_count
    if row_count == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="reverse",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            return _reverse_gpu(owned)
        except Exception:
            pass

    # CPU fallback: reverse within offset spans
    new_families = {}
    for family, buf in owned.families.items():
        if buf.row_count == 0:
            new_families[family] = buf
            continue

        x_out = buf.x.copy()
        y_out = buf.y.copy()

        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            offsets = buf.ring_offsets
        elif family is GeometryFamily.MULTILINESTRING:
            offsets = buf.part_offsets
        else:
            offsets = buf.geometry_offsets

        if offsets is not None:
            for s in range(len(offsets) - 1):
                start, end = int(offsets[s]), int(offsets[s + 1])
                x_out[start:end] = x_out[start:end][::-1]
                y_out[start:end] = y_out[start:end][::-1]

        new_families[family] = FamilyGeometryBuffer(
            family=family,
            schema=buf.schema,
            row_count=buf.row_count,
            x=x_out,
            y=y_out,
            geometry_offsets=buf.geometry_offsets,
            empty_mask=buf.empty_mask,
            part_offsets=buf.part_offsets,
            ring_offsets=buf.ring_offsets,
            bounds=buf.bounds,
        )

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=owned.tags.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
        families=new_families,
        residency=Residency.HOST,
    )
