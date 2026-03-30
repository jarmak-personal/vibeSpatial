"""GPU-accelerated affine transformations for all geometry types.

Single CUDA kernel applies a 2x3 affine matrix to every coordinate:
    x_out = a*x + b*y + xoff
    y_out = d*x + e*y + yoff

Covers: affine_transform, translate, rotate, scale, skew.
All are specializations of the same matrix multiply.

ADR-0033: Tier 1 NVRTC, 1 thread per coordinate.
ADR-0002: COARSE class — coordinate transform is exact in fp64,
acceptable error budget in fp32 for visualization-grade work.
"""

from __future__ import annotations

import math
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
from vibespatial.cuda.preamble import PRECISION_PREAMBLE
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# NVRTC kernel source
# ---------------------------------------------------------------------------

_AFFINE_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void affine_transform_coords(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double a, double b, double xoff,
    double d, double e, double yoff,
    double center_x, double center_y,
    int coord_count
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= coord_count) return;

    const double xi = x_in[i];
    const double yi = y_in[i];
    x_out[i] = a * xi + b * yi + xoff;
    y_out[i] = d * xi + e * yi + yoff;
}}
"""

_AFFINE_KERNEL_NAMES = ("affine_transform_coords",)
_AFFINE_FP64 = _AFFINE_KERNEL_SOURCE.format(compute_type="double")
_AFFINE_FP32 = _AFFINE_KERNEL_SOURCE.format(compute_type="float")

# Background precompilation
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("affine-transform-fp64", _AFFINE_FP64, _AFFINE_KERNEL_NAMES),
    ("affine-transform-fp32", _AFFINE_FP32, _AFFINE_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Matrix builders for each affine operation
# ---------------------------------------------------------------------------

def _affine_matrix(matrix) -> tuple[float, float, float, float, float, float]:
    """Convert a shapely-compatible affine matrix to (a, b, xoff, d, e, yoff)."""
    if len(matrix) == 6:
        a, b, d, e, xoff, yoff = (float(v) for v in matrix)
    elif len(matrix) == 12:
        a, b, _, d, e, _, _, _, _, xoff, yoff, _ = (float(v) for v in matrix)
    else:
        raise ValueError(f"affine matrix must have 6 or 12 elements, got {len(matrix)}")
    return a, b, xoff, d, e, yoff


def _translate_matrix(
    xoff: float = 0.0, yoff: float = 0.0, zoff: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, float(xoff), 0.0, 1.0, float(yoff))


def _rotate_matrix(
    angle: float,
    origin: str | tuple[float, float] = "center",
    use_radians: bool = False,
) -> tuple[float, float, float, float, float, float]:
    if not use_radians:
        angle = math.radians(angle)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # For non-center origins, rotation is applied at the origin;
    # the caller must compose translate(-ox,-oy) + rotate + translate(ox,oy).
    # When origin="center", the caller computes the centroid and composes.
    return (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0)


def _scale_matrix(
    xfact: float = 1.0,
    yfact: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    return (float(xfact), 0.0, 0.0, 0.0, float(yfact), 0.0)


def _skew_matrix(
    xs: float = 0.0,
    ys: float = 0.0,
    use_radians: bool = False,
) -> tuple[float, float, float, float, float, float]:
    if not use_radians:
        xs = math.radians(xs)
        ys = math.radians(ys)
    return (1.0, math.tan(xs), 0.0, math.tan(ys), 1.0, 0.0)


def _compose_around_origin(
    matrix: tuple[float, float, float, float, float, float],
    ox: float,
    oy: float,
) -> tuple[float, float, float, float, float, float]:
    """Compose translate(-ox,-oy) -> matrix -> translate(ox,oy)."""
    a, b, xoff, d, e, yoff = matrix
    # T(ox,oy) . M . T(-ox,-oy)
    new_xoff = a * (-ox) + b * (-oy) + xoff + ox
    new_yoff = d * (-ox) + e * (-oy) + yoff + oy
    return (a, b, new_xoff, d, e, new_yoff)


# ---------------------------------------------------------------------------
# GPU kernel launcher
# ---------------------------------------------------------------------------

def _launch_affine_family(
    runtime,
    device_buf: DeviceFamilyGeometryBuffer,
    a: float, b: float, xoff: float,
    d: float, e: float, yoff: float,
    compute_type: str = "double",
) -> tuple:
    """Apply affine transform to one family's coordinates on device.

    Returns (d_x_out, d_y_out) device arrays. Caller owns them.
    """
    source = _AFFINE_FP64 if compute_type == "double" else _AFFINE_FP32
    suffix = "fp64" if compute_type == "double" else "fp32"
    kernels = compile_kernel_group(f"affine-transform-{suffix}", source, _AFFINE_KERNEL_NAMES)
    kernel = kernels["affine_transform_coords"]

    coord_count = int(device_buf.x.shape[0])
    if coord_count == 0:
        return device_buf.x, device_buf.y  # share empty arrays

    d_x_out = runtime.allocate((coord_count,), np.float64)
    d_y_out = runtime.allocate((coord_count,), np.float64)

    ptr = runtime.pointer
    params = (
        (ptr(device_buf.x), ptr(device_buf.y),
         ptr(d_x_out), ptr(d_y_out),
         a, b, xoff, d, e, yoff,
         0.0, 0.0,  # center_x, center_y (unused for affine)
         coord_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
         KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
         KERNEL_PARAM_F64, KERNEL_PARAM_F64,
         KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, coord_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_x_out, d_y_out


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "affine_transform",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "affine", "transform"),
)
def _affine_transform_gpu(
    owned: OwnedGeometryArray,
    a: float, b: float, xoff: float,
    d: float, e: float, yoff: float,
) -> OwnedGeometryArray:
    """GPU affine transform — returns device-resident OwnedGeometryArray.

    Applies the 2x3 matrix [[a,b,xoff],[d,e,yoff]] to all coordinates.
    All offset/metadata arrays are shared from the input (zero-copy).
    Only coordinate buffers are new.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    for family, device_buf in d_state.families.items():
        d_x_out, d_y_out = _launch_affine_family(
            runtime, device_buf, a, b, xoff, d, e, yoff,
        )
        # Share all offset/metadata buffers from input — only coords change
        new_device_families[family] = DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=None,  # invalidated by transform
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=owned.tags.copy(),
        validity=owned.validity.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
    )


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "affine_transform",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "constructive", "affine"),
)
def _affine_transform_cpu(
    owned: OwnedGeometryArray,
    a: float, b: float, xoff: float,
    d: float, e: float, yoff: float,
) -> OwnedGeometryArray:
    """CPU affine transform — vectorized NumPy on coordinate buffers."""

    new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, buf in owned.families.items():
        if buf.row_count == 0:
            new_families[family] = buf
            continue
        x_out = a * buf.x + b * buf.y + xoff
        y_out = d * buf.x + e * buf.y + yoff
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
            bounds=None,
        )

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=owned.tags.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
        families=new_families,
        residency=Residency.HOST,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def affine_transform_owned(
    owned: OwnedGeometryArray,
    matrix,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Apply a 2D affine transformation to all geometries.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    matrix : sequence of 6 or 12 floats
        Affine matrix [a, b, d, e, xoff, yoff] (Shapely convention).
    """
    a, b, xoff, d, e, yoff = _affine_matrix(matrix)
    row_count = owned.row_count
    if row_count == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="affine_transform",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _affine_transform_gpu(owned, a, b, xoff, d, e, yoff)
        except Exception:
            pass
        else:
            record_dispatch_event(
                surface="geopandas.array.affine_transform",
                operation="affine_transform",
                implementation="affine_transform_gpu_nvrtc",
                reason=selection.reason,
                detail=f"rows={row_count}",
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _affine_transform_cpu(owned, a, b, xoff, d, e, yoff)
    record_dispatch_event(
        surface="geopandas.array.affine_transform",
        operation="affine_transform",
        implementation="affine_transform_cpu_numpy",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def translate_owned(
    owned: OwnedGeometryArray,
    xoff: float = 0.0,
    yoff: float = 0.0,
    zoff: float = 0.0,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Translate all geometries by (xoff, yoff)."""
    a, b, x, d, e, y = _translate_matrix(xoff, yoff, zoff)
    return affine_transform_owned(owned, [a, b, d, e, x, y], dispatch_mode=dispatch_mode)


def rotate_owned(
    owned: OwnedGeometryArray,
    angle: float,
    origin: str | tuple[float, float] = "center",
    use_radians: bool = False,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Rotate all geometries by angle around origin."""
    a, b, xoff, d, e, yoff = _rotate_matrix(angle, origin="center", use_radians=use_radians)

    if origin == "center":
        # Compute centroid of all coordinates
        all_x = np.concatenate([buf.x for buf in owned.families.values() if buf.row_count > 0])
        all_y = np.concatenate([buf.y for buf in owned.families.values() if buf.row_count > 0])
        ox = float(np.mean(all_x)) if len(all_x) > 0 else 0.0
        oy = float(np.mean(all_y)) if len(all_y) > 0 else 0.0
    elif origin == "centroid":
        all_x = np.concatenate([buf.x for buf in owned.families.values() if buf.row_count > 0])
        all_y = np.concatenate([buf.y for buf in owned.families.values() if buf.row_count > 0])
        ox = float(np.mean(all_x)) if len(all_x) > 0 else 0.0
        oy = float(np.mean(all_y)) if len(all_y) > 0 else 0.0
    elif isinstance(origin, (tuple, list)):
        ox, oy = float(origin[0]), float(origin[1])
    else:
        raise ValueError(f"unsupported origin: {origin!r}")

    a, b, xoff, d, e, yoff = _compose_around_origin((a, b, xoff, d, e, yoff), ox, oy)
    return affine_transform_owned(owned, [a, b, d, e, xoff, yoff], dispatch_mode=dispatch_mode)


def scale_owned(
    owned: OwnedGeometryArray,
    xfact: float = 1.0,
    yfact: float = 1.0,
    zfact: float = 1.0,
    origin: str | tuple[float, float] = "center",
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Scale all geometries by (xfact, yfact) around origin."""
    a, b, xoff, d, e, yoff = _scale_matrix(xfact, yfact)

    if origin == "center" or origin == "centroid":
        all_x = np.concatenate([buf.x for buf in owned.families.values() if buf.row_count > 0])
        all_y = np.concatenate([buf.y for buf in owned.families.values() if buf.row_count > 0])
        ox = float(np.mean(all_x)) if len(all_x) > 0 else 0.0
        oy = float(np.mean(all_y)) if len(all_y) > 0 else 0.0
    elif isinstance(origin, (tuple, list)):
        ox, oy = float(origin[0]), float(origin[1])
    else:
        raise ValueError(f"unsupported origin: {origin!r}")

    a, b, xoff, d, e, yoff = _compose_around_origin((a, b, xoff, d, e, yoff), ox, oy)
    return affine_transform_owned(owned, [a, b, d, e, xoff, yoff], dispatch_mode=dispatch_mode)


def skew_owned(
    owned: OwnedGeometryArray,
    xs: float = 0.0,
    ys: float = 0.0,
    origin: str | tuple[float, float] = "center",
    use_radians: bool = False,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Skew all geometries by (xs, ys) angles around origin."""
    a, b, xoff, d, e, yoff = _skew_matrix(xs, ys, use_radians=use_radians)

    if origin == "center" or origin == "centroid":
        all_x = np.concatenate([buf.x for buf in owned.families.values() if buf.row_count > 0])
        all_y = np.concatenate([buf.y for buf in owned.families.values() if buf.row_count > 0])
        ox = float(np.mean(all_x)) if len(all_x) > 0 else 0.0
        oy = float(np.mean(all_y)) if len(all_y) > 0 else 0.0
    elif isinstance(origin, (tuple, list)):
        ox, oy = float(origin[0]), float(origin[1])
    else:
        raise ValueError(f"unsupported origin: {origin!r}")

    a, b, xoff, d, e, yoff = _compose_around_origin((a, b, xoff, d, e, yoff), ox, oy)
    return affine_transform_owned(owned, [a, b, d, e, xoff, yoff], dispatch_mode=dispatch_mode)
