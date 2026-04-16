"""GPU-accelerated polygon orientation (ring winding order).

Ensures exterior rings are counter-clockwise and interior rings are clockwise
(or vice versa when exterior_cw=True). Uses shoelace signed area to detect
current orientation and reverses rings with wrong winding.

ADR-0033: Tier 1 NVRTC, 1 thread per ring.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.orient_kernels import (
    _ORIENT_FP64,
    _ORIENT_KERNEL_NAMES,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    forward_result_metadata,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

request_nvrtc_warmup([
    ("orient-rings-fp64", _ORIENT_FP64, _ORIENT_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _build_is_exterior(device_buf, family):
    """Build a device int32 array marking exterior rings (1) vs interior (0).

    For Polygon:      geometry_offsets indexes into rings directly.
                      Ring geometry_offsets[g] is exterior for each geometry g.
    For MultiPolygon: geometry_offsets -> parts, part_offsets -> rings.
                      Ring part_offsets[p] is exterior for each part p.

    All operations stay on-device (no D->H->D round-trip).
    """
    ring_count = int(device_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        return cp.empty(0, dtype=cp.int32)

    if family is GeometryFamily.POLYGON:
        # geometry_offsets[g] = index of first ring (exterior) of geometry g
        # cp.asarray on an existing device array is a zero-copy no-op.
        d_offsets = cp.asarray(device_buf.geometry_offsets)
    elif family is GeometryFamily.MULTIPOLYGON:
        # part_offsets[p] = index of first ring (exterior) of part p
        d_offsets = cp.asarray(device_buf.part_offsets)
    else:
        raise ValueError(f"orient is only meaningful for polygon families, got {family}")

    d_is_exterior = cp.zeros(ring_count, dtype=cp.int32)
    # Mark the first ring of each geometry (Polygon) or part (MultiPolygon)
    # as exterior. The sentinel value at d_offsets[-1] equals ring_count,
    # so we exclude it with [:-1].
    d_exterior_indices = d_offsets[:-1]
    # Guard against out-of-range indices (empty geometries may have
    # duplicate offsets at the boundary).
    d_valid = d_exterior_indices < ring_count
    d_is_exterior[d_exterior_indices[d_valid]] = 1
    return d_is_exterior


def _orient_family_gpu(runtime, device_buf, family, exterior_cw):
    """Orient rings of one polygon-type family on the GPU."""
    coord_count = int(device_buf.x.shape[0])
    if coord_count == 0:
        return device_buf.x, device_buf.y

    ring_count = int(device_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        return device_buf.x, device_buf.y

    kernels = compile_kernel_group(
        "orient-rings-fp64", _ORIENT_FP64, _ORIENT_KERNEL_NAMES,
    )
    kernel = kernels["orient_rings"]

    d_x_out = runtime.allocate((coord_count,), np.float64)
    d_y_out = runtime.allocate((coord_count,), np.float64)

    d_is_exterior = _build_is_exterior(device_buf, family)

    exterior_ccw = 0 if exterior_cw else 1

    ptr = runtime.pointer
    params = (
        (
            ptr(device_buf.x),
            ptr(device_buf.y),
            ptr(device_buf.ring_offsets),
            ptr(d_is_exterior),
            ptr(d_x_out),
            ptr(d_y_out),
            exterior_ccw,
            ring_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, ring_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_x_out, d_y_out


@register_kernel_variant(
    "orient",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "orient"),
)
def _orient_gpu(
    owned: OwnedGeometryArray,
    *,
    exterior_cw: bool = False,
) -> OwnedGeometryArray:
    """GPU orient -- returns device-resident OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            d_x_out, d_y_out = _orient_family_gpu(
                runtime, device_buf, family, exterior_cw,
            )
        else:
            # Non-polygon families pass through unchanged
            d_x_out = device_buf.x
            d_y_out = device_buf.y

        new_device_families[family] = DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=device_buf.bounds,  # bounds unchanged by orientation
        )

    tags, validity, family_row_offsets = forward_result_metadata(owned)

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# CPU implementation (correct exterior/interior handling)
# ---------------------------------------------------------------------------

def _shoelace_area_2x(x, y, start, end):
    """Compute 2x signed area of a ring. Positive = CCW."""
    xs = x[start:end]
    ys = y[start:end]
    return float(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]))


def _orient_polygon_cpu(buf, exterior_cw):
    """Orient a Polygon family buffer. Returns new x, y arrays."""
    x_out = buf.x.copy()
    y_out = buf.y.copy()

    for g in range(buf.row_count):
        ring_start_idx = int(buf.geometry_offsets[g])
        ring_end_idx = int(buf.geometry_offsets[g + 1])

        for ring_idx, r in enumerate(range(ring_start_idx, ring_end_idx)):
            coord_start = int(buf.ring_offsets[r])
            coord_end = int(buf.ring_offsets[r + 1])

            area2 = _shoelace_area_2x(buf.x, buf.y, coord_start, coord_end)
            is_exterior = (ring_idx == 0)

            if is_exterior:
                # Exterior: should be CW if exterior_cw, CCW otherwise
                want_positive = not exterior_cw  # positive area = CCW
                if (area2 > 0) != want_positive:
                    x_out[coord_start:coord_end] = x_out[coord_start:coord_end][::-1]
                    y_out[coord_start:coord_end] = y_out[coord_start:coord_end][::-1]
            else:
                # Interior (holes): opposite of exterior
                want_positive = exterior_cw  # holes are opposite of exterior
                if (area2 > 0) != want_positive:
                    x_out[coord_start:coord_end] = x_out[coord_start:coord_end][::-1]
                    y_out[coord_start:coord_end] = y_out[coord_start:coord_end][::-1]

    return x_out, y_out


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def orient_owned(
    owned: OwnedGeometryArray,
    *,
    exterior_cw: bool = False,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Orient polygon rings to specified winding order.

    Parameters
    ----------
    exterior_cw : bool
        If True, exterior rings are clockwise. Default False (CCW).
    """
    row_count = owned.row_count
    if row_count == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="orient",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        current_residency=owned.residency,
    )

    if selection.selected is ExecutionMode.GPU:
        return _orient_gpu(owned, exterior_cw=exterior_cw)

    # CPU fallback (correct exterior/interior handling)
    new_families = {}
    for family, buf in owned.families.items():
        if buf.row_count == 0:
            new_families[family] = buf
            continue

        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            x_out, y_out = _orient_polygon_cpu(buf, exterior_cw)
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
        else:
            new_families[family] = buf

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=owned.tags.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
        families=new_families,
        residency=Residency.HOST,
    )
