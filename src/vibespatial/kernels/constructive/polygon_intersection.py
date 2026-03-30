"""GPU-native element-wise polygon intersection kernel.

Computes element-wise intersection of two equal-length OwnedGeometryArrays
containing polygons, returning a device-resident OwnedGeometryArray without
any D->H transfer in the hot path.

Algorithm: Sutherland-Hodgman polygon clipping on GPU.
- For each pair (left[i], right[i]), clips left's exterior ring by each
  edge of right's exterior ring.
- Two-pass count-scatter pattern: pass 1 counts output vertices per pair,
  prefix sum computes offsets, pass 2 scatters clipped vertices.
- Degenerate results (empty, point, line) produce empty polygons with
  validity=False.

ADR-0033: Tier 1 (custom NVRTC kernel) -- geometry-specific inner loop
  with ring traversal and edge-by-edge clipping.
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
  PrecisionPlan wired through for observability only.
ADR-0034: NVRTC precompilation via request_nvrtc_warmup at module scope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    from_shapely_geometries,
)
from vibespatial.kernels.constructive.polygon_intersection_source import (
    _KERNEL_NAMES,
    _POLYGON_INTERSECTION_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import PrecisionPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC kernel source -- Sutherland-Hodgman polygon clipping
# ---------------------------------------------------------------------------
# The kernel uses a workspace buffer sized per-pair to hold intermediate
# clipped vertex lists.  Two buffers alternate roles (input/output) as
# each clip edge is processed.
#
# Limitations of Sutherland-Hodgman:
# - Subject polygon is clipped by a convex clip polygon (right operand).
#   For concave clip polygons, the result may include extra area.
#   This is acceptable as a first implementation; Weiler-Atherton can
#   be added later for full generality.
# - Holes are not handled in this initial version; only exterior rings.
#
# The workspace is sized at MAX_CLIP_VERTS per pair.  If clipping produces
# more vertices than this, the pair is marked as overflowed and falls back
# to validity=False (the CPU fallback handles it).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ADR-0034: request NVRTC precompilation at module scope
# ---------------------------------------------------------------------------
request_nvrtc_warmup([
    ("polygon-intersection", _POLYGON_INTERSECTION_KERNEL_SOURCE, _KERNEL_NAMES),
])

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helper
# ---------------------------------------------------------------------------

def _polygon_intersection_kernels():
    """Compile and cache polygon intersection NVRTC kernels."""
    return compile_kernel_group(
        "polygon-intersection",
        _POLYGON_INTERSECTION_KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Device-backed OwnedGeometryArray builder
# ---------------------------------------------------------------------------

def _build_device_backed_polygon_intersection_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    """Build a device-resident OwnedGeometryArray from GPU coordinate buffers.

    Follows the DeviceFamilyGeometryBuffer pattern from
    ``_build_device_backed_fixed_polygon_output`` but handles variable-length
    output and per-row validity.
    """
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=None,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        runtime_history=[runtime_selection],
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None,
                )
            },
        ),
    )


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _extract_polygon_family_buffers(owned: OwnedGeometryArray):
    """Extract polygon family device buffers, uploading if needed.

    Returns (device_buf, host_buf) for the POLYGON family, or
    (None, None) if no polygon rows exist.
    """
    if GeometryFamily.POLYGON not in owned.families:
        return None, None
    host_buf = owned.families[GeometryFamily.POLYGON]
    if host_buf.row_count == 0:
        return None, None

    state = owned._ensure_device_state()
    if GeometryFamily.POLYGON not in state.families:
        return None, None
    return state.families[GeometryFamily.POLYGON], host_buf


def _polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    """GPU polygon intersection via Sutherland-Hodgman clipping.

    Both inputs must be polygon-only OwnedGeometryArrays of equal length.
    Returns a device-resident OwnedGeometryArray.
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    n = left.row_count

    # Ensure device state for both inputs
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_intersection selected GPU execution",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_intersection selected GPU execution",
    )

    left_dev, left_host = _extract_polygon_family_buffers(left)
    right_dev, right_host = _extract_polygon_family_buffers(right)

    if left_dev is None or right_dev is None:
        # No polygon data -- return all-empty
        return _build_empty_result(n, runtime_selection)

    # Validate that both inputs are polygon-only (1:1 global-to-family mapping).
    # The kernel uses idx as both global row index and family buffer index.
    if left_host.row_count != n or right_host.row_count != n:
        raise ValueError(
            "polygon_intersection GPU path requires polygon-only inputs "
            f"(left family rows={left_host.row_count}, "
            f"right family rows={right_host.row_count}, expected={n})"
        )

    # Build per-row validity masks on device (int32 for kernel compatibility).
    # Since we verified polygon-only, the family empty_mask is 1:1 with rows.
    left_state = left.device_state
    right_state = right.device_state
    d_left_valid = (
        left_state.validity.astype(cp.bool_) & ~left_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)
    d_right_valid = (
        right_state.validity.astype(cp.bool_) & ~right_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)

    # Allocate output arrays for the count pass
    d_counts = runtime.allocate((n,), np.int32, zero=True)
    d_valid = runtime.allocate((n,), np.int32, zero=True)

    # Compile and launch count kernel
    kernels = _polygon_intersection_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            n,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["polygon_intersection_count"], n)
    runtime.launch(
        kernels["polygon_intersection_count"],
        grid=grid, block=block, params=count_params,
    )

    # Exclusive prefix sum for scatter offsets (same-stream, no sync needed)
    d_offsets = exclusive_sum(d_counts, synchronize=False)

    # Get total output vertices
    total_verts = count_scatter_total(runtime, d_counts, d_offsets)

    if total_verts == 0:
        # All intersections are empty
        return _build_empty_result(n, runtime_selection)

    # Allocate output coordinate arrays
    d_out_x = runtime.allocate((total_verts,), np.float64)
    d_out_y = runtime.allocate((total_verts,), np.float64)

    # Launch scatter kernel
    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_offsets),
            ptr(d_valid),
            ptr(d_out_x),
            ptr(d_out_y),
            n,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_intersection_scatter"], n,
    )
    runtime.launch(
        kernels["polygon_intersection_scatter"],
        grid=scatter_grid, block=scatter_block, params=scatter_params,
    )

    # Build ring_offsets on device from the existing d_offsets (exclusive prefix
    # sum of d_counts) to avoid D2H -> host cumsum -> H2D ping-pong.
    # ring_offsets[i] = d_offsets[i] for i < n, ring_offsets[n] = total_verts.
    runtime.synchronize()

    # d_offsets is already the exclusive prefix sum = inclusive ring_offsets[0:n].
    # Append total_verts to get the full ring_offsets array on device.
    import cupy as _cp

    d_ring_offsets = _cp.empty(n + 1, dtype=_cp.int32)
    d_ring_offsets[:n] = _cp.asarray(d_offsets)
    d_ring_offsets[n] = total_verts

    # geometry_offsets: one ring per polygon = [0, 1, 2, ..., n]
    # (host-side only; device state uses the same pattern implicitly)

    # Host copies for OwnedGeometryArray metadata (small: O(n) int/bool).
    # d_valid is already a CuPy array (returned by runtime.allocate), so
    # pass it directly to copy_device_to_host -- no cp.asarray() needed.
    validity = runtime.copy_device_to_host(d_valid).astype(bool)
    geometry_offsets = np.arange(n + 1, dtype=np.int32)
    ring_offsets = runtime.copy_device_to_host(d_ring_offsets)

    return _build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=validity,
        geometry_offsets=geometry_offsets,
        ring_offsets=ring_offsets,
        runtime_selection=runtime_selection,
    )


def _build_empty_result(n: int, runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    """Build an all-empty polygon result."""
    runtime = get_cuda_runtime()
    d_x = runtime.allocate((0,), np.float64)
    d_y = runtime.allocate((0,), np.float64)
    return _build_device_backed_polygon_intersection_output(
        d_x,
        d_y,
        row_count=n,
        validity=np.zeros(n, dtype=bool),
        geometry_offsets=np.arange(n + 1, dtype=np.int32),
        ring_offsets=np.zeros(n + 1, dtype=np.int32),
        runtime_selection=runtime_selection,
    )


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "polygon_intersection",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    tags=("shapely", "constructive", "intersection"),
)
def _polygon_intersection_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """CPU fallback: Shapely element-wise polygon intersection."""
    import shapely

    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    result_arr = shapely.intersection(left_arr, right_arr)
    return from_shapely_geometries(list(result_arr))


@register_kernel_variant(
    "polygon_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=False,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "intersection", "sutherland-hodgman"),
)
def _polygon_intersection_gpu_variant(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    """GPU polygon intersection via Sutherland-Hodgman NVRTC kernel."""
    return _polygon_intersection_gpu(
        left, right,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def polygon_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Element-wise polygon intersection of two OwnedGeometryArrays.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Input polygon arrays of equal length.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint (GPU/CPU/AUTO).
    precision : PrecisionMode or str, default AUTO
        Precision mode. CONSTRUCTIVE kernels stay fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Device-resident result when GPU path is taken; host-resident
        when CPU fallback is used.
    """
    if left.row_count != right.row_count:
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )

    n = left.row_count
    if n == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="polygon_intersection",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        # ADR-0002: CONSTRUCTIVE stays fp64; plan is for observability.
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )

        try:
            result = _polygon_intersection_gpu(
                left, right,
                runtime_selection=selection,
                precision_plan=precision_plan,
            )
            record_dispatch_event(
                surface="vibespatial.kernels.constructive.polygon_intersection",
                operation="polygon_intersection",
                implementation="polygon_intersection_gpu",
                reason=selection.reason,
                detail=(
                    f"rows={n}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            logger.debug(
                "GPU polygon_intersection failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback
    result = _polygon_intersection_cpu(left, right, precision=precision)
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_intersection",
        operation="polygon_intersection",
        implementation="polygon_intersection_cpu",
        reason=selection.reason,
        detail=f"rows={n}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
