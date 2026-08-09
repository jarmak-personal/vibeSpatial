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

from vibespatial.constructive.polygon_intersection_cpu import (
    polygon_intersection_cpu as _polygon_intersection_cpu,
)
from vibespatial.constructive.polygon_intersection_output import (
    build_device_backed_polygon_intersection_output,
    build_empty_device_backed_polygon_intersection_output,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.kernels.constructive.polygon_intersection_source import (
    _KERNEL_NAMES,
    _MAX_CLIP_VERTS,  # Re-exported for overlay/gpu.py and binary_constructive.py.
    _POLYGON_INTERSECTION_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_pairwise_product_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
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
# The workspace is sized at MAX_CLIP_VERTS per pair. Rows with overflow,
# lower-dimensional output, or numerically uncertain source incidences expose
# ``False`` in ``_polygon_intersection_sh_supported``. The native capacity
# partitioner routes only those rows to exact row-isolated topology.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ADR-0034: request NVRTC precompilation at module scope
# ---------------------------------------------------------------------------
request_nvrtc_warmup(
    [
        ("polygon-intersection", _POLYGON_INTERSECTION_KERNEL_SOURCE, _KERNEL_NAMES),
    ]
)

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


def polygon_intersection_sh_eligible_mask(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
):
    """Return a device mask for rows admissible to the SH intersection kernel.

    This is a native row-indirected admissibility carrier: indexed views keep
    their device family-row mapping, and no host metadata or physical row copy is
    required before deciding which rows can use Sutherland-Hodgman. The mask is
    deliberately convex-convex: intersecting a concave simple polygon with a
    convex clip can require multipart output, which this single-ring SH kernel
    cannot represent.
    """
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        import cupy as cp

        return cp.zeros(0, dtype=cp.bool_)

    import cupy as cp

    if left.residency is not Residency.DEVICE:
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_intersection SH eligibility selected GPU execution",
        )
    if right.residency is not Residency.DEVICE:
        right.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_intersection SH eligibility selected GPU execution",
        )

    left_dev, left_state = _extract_polygon_family_buffers(left)
    right_dev, right_state = _extract_polygon_family_buffers(right)
    n = int(left.row_count)
    if (
        left_dev is None
        or right_dev is None
        or left_dev.ring_offsets is None
        or right_dev.ring_offsets is None
    ):
        return cp.zeros(n, dtype=cp.bool_)

    left_polygon_rows = int(left_dev.geometry_offsets.size) - 1
    right_polygon_rows = int(right_dev.geometry_offsets.size) - 1
    if left_polygon_rows <= 0 or right_polygon_rows <= 0:
        return cp.zeros(n, dtype=cp.bool_)

    runtime = get_cuda_runtime()
    d_eligible = runtime.allocate((n,), cp.bool_, zero=True)
    kernels = _polygon_intersection_kernels()
    ptr = runtime.pointer
    d_left_validity = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)
    d_left_family_rows = cp.asarray(left_state.family_row_offsets, dtype=cp.int32)
    d_right_validity = cp.asarray(right_state.validity, dtype=cp.bool_)
    d_right_tags = cp.asarray(right_state.tags, dtype=cp.int8)
    d_right_family_rows = cp.asarray(right_state.family_row_offsets, dtype=cp.int32)
    params = (
        (
            ptr(d_left_validity),
            ptr(d_left_tags),
            ptr(d_left_family_rows),
            ptr(d_right_validity),
            ptr(d_right_tags),
            ptr(d_right_family_rows),
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(left_dev.empty_mask),
            left_polygon_rows,
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(right_dev.empty_mask),
            right_polygon_rows,
            int(FAMILY_TAGS[GeometryFamily.POLYGON]),
            n,
            ptr(d_eligible),
        ),
        (
            # Logical row metadata.
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            # Left polygon buffers and physical row count.
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            # Right polygon buffers and physical row count.
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            # Family tag, logical row count, and output.
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        ),
    )
    grid, block = runtime.launch_config(kernels["polygon_intersection_sh_eligible"], n)
    runtime.launch(
        kernels["polygon_intersection_sh_eligible"],
        grid=grid,
        block=block,
        params=params,
    )
    return d_eligible


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------


def _extract_polygon_family_buffers(owned: OwnedGeometryArray):
    """Extract polygon family device buffers without flattening row views."""
    state = owned._ensure_device_state(preserve_indexed_view=True)
    device_buf = state.families.get(GeometryFamily.POLYGON)
    if device_buf is None:
        return None, state
    if int(device_buf.geometry_offsets.size) <= 1:
        return None, state
    return device_buf, state


def _polygon_intersection_input_coordinate_capacity(
    owned: OwnedGeometryArray,
    device_buffer,
    row_count: int,
) -> int:
    """Bound logical input coordinates without reading device offsets."""
    fixed_size = getattr(device_buffer, "fixed_size", None)
    fixed_width = None if fixed_size is None else fixed_size.coord_count_per_row
    if fixed_width is not None:
        return int(row_count) * int(fixed_width)
    if not owned.is_indexed_view and int(device_buffer.geometry_offsets.size) - 1 == int(row_count):
        return int(device_buffer.x.size)
    return int(row_count) * int(_MAX_CLIP_VERTS)


def _polygon_intersection_vertex_capacity(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    left_buffer,
    right_buffer,
) -> int:
    """Return the SH output capacity from physical input shape proofs."""
    row_count = int(left.row_count)
    workspace_bound = row_count * (int(_MAX_CLIP_VERTS) + 1)
    input_bound = (
        _polygon_intersection_input_coordinate_capacity(
            left,
            left_buffer,
            row_count,
        )
        + _polygon_intersection_input_coordinate_capacity(
            right,
            right_buffer,
            row_count,
        )
        + row_count
    )
    return min(workspace_bound, input_bound)


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

    # Ensure device state for both inputs without flattening an existing
    # row-indirected carrier.
    if left.residency is not Residency.DEVICE:
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_intersection selected GPU execution",
        )
    else:
        left._ensure_device_state(preserve_indexed_view=True)
    if right.residency is not Residency.DEVICE:
        right.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_intersection selected GPU execution",
        )
    else:
        right._ensure_device_state(preserve_indexed_view=True)

    left_dev, left_state = _extract_polygon_family_buffers(left)
    right_dev, right_state = _extract_polygon_family_buffers(right)

    if left_dev is None or right_dev is None:
        # No polygon data -- return all-empty
        return _build_empty_result(n, runtime_selection)

    # Build per-row validity masks on device (int32 for kernel compatibility).
    # Logical rows may be row-indirected into compact source family buffers.
    left_polygon_rows = int(left_dev.geometry_offsets.size) - 1
    right_polygon_rows = int(right_dev.geometry_offsets.size) - 1
    if left_polygon_rows <= 0 or right_polygon_rows <= 0:
        return _build_empty_result(n, runtime_selection)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_family_rows = cp.asarray(left_state.family_row_offsets, dtype=cp.int32)
    d_right_family_rows = cp.asarray(right_state.family_row_offsets, dtype=cp.int32)
    d_left_safe_rows = cp.clip(
        d_left_family_rows,
        cp.int32(0),
        cp.int32(left_polygon_rows - 1),
    ).astype(cp.int64, copy=False)
    d_right_safe_rows = cp.clip(
        d_right_family_rows,
        cp.int32(0),
        cp.int32(right_polygon_rows - 1),
    ).astype(cp.int64, copy=False)
    d_left_family_valid = (
        (cp.asarray(left_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_left_family_rows >= 0)
        & (d_left_family_rows < left_polygon_rows)
    )
    d_right_family_valid = (
        (cp.asarray(right_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_right_family_rows >= 0)
        & (d_right_family_rows < right_polygon_rows)
    )
    d_left_valid = (
        cp.asarray(left_state.validity, dtype=cp.bool_)
        & d_left_family_valid
        & ~cp.asarray(left_dev.empty_mask, dtype=cp.bool_)[d_left_safe_rows]
    ).astype(cp.int32)
    d_right_valid = (
        cp.asarray(right_state.validity, dtype=cp.bool_)
        & d_right_family_valid
        & ~cp.asarray(right_dev.empty_mask, dtype=cp.bool_)[d_right_safe_rows]
    ).astype(cp.int32)

    # Allocate output arrays for the count pass
    d_counts = runtime.allocate((n,), cp.int32, zero=True)
    d_valid = runtime.allocate((n,), cp.int32, zero=True)
    d_supported = runtime.allocate((n,), cp.int32, zero=True)

    # Compile and launch count kernel
    kernels = _polygon_intersection_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_left_family_rows),
            left_polygon_rows,
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_right_family_rows),
            right_polygon_rows,
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_supported),
            n,
        ),
        (
            # Left coordinate/offset buffers and polygon-row count.
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            # Right coordinate/offset buffers and polygon-row count.
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            # Input validity, count/valid/support outputs, and row count.
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
        grid=grid,
        block=block,
        params=count_params,
    )

    # Exclusive prefix sum for scatter offsets (same-stream, no sync needed)
    d_offsets = exclusive_sum(d_counts, synchronize=False)

    vertex_capacity = _polygon_intersection_vertex_capacity(
        left,
        right,
        left_dev,
        right_dev,
    )
    d_out_x = runtime.allocate((vertex_capacity,), cp.float64)
    d_out_y = runtime.allocate((vertex_capacity,), cp.float64)

    # Launch scatter kernel
    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_left_family_rows),
            left_polygon_rows,
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_right_family_rows),
            right_polygon_rows,
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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
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
        kernels["polygon_intersection_scatter"],
        n,
    )
    runtime.launch(
        kernels["polygon_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    # Build ring_offsets on device from the existing d_offsets (exclusive prefix
    # sum of d_counts) to avoid D2H -> host cumsum -> H2D ping-pong.
    # ring_offsets[i] = d_offsets[i] for i < n, ring_offsets[n] = total_verts.
    # d_offsets is already the exclusive prefix sum = inclusive ring_offsets[0:n].
    # Append the device logical total to get the full ring_offsets array.
    import cupy as _cp

    d_ring_offsets = _cp.empty(n + 1, dtype=_cp.int32)
    if n:
        d_ring_offsets[:n] = _cp.asarray(d_offsets)
        d_ring_offsets[n] = _cp.asarray(d_offsets)[-1] + d_counts[-1]
    else:
        d_ring_offsets[0] = 0

    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid.astype(_cp.bool_),
        ring_offsets=d_ring_offsets,
        runtime_selection=runtime_selection,
    )
    result._polygon_intersection_sh_supported = d_supported.astype(_cp.bool_)
    return result


def _build_empty_result(n: int, runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    """Build an all-empty polygon result."""
    return build_empty_device_backed_polygon_intersection_output(
        row_count=n,
        runtime_selection=runtime_selection,
    )


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------


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
        left,
        right,
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
        raise ValueError(f"row count mismatch: left={left.row_count}, right={right.row_count}")

    n = left.row_count
    if n == 0:
        return from_shapely_geometries([])

    pair_work = estimate_pairwise_product_work_from_owned(
        left,
        right,
        pair_unit="segment",
        output_row_count=n,
        primary_unit_name="polygon-intersection-segment-pair",
    )
    output_coordinate_capacity = n * (int(_MAX_CLIP_VERTS) + 1)
    selection = plan_dispatch_selection(
        kernel_name="polygon_intersection",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        work_estimate=PhysicalWorkEstimate(
            row_count=n,
            coordinate_count=pair_work.coordinate_count,
            segment_count=pair_work.segment_count,
            segment_pair_count=pair_work.segment_pair_count,
            part_count=pair_work.part_count,
            ring_count=pair_work.ring_count,
            output_row_count=n,
            output_byte_count=output_coordinate_capacity * 16,
            temporary_byte_count=n * int(_MAX_CLIP_VERTS) * 32,
            primary_unit_count=max(
                pair_work.dispatch_unit_count(),
                output_coordinate_capacity,
            ),
            primary_unit_name="polygon-intersection-segment-pair",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=combined_residency(left, right),
    )

    if selection.selected is ExecutionMode.GPU:
        # ADR-0002: CONSTRUCTIVE stays fp64; plan is for observability.
        precision_plan = selection.precision_plan

        try:
            result = _polygon_intersection_gpu(
                left,
                right,
                runtime_selection=selection,
                precision_plan=precision_plan,
            )
            record_dispatch_event(
                surface="vibespatial.kernels.constructive.polygon_intersection",
                operation="polygon_intersection",
                implementation="polygon_intersection_gpu",
                reason=selection.reason,
                detail=(f"rows={n}, precision={precision_plan.compute_precision.value}"),
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
