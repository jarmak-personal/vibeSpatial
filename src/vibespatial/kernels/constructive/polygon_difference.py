"""GPU-native element-wise polygon difference kernel.

Computes left - right for aligned polygon/multipolygon OwnedGeometryArrays
entirely on the GPU, returning a device-resident OwnedGeometryArray with no
D->H transfers on the critical path.

Algorithm: Reuses the overlay topology pipeline (ADR-0016):
    extract_segments -> classify_segment_intersections ->
    build_gpu_split_events -> build_gpu_atomic_edges ->
    build_gpu_half_edge_graph -> build_gpu_overlay_faces ->
    face selection (left_covered & ~right_covered) ->
    face-to-polygon assembly

ADR-0033: Tier 3 pipeline orchestrating Tier 1 NVRTC kernels and
    Tier 3a CCCL primitives.
ADR-0002: CONSTRUCTIVE class -- stays fp64 per policy; precision plan
    wired through for observability.
"""

from __future__ import annotations

from vibespatial.constructive.polygon_difference_cpu import (
    polygon_difference_cpu as _polygon_difference_cpu,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_pairwise_product_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)
from vibespatial.runtime.residency import Residency

# Polygon-family types that can enter the overlay pipeline
_POLYGONAL_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})


def _is_polygon_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Polygon or MultiPolygon."""
    has_polygon_rows = False
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            if family not in _POLYGONAL_FAMILIES:
                return False
            has_polygon_rows = True
    return has_polygon_rows


def _polygon_difference_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Execute canonical row-capacity polygon difference topology."""
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_difference_overlay_batched_gpu,
    )

    result = _dispatch_polygon_difference_overlay_batched_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result is None or result.row_count != left.row_count:
        raise RuntimeError("canonical polygon difference topology declined admitted rows")
    return result


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "polygon_difference",
    "gpu-overlay",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "overlay"),
)
def _polygon_difference_gpu_variant(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """GPU polygon difference via overlay topology pipeline."""
    return _polygon_difference_gpu(left, right)


def polygon_difference(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Element-wise polygon difference: left - right.

    Computes the geometric difference of aligned polygon/multipolygon arrays.
    When both inputs are polygonal and GPU is available, executes the full
    overlay topology pipeline on GPU. Falls back to Shapely for non-polygonal
    inputs or when GPU is unavailable.

    Parameters
    ----------
    left : OwnedGeometryArray
        Left geometry array (the "base" polygons).
    right : OwnedGeometryArray
        Right geometry array (the polygons to subtract).
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode. CONSTRUCTIVE kernels stay fp64 per ADR-0002;
        the plan is computed for observability only.

    Returns
    -------
    OwnedGeometryArray
        Result geometries. May contain MultiPolygon when the difference
        splits a polygon. Empty geometry when left is fully inside right.
        Original left geometry when there is no overlap.

    Raises
    ------
    ValueError
        If row counts do not match.
    """
    if left.row_count != right.row_count:
        raise ValueError(f"row count mismatch: left={left.row_count}, right={right.row_count}")

    if left.row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="polygon_difference",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        work_estimate=estimate_pairwise_product_work_from_owned(
            left,
            right,
            pair_unit="segment",
            output_row_count=left.row_count,
            primary_unit_name="polygon-difference-segment-pair",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=combined_residency(left, right),
    )

    # ADR-0002: CONSTRUCTIVE kernels stay fp64. precision_plan is computed
    # for observability (dispatch event detail) only.
    precision_plan = selection.precision_plan

    if selection.selected is ExecutionMode.GPU:
        if _is_polygon_only(left) and _is_polygon_only(right):
            result = _polygon_difference_gpu(left, right)
            record_dispatch_event(
                surface="polygon_difference",
                operation="difference",
                implementation="polygon_difference_capacity_topology_gpu",
                reason=selection.reason,
                detail=(
                    f"rows={left.row_count}, precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    # CPU fallback
    if selection.selected is ExecutionMode.GPU and not (
        _is_polygon_only(left) and _is_polygon_only(right)
    ):
        fallback_reason = "non-polygonal input families"
    else:
        fallback_reason = selection.reason

    result = _polygon_difference_cpu(left, right)
    record_dispatch_event(
        surface="polygon_difference",
        operation="difference",
        implementation="polygon_difference_cpu",
        reason=fallback_reason,
        detail=f"rows={left.row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
