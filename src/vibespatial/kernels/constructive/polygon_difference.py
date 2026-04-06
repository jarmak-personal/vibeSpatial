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

import logging

from vibespatial.constructive.polygon_difference_cpu import (
    polygon_difference_cpu as _polygon_difference_cpu,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)
from vibespatial.runtime.residency import Residency

logger = logging.getLogger(__name__)

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
    """Execute the GPU overlay topology pipeline for polygon difference.

    Orchestrates the 8-stage overlay pipeline from gpu.py, selecting only
    faces that are left-covered and NOT right-covered (the "difference"
    face label).

    All intermediate data stays on device. The only D->H transfer is the
    final coordinate materialization at the face assembly boundary (same
    transfer point as all other overlay operations).
    """
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    runtime_selection = RuntimeSelection(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        reason="GPU polygon_difference kernel selected",
    )

    plan = _build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    result, _ = _materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
    )
    result.runtime_history.append(runtime_selection)
    return result


def _single_row_polygon_difference_needs_exact_fallback(
    left: OwnedGeometryArray,
    result: OwnedGeometryArray,
) -> bool:
    """Return True when a single-row difference result is structurally impossible."""
    from vibespatial.constructive.measurement import area_owned
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    if result.row_count == 0:
        return False
    if result.row_count != 1:
        return True

    result_state = result._ensure_device_state()
    if not bool(result_state.validity[0]) or not _is_polygon_only(result):
        return True

    left_area = float(area_owned(left)[0])
    result_area = float(area_owned(result)[0])
    tol = 1e-9
    if result_area > left_area + tol:
        return True

    if abs(result_area - left_area) <= tol:
        left_bounds = compute_geometry_bounds(left)
        result_bounds = compute_geometry_bounds(result)
        for got, expected in zip(result_bounds[0], left_bounds[0], strict=True):
            if abs(float(got) - float(expected)) > tol:
                return True
    return False


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
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )

    if left.row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="polygon_difference",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    # ADR-0002: CONSTRUCTIVE kernels stay fp64. precision_plan is computed
    # for observability (dispatch event detail) only.
    precision_plan = selection.precision_plan

    gpu_attempted = False
    if selection.selected is ExecutionMode.GPU:
        if _is_polygon_only(left) and _is_polygon_only(right):
            gpu_attempted = True
            try:
                result = _polygon_difference_gpu(left, right)
                detail = (
                    f"rows={left.row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                )
                if left.row_count == 1 and _single_row_polygon_difference_needs_exact_fallback(
                    left,
                    result,
                ):
                    from vibespatial.constructive.binary_constructive import (
                        _single_row_polygon_difference_exact_correction,
                    )
                    from vibespatial.overlay.gpu import _overlay_owned

                    correction = _single_row_polygon_difference_exact_correction(
                        left,
                        right,
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    if correction is not None:
                        result = correction
                        detail += ", corrected=exact_edge_case"
                    else:
                        result = _overlay_owned(
                            left,
                            right,
                            operation="difference",
                            dispatch_mode=ExecutionMode.GPU,
                            _row_isolated=True,
                        )
                        detail += ", corrected=row_isolated_overlay"
                record_dispatch_event(
                    surface="polygon_difference",
                    operation="difference",
                    implementation="polygon_difference_gpu",
                    reason=selection.reason,
                    detail=detail,
                    requested=selection.requested,
                    selected=ExecutionMode.GPU,
                )
                return result
            except Exception:
                logger.debug(
                    "GPU polygon_difference failed, falling back to CPU",
                    exc_info=True,
                )

    # CPU fallback
    if gpu_attempted:
        fallback_reason = "GPU kernel failed, CPU fallback"
    elif selection.selected is ExecutionMode.GPU and not (
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
