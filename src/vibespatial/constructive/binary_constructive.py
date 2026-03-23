"""GPU-accelerated binary constructive operations.

Operations: intersection(a,b), union(a,b), difference(a,b), symmetric_difference(a,b)

Element-wise binary constructive operations dispatched per family pair:
- Point-Polygon: PIP kernel for intersection/difference
- Polygon-Polygon: overlay pipeline (face selection)
- Simpler pairs: trivial per-geometry kernels

Currently implements an owned-array dispatch wrapper with CPU fallback.
GPU acceleration leverages the existing overlay pipeline for Polygon-Polygon
and extends progressively to other family pairs.

ADR-0033: Tier 3 — complex multi-stage pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import shapely

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    select_precision_plan,
)

logger = logging.getLogger(__name__)

# Constructive operations that this module handles
_CONSTRUCTIVE_OPS = frozenset({"intersection", "union", "difference", "symmetric_difference"})

# Polygon-family types supported by the GPU overlay pipeline
_POLYGONAL_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# Point-Polygon constructive operations supported by the PIP fast path
_POINT_POLYGON_OPS = frozenset({"intersection", "difference"})


def is_constructive_op(op: str) -> bool:
    """Check if an operation name is a binary constructive operation."""
    return op in _CONSTRUCTIVE_OPS


def _is_polygon_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Polygon or MultiPolygon."""
    has_polygon_rows = False
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            if family not in _POLYGONAL_FAMILIES:
                return False
            has_polygon_rows = True
    return has_polygon_rows


def _is_point_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Point."""
    has_point_rows = False
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            if family is not GeometryFamily.POINT:
                return False
            has_point_rows = True
    return has_point_rows


def _build_point_polygon_result(
    points: OwnedGeometryArray,
    new_validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray that shares the Point coordinate buffers.

    The result keeps the same family buffers (coordinates, offsets, etc.)
    as *points* but replaces the top-level validity mask.  Rows where
    ``new_validity[i]`` is False become NULL in the output.

    Host family buffers are shared directly (no copy).  If the input
    has a device state, the result gets a new ``OwnedGeometryDeviceState``
    whose family buffers are shared but whose ``validity`` DeviceArray
    reflects the new mask — ensuring downstream GPU consumers see the
    correct null rows without re-uploading coordinates.
    """
    from vibespatial.geometry.owned import OwnedGeometryDeviceState

    new_device_state = None
    if points.device_state is not None:
        from vibespatial.cuda._runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        new_device_state = OwnedGeometryDeviceState(
            validity=runtime.from_host(new_validity),
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
            families=dict(points.device_state.families),
        )

    # Tags and family_row_offsets still index into the same family buffer.
    # For rows that are now invalid, the consumer will skip them via validity.
    result = OwnedGeometryArray(
        validity=new_validity,
        tags=points.tags.copy(),
        family_row_offsets=points.family_row_offsets.copy(),
        families=dict(points.families),
        residency=points.residency,
        device_state=new_device_state,
    )
    return result


def _intersection_point_polygon_gpu(
    points: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """GPU Point-Polygon intersection via PIP kernel + validity masking.

    For each element-wise pair (point_i, polygon_i):
    - point inside polygon  -> keep the point
    - point outside polygon -> NULL
    - either input NULL     -> NULL

    Uses the existing fused PIP kernel with ``_return_device=True`` to
    obtain a device-resident boolean mask, then builds the output by
    copying the input Point buffers and masking validity.  The only
    D->H transfer is the boolean mask (N bytes).

    ADR-0033: Tier 2 (CuPy element-wise mask) over Tier 1 PIP kernel.
    """
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    # PIP kernel returns CuPy bool array (GPU) or numpy bool array (CPU).
    pip_mask = point_in_polygon(points, polygons, _return_device=True)

    # Single D->H transfer of the boolean mask.  This is N booleans and
    # is necessary because OwnedGeometryArray.validity is host-resident.
    if hasattr(pip_mask, "get"):
        # CuPy array — transfer to host
        h_pip = pip_mask.get()
    else:
        # numpy array (CPU fallback)
        h_pip = np.asarray(pip_mask, dtype=bool)

    # intersection: output is valid only where PIP is True.
    # pip_mask already encodes null handling (False when either input null).
    new_validity = h_pip.astype(bool, copy=False)
    return _build_point_polygon_result(points, new_validity)


def _difference_point_polygon_gpu(
    points: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """GPU Point-Polygon difference via PIP kernel + inverted validity masking.

    For each element-wise pair (point_i, polygon_i):
    - point outside polygon -> keep the point
    - point inside polygon  -> NULL
    - left (point) NULL     -> NULL
    - right (polygon) NULL  -> keep the point (difference with NULL = identity)

    ADR-0033: Tier 2 (CuPy element-wise mask) over Tier 1 PIP kernel.
    """
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pip_mask = point_in_polygon(points, polygons, _return_device=True)

    if hasattr(pip_mask, "get"):
        h_pip = pip_mask.get()
    else:
        h_pip = np.asarray(pip_mask, dtype=bool)

    # difference: output valid when left is valid AND point is NOT inside.
    # pip_mask is False for null right polygons, so ~pip_mask is True for
    # those rows — preserving the identity semantics (point - NULL = point).
    new_validity = points.validity & ~h_pip
    return _build_point_polygon_result(points, new_validity)


def _dispatch_overlay_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch to the GPU overlay pipeline for Polygon-Polygon pairs.

    Imports are lazy to avoid circular dependencies between constructive
    and overlay modules.
    """
    from vibespatial.overlay.gpu import (
        overlay_difference_owned,
        overlay_intersection_owned,
        overlay_symmetric_difference_owned,
        overlay_union_owned,
    )

    dispatch = {
        "intersection": overlay_intersection_owned,
        "union": overlay_union_owned,
        "difference": overlay_difference_owned,
        "symmetric_difference": overlay_symmetric_difference_owned,
    }
    fn = dispatch[op]
    return fn(left, right)


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "binary_constructive",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely",),
)
def _binary_constructive_cpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
) -> OwnedGeometryArray:
    """CPU fallback: Shapely element-wise binary constructive."""
    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    kwargs = {}
    if grid_size is not None:
        kwargs["grid_size"] = grid_size

    result_arr = getattr(shapely, op)(left_arr, right_arr, **kwargs)
    return from_shapely_geometries(result_arr.tolist())


@register_kernel_variant(
    "binary_constructive",
    "gpu-overlay-pip",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "overlay", "pip"),
)
def _binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """GPU binary constructive: overlay for poly-poly, PIP for point-poly.

    Returns None when the family combination is not GPU-accelerated,
    signalling the caller to fall back to CPU.
    """
    # --- Point-Polygon fast path (PIP kernel) ---
    if op in _POINT_POLYGON_OPS:
        try:
            if _is_point_only(left) and _is_polygon_only(right):
                if op == "intersection":
                    return _intersection_point_polygon_gpu(left, right)
                else:  # difference
                    return _difference_point_polygon_gpu(left, right)
            elif _is_polygon_only(left) and _is_point_only(right):
                if op == "intersection":
                    return _intersection_point_polygon_gpu(right, left)
        except Exception:
            logger.debug(
                "Point-Polygon GPU %s failed, falling back to CPU",
                op,
                exc_info=True,
            )

    # --- Polygon-Polygon overlay path ---
    if _is_polygon_only(left) and _is_polygon_only(right):
        try:
            result = _dispatch_overlay_gpu(op, left, right)
            if result.row_count == left.row_count:
                return result
            logger.debug(
                "overlay GPU dispatch returned %d rows (expected %d) for %s",
                result.row_count,
                left.row_count,
                op,
            )
        except Exception:
            logger.debug(
                "overlay GPU dispatch failed for %s, falling back to CPU",
                op,
                exc_info=True,
            )

    return None


def binary_constructive_owned(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Element-wise binary constructive operation on owned arrays.

    Uses the standard dispatch framework: ``plan_dispatch_selection`` for
    GPU/CPU routing, ``select_precision_plan`` for precision, and
    ``record_dispatch_event`` for observability.

    GPU paths:
    - Polygon-Polygon pairs: overlay pipeline (face selection).
    - Point-Polygon intersection/difference: PIP kernel + validity masking.

    Parameters
    ----------
    op : str
        One of 'intersection', 'union', 'difference', 'symmetric_difference'.
    left, right : OwnedGeometryArray
        Input geometry arrays (must have same row count).
    grid_size : float or None
        Snap grid size for GEOS precision model.  When set, forces the
        CPU/Shapely path because the GPU pipeline does not support
        snapped precision.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode for GPU path.
    """
    if op not in _CONSTRUCTIVE_OPS:
        raise ValueError(f"unsupported constructive operation: {op}")

    if left.row_count != right.row_count:
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )

    if left.row_count == 0:
        return from_shapely_geometries([])

    # Force CPU when grid_size is set (GPU pipeline doesn't support snapped precision)
    effective_mode = dispatch_mode
    if grid_size is not None:
        effective_mode = ExecutionMode.CPU

    selection = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        requested_mode=effective_mode,
    )

    gpu_attempted = False
    if selection.selected is ExecutionMode.GPU:
        # ADR-0002: CONSTRUCTIVE kernels stay fp64.  precision_plan is
        # computed for observability (dispatch event detail) only; the
        # overlay and PIP kernels manage their own precision internally.
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        gpu_attempted = True
        result = _binary_constructive_gpu(op, left, right)
        if result is not None:
            record_dispatch_event(
                surface=f"geopandas.array.{op}",
                operation=op,
                implementation="binary_constructive_gpu",
                reason=selection.reason,
                detail=(
                    f"rows={left.row_count}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    # CPU fallback: Shapely element-wise
    if grid_size is not None:
        fallback_reason = "grid_size requires GEOS precision model"
    elif gpu_attempted:
        fallback_reason = "GPU kernel returned None (unsupported family pair)"
    else:
        fallback_reason = selection.reason

    result = _binary_constructive_cpu(op, left, right, grid_size=grid_size)
    record_dispatch_event(
        surface=f"geopandas.array.{op}",
        operation=op,
        implementation="binary_constructive_cpu",
        reason=fallback_reason,
        detail=f"rows={left.row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
