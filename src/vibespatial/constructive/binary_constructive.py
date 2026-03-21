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


def binary_constructive_owned(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
) -> OwnedGeometryArray:
    """Element-wise binary constructive operation on owned arrays.

    When both inputs are Polygon/MultiPolygon-only and no grid_size is
    requested, dispatches to the GPU overlay pipeline which handles its
    own GPU/CPU selection internally.  Otherwise falls back to Shapely.

    Parameters
    ----------
    op : str
        One of 'intersection', 'union', 'difference', 'symmetric_difference'.
    left, right : OwnedGeometryArray
        Input geometry arrays (must have same row count).
    grid_size : float or None
        Snap grid size for GEOS precision model. When set, forces the
        Shapely/GEOS path because the overlay pipeline does not support
        snapped precision.
    """
    if op not in _CONSTRUCTIVE_OPS:
        raise ValueError(f"unsupported constructive operation: {op}")

    if left.row_count != right.row_count:
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )

    if left.row_count == 0:
        return from_shapely_geometries([])

    # --- GPU fast path for Point-Polygon intersection/difference ---
    # The PIP kernel does not support grid_size, so skip when set.
    # union/symmetric_difference of Point-Polygon produce GeometryCollections
    # which are complex — leave those to Shapely.
    if grid_size is None and op in _POINT_POLYGON_OPS:
        try:
            if _is_point_only(left) and _is_polygon_only(right):
                if op == "intersection":
                    return _intersection_point_polygon_gpu(left, right)
                else:  # difference
                    return _difference_point_polygon_gpu(left, right)
            elif _is_polygon_only(left) and _is_point_only(right):
                # Symmetric case: intersection is commutative, so swap operands.
                # difference(polygon, point) is not handled here — it produces
                # polygons with point-holes which is complex.
                if op == "intersection":
                    return _intersection_point_polygon_gpu(right, left)
        except Exception:
            logger.debug(
                "Point-Polygon GPU %s failed, falling back to Shapely",
                op,
                exc_info=True,
            )

    # --- GPU overlay fast path for Polygon-Polygon pairs ---
    # The overlay pipeline does not support grid_size, so skip when set.
    if grid_size is None and _is_polygon_only(left) and _is_polygon_only(right):
        try:
            result = _dispatch_overlay_gpu(op, left, right)
            # Verify the overlay pipeline preserved row count.  Invalid
            # geometries (e.g. self-intersecting bowties) can cause the
            # pipeline to silently drop rows.  Fall through to Shapely
            # when the count does not match so the caller sees correct
            # results or GEOS-level errors.
            if result.row_count == left.row_count:
                return result
            logger.debug(
                "overlay GPU dispatch returned %d rows (expected %d) for %s, "
                "falling back to Shapely",
                result.row_count,
                left.row_count,
                op,
            )
        except Exception:
            # Any failure in the overlay pipeline (unsupported geometry mix,
            # GPU unavailable, etc.) falls through to the Shapely path.
            logger.debug(
                "overlay GPU dispatch failed for %s, falling back to Shapely",
                op,
                exc_info=True,
            )

    # --- Shapely fallback for non-polygon pairs or grid_size usage ---
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
    result_list = result_arr.tolist()

    return from_shapely_geometries(result_list)
