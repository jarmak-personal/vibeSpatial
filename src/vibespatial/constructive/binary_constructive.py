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
    FAMILY_TAGS,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode

logger = logging.getLogger(__name__)

# Constructive operations that this module handles
_CONSTRUCTIVE_OPS = frozenset({"intersection", "union", "difference", "symmetric_difference"})

# Polygon-family types supported by the GPU overlay pipeline
_POLYGONAL_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})


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
