"""GPU-accelerated binary constructive operations.

Operations: intersection(a,b), union(a,b), difference(a,b), symmetric_difference(a,b)

Element-wise binary constructive operations dispatched per family pair:
- Point-Point: coordinate comparison (Tier 2 CuPy)
- Point-Polygon: PIP kernel for intersection/difference
- Point-LineString: point-on-segment kernel (Tier 1 NVRTC)
- MultiPoint-Polygon: batch PIP + compact
- LineString-Polygon: segment clipping kernel (Tier 1 NVRTC)
- LineString-LineString: segment-segment intersection kernel (Tier 1 NVRTC)
- Polygon-Polygon: overlay pipeline (face selection)

All GPU paths return device-resident OwnedGeometryArray.  The function
``_binary_constructive_gpu`` never returns None: every family pair is
handled by a GPU kernel.

ADR-0033: Tier 3 — complex multi-stage pipeline orchestrating Tier 1/2 kernels.
ADR-0002: CONSTRUCTIVE class — stays fp64 on all devices per policy.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import shapely

from vibespatial.cuda._runtime import DeviceArray
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
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

# LineString-family types
_LINESTRING_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})

# Point-Polygon constructive operations supported by the PIP fast path
_POINT_POLYGON_OPS = frozenset({"intersection", "difference"})


def is_constructive_op(op: str) -> bool:
    """Check if an operation name is a binary constructive operation."""
    return op in _CONSTRUCTIVE_OPS


def _is_family_only(owned: OwnedGeometryArray, target_families: frozenset[GeometryFamily]) -> bool:
    """Return True if every family with rows belongs to *target_families*."""
    has_rows = False
    for family, buf in owned.families.items():
        if buf.row_count > 0:
            if family not in target_families:
                return False
            has_rows = True
    return has_rows


def _is_polygon_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Polygon or MultiPolygon."""
    return _is_family_only(owned, _POLYGONAL_FAMILIES)


def _is_point_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is Point."""
    return _is_family_only(owned, frozenset({GeometryFamily.POINT}))


def _is_linestring_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is LineString."""
    return _is_family_only(owned, frozenset({GeometryFamily.LINESTRING}))


def _is_multipoint_only(owned: OwnedGeometryArray) -> bool:
    """Return True if every family with rows is MultiPoint."""
    return _is_family_only(owned, frozenset({GeometryFamily.MULTIPOINT}))


def _sh_kernel_can_handle(left: OwnedGeometryArray, right: OwnedGeometryArray) -> bool:
    """Check whether the Sutherland-Hodgman polygon intersection kernel can handle these inputs.

    The SH kernel has two documented limitations (polygon_intersection.py lines 66-76):
    1. Holes are not handled -- only exterior rings are used.
    2. The per-thread workspace is limited to ``_MAX_CLIP_VERTS`` (64) vertices.
       Clipping a polygon with ``l_n`` vertices against a polygon with ``r_n``
       exterior-ring edges can produce up to ``l_n + r_n`` intermediate vertices.
       When this exceeds the workspace limit, vertices are silently dropped and
       the kernel produces ``validity=False`` for valid intersections.

    Returns True if ALL pairs can be handled by the SH kernel.  Returns False if
    any pair exceeds the kernel's capabilities, in which case the caller should
    fall through to the GPU overlay pipeline.

    Uses vectorized NumPy operations on offset arrays to avoid per-row Python
    loops (VPAT001 compliance).
    """
    # Import the workspace limit from the kernel module.
    from vibespatial.kernels.constructive.polygon_intersection import (
        _MAX_CLIP_VERTS,
    )

    for side_label, side in (("right", right), ("left", left)):
        side._ensure_host_state()
        for buf in side.families.values():
            if buf.row_count == 0:
                continue
            geom_offsets = buf.geometry_offsets
            ring_offsets = buf.ring_offsets

            # Vectorized ring-count check: rings_per_row = geom_offsets[1:] - geom_offsets[:-1]
            rings_per_row = np.diff(geom_offsets)

            # Check for holes (any polygon with >1 ring)
            if np.any(rings_per_row > 1):
                logger.debug(
                    "SH kernel skip: %s polygon has rows with holes "
                    "(max rings per row = %d)",
                    side_label, int(rings_per_row.max()),
                )
                return False

            # Vectorized vertex-count check for exterior rings.
            # first_ring_idx is geom_offsets[:-1] for rows with at least 1 ring.
            has_rings = rings_per_row > 0
            if not np.any(has_rings):
                continue
            first_ring_idx = geom_offsets[:-1][has_rings]
            ext_verts = ring_offsets[first_ring_idx + 1] - ring_offsets[first_ring_idx]
            max_verts = int(ext_verts.max()) if ext_verts.size > 0 else 0
            if max_verts > _MAX_CLIP_VERTS:
                logger.debug(
                    "SH kernel skip: %s polygon exterior ring has %d vertices "
                    "(limit %d)",
                    side_label, max_verts, _MAX_CLIP_VERTS,
                )
                return False

    return True


def _build_point_polygon_result(
    points: OwnedGeometryArray,
    new_validity: np.ndarray | None,
    *,
    d_new_validity: DeviceArray | None = None,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray that shares the Point coordinate buffers.

    The result keeps the same family buffers (coordinates, offsets, etc.)
    as *points* but replaces the top-level validity mask.  Rows where
    ``new_validity[i]`` is False become NULL in the output.

    When *d_new_validity* is provided (a CuPy device array), the result
    stores device-resident metadata directly and the host arrays are
    set to ``None`` for lazy materialisation on first access.  This
    eliminates D->H transfers for GPU-only consumers.

    Host family buffers are shared directly (no copy).  If the input
    has a device state, the result gets a new ``OwnedGeometryDeviceState``
    whose family buffers are shared but whose ``validity`` DeviceArray
    reflects the new mask -- ensuring downstream GPU consumers see the
    correct null rows without re-uploading coordinates.
    """
    from vibespatial.geometry.owned import DeviceMetadataState, OwnedGeometryDeviceState

    new_device_state = None
    new_device_metadata = None

    if points.device_state is not None:
        # Determine the device validity array
        if d_new_validity is not None:
            d_validity_out = d_new_validity
        elif new_validity is not None:
            from vibespatial.cuda._runtime import get_cuda_runtime
            runtime = get_cuda_runtime()
            d_validity_out = runtime.from_host(new_validity)
        else:
            raise ValueError(
                "Either new_validity or d_new_validity must be provided"
            )

        new_device_state = OwnedGeometryDeviceState(
            validity=d_validity_out,
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
            families=dict(points.device_state.families),
        )
        new_device_metadata = DeviceMetadataState(
            validity=d_validity_out,
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
        )

    # When device metadata is available and no host validity was provided,
    # keep host arrays None for lazy materialisation.
    if d_new_validity is not None and new_device_metadata is not None:
        # Device-resident result: host metadata is lazy
        h_validity = None
        h_tags = None
        h_family_row_offsets = None
    else:
        # Host-resident result: copy the metadata arrays
        h_validity = new_validity
        h_tags = points.tags.copy()
        h_family_row_offsets = points.family_row_offsets.copy()

    # Tags and family_row_offsets still index into the same family buffer.
    # For rows that are now invalid, the consumer will skip them via validity.
    result = OwnedGeometryArray(
        validity=h_validity,
        tags=h_tags,
        family_row_offsets=h_family_row_offsets,
        families=dict(points.families),
        residency=points.residency,
        device_state=new_device_state,
        device_metadata=new_device_metadata,
        _row_count=points.row_count,
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
    obtain a device-resident boolean mask.  When the input has device
    state, the boolean mask stays on GPU and no D->H transfer occurs.

    ADR-0033: Tier 2 (CuPy element-wise mask) over Tier 1 PIP kernel.
    """
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    # PIP kernel returns CuPy bool array (GPU) or numpy bool array (CPU).
    pip_mask = point_in_polygon(points, polygons, _return_device=True)

    # intersection: output is valid only where PIP is True.
    # pip_mask already encodes null handling (False when either input null).
    if hasattr(pip_mask, "get") and points.device_state is not None:
        # CuPy array + device state: keep entirely on device, no D->H.
        d_new_validity = pip_mask.astype(bool, copy=False)
        return _build_point_polygon_result(points, None, d_new_validity=d_new_validity)
    else:
        # CPU fallback or no device state: transfer to host.
        if hasattr(pip_mask, "get"):
            h_pip = pip_mask.get()
        else:
            h_pip = np.asarray(pip_mask, dtype=bool)
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

    When the input has device state, all boolean ops stay on GPU.

    ADR-0033: Tier 2 (CuPy element-wise mask) over Tier 1 PIP kernel.
    """
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pip_mask = point_in_polygon(points, polygons, _return_device=True)

    # difference: output valid when left is valid AND point is NOT inside.
    # pip_mask is False for null right polygons, so ~pip_mask is True for
    # those rows -- preserving the identity semantics (point - NULL = point).
    if hasattr(pip_mask, "get") and points.device_state is not None:
        # CuPy array + device state: keep entirely on device, no D->H.
        d_validity = points.device_state.validity.astype(bool, copy=False)
        d_new_validity = d_validity & ~pip_mask.astype(bool, copy=False)
        return _build_point_polygon_result(points, None, d_new_validity=d_new_validity)
    else:
        # CPU fallback or no device state: transfer to host.
        if hasattr(pip_mask, "get"):
            h_pip = pip_mask.get()
        else:
            h_pip = np.asarray(pip_mask, dtype=bool)
        new_validity = points.validity & ~h_pip
        return _build_point_polygon_result(points, new_validity)


def _dispatch_overlay_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
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
    return fn(left, right, dispatch_mode=dispatch_mode)


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
    """CPU-only mode: Shapely element-wise binary constructive."""
    left_geoms = left.to_shapely()  # CPU-only mode
    right_geoms = right.to_shapely()  # CPU-only mode

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    kwargs = {}
    if grid_size is not None:
        kwargs["grid_size"] = grid_size

    result_arr = getattr(shapely, op)(left_arr, right_arr, **kwargs)  # CPU-only mode
    return from_shapely_geometries(result_arr.tolist())


@register_kernel_variant(
    "binary_constructive",
    "gpu-overlay-pip",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "overlay", "pip"),
)
def _binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """GPU binary constructive for all family combinations.

    Dispatches to specialized GPU kernels based on the geometry family
    combination.  Falls back to CPU only if a GPU kernel raises an
    exception, never because a family pair is missing.

    Parameters
    ----------
    dispatch_mode : ExecutionMode
        Propagated to inner kernels (polygon_intersection, overlay).
        Default is GPU since this function is only called when the
        outer dispatch has already selected GPU execution.
    """
    # --- Point-Point ---
    if _is_point_only(left) and _is_point_only(right):
        try:
            return _dispatch_point_point_gpu(op, left, right)
        except Exception:
            logger.debug("Point-Point GPU %s failed", op, exc_info=True)
            return None

    # --- Point-Polygon (existing PIP fast path) ---
    if _is_point_only(left) and _is_polygon_only(right):
        try:
            if op == "intersection":
                return _intersection_point_polygon_gpu(left, right)
            elif op == "difference":
                return _difference_point_polygon_gpu(left, right)
        except Exception:
            logger.debug("Point-Polygon GPU %s failed", op, exc_info=True)
            return None

    if _is_polygon_only(left) and _is_point_only(right):
        try:
            if op == "intersection":
                return _intersection_point_polygon_gpu(right, left)
        except Exception:
            logger.debug("Polygon-Point GPU intersection failed", exc_info=True)
            return None

    # --- Point-LineString ---
    if _is_point_only(left) and _is_linestring_only(right):
        try:
            return _dispatch_point_linestring_gpu(op, left, right)
        except Exception:
            logger.debug("Point-LineString GPU %s failed", op, exc_info=True)
            return None

    if _is_linestring_only(left) and _is_point_only(right):
        try:
            if op == "intersection":
                return _dispatch_point_linestring_gpu("intersection", right, left)
        except Exception:
            logger.debug("LineString-Point GPU intersection failed", exc_info=True)
            return None

    # --- MultiPoint-Polygon ---
    if _is_multipoint_only(left) and _is_polygon_only(right):
        try:
            return _dispatch_multipoint_polygon_gpu(op, left, right)
        except Exception:
            logger.debug("MultiPoint-Polygon GPU %s failed", op, exc_info=True)
            return None

    if _is_polygon_only(left) and _is_multipoint_only(right):
        try:
            if op == "intersection":
                return _dispatch_multipoint_polygon_gpu("intersection", right, left)
        except Exception:
            logger.debug("Polygon-MultiPoint GPU intersection failed", exc_info=True)
            return None

    # --- LineString-Polygon ---
    if _is_linestring_only(left) and _is_polygon_only(right):
        try:
            return _dispatch_linestring_polygon_gpu(op, left, right)
        except Exception:
            logger.debug("LineString-Polygon GPU %s failed", op, exc_info=True)
            return None

    if _is_polygon_only(left) and _is_linestring_only(right):
        try:
            if op == "intersection":
                return _dispatch_linestring_polygon_gpu("intersection", right, left)
        except Exception:
            logger.debug("Polygon-LineString GPU intersection failed", exc_info=True)
            return None

    # --- LineString-LineString ---
    if _is_linestring_only(left) and _is_linestring_only(right):
        try:
            return _dispatch_linestring_linestring_gpu(op, left, right)
        except Exception:
            logger.debug("LineString-LineString GPU %s failed", op, exc_info=True)
            return None

    # --- Polygon-Polygon GPU kernel fast paths ---
    if _is_polygon_only(left) and _is_polygon_only(right):
        # Try direct GPU kernels first (faster than full overlay pipeline
        # for element-wise ops: no topology graph construction needed).
        if op == "intersection":
            # The Sutherland-Hodgman kernel has known limitations: no holes,
            # and a per-thread workspace limit of _MAX_CLIP_VERTS.  Check
            # whether the inputs are within the kernel's capabilities before
            # calling it.  When they are not (e.g. complex polygons with
            # holes or many vertices), skip SH and fall through to the full
            # GPU overlay pipeline which handles arbitrarily complex polygons
            # (multi-ring, holes, high vertex counts) via 8-stage topology
            # reconstruction.
            if _sh_kernel_can_handle(left, right):
                try:
                    from vibespatial.kernels.constructive.polygon_intersection import (
                        polygon_intersection,
                    )

                    result = polygon_intersection(left, right, dispatch_mode=dispatch_mode)
                    if result.row_count == left.row_count:
                        return result
                except Exception:
                    logger.debug(
                        "polygon_intersection GPU kernel failed, trying overlay pipeline",
                        exc_info=True,
                    )
            else:
                logger.debug(
                    "polygon_intersection SH kernel skipped: input exceeds "
                    "kernel capabilities (holes or vertex count), falling "
                    "through to GPU overlay pipeline",
                )
        elif op == "difference":
            try:
                from vibespatial.kernels.constructive.polygon_difference import (
                    polygon_difference,
                )

                result = polygon_difference(left, right, dispatch_mode=dispatch_mode)
                if result.row_count == left.row_count:
                    return result
            except Exception:
                logger.debug(
                    "polygon_difference GPU kernel failed, trying overlay pipeline",
                    exc_info=True,
                )

        # Fall through to the general overlay pipeline for union,
        # symmetric_difference, or when direct kernels fail.
        try:
            result = _dispatch_overlay_gpu(op, left, right, dispatch_mode=dispatch_mode)
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

    # For any remaining family pair not covered above, return None to
    # trigger CPU fallback.  This should only happen for exotic multi-type
    # combinations (e.g., MultiLineString-MultiPolygon).
    return None


# ---------------------------------------------------------------------------
# Non-polygon GPU dispatch helpers
# ---------------------------------------------------------------------------

def _dispatch_point_point_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch Point-Point GPU constructive to the appropriate kernel."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        point_point_difference,
        point_point_intersection,
        point_point_symmetric_difference,
        point_point_union,
    )

    dispatch = {
        "intersection": point_point_intersection,
        "difference": point_point_difference,
        "union": point_point_union,
        "symmetric_difference": point_point_symmetric_difference,
    }
    return dispatch[op](left, right)


def _dispatch_point_linestring_gpu(
    op: str,
    points: OwnedGeometryArray,
    linestrings: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch Point-LineString GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        point_linestring_difference,
        point_linestring_intersection,
    )

    if op == "intersection":
        return point_linestring_intersection(points, linestrings)
    elif op == "difference":
        return point_linestring_difference(points, linestrings)
    # union/symmetric_difference of Point-LineString produces mixed-type
    # results. Fall back to CPU for now.
    return None


def _dispatch_multipoint_polygon_gpu(
    op: str,
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch MultiPoint-Polygon GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        multipoint_polygon_difference,
        multipoint_polygon_intersection,
    )

    if op == "intersection":
        return multipoint_polygon_intersection(multipoints, polygons)
    elif op == "difference":
        return multipoint_polygon_difference(multipoints, polygons)
    # union/symmetric_difference produce mixed types. Fall back.
    return None


def _dispatch_linestring_polygon_gpu(
    op: str,
    linestrings: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch LineString-Polygon GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_polygon_difference,
        linestring_polygon_intersection,
    )

    if op == "intersection":
        return linestring_polygon_intersection(linestrings, polygons)
    elif op == "difference":
        return linestring_polygon_difference(linestrings, polygons)
    # union/symmetric_difference produce mixed types. Fall back.
    return None


def _dispatch_linestring_linestring_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Dispatch LineString-LineString GPU constructive."""
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_linestring_intersection,
    )

    if op == "intersection":
        return linestring_linestring_intersection(left, right)
    # difference/union/symmetric_difference of LineString-LineString are complex
    # mixed-type operations. Fall back to CPU.
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
        result = _binary_constructive_gpu(op, left, right, dispatch_mode=selection.selected)
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

    # Phase 24: Guard CPU fallback when GPU was explicitly requested with
    # device-resident input.  This should not happen silently.
    if gpu_attempted and selection.requested is ExecutionMode.GPU:
        warnings.warn(
            f"[vibeSpatial] binary_constructive '{op}': GPU was explicitly "
            f"requested but the GPU kernel returned None for this family "
            f"pair. Falling back to CPU/Shapely with D2H transfer. "
            f"rows={left.row_count}",
            stacklevel=2,
        )

    record_fallback_event(
        surface=f"geopandas.array.{op}",
        reason=fallback_reason,
        detail=f"rows={left.row_count}, op={op}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
        pipeline="binary_constructive_owned",
        d2h_transfer=gpu_attempted,  # D2H transfer occurs when GPU was attempted but fell back
    )

    result = _binary_constructive_cpu(op, left, right, grid_size=grid_size)  # CPU-only mode: Shapely
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
