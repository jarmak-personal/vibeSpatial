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
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.constructive.binary_constructive_cpu import binary_constructive_cpu
from vibespatial.constructive.nonpolygon_binary_output import (
    build_point_result_from_source,
)
from vibespatial.cuda._runtime import DeviceArray

if TYPE_CHECKING:
    from vibespatial.spatial.segment_primitives import DeviceSegmentTable

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    NULL_TAG,
    TAG_FAMILIES,
    OwnedGeometryArray,
    _concat_device_family_buffers,
    build_device_resident_owned,
    from_shapely_geometries,
    materialize_broadcast,
    tile_single_row,
    unique_tag_pairs,
)
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import WorkloadShape
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import (
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)
from vibespatial.runtime.residency import Residency, TransferTrigger

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
    2. The clip polygon (right operand) must be convex.
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

    def _is_convex_ring(x: np.ndarray, y: np.ndarray, start: int, end: int) -> bool:
        """Return True when the closed ring [start:end] is convex.

        The SH kernel clips the left subject polygon against the right clip
        polygon edge-by-edge. That only holds for convex clip polygons. A
        concave right-hand exterior ring can silently produce the small
        central kernel of the polygon instead of the full clipped area.
        """
        if end - start < 4:
            return False

        ring_x = np.asarray(x[start:end], dtype=np.float64)
        ring_y = np.asarray(y[start:end], dtype=np.float64)

        if ring_x.size >= 2 and ring_x[0] == ring_x[-1] and ring_y[0] == ring_y[-1]:
            ring_x = ring_x[:-1]
            ring_y = ring_y[:-1]

        if ring_x.size < 3:
            return False

        prev_x = np.roll(ring_x, 1)
        prev_y = np.roll(ring_y, 1)
        next_x = np.roll(ring_x, -1)
        next_y = np.roll(ring_y, -1)
        cross = (ring_x - prev_x) * (next_y - ring_y) - (ring_y - prev_y) * (next_x - ring_x)
        non_collinear = cross[np.abs(cross) > 1e-12]
        if non_collinear.size == 0:
            return False
        return bool(np.all(non_collinear > 0.0) or np.all(non_collinear < 0.0))

    for side_label, side in (("right", right), ("left", left)):
        for family, buf in side.families.items():
            if buf.row_count == 0:
                continue
            side._ensure_host_family_structure(family)
            buf = side.families[family]
            geom_offsets = buf.geometry_offsets
            ring_offsets = buf.ring_offsets
            if ring_offsets is None:
                logger.debug(
                    "SH kernel skip: %s %s family is missing ring offsets",
                    side_label, family.value,
                )
                return False

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

            if side_label == "right" and family == GeometryFamily.MULTIPOLYGON:
                logger.debug(
                    "SH kernel skip: right clip polygon family is multipolygon",
                )
                return False

            # Vectorized vertex-count check for exterior rings.
            # first_ring_idx is geom_offsets[:-1] for rows with at least 1 ring.
            has_rings = rings_per_row > 0
            if not np.any(has_rings):
                continue
            first_ring_idx = geom_offsets[:-1][has_rings]

            if side_label == "right":
                for ring_idx in first_ring_idx:
                    start = int(ring_offsets[ring_idx])
                    end = int(ring_offsets[ring_idx + 1])
                    if not _is_convex_ring(buf.x, buf.y, start, end):
                        logger.debug(
                            "SH kernel skip: right clip polygon exterior ring is concave",
                        )
                        return False

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
    return build_point_result_from_source(
        points,
        new_validity,
        d_new_validity=d_new_validity,
    )


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

    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon intersection GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon intersection GPU",
    )

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

    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon difference GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="binary_constructive point_polygon difference GPU",
    )

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
    _cached_right_segments: DeviceSegmentTable | None = None,
    _row_isolated: bool = False,
) -> OwnedGeometryArray:
    """Dispatch to the GPU overlay pipeline for Polygon-Polygon pairs.

    Imports are lazy to avoid circular dependencies between constructive
    and overlay modules.
    """
    from vibespatial.overlay.gpu import _overlay_owned

    return _overlay_owned(
        left, right, dispatch_mode=dispatch_mode,
        operation=op,
        _cached_right_segments=_cached_right_segments,
        _row_isolated=_row_isolated,
    )


def _dispatch_polygon_overlay_rowwise_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Preserve pairwise row cardinality for polygon constructive ops via overlay.

    The general overlay pipeline is optimized for spatial-overlay semantics and
    drops empty rows from the output.  That is correct for overlay surfaces,
    but it breaks the element-wise constructive contract for polygon
    operations, where the output must remain aligned 1:1 with the input rows.

    This helper executes overlay one pair at a time, then scatters each
    single-row result back into an ``N``-row device-resident
    OwnedGeometryArray while keeping null rows for empty results.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    if left.row_count != right.row_count:
        return None

    def _collapse_to_single_row(sub_result: OwnedGeometryArray) -> OwnedGeometryArray | None:
        if sub_result.row_count <= 1:
            return sub_result
        if not _is_polygon_only(sub_result):
            return None

        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        collapsed = segmented_union_all(
            sub_result,
            np.asarray([0, sub_result.row_count], dtype=np.int64),
        )
        if collapsed.row_count != 1:
            return None
        return collapsed

    if left.row_count > 1:
        try:
            batch_result = _dispatch_overlay_gpu(
                op,
                left,
                right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=_cached_right_segments,
                _row_isolated=True,
            )
            if batch_result.row_count == left.row_count:
                return batch_result
        except Exception:
            logger.debug(
                "row-isolated batched polygon %s overlay failed; falling back to per-row path",
                op,
                exc_info=True,
            )

    out_validity = cp.zeros(left.row_count, dtype=cp.bool_)
    out_tags = cp.full(left.row_count, NULL_TAG, dtype=cp.int8)
    out_family_row_offsets = cp.full(left.row_count, -1, dtype=cp.int32)
    family_buffers: dict[GeometryFamily, list] = {}
    family_row_bases: dict[GeometryFamily, int] = {}

    for row_index in range(left.row_count):
        if not (bool(left.validity[row_index]) and bool(right.validity[row_index])):
            continue

        d_row = cp.asarray([row_index], dtype=cp.int64)
        left_row = left.take(d_row)
        right_row = right.take(d_row)

        sub_result = _dispatch_overlay_gpu(
            op,
            left_row,
            right_row,
            dispatch_mode=dispatch_mode,
            # The cached segment table is built for the original aligned
            # multi-row right operand. Reusing it after right.take([i])
            # breaks row isolation and can leak full-batch topology into a
            # single-row overlay subcall.
            _cached_right_segments=None,
        )
        sub_result = _collapse_to_single_row(sub_result)
        if sub_result is None:
            logger.debug(
                "row-preserving polygon %s fallback could not collapse multi-row result",
                op,
            )
            return None
        if sub_result.row_count == 0:
            continue
        if sub_result.row_count != 1:
            logger.debug(
                "row-preserving polygon %s fallback expected 1 row, got %d",
                op,
                sub_result.row_count,
            )
            return None

        sub_state = sub_result._ensure_device_state()
        if not bool(sub_state.validity[0]):
            continue

        tag_value = int(sub_state.tags[0])
        family = TAG_FAMILIES.get(tag_value)
        if family is None:
            logger.debug(
                "row-preserving polygon %s fallback saw unsupported tag %d",
                op,
                tag_value,
            )
            return None

        buffer = sub_state.families.get(family)
        if buffer is None:
            logger.debug(
                "row-preserving polygon %s fallback missing %s family buffer",
                op,
                family.value,
            )
            return None

        base_row = family_row_bases.get(family, 0)
        out_validity[row_index] = sub_state.validity[0]
        out_tags[row_index] = sub_state.tags[0]
        out_family_row_offsets[row_index] = base_row
        family_buffers.setdefault(family, []).append(buffer)
        family_row_bases[family] = base_row + (int(buffer.geometry_offsets.size) - 1)

    device_families = {
        family: _concat_device_family_buffers(family, buffers)
        for family, buffers in family_buffers.items()
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=left.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _dispatch_polygon_contraction_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Try the contraction-based polygon constructive path for aligned batches."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count <= 1:
        return None
    try:
        from vibespatial.overlay.contraction import overlay_contraction_owned

        result = overlay_contraction_owned(
            left,
            right,
            operation=op,
            dispatch_mode=dispatch_mode,
        )
    except Exception:
        logger.debug(
            "contraction polygon %s GPU path failed; falling back to overlay helpers",
            op,
            exc_info=True,
        )
        return None
    if result.row_count != left.row_count:
        logger.debug(
            "contraction polygon %s GPU path returned %d rows (expected %d)",
            op,
            result.row_count,
            left.row_count,
        )
        return None
    return result


def _dispatch_polygon_intersection_overlay_rowwise_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    if left.row_count == right.row_count == 1:
        result = _dispatch_overlay_gpu(
            "intersection",
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
            _row_isolated=True,
        )
        if result.row_count == 1:
            return result
    return _dispatch_polygon_overlay_rowwise_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def _dispatch_polygon_difference_overlay_rowwise_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Preserve pairwise row cardinality for polygon difference via overlay."""
    try:
        batched_result = _dispatch_polygon_difference_overlay_batched_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
        )
    except Exception:
        logger.debug(
            "batched polygon difference overlay fast path failed; falling back to legacy rowwise path",
            exc_info=True,
        )
        batched_result = None
    if batched_result is not None:
        return batched_result
    return _dispatch_polygon_difference_overlay_rowwise_gpu_legacy(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def _dispatch_polygon_difference_overlay_batched_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    if left.row_count != right.row_count:
        return None

    def _collapse_to_single_row(sub_result: OwnedGeometryArray) -> OwnedGeometryArray | None:
        if sub_result.row_count <= 1:
            return sub_result
        if not _is_polygon_only(sub_result):
            return None

        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        collapsed = segmented_union_all(
            sub_result,
            np.asarray([0, sub_result.row_count], dtype=np.int64),
        )
        if collapsed.row_count != 1:
            return None
        return collapsed

    left_valid = np.asarray(left.validity, dtype=bool)
    right_valid = np.asarray(right.validity, dtype=bool)
    active_rows = left_valid & right_valid

    if not active_rows.any():
        return _empty_device_constructive_output(left.row_count)

    active_row_indices = np.flatnonzero(active_rows).astype(np.int64, copy=False)
    active_left = left.take(active_row_indices)
    active_right = right.take(active_row_indices)

    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    # Materializing an overlay execution plan is not a pure read-only
    # operation for every downstream consumer. The row-preserving batched
    # difference helper needs both an area-intersection classification pass
    # and the final difference result; reuse of the same plan across both
    # materializations can leak mutated assembly state into the second call
    # on complex aligned batches. Build a fresh plan for the difference
    # phase so the topology result is computed from pristine inputs.
    intersection_plan = _build_overlay_execution_plan(
        active_left,
        active_right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=None,
        _row_isolated=True,
    )
    area_intersection, _ = _materialize_overlay_execution_plan(
        intersection_plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
        preserve_row_count=active_left.row_count,
    )
    if area_intersection.row_count != active_left.row_count:
        return None

    area_state = area_intersection._ensure_device_state()
    d_area_valid = area_state.validity.astype(cp.bool_, copy=False)

    d_preserve_rows = cp.flatnonzero(cp.asarray(left_valid & ~right_valid)).astype(cp.int64)
    d_fallback_local = cp.flatnonzero(~d_area_valid).astype(cp.int64, copy=False)
    d_partial_local = cp.flatnonzero(d_area_valid).astype(cp.int64, copy=False)

    partial_result: OwnedGeometryArray | None = None
    if int(d_partial_local.size) > 0:
        difference_plan = _build_overlay_execution_plan(
            active_left,
            active_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
            _row_isolated=True,
        )
        partial_result, _ = _materialize_overlay_execution_plan(
            difference_plan,
            operation="difference",
            requested=ExecutionMode.GPU,
            preserve_row_count=active_left.row_count,
        )
        if partial_result.row_count != active_left.row_count:
            return None

    fallback_result: OwnedGeometryArray | None = None
    if int(d_fallback_local.size) > 0:
        fallback_left = active_left.take(d_fallback_local)
        fallback_right = active_right.take(d_fallback_local)
        # Rows with invalid polygonal area-intersection are exactly the cases
        # that the batched difference planner classifies as structurally tricky
        # (touch-only, disjoint, lower-dimensional, or other degenerate
        # topologies). Route that compacted subset through the proven rowwise
        # fallback instead of trying to preserve compacted row identity through
        # another batched overlay materialization.
        fallback_result = _dispatch_polygon_difference_overlay_rowwise_gpu_legacy(
            fallback_left,
            fallback_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
        )
        if fallback_result is None or fallback_result.row_count != fallback_left.row_count:
            return None

    preserve_rows = cp.asnumpy(d_preserve_rows)  # zcopy:ok(final row-assembly metadata after batched GPU classification; tiny row-id materialization only)
    partial_local = cp.asnumpy(d_partial_local)
    fallback_local = cp.asnumpy(d_fallback_local)
    out_validity = cp.zeros(left.row_count, dtype=cp.bool_)
    out_tags = cp.full(left.row_count, NULL_TAG, dtype=cp.int8)
    out_family_row_offsets = cp.full(left.row_count, -1, dtype=cp.int32)
    family_buffers: dict[GeometryFamily, list] = {}
    family_row_bases: dict[GeometryFamily, int] = {}
    partial_pos = {
        int(active_row_indices[int(local_pos)]): int(local_pos)
        for local_pos in partial_local.tolist()
    }
    fallback_pos = {
        int(active_row_indices[int(local_pos)]): compact_pos
        for compact_pos, local_pos in enumerate(fallback_local.tolist())
    }
    preserve_set = set(preserve_rows.tolist())

    for row_index in range(left.row_count):
        if not left_valid[row_index]:
            continue

        if row_index in preserve_set:
            sub_result = left.take(cp.asarray([row_index], dtype=cp.int64))
        elif row_index in partial_pos:
            if partial_result is None:
                return None
            sub_result = partial_result.take(
                cp.asarray([partial_pos[row_index]], dtype=cp.int64)
            )
        elif row_index in fallback_pos:
            if fallback_result is None:
                return None
            sub_result = fallback_result.take(
                cp.asarray([fallback_pos[row_index]], dtype=cp.int64)
            )
        else:
            continue

        sub_result = _collapse_to_single_row(sub_result)
        if sub_result is None:
            return None
        if sub_result.row_count != 1:
            return None
        sub_state = sub_result._ensure_device_state()
        if not bool(sub_state.validity[0]):
            continue

        tag_value = int(sub_state.tags[0])
        family = TAG_FAMILIES.get(tag_value)
        if family is None:
            return None

        buffer = sub_state.families.get(family)
        if buffer is None:
            return None

        base_row = family_row_bases.get(family, 0)
        out_validity[row_index] = sub_state.validity[0]
        out_tags[row_index] = sub_state.tags[0]
        out_family_row_offsets[row_index] = base_row
        family_buffers.setdefault(family, []).append(buffer)
        family_row_bases[family] = base_row + (int(buffer.geometry_offsets.size) - 1)

    device_families = {
        family: _concat_device_family_buffers(family, buffers)
        for family, buffers in family_buffers.items()
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=left.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _dispatch_polygon_difference_overlay_rowwise_gpu_legacy(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Legacy per-row fallback for polygon difference via overlay."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    if left.row_count != right.row_count:
        return None

    def _collapse_to_single_row(sub_result: OwnedGeometryArray) -> OwnedGeometryArray | None:
        if sub_result.row_count <= 1:
            return sub_result
        if not _is_polygon_only(sub_result):
            return None

        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        collapsed = segmented_union_all(
            sub_result,
            np.asarray([0, sub_result.row_count], dtype=np.int64),
        )
        if collapsed.row_count != 1:
            return None
        return collapsed

    from vibespatial.kernels.constructive.polygon_difference import (
        polygon_difference as _single_row_polygon_difference,
    )
    empty_row = _empty_device_constructive_output(1)
    row_results: list[OwnedGeometryArray] = []

    for row_index in range(left.row_count):
        if not bool(left.validity[row_index]):
            row_results.append(empty_row)
            continue

        d_row = cp.asarray([row_index], dtype=cp.int64)
        left_row = left.take(d_row)
        right_row = right.take(d_row)
        try:
            sub_result = _single_row_polygon_difference(
                left_row,
                right_row,
                dispatch_mode=dispatch_mode,
            )
        except Exception:
            logger.debug(
                "single-row polygon_difference kernel failed; falling back to overlay",
                exc_info=True,
            )
            sub_result = _dispatch_overlay_gpu(
                "difference",
                left_row,
                right_row,
                dispatch_mode=dispatch_mode,
                # The cached segment table is built for the original aligned
                # multi-row right operand. Reusing it after right.take([i])
                # breaks row isolation and can leak full-batch topology into a
                # single-row overlay subcall.
                _cached_right_segments=None,
            )
        sub_result = _collapse_to_single_row(sub_result)
        if sub_result is None:
            logger.debug(
                "row-preserving polygon difference fallback could not collapse multi-row result",
            )
            return None
        if sub_result.row_count == 0:
            row_results.append(empty_row)
            continue
        if sub_result.row_count != 1:
            logger.debug(
                "row-preserving polygon difference fallback expected 1 row, got %d",
                sub_result.row_count,
            )
            return None

        sub_state = sub_result._ensure_device_state()
        if not bool(sub_state.validity[0]):
            row_results.append(empty_row)
            continue

        row_results.append(sub_result)

    return OwnedGeometryArray.concat(row_results)


def _pair_supports_gpu_constructive(
    op: str,
    left_family: GeometryFamily,
    right_family: GeometryFamily,
) -> bool:
    """Return True when the current GPU dispatcher can handle this family pair."""
    if left_family is GeometryFamily.POINT and right_family is GeometryFamily.POINT:
        return op in _CONSTRUCTIVE_OPS
    if left_family is GeometryFamily.POINT and right_family in _POLYGONAL_FAMILIES:
        return op in _POINT_POLYGON_OPS
    if left_family in _POLYGONAL_FAMILIES and right_family is GeometryFamily.POINT:
        return op in {"intersection", "difference"}
    if left_family is GeometryFamily.POINT and right_family in _LINESTRING_FAMILIES:
        return op in {"intersection", "difference"}
    if left_family in _LINESTRING_FAMILIES and right_family is GeometryFamily.POINT:
        return op == "intersection"
    if left_family is GeometryFamily.MULTIPOINT and right_family in _POLYGONAL_FAMILIES:
        return op in _POINT_POLYGON_OPS
    if left_family in _POLYGONAL_FAMILIES and right_family is GeometryFamily.MULTIPOINT:
        return op in {"intersection", "difference"}
    if left_family in _LINESTRING_FAMILIES and right_family in _POLYGONAL_FAMILIES:
        return op in {"intersection", "difference"}
    if left_family in _POLYGONAL_FAMILIES and right_family in _LINESTRING_FAMILIES:
        return op == "intersection"
    if left_family in _LINESTRING_FAMILIES and right_family in _LINESTRING_FAMILIES:
        return op == "intersection"
    if left_family in _POLYGONAL_FAMILIES and right_family in _POLYGONAL_FAMILIES:
        return op in _CONSTRUCTIVE_OPS
    return False


def _supports_gpu_constructive(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    """Return True when every valid family pair in the workload is GPU-supported."""
    valid_mask = left.validity & right.validity
    if not valid_mask.any():
        return True

    left_tags = left.tags[valid_mask]
    right_tags = right.tags[valid_mask]
    for left_tag, right_tag in unique_tag_pairs(left_tags, right_tags):
        left_family = TAG_FAMILIES.get(left_tag)
        right_family = TAG_FAMILIES.get(right_tag)
        if left_family is None or right_family is None:
            return False
        if not _pair_supports_gpu_constructive(op, left_family, right_family):
            return False
    return True


def _needs_grouped_gpu_dispatch(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    """Return True when the workload spans multiple valid family pairs."""
    if _is_polygon_only(left) and _is_polygon_only(right):
        return False
    valid_mask = left.validity & right.validity
    if not valid_mask.any():
        return False
    return len(unique_tag_pairs(left.tags[valid_mask], right.tags[valid_mask])) > 1


def _empty_device_constructive_output(row_count: int) -> OwnedGeometryArray:
    """Build an all-null device-resident constructive result."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for GPU constructive output assembly")
    return build_device_resident_owned(
        device_families={},
        row_count=row_count,
        tags=cp.full(row_count, NULL_TAG, dtype=cp.int8),
        validity=cp.zeros(row_count, dtype=cp.bool_),
        family_row_offsets=cp.full(row_count, -1, dtype=cp.int32),
        execution_mode="gpu",
    )


def _single_row_polygon_difference_exact_correction(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Correct single-row touch/disjoint/containment difference edge cases."""
    try:
        area_intersection = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )
        if area_intersection is None or area_intersection.row_count != 1:
            return None

        area_state = area_intersection._ensure_device_state()
        if (
            not bool(area_state.validity[0])
            or not _is_polygon_only(area_intersection)
        ):
            from vibespatial.constructive.representative_point import (
                representative_point_owned,
            )
            from vibespatial.kernels.core.geometry_analysis import (
                compute_geometry_bounds,
            )
            from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

            left_bounds = compute_geometry_bounds(left)
            right_bounds = compute_geometry_bounds(right)
            if (
                left_bounds[0, 0] >= right_bounds[0, 0]
                and left_bounds[0, 1] >= right_bounds[0, 1]
                and left_bounds[0, 2] <= right_bounds[0, 2]
                and left_bounds[0, 3] <= right_bounds[0, 3]
            ):
                rep = representative_point_owned(
                    left,
                    dispatch_mode=ExecutionMode.GPU,
                )
                inside_mask = point_in_polygon(rep, right)
                if bool(inside_mask[0]):
                    return _empty_device_constructive_output(1)
            return left

        from vibespatial.constructive.measurement import area_owned
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

        left_bounds = compute_geometry_bounds(left)
        inter_bounds = compute_geometry_bounds(area_intersection)
        if np.allclose(left_bounds, inter_bounds, atol=1e-9, rtol=1e-9):
            left_area = float(area_owned(left)[0])
            inter_area = float(area_owned(area_intersection)[0])
            if np.isclose(left_area, inter_area, atol=1e-9, rtol=1e-9):
                return _empty_device_constructive_output(1)
    except Exception:
        logger.debug(
            "single-row polygon difference exact correction failed",
            exc_info=True,
        )
    return None


def _dispatch_mixed_binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Group mixed-family workloads by tag pair and reuse homogeneous GPU kernels."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()

    d_valid_rows = cp.flatnonzero(left_state.validity & right_state.validity)
    if int(d_valid_rows.size) == 0:
        return _empty_device_constructive_output(left.row_count)

    valid_left_tags = left_state.tags[d_valid_rows]
    valid_right_tags = right_state.tags[d_valid_rows]

    out_validity = cp.zeros(left.row_count, dtype=cp.bool_)
    out_tags = cp.full(left.row_count, NULL_TAG, dtype=cp.int8)
    out_family_row_offsets = cp.full(left.row_count, -1, dtype=cp.int32)
    family_buffers: dict[GeometryFamily, list] = {}
    family_row_bases: dict[GeometryFamily, int] = {}

    for left_tag, right_tag in unique_tag_pairs(valid_left_tags, valid_right_tags):
        sub_mask = (valid_left_tags == left_tag) & (valid_right_tags == right_tag)
        d_sub_rows = d_valid_rows[sub_mask].astype(cp.int64, copy=False)
        if int(d_sub_rows.size) == 0:
            continue

        left_subset = left.take(d_sub_rows)
        right_subset = right.take(d_sub_rows)
        sub_result = _binary_constructive_gpu(
            op,
            left_subset,
            right_subset,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )
        if sub_result is None:
            return None

        sub_state = sub_result._ensure_device_state()
        out_validity[d_sub_rows] = sub_state.validity
        out_tags[d_sub_rows] = sub_state.tags

        for family, buffer in sub_state.families.items():
            family_mask = sub_state.tags == FAMILY_TAGS[family]
            if not bool(cp.any(family_mask)):
                continue
            base_row = family_row_bases.get(family, 0)
            out_family_row_offsets[d_sub_rows[family_mask]] = (
                sub_state.family_row_offsets[family_mask] + base_row
            )
            family_buffers.setdefault(family, []).append(buffer)
            family_row_bases[family] = base_row + (int(buffer.geometry_offsets.size) - 1)

    device_families = {
        family: _concat_device_family_buffers(family, buffers)
        for family, buffers in family_buffers.items()
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=left.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------


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
    _cached_right_segments: DeviceSegmentTable | None = None,
    _prefer_exact_polygon_intersection: bool = False,
    _prefer_rowwise_polygon_difference_overlay: bool = False,
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
    _cached_right_segments : DeviceSegmentTable, optional
        Pre-extracted right-side segments for reuse (lyy.15).
    """
    if _needs_grouped_gpu_dispatch(left, right):
        try:
            return _dispatch_mixed_binary_constructive_gpu(
                op,
                left,
                right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=_cached_right_segments,
            )
        except Exception:
            logger.debug("Mixed-family GPU %s failed", op, exc_info=True)
            return None

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
            elif op == "difference":
                return left
        except Exception:
            logger.debug("Polygon-Point GPU %s failed", op, exc_info=True)
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
            elif op == "difference":
                return left
        except Exception:
            logger.debug("Polygon-MultiPoint GPU %s failed", op, exc_info=True)
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
            logger.debug("Polygon-LineString GPU %s failed", op, exc_info=True)
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
        if op == "intersection" and _prefer_exact_polygon_intersection:
            try:
                rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=_cached_right_segments,
                )
                if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                    return rowwise_result
            except Exception:
                logger.debug(
                    "preferred exact rowwise polygon intersection GPU path failed",
                    exc_info=True,
                )

        if op == "difference" and left.row_count == 1:
            correction = _single_row_polygon_difference_exact_correction(
                left,
                right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=_cached_right_segments,
            )
            if correction is not None:
                return correction

        # Element-wise constructive semantics require per-row isolation.
        # The general overlay pipeline is optimized for overlay surfaces and
        # can return row-count-preserving but cross-row-contaminated results
        # on aligned multi-row workloads. For multi-row polygon batches,
        # prefer the existing rowwise GPU overlay helpers once we leave the
        # direct single-pair kernels.
        prefer_rowwise_overlay = left.row_count > 1
        if op == "difference" and _prefer_rowwise_polygon_difference_overlay:
            prefer_rowwise_overlay = True

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
            if not prefer_rowwise_overlay:
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

        if prefer_rowwise_overlay:
            try:
                contraction_result = None
                if not (
                    op == "difference"
                    and _prefer_rowwise_polygon_difference_overlay
                ):
                    contraction_result = _dispatch_polygon_contraction_gpu(
                        op,
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                    )
                if contraction_result is not None and contraction_result.row_count == left.row_count:
                    return contraction_result

                if op == "intersection":
                    rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                elif op == "difference":
                    rowwise_result = _dispatch_polygon_difference_overlay_rowwise_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                else:
                    rowwise_result = _dispatch_polygon_overlay_rowwise_gpu(
                        op,
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                    return rowwise_result
            except Exception:
                logger.debug(
                    "row-preserving overlay GPU fast path failed for %s",
                    op,
                    exc_info=True,
                )

        # Fall through to the general overlay pipeline for union,
        # symmetric_difference, or when direct kernels fail.
        try:
            result = _dispatch_overlay_gpu(
                op, left, right, dispatch_mode=dispatch_mode,
                _cached_right_segments=_cached_right_segments,
            )
            if result.row_count == left.row_count:
                return result
            logger.debug(
                "overlay GPU dispatch returned %d rows (expected %d) for %s",
                result.row_count,
                left.row_count,
                op,
            )
            if op == "intersection":
                rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=_cached_right_segments,
                )
                if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                    return rowwise_result
            if op == "difference":
                rowwise_result = _dispatch_polygon_difference_overlay_rowwise_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=_cached_right_segments,
                )
                if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                    return rowwise_result
        except Exception:
            logger.debug(
                "overlay GPU dispatch failed for %s, falling back to CPU",
                op,
                exc_info=True,
            )
            if op == "intersection":
                try:
                    rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                    if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                        return rowwise_result
                except Exception:
                    logger.debug(
                        "row-preserving overlay GPU fallback failed for %s",
                        op,
                        exc_info=True,
                    )
            if op == "difference":
                try:
                    rowwise_result = _dispatch_polygon_difference_overlay_rowwise_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                    if rowwise_result is not None and rowwise_result.row_count == left.row_count:
                        return rowwise_result
                except Exception:
                    logger.debug(
                        "row-preserving overlay GPU fallback failed for %s",
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
    from vibespatial.constructive.multipoint_polygon_constructive import (
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
    _cached_right_segments: DeviceSegmentTable | None = None,
    workload_shape: WorkloadShape | None = None,
    _prefer_exact_polygon_intersection: bool = False,
    _prefer_rowwise_polygon_difference_overlay: bool = False,
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
    _cached_right_segments : DeviceSegmentTable, optional
        Pre-extracted right-side device segments for reuse (lyy.15).
        Passed through to the overlay pipeline to avoid redundant
        segment extraction in N-vs-1 overlay loops.
    """
    if op not in _CONSTRUCTIVE_OPS:
        raise ValueError(f"unsupported constructive operation: {op}")

    from vibespatial.runtime.crossover import detect_workload_shape

    workload = workload_shape or detect_workload_shape(left.row_count, right.row_count)

    # Broadcast-right: tile the 1-row right to match left.  The tiling
    # only replicates the tiny metadata arrays (validity, tags,
    # family_row_offsets) and shares the coordinate buffers, so this is
    # O(N) in int8/int32/bool, not O(N * vertex_count) in fp64.
    is_broadcast = workload in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT)
    if is_broadcast:
        right = tile_single_row(right, left.row_count)

    if left.row_count == 0:
        return from_shapely_geometries([])

    # Force CPU when grid_size is set (GPU pipeline doesn't support snapped precision)
    effective_mode = dispatch_mode
    if grid_size is not None:
        effective_mode = ExecutionMode.CPU
    normalized_effective_mode = (
        effective_mode if isinstance(effective_mode, ExecutionMode) else ExecutionMode(effective_mode)
    )
    strict_gpu_promotion = (
        normalized_effective_mode is ExecutionMode.AUTO
        and strict_native_mode_enabled()
        and _supports_gpu_constructive(op, left, right)
    )
    if strict_gpu_promotion:
        effective_mode = ExecutionMode.GPU

    selection = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        requested_mode=effective_mode,
        requested_precision=precision,
        workload_shape=workload,
    )
    selection_reason = selection.reason
    if strict_gpu_promotion:
        selection_reason = (
            f"{selection_reason}; strict-native promoted {workload.value} {op} to GPU"
        )

    gpu_attempted = False
    if selection.selected is ExecutionMode.GPU:
        # Broadcast-right tiles share coordinate buffers (family
        # row_count == 1) which GPU kernels cannot index by global row.
        # Materialize physically replicated coordinate buffers so that
        # family row_count == n, enabling direct kernel indexing.
        if is_broadcast:
            right = materialize_broadcast(right)
        # ADR-0002: CONSTRUCTIVE kernels stay fp64.  precision_plan is
        # computed for observability (dispatch event detail) only; the
        # overlay and PIP kernels manage their own precision internally.
        precision_plan = selection.precision_plan
        gpu_attempted = True
        result = _binary_constructive_gpu(
            op, left, right, dispatch_mode=selection.selected,
            _cached_right_segments=_cached_right_segments,
            _prefer_exact_polygon_intersection=_prefer_exact_polygon_intersection,
            _prefer_rowwise_polygon_difference_overlay=_prefer_rowwise_polygon_difference_overlay,
        )
        if result is not None:
            record_dispatch_event(
                surface=f"geopandas.array.{op}",
                operation=op,
                implementation="binary_constructive_gpu",
                reason=selection_reason,
                detail=(
                    f"rows={left.row_count}, "
                    f"precision={precision_plan.compute_precision.value}, "
                    f"workload={workload.value}"
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
        fallback_reason = selection_reason

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
        detail=f"rows={left.row_count}, op={op}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
        pipeline="binary_constructive_owned",
        d2h_transfer=gpu_attempted,  # D2H transfer occurs when GPU was attempted but fell back
    )

    result = binary_constructive_cpu(op, left, right, grid_size=grid_size)
    record_dispatch_event(
        surface=f"geopandas.array.{op}",
        operation=op,
        implementation="binary_constructive_cpu",
        reason=fallback_reason,
        detail=f"rows={left.row_count}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
