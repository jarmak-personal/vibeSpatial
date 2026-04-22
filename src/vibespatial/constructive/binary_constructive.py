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
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    _concat_device_family_buffers,
    _device_gather_offset_slices,
    build_device_resident_owned,
    device_concat_owned_scatter,
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
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency

logger = logging.getLogger(__name__)

# Constructive operations that this module handles
_CONSTRUCTIVE_OPS = frozenset({"intersection", "union", "difference", "symmetric_difference"})

# Polygon-family types supported by the GPU overlay pipeline
_POLYGONAL_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# LineString-family types
_LINESTRING_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})

# Point-Polygon constructive operations supported by the PIP fast path
_POINT_POLYGON_OPS = frozenset({"intersection", "difference"})

# Contraction has a high fixed setup cost from microcell labeling and
# reconstruction. Small aligned polygon batches are faster on the existing
# row-isolated overlay helpers than on the contraction path.
_POLYGON_CONTRACTION_MIN_ROWS = 32

# Tiny mixed polygonal intersection batches spend more time partitioning by
# tag pair and exploding multipolygons than they do running one row-isolated
# exact overlay across the aligned batch.
_MIXED_POLYGON_INTERSECTION_ROWWISE_MAX = 16

# For aligned polygon intersections, the direct Sutherland-Hodgman kernel is
# much cheaper than full topology overlay, but the legacy dispatch was
# all-or-nothing for the entire batch. Partition only when there is enough
# eligible work to amortize the scatter and remainder overlay.
_PARTITIONED_SH_INTERSECTION_MIN_ROWS = 32

# Direct multipart intersection packing is only exact when fragments from the
# same source row cannot overlap or touch.  Keep the proof cheap and bounded; if
# a group needs a larger pairwise disjointness probe, use the exact union plan.
_DIRECT_MULTIPART_PACK_MAX_PAIR_PROBE = 512

# Exact polygon union rescue is intentionally a tiny-batch fallback.  Larger
# aligned batches must use the batched partition/coverage path so union does
# not devolve into one topology graph per row.
_POLYGON_UNION_EXACT_ROW_FALLBACK_MAX = 8

# Overlay can emit sub-microscopic near-collinear polygons that are valid but
# have no material area contribution at projected-coordinate scale.  If exact
# partition union cannot produce polygon pieces for such a pair, preserve the
# dominant area operand instead of fragmenting the dissolve with a sliver part.
_POLYGON_UNION_DEGENERATE_AREA_RTOL = 1.0e-9


def _sync_hotpath() -> None:
    if hotpath_trace_enabled():
        from vibespatial.cuda._runtime import get_cuda_runtime

        get_cuda_runtime().synchronize()


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


def _resolve_indexed_polygon_fast_path_candidate(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Materialize indexed polygon batches before exact GPU fast-path probes.

    Gathered pair batches often arrive as device-side indexed views whose
    family buffers still carry compact row counts from the source arrays.
    Rectangle/exact polygon kernels expect physically materialized family
    rows, so resolve the indexed view on the active residency before probing.
    """
    if not owned.is_indexed_view:
        return owned
    if owned.residency is Residency.DEVICE or owned.device_state is not None:
        return owned._device_resolve()
    return owned._resolve()


def _host_single_ring_polygon_mask(
    owned: OwnedGeometryArray,
    *,
    max_input_vertices: int = 256,
) -> np.ndarray | None:
    """Classify rows that fit the single-ring polygon kernel contract."""
    if set(owned.families) != {GeometryFamily.POLYGON}:
        return None

    owned._ensure_host_state()
    polygon_buf = owned.families[GeometryFamily.POLYGON]
    if polygon_buf.row_count != owned.row_count or polygon_buf.ring_offsets is None:
        return None
    if len(polygon_buf.geometry_offsets) != owned.row_count + 1:
        return None

    mask = np.zeros(owned.row_count, dtype=bool)
    geom_offsets = np.asarray(polygon_buf.geometry_offsets, dtype=np.int64)
    ring_offsets = np.asarray(polygon_buf.ring_offsets, dtype=np.int64)
    x = np.asarray(polygon_buf.x, dtype=np.float64)
    y = np.asarray(polygon_buf.y, dtype=np.float64)

    for row in range(owned.row_count):
        ring_start = int(geom_offsets[row])
        ring_end = int(geom_offsets[row + 1])
        if ring_end - ring_start != 1:
            continue
        start = int(ring_offsets[ring_start])
        end = int(ring_offsets[ring_start + 1])
        n = end - start
        if n > 1:
            dx = float(x[start] - x[end - 1])
            dy = float(y[start] - y[end - 1])
            if (dx * dx + dy * dy) <= 1e-24:
                n -= 1
        if 3 <= n <= max_input_vertices:
            mask[row] = True
    return mask


def _host_rectangle_polygon_mask(owned: OwnedGeometryArray) -> np.ndarray | None:
    """Classify rows that are exact axis-aligned rectangles."""
    single_ring_mask = _host_single_ring_polygon_mask(owned, max_input_vertices=4)
    if single_ring_mask is None:
        return None

    polygon_buf = owned.families[GeometryFamily.POLYGON]
    geom_offsets = np.asarray(polygon_buf.geometry_offsets, dtype=np.int64)
    ring_offsets = np.asarray(polygon_buf.ring_offsets, dtype=np.int64)
    x = np.asarray(polygon_buf.x, dtype=np.float64)
    y = np.asarray(polygon_buf.y, dtype=np.float64)

    rect_mask = np.zeros(owned.row_count, dtype=bool)
    for row in np.flatnonzero(single_ring_mask):
        ring_row = int(geom_offsets[int(row)])
        start = int(ring_offsets[ring_row])
        end = int(ring_offsets[ring_row + 1])
        coords_x = x[start:end]
        coords_y = y[start:end]
        if coords_x.size > 1:
            dx = float(coords_x[0] - coords_x[-1])
            dy = float(coords_y[0] - coords_y[-1])
            if (dx * dx + dy * dy) <= 1e-24:
                coords_x = coords_x[:-1]
                coords_y = coords_y[:-1]
        if coords_x.size != 4:
            continue
        xmin = float(np.min(coords_x))
        xmax = float(np.max(coords_x))
        ymin = float(np.min(coords_y))
        ymax = float(np.max(coords_y))
        if not (xmin < xmax and ymin < ymax):
            continue
        corners = {
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymin),
            (xmax, ymax),
        }
        row_points = list(zip(coords_x.tolist(), coords_y.tolist(), strict=True))
        row_corners = {
            (float(px), float(py))
            for px, py in row_points
        }
        if row_corners != corners:
            continue
        is_axis_aligned = True
        for index, (x0, y0) in enumerate(row_points):
            x1, y1 = row_points[(index + 1) % len(row_points)]
            same_x = abs(float(x0) - float(x1)) <= 1e-12
            same_y = abs(float(y0) - float(y1)) <= 1e-12
            if same_x == same_y:
                is_axis_aligned = False
                break
        rect_mask[int(row)] = is_axis_aligned
    return rect_mask


def _host_single_ring_and_rectangle_polygon_masks(
    owned: OwnedGeometryArray,
    *,
    max_input_vertices: int = 256,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Classify polygon rows inside pure or mixed polygonal owned batches.

    The public overlay planner often carries mixed ``Polygon``/``MultiPolygon``
    batches after an earlier clip.  Rectangle fast paths should still peel off
    the plain polygon rows instead of forcing the whole batch through exact
    overlay.
    """
    if GeometryFamily.POLYGON not in owned.families:
        empty = np.zeros(owned.row_count, dtype=bool)
        return empty, empty.copy()
    if set(owned.families) == {GeometryFamily.POLYGON}:
        simple = _host_single_ring_polygon_mask(
            owned,
            max_input_vertices=max_input_vertices,
        )
        if simple is None:
            return None
        rect = _host_rectangle_polygon_mask(owned)
        if rect is None:
            rect = np.zeros(owned.row_count, dtype=bool)
        return simple, rect

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    tags = np.asarray(owned.tags)
    validity = np.asarray(owned.validity, dtype=bool)
    polygon_rows = np.flatnonzero((tags == poly_tag) & validity).astype(
        np.intp,
        copy=False,
    )
    simple = np.zeros(owned.row_count, dtype=bool)
    rect = np.zeros(owned.row_count, dtype=bool)
    if polygon_rows.size == 0:
        return simple, rect

    polygon_owned = owned.take(polygon_rows)
    if set(polygon_owned.families) != {GeometryFamily.POLYGON}:
        return simple, rect
    polygon_owned._ensure_host_state()
    polygon_buf = polygon_owned.families[GeometryFamily.POLYGON]
    if (
        polygon_buf.row_count != polygon_owned.row_count
        or polygon_buf.ring_offsets is None
        or len(polygon_buf.geometry_offsets) != polygon_owned.row_count + 1
    ):
        return None

    geom_offsets = np.asarray(polygon_buf.geometry_offsets, dtype=np.int64)
    ring_offsets = np.asarray(polygon_buf.ring_offsets, dtype=np.int64)
    x = np.asarray(polygon_buf.x, dtype=np.float64)
    y = np.asarray(polygon_buf.y, dtype=np.float64)

    for local_row, global_row in enumerate(polygon_rows):
        ring_start = int(geom_offsets[local_row])
        ring_end = int(geom_offsets[local_row + 1])
        if ring_end - ring_start != 1:
            continue
        start = int(ring_offsets[ring_start])
        end = int(ring_offsets[ring_start + 1])
        coords_x = x[start:end]
        coords_y = y[start:end]
        if coords_x.size > 1:
            dx = float(coords_x[0] - coords_x[-1])
            dy = float(coords_y[0] - coords_y[-1])
            if (dx * dx + dy * dy) <= 1e-24:
                coords_x = coords_x[:-1]
                coords_y = coords_y[:-1]
        if 3 <= coords_x.size <= max_input_vertices:
            simple[int(global_row)] = True
        if coords_x.size != 4:
            continue
        xmin = float(np.min(coords_x))
        xmax = float(np.max(coords_x))
        ymin = float(np.min(coords_y))
        ymax = float(np.max(coords_y))
        if not (xmin < xmax and ymin < ymax):
            continue
        row_points = list(zip(coords_x.tolist(), coords_y.tolist(), strict=True))
        row_corners = {(float(px), float(py)) for px, py in row_points}
        corners = {
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymin),
            (xmax, ymax),
        }
        if row_corners != corners:
            continue
        is_axis_aligned = True
        for index, (x0, y0) in enumerate(row_points):
            x1, y1 = row_points[(index + 1) % len(row_points)]
            same_x = abs(float(x0) - float(x1)) <= 1e-12
            same_y = abs(float(y0) - float(y1)) <= 1e-12
            if same_x == same_y:
                is_axis_aligned = False
                break
        rect[int(global_row)] = is_axis_aligned
    return simple, rect


def _dispatch_mixed_polygon_rect_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Use the rectangle kernel on row subsets within mixed polygon batches."""
    if cp is None or left.row_count != right.row_count or left.row_count == 0:
        return None
    if left.row_count > 4096:
        return None

    left = _resolve_indexed_polygon_fast_path_candidate(left)
    right = _resolve_indexed_polygon_fast_path_candidate(right)

    left_masks = _host_single_ring_and_rectangle_polygon_masks(left)
    right_masks = _host_single_ring_and_rectangle_polygon_masks(right)
    if left_masks is None or right_masks is None:
        return None
    left_simple, left_rect = left_masks
    right_simple, right_rect = right_masks

    right_rect_rows = left_simple & right_rect
    left_rect_rows = right_simple & left_rect & ~right_rect_rows
    handled_rows = right_rect_rows | left_rect_rows
    if not handled_rows.any():
        return None

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection,
    )

    result = _empty_device_constructive_output(left.row_count)
    boundary_overlap = cp.zeros(left.row_count, dtype=cp.bool_)
    exact_polygon_only = cp.zeros(left.row_count, dtype=cp.bool_)
    used_rect_kernel = False

    if right_rect_rows.any():
        right_rect_indices = np.flatnonzero(right_rect_rows)
        d_rows = cp.asarray(right_rect_indices, dtype=cp.int64)
        subset = polygon_rect_intersection(
            left.take(d_rows),
            right.take(d_rows),
            dispatch_mode=ExecutionMode.GPU,
        )
        result = device_concat_owned_scatter(result, subset, d_rows)
        subset_overlap = getattr(subset, "_polygon_rect_boundary_overlap", None)
        if subset_overlap is not None:
            boundary_overlap[d_rows] = cp.asarray(subset_overlap, dtype=cp.bool_)
        exact_polygon_only[d_rows] = cp.asarray(left_rect[right_rect_indices], dtype=cp.bool_)
        used_rect_kernel = True

    if left_rect_rows.any():
        d_rows = cp.asarray(np.flatnonzero(left_rect_rows), dtype=cp.int64)
        subset = polygon_rect_intersection(
            right.take(d_rows),
            left.take(d_rows),
            dispatch_mode=ExecutionMode.GPU,
        )
        result = device_concat_owned_scatter(result, subset, d_rows)
        subset_overlap = getattr(subset, "_polygon_rect_boundary_overlap", None)
        if subset_overlap is not None:
            boundary_overlap[d_rows] = cp.asarray(subset_overlap, dtype=cp.bool_)
        used_rect_kernel = True

    remainder_rows = np.flatnonzero(~handled_rows)
    if remainder_rows.size > 0:
        d_rows = cp.asarray(remainder_rows, dtype=cp.int64)
        remainder = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            left.take(d_rows),
            right.take(d_rows),
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
        )
        if remainder is None or remainder.row_count != remainder_rows.size:
            return None
        result = device_concat_owned_scatter(result, remainder, d_rows)
    if used_rect_kernel:
        result._polygon_rect_boundary_overlap = boundary_overlap
        result._polygon_rect_exact_polygon_only = exact_polygon_only

    return result


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

            rings_per_row = np.diff(geom_offsets)

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


def _aligned_sh_eligible_polygon_rows(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """Return aligned row positions safe for the direct SH intersection kernel."""
    from vibespatial.kernels.constructive.polygon_intersection import (
        _MAX_CLIP_VERTS,
    )
    from vibespatial.overlay.bypass import _is_convex_ring_xy

    if left.row_count != right.row_count or left.row_count == 0:
        return np.empty(0, dtype=np.int64)
    if (
        GeometryFamily.POLYGON not in left.families
        or GeometryFamily.POLYGON not in right.families
    ):
        return np.empty(0, dtype=np.int64)

    left._ensure_host_family_structure(GeometryFamily.POLYGON)
    right._ensure_host_family_structure(GeometryFamily.POLYGON)
    left_buf = left.families[GeometryFamily.POLYGON]
    right_buf = right.families[GeometryFamily.POLYGON]
    if left_buf.row_count == 0 or right_buf.row_count == 0:
        return np.empty(0, dtype=np.int64)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    valid = np.asarray(left.validity, dtype=bool) & np.asarray(right.validity, dtype=bool)
    candidate_mask = (
        valid
        & (np.asarray(left.tags) == poly_tag)
        & (np.asarray(right.tags) == poly_tag)
    )
    candidate_rows = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    if candidate_rows.size == 0:
        return candidate_rows

    left_fro = np.asarray(left.family_row_offsets, dtype=np.int64)[candidate_rows]
    right_fro = np.asarray(right.family_row_offsets, dtype=np.int64)[candidate_rows]
    if (
        np.any(left_fro < 0)
        or np.any(right_fro < 0)
        or np.any(left_fro >= left_buf.row_count)
        or np.any(right_fro >= right_buf.row_count)
    ):
        return np.empty(0, dtype=np.int64)

    left_geom_offsets = np.asarray(left_buf.geometry_offsets, dtype=np.int64)
    right_geom_offsets = np.asarray(right_buf.geometry_offsets, dtype=np.int64)
    left_ring_offsets = np.asarray(left_buf.ring_offsets, dtype=np.int64)
    right_ring_offsets = np.asarray(right_buf.ring_offsets, dtype=np.int64)
    left_rings_per_row = np.diff(left_geom_offsets)
    right_rings_per_row = np.diff(right_geom_offsets)

    left_single_ring = left_rings_per_row[left_fro] == 1
    right_single_ring = right_rings_per_row[right_fro] == 1
    single_ring = left_single_ring & right_single_ring
    if not bool(np.any(single_ring)):
        return np.empty(0, dtype=np.int64)

    left_first_ring = left_geom_offsets[left_fro]
    right_first_ring = right_geom_offsets[right_fro]
    left_vertex_counts = (
        left_ring_offsets[left_first_ring + 1] - left_ring_offsets[left_first_ring]
    )
    right_vertex_counts = (
        right_ring_offsets[right_first_ring + 1] - right_ring_offsets[right_first_ring]
    )
    workspace_ok = (left_vertex_counts + right_vertex_counts) <= _MAX_CLIP_VERTS
    local_mask = single_ring & workspace_ok
    if not bool(np.any(local_mask)):
        return np.empty(0, dtype=np.int64)

    convex_right = np.zeros(candidate_rows.size, dtype=bool)
    unique_right_rows = np.unique(right_fro[local_mask])
    right_convex_cache: dict[int, bool] = {}
    right_x = np.asarray(right_buf.x, dtype=np.float64)
    right_y = np.asarray(right_buf.y, dtype=np.float64)
    for right_row in unique_right_rows:
        right_row_int = int(right_row)
        ring_idx = int(right_geom_offsets[right_row_int])
        start = int(right_ring_offsets[ring_idx])
        end = int(right_ring_offsets[ring_idx + 1])
        right_convex_cache[right_row_int] = _is_convex_ring_xy(
            right_x,
            right_y,
            start,
            end,
        )
    for right_row, is_convex in right_convex_cache.items():
        if is_convex:
            convex_right[right_fro == right_row] = True

    eligible = local_mask & convex_right
    return candidate_rows[eligible].astype(np.int64, copy=False)


def _dispatch_partitioned_polygon_intersection_sh_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Run SH-eligible aligned rows through the direct kernel, overlay the rest."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count < _PARTITIONED_SH_INTERSECTION_MIN_ROWS:
        return None

    sh_rows = _aligned_sh_eligible_polygon_rows(left, right)
    sh_count = int(sh_rows.size)
    if (
        sh_count < _PARTITIONED_SH_INTERSECTION_MIN_ROWS
        or sh_count == left.row_count
    ):
        return None

    valid_rows = np.flatnonzero(
        np.asarray(left.validity, dtype=bool) & np.asarray(right.validity, dtype=bool)
    ).astype(np.int64, copy=False)
    remainder_rows = np.setdiff1d(valid_rows, sh_rows, assume_unique=True)

    try:
        from vibespatial.kernels.constructive.polygon_intersection import (
            polygon_intersection,
        )

        out = _empty_device_constructive_output(left.row_count)
        d_sh_rows = cp.asarray(sh_rows, dtype=cp.int64)
        sh_result = polygon_intersection(
            left.take(d_sh_rows),
            right.take(d_sh_rows),
            dispatch_mode=dispatch_mode,
        )
        if sh_result.row_count != sh_count:
            return None
        out = device_concat_owned_scatter(out, sh_result, d_sh_rows)

        if remainder_rows.size > 0:
            d_remainder_rows = cp.asarray(remainder_rows, dtype=cp.int64)
            remainder = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                left.take(d_remainder_rows),
                right.take(d_remainder_rows),
                dispatch_mode=dispatch_mode,
                _cached_right_segments=None,
            )
            if remainder is None or remainder.row_count != int(remainder_rows.size):
                return None
            out = device_concat_owned_scatter(out, remainder, d_remainder_rows)

        record_dispatch_event(
            surface="vibespatial.constructive.binary_constructive",
            operation="intersection",
            implementation="partitioned_sh_plus_overlay_gpu",
            reason=(
                "aligned polygon intersection partitioned SH-eligible rows "
                "from complex overlay rows"
            ),
            detail=(
                f"rows={left.row_count}, sh_rows={sh_count}, "
                f"overlay_rows={int(remainder_rows.size)}"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
        return out
    except Exception:
        logger.debug(
            "partitioned SH polygon intersection GPU path failed; "
            "falling back to existing exact overlay path",
            exc_info=True,
        )
        return None


def _dispatch_polygon_intersection_sh_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Run the direct SH kernel, swapping operands when the clip side is convex."""
    try:
        from vibespatial.kernels.constructive.polygon_intersection import (
            polygon_intersection,
        )

        if _sh_kernel_can_handle(left, right):
            result = polygon_intersection(left, right, dispatch_mode=dispatch_mode)
            if result.row_count == left.row_count:
                return result

        # Intersection is commutative. Sutherland-Hodgman only requires the
        # clip operand to be convex, so mask-clip workloads with convex source
        # rows and one concave mask can still use the direct kernel by swapping.
        if _sh_kernel_can_handle(right, left):
            result = polygon_intersection(right, left, dispatch_mode=dispatch_mode)
            if result.row_count == left.row_count:
                record_dispatch_event(
                    surface="vibespatial.constructive.binary_constructive",
                    operation="intersection",
                    implementation="swapped_sh_polygon_intersection_gpu",
                    reason=(
                        "polygon intersection used commutative SH dispatch "
                        "because the original left side was the convex clip operand"
                    ),
                    detail=f"rows={left.row_count}",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return result
    except Exception:
        logger.debug(
            "direct SH polygon intersection GPU path failed; "
            "falling back to exact overlay",
            exc_info=True,
        )
    return None


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


def _collapse_polygon_overlay_sub_result(
    sub_result: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Collapse a polygonal overlay sub-result back to one logical row."""
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


def _free_device_segment_table(device_segments: DeviceSegmentTable) -> None:
    """Release a cached device segment table allocated by _extract_segments_gpu."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime.free(device_segments.x0)
    runtime.free(device_segments.y0)
    runtime.free(device_segments.x1)
    runtime.free(device_segments.y1)
    runtime.free(device_segments.row_indices)
    runtime.free(device_segments.segment_indices)
    if device_segments.part_indices is not None:
        runtime.free(device_segments.part_indices)
    if device_segments.ring_indices is not None:
        runtime.free(device_segments.ring_indices)


def _broadcast_right_cached_segments(
    right: OwnedGeometryArray,
    row_count: int,
    *,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> DeviceSegmentTable | None:
    """Expand a scalar-right segment table to N logical right rows on device."""
    if cp is None or row_count <= 1 or right.row_count != 1:
        return _cached_right_segments

    from vibespatial.spatial.segment_primitives import (
        DeviceSegmentTable,
        _extract_segments_gpu,
    )

    source_segments = _cached_right_segments
    owns_source_segments = False
    if source_segments is None:
        source_segments = _extract_segments_gpu(right)
        owns_source_segments = True

    segment_count = int(source_segments.count)
    if segment_count == 0:
        return source_segments

    try:
        broadcast = DeviceSegmentTable(
            row_indices=cp.repeat(cp.arange(row_count, dtype=cp.int32), segment_count),
            segment_indices=cp.tile(
                cp.asarray(source_segments.segment_indices, dtype=cp.int32),
                row_count,
            ),
            x0=cp.tile(source_segments.x0, row_count),
            y0=cp.tile(source_segments.y0, row_count),
            x1=cp.tile(source_segments.x1, row_count),
            y1=cp.tile(source_segments.y1, row_count),
            count=segment_count * row_count,
            part_indices=(
                None
                if source_segments.part_indices is None
                else cp.tile(
                    cp.asarray(source_segments.part_indices, dtype=cp.int32),
                    row_count,
                )
            ),
            ring_indices=(
                None
                if source_segments.ring_indices is None
                else cp.tile(
                    cp.asarray(source_segments.ring_indices, dtype=cp.int32),
                    row_count,
                )
            ),
        )
    finally:
        if owns_source_segments and source_segments is not None:
            _free_device_segment_table(source_segments)

    return broadcast


def _expand_right_segments_for_pair_rows(
    right: OwnedGeometryArray,
    source_rows: np.ndarray,
) -> DeviceSegmentTable | None:
    """Expand unique right-row segments across pair rows without re-extraction."""
    if cp is None:
        return None
    source_rows = np.asarray(source_rows, dtype=np.int32)
    if source_rows.size == 0:
        return None

    from vibespatial.spatial.segment_primitives import (
        DeviceSegmentTable,
        _extract_segments_gpu,
    )

    unique_rows, inverse = np.unique(source_rows, return_inverse=True)
    right_unique = right.take(unique_rows.astype(np.intp, copy=False))
    base_segments = _extract_segments_gpu(right_unique)
    if int(base_segments.count) == 0:
        return base_segments

    try:
        d_base_row_indices = cp.asarray(base_segments.row_indices, dtype=cp.int32)
        if int(d_base_row_indices.size) != int(base_segments.count):
            raise RuntimeError("segment row-index count mismatch")
        if int(d_base_row_indices.size) > 1:
            monotonic = bool(cp.all(d_base_row_indices[1:] >= d_base_row_indices[:-1]).item())
            if not monotonic:
                raise RuntimeError("segment rows are not grouped by right row")

        group_count = int(unique_rows.size)
        pair_counts = np.bincount(inverse, minlength=group_count).astype(np.int64, copy=False)
        d_pair_counts = cp.asarray(pair_counts, dtype=cp.int64)
        d_segment_counts = cp.bincount(
            d_base_row_indices.astype(cp.int32, copy=False),
            minlength=group_count,
        ).astype(cp.int64, copy=False)
        d_expanded_counts = d_pair_counts * d_segment_counts
        total = int(cp.sum(d_expanded_counts).item())
        if total == 0:
            return None

        d_pair_offsets = cp.empty(group_count + 1, dtype=cp.int64)
        d_pair_offsets[0] = 0
        cp.cumsum(d_pair_counts, out=d_pair_offsets[1:])

        d_segment_offsets = cp.empty(group_count + 1, dtype=cp.int64)
        d_segment_offsets[0] = 0
        cp.cumsum(d_segment_counts, out=d_segment_offsets[1:])

        d_expanded_offsets = cp.empty(group_count + 1, dtype=cp.int64)
        d_expanded_offsets[0] = 0
        cp.cumsum(d_expanded_counts, out=d_expanded_offsets[1:])

        pair_order = np.argsort(inverse, kind="stable").astype(np.int32, copy=False)
        d_pair_row_ids = cp.asarray(pair_order, dtype=cp.int32)
        d_out_positions = cp.arange(total, dtype=cp.int64)
        d_group_ids = cp.searchsorted(
            d_expanded_offsets[1:],
            d_out_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        d_local = d_out_positions - d_expanded_offsets[d_group_ids]
        d_group_segment_counts = d_segment_counts[d_group_ids]
        d_pair_local = d_local // d_group_segment_counts
        d_segment_local = d_local - (d_pair_local * d_group_segment_counts)
        d_pair_indices = d_pair_offsets[d_group_ids] + d_pair_local
        d_segment_indices = d_segment_offsets[d_group_ids] + d_segment_local

        return DeviceSegmentTable(
            row_indices=d_pair_row_ids[d_pair_indices].astype(cp.int32, copy=False),
            segment_indices=cp.asarray(
                base_segments.segment_indices,
                dtype=cp.int32,
            )[d_segment_indices],
            x0=base_segments.x0[d_segment_indices],
            y0=base_segments.y0[d_segment_indices],
            x1=base_segments.x1[d_segment_indices],
            y1=base_segments.y1[d_segment_indices],
            count=total,
            part_indices=(
                None
                if base_segments.part_indices is None
                else cp.asarray(base_segments.part_indices, dtype=cp.int32)[d_segment_indices]
            ),
            ring_indices=(
                None
                if base_segments.ring_indices is None
                else cp.asarray(base_segments.ring_indices, dtype=cp.int32)[d_segment_indices]
            ),
        )
    except Exception:
        logger.debug(
            "vectorized pair-row right-segment expansion failed; falling back to legacy path",
            exc_info=True,
        )
        return _expand_right_segments_for_pair_rows_legacy(right, source_rows)
    finally:
        _free_device_segment_table(base_segments)


def _expand_right_segments_for_pair_rows_legacy(
    right: OwnedGeometryArray,
    source_rows: np.ndarray,
) -> DeviceSegmentTable | None:
    """Expand unique right-row segments across pair rows without re-extraction."""
    if cp is None:
        return None
    source_rows = np.asarray(source_rows, dtype=np.int32)
    if source_rows.size == 0:
        return None

    from vibespatial.spatial.segment_primitives import (
        DeviceSegmentTable,
        _extract_segments_gpu,
    )

    unique_rows, inverse = np.unique(source_rows, return_inverse=True)
    right_unique = right.take(unique_rows.astype(np.intp, copy=False))
    base_segments = _extract_segments_gpu(right_unique)
    if int(base_segments.count) == 0:
        return base_segments

    base_row_indices = cp.asarray(base_segments.row_indices, dtype=cp.int32)
    segment_indices_parts = []
    x0_parts = []
    y0_parts = []
    x1_parts = []
    y1_parts = []
    row_index_parts = []
    part_index_parts = [] if base_segments.part_indices is not None else None
    ring_index_parts = [] if base_segments.ring_indices is not None else None

    for unique_local_row in range(unique_rows.size):
        pair_rows = np.flatnonzero(inverse == unique_local_row).astype(np.int32, copy=False)
        if pair_rows.size == 0:
            continue
        d_segment_rows = cp.flatnonzero(base_row_indices == unique_local_row).astype(
            cp.int64,
            copy=False,
        )
        if int(d_segment_rows.size) == 0:
            continue

        d_pair_rows = cp.asarray(pair_rows, dtype=cp.int32)
        row_index_parts.append(cp.repeat(d_pair_rows, int(d_segment_rows.size)))
        segment_indices_parts.append(
            cp.tile(
                cp.asarray(base_segments.segment_indices[d_segment_rows], dtype=cp.int32),
                pair_rows.size,
            )
        )
        x0_parts.append(cp.tile(base_segments.x0[d_segment_rows], pair_rows.size))
        y0_parts.append(cp.tile(base_segments.y0[d_segment_rows], pair_rows.size))
        x1_parts.append(cp.tile(base_segments.x1[d_segment_rows], pair_rows.size))
        y1_parts.append(cp.tile(base_segments.y1[d_segment_rows], pair_rows.size))
        if part_index_parts is not None:
            part_index_parts.append(
                cp.tile(
                    cp.asarray(base_segments.part_indices[d_segment_rows], dtype=cp.int32),
                    pair_rows.size,
                )
            )
        if ring_index_parts is not None:
            ring_index_parts.append(
                cp.tile(
                    cp.asarray(base_segments.ring_indices[d_segment_rows], dtype=cp.int32),
                    pair_rows.size,
                )
            )

    try:
        if not row_index_parts:
            return None
        return DeviceSegmentTable(
            row_indices=cp.concatenate(row_index_parts),
            segment_indices=cp.concatenate(segment_indices_parts),
            x0=cp.concatenate(x0_parts),
            y0=cp.concatenate(y0_parts),
            x1=cp.concatenate(x1_parts),
            y1=cp.concatenate(y1_parts),
            count=int(sum(int(part.size) for part in row_index_parts)),
            part_indices=(
                None if part_index_parts is None else cp.concatenate(part_index_parts)
            ),
            ring_indices=(
                None if ring_index_parts is None else cp.concatenate(ring_index_parts)
            ),
        )
    finally:
        _free_device_segment_table(base_segments)


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

    if left.row_count > 1 and op != "union":
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

    if op == "union":
        return _dispatch_polygon_union_repair_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
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

        sub_result = None

        if sub_result is None:
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
        sub_result = _collapse_polygon_overlay_sub_result(sub_result)
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


def _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Preserve row cardinality for polygon intersection against a scalar right polygon.

    This keeps the right operand truly broadcast-right: one right geometry,
    one extracted right-side segment table, many left rows. It avoids
    materializing a tiled right OwnedGeometryArray just to feed the exact
    overlay path.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if right.row_count != 1:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    right_valid = bool(right.validity[0])
    if not right_valid:
        return _empty_device_constructive_output(left.row_count)

    cached_right_segments = _cached_right_segments
    owns_cached_right_segments = False
    if cached_right_segments is None and left.row_count > 1:
        from vibespatial.spatial.segment_primitives import _extract_segments_gpu

        cached_right_segments = _extract_segments_gpu(right)
        owns_cached_right_segments = True

    out_validity = cp.zeros(left.row_count, dtype=cp.bool_)
    out_tags = cp.full(left.row_count, NULL_TAG, dtype=cp.int8)
    out_family_row_offsets = cp.full(left.row_count, -1, dtype=cp.int32)
    family_buffers: dict[GeometryFamily, list] = {}
    family_row_bases: dict[GeometryFamily, int] = {}

    try:
        for row_index in range(left.row_count):
            if not bool(left.validity[row_index]):
                continue

            d_row = cp.asarray([row_index], dtype=cp.int64)
            left_row = left.take(d_row)
            sub_result = _dispatch_overlay_gpu(
                "intersection",
                left_row,
                right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=cached_right_segments,
                _row_isolated=True,
            )
            sub_result = _collapse_polygon_overlay_sub_result(sub_result)
            if sub_result is None:
                logger.debug(
                    "broadcast-right polygon intersection could not collapse "
                    "multi-row exact sub-result",
                )
                return None
            if sub_result.row_count == 0:
                continue
            if sub_result.row_count != 1:
                logger.debug(
                    "broadcast-right polygon intersection expected 1 row, got %d",
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
                    "broadcast-right polygon intersection saw unsupported tag %d",
                    tag_value,
                )
                return None

            buffer = sub_state.families.get(family)
            if buffer is None:
                logger.debug(
                    "broadcast-right polygon intersection missing %s family buffer",
                    family.value,
                )
                return None

            base_row = family_row_bases.get(family, 0)
            out_validity[row_index] = sub_state.validity[0]
            out_tags[row_index] = sub_state.tags[0]
            out_family_row_offsets[row_index] = base_row
            family_buffers.setdefault(family, []).append(buffer)
            family_row_bases[family] = base_row + (int(buffer.geometry_offsets.size) - 1)
    finally:
        if owns_cached_right_segments and cached_right_segments is not None:
            _free_device_segment_table(cached_right_segments)

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
    if left.row_count < _POLYGON_CONTRACTION_MIN_ROWS:
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


def _explode_multipolygon_rows_to_polygons_gpu(
    owned: OwnedGeometryArray,
) -> tuple[OwnedGeometryArray, DeviceArray]:
    """Explode device-resident multipolygon rows into polygon rows on GPU.

    Returns
    -------
    exploded : OwnedGeometryArray
        Polygon-only rows, one per polygon part.
    source_rows : DeviceArray
        Global source row id for each exploded polygon row.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for multipolygon explosion")

    state = owned._ensure_device_state()
    multipolygon_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    d_global_rows = cp.flatnonzero(
        state.validity & (state.tags == multipolygon_tag)
    ).astype(cp.int64, copy=False)
    if int(d_global_rows.size) == 0:
        return _empty_device_constructive_output(0), cp.empty(0, dtype=cp.int64)

    buffer = state.families.get(GeometryFamily.MULTIPOLYGON)
    if buffer is None:
        return _empty_device_constructive_output(0), cp.empty(0, dtype=cp.int64)

    d_family_rows = state.family_row_offsets[d_global_rows].astype(cp.int64, copy=False)
    d_part_counts = (
        buffer.geometry_offsets[d_family_rows + 1] - buffer.geometry_offsets[d_family_rows]
    ).astype(cp.int32, copy=False)
    d_nonempty = d_part_counts > 0
    if not bool(cp.any(d_nonempty)):
        return _empty_device_constructive_output(0), cp.empty(0, dtype=cp.int64)

    d_global_rows = d_global_rows[d_nonempty]
    d_family_rows = d_family_rows[d_nonempty]
    d_part_counts = d_part_counts[d_nonempty]

    total_parts = int(d_part_counts.sum())
    d_part_starts = buffer.geometry_offsets[d_family_rows].astype(cp.int64, copy=False)
    d_part_offsets = cp.empty(int(d_part_counts.size) + 1, dtype=cp.int64)
    d_part_offsets[0] = 0
    cp.cumsum(d_part_counts.astype(cp.int64, copy=False), out=d_part_offsets[1:])
    d_positions = cp.arange(total_parts, dtype=cp.int64)
    d_segment_ids = cp.searchsorted(d_part_offsets, d_positions, side="right") - 1
    d_source_rows = d_global_rows[d_segment_ids]
    d_part_indices = (
        d_positions
        - d_part_offsets[d_segment_ids]
        + d_part_starts[d_segment_ids]
    ).astype(cp.int64, copy=False)

    d_ring_space = cp.arange(buffer.ring_offsets.size, dtype=cp.int32)
    d_ring_indices, d_geom_offsets = _device_gather_offset_slices(
        d_ring_space,
        buffer.part_offsets,
        d_part_indices,
    )
    d_ring_indices = d_ring_indices.astype(cp.int64, copy=False)

    d_coords = (
        cp.column_stack([buffer.x, buffer.y])
        if int(buffer.x.size) > 0
        else cp.empty((0, 2), dtype=cp.float64)
    )
    d_gathered, d_ring_offsets = _device_gather_offset_slices(
        d_coords,
        buffer.ring_offsets,
        d_ring_indices,
    )
    d_x = d_gathered[:, 0].copy() if int(d_gathered.size) else cp.empty(0, dtype=cp.float64)
    d_y = d_gathered[:, 1].copy() if int(d_gathered.size) else cp.empty(0, dtype=cp.float64)

    polygon_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=d_x,
        y=d_y,
        geometry_offsets=d_geom_offsets,
        empty_mask=cp.zeros(total_parts, dtype=cp.bool_),
        ring_offsets=d_ring_offsets,
        bounds=None,
    )
    exploded = build_device_resident_owned(
        device_families={GeometryFamily.POLYGON: polygon_buffer},
        row_count=total_parts,
        tags=cp.full(total_parts, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(total_parts, dtype=cp.bool_),
        family_row_offsets=cp.arange(total_parts, dtype=cp.int32),
        execution_mode="gpu",
    )
    return exploded, d_source_rows


def _dispatch_multipolygon_polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact GPU rescue for homogeneous MultiPolygon-Polygon intersection.

    Explodes multipolygon rows into polygon parts on device, intersects each
    part against its aligned polygon row using the existing exact polygon
    path, then unions the valid part intersections back per original row.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    left_is_mpoly = _is_family_only(left, frozenset({GeometryFamily.MULTIPOLYGON}))
    right_is_poly = _is_family_only(right, frozenset({GeometryFamily.POLYGON}))
    left_is_poly = _is_family_only(left, frozenset({GeometryFamily.POLYGON}))
    right_is_mpoly = _is_family_only(right, frozenset({GeometryFamily.MULTIPOLYGON}))

    if not ((left_is_mpoly and right_is_poly) or (left_is_poly and right_is_mpoly)):
        return None

    if right_is_mpoly:
        return _dispatch_multipolygon_polygon_intersection_gpu(
            right,
            left,
            dispatch_mode=dispatch_mode,
        )

    _sync_hotpath()
    with hotpath_stage("constructive.intersection.multipart_explode", category="setup"):
        exploded_left, d_source_rows = _explode_multipolygon_rows_to_polygons_gpu(left)
    _sync_hotpath()
    if exploded_left.row_count == 0:
        return _empty_device_constructive_output(left.row_count)

    exploded_right = right.take(d_source_rows)
    try:
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection,
            polygon_rect_intersection_can_handle,
        )

        if polygon_rect_intersection_can_handle(exploded_left, exploded_right):
            part_result = polygon_rect_intersection(
                exploded_left,
                exploded_right,
                dispatch_mode=ExecutionMode.GPU,
            )
            if part_result is not None and part_result.row_count == exploded_left.row_count:
                d_valid_parts = cp.flatnonzero(
                    part_result._ensure_device_state().validity,
                ).astype(cp.int64, copy=False)
                if int(d_valid_parts.size) == 0:
                    return _empty_device_constructive_output(left.row_count)
                return _regroup_intersection_parts_with_grouped_union_gpu(
                    part_result.take(d_valid_parts),
                    d_source_rows[d_valid_parts].astype(cp.int64, copy=False),
                    output_row_count=left.row_count,
                    dispatch_mode=dispatch_mode,
                )

        if polygon_rect_intersection_can_handle(exploded_right, exploded_left):
            part_result = polygon_rect_intersection(
                exploded_right,
                exploded_left,
                dispatch_mode=ExecutionMode.GPU,
            )
            if part_result is not None and part_result.row_count == exploded_left.row_count:
                d_valid_parts = cp.flatnonzero(
                    part_result._ensure_device_state().validity,
                ).astype(cp.int64, copy=False)
                if int(d_valid_parts.size) == 0:
                    return _empty_device_constructive_output(left.row_count)
                return _regroup_intersection_parts_with_grouped_union_gpu(
                    part_result.take(d_valid_parts),
                    d_source_rows[d_valid_parts].astype(cp.int64, copy=False),
                    output_row_count=left.row_count,
                    dispatch_mode=dispatch_mode,
                )
    except Exception:
        logger.debug(
            "multipart-polygon rectangle intersection GPU rescue failed",
            exc_info=True,
        )

    part_result = _dispatch_polygon_intersection_sh_gpu(
        exploded_left,
        exploded_right,
        dispatch_mode=dispatch_mode,
    )
    if part_result is None:
        part_result = _dispatch_partitioned_polygon_intersection_sh_gpu(
            exploded_left,
            exploded_right,
            dispatch_mode=dispatch_mode,
        )
    if part_result is None:
        part_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            exploded_left,
            exploded_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
        )
    if part_result is None or part_result.row_count != exploded_left.row_count:
        return None

    part_state = part_result._ensure_device_state()
    d_valid_parts = cp.flatnonzero(part_state.validity).astype(cp.int64, copy=False)
    if int(d_valid_parts.size) == 0:
        return _empty_device_constructive_output(left.row_count)

    valid_parts = part_result.take(d_valid_parts)
    d_valid_source_rows = d_source_rows[d_valid_parts].astype(cp.int64, copy=False)
    return _regroup_intersection_parts_with_grouped_union_gpu(
        valid_parts,
        d_valid_source_rows,
        output_row_count=left.row_count,
        dispatch_mode=dispatch_mode,
    )


def _regroup_intersection_parts_with_grouped_union_gpu(
    valid_parts: OwnedGeometryArray,
    d_valid_source_rows: DeviceArray,
    *,
    output_row_count: int,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Regroup multipart intersection pieces with one grouped overlay union plan.

    Each original multipolygon row contributes one or more polygonal
    intersection fragments in ``valid_parts``. The correct high-performance
    shape is:

    - choose one seed fragment per non-empty source row
    - group all remaining fragments by that same source row
    - run one row-isolated overlay union plan across the full grouped batch

    This replaces the previous ``segmented_union_all`` tree reduction, which
    split back into many single-row pairwise union overlay executions.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    direct = _pack_disjoint_multipart_intersection_parts_gpu(
        valid_parts,
        d_valid_source_rows,
        output_row_count=output_row_count,
    )
    if direct is not None:
        return direct

    from vibespatial.kernels.constructive.segmented_union import segmented_union_all
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    _sync_hotpath()
    with hotpath_stage("constructive.intersection.multipart_group_counts", category="setup"):
        d_counts = cp.bincount(
            d_valid_source_rows,
            minlength=output_row_count,
        ).astype(cp.int64, copy=False)
        d_nonempty_rows = cp.flatnonzero(d_counts > 0).astype(cp.int64, copy=False)
    _sync_hotpath()
    if int(d_nonempty_rows.size) == 0:
        return _empty_device_constructive_output(output_row_count)

    d_group_counts = d_counts[d_nonempty_rows]
    d_seed_positions = cp.cumsum(d_group_counts, dtype=cp.int64) - d_group_counts
    seed_parts = valid_parts.take(d_seed_positions)
    compact: OwnedGeometryArray | None = None

    if int(valid_parts.row_count) == int(seed_parts.row_count):
        compact = seed_parts
    else:
        d_rest_counts = d_group_counts - 1
        _sync_hotpath()
        with hotpath_stage("constructive.intersection.multipart_union.group_rows", category="setup"):
            d_rest_offsets = cp.empty(d_rest_counts.size + 1, dtype=cp.int64)
            d_rest_offsets[0] = 0
            cp.cumsum(d_rest_counts, out=d_rest_offsets[1:])
            rest_total = int(d_rest_offsets[-1])
            d_rest_positions = cp.arange(rest_total, dtype=cp.int64)
            d_rest_group_ids = (
                cp.searchsorted(d_rest_offsets, d_rest_positions, side="right").astype(cp.int64)
                - 1
            )
            d_rest_positions = (
                d_rest_positions
                - d_rest_offsets[d_rest_group_ids].astype(cp.int64, copy=False)
                + d_seed_positions[d_rest_group_ids].astype(cp.int64, copy=False)
                + 1
            )
            rest_parts = valid_parts.take(d_rest_positions)
            d_right_group_rows = d_rest_group_ids.astype(cp.int32, copy=False)
        _sync_hotpath()
        try:
            _sync_hotpath()
            with hotpath_stage("constructive.intersection.multipart_union.plan.build", category="setup"):
                plan = _build_overlay_execution_plan(
                    seed_parts,
                    rest_parts,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=None,
                    _row_isolated=True,
                    _right_geometry_source_rows=d_right_group_rows,
                )
            _sync_hotpath()
            with hotpath_stage("constructive.intersection.multipart_union.plan.materialize", category="refine"):
                compact, _selected = _materialize_overlay_execution_plan(
                    plan,
                    operation="union",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=seed_parts.row_count,
                )
            _sync_hotpath()
        except Exception:
            logger.debug(
                "grouped multipart intersection union failed; "
                "falling back to segmented_union_all",
                exc_info=True,
            )
            d_group_offsets = cp.empty(d_group_counts.size + 1, dtype=cp.int64)
            d_group_offsets[0] = 0
            cp.cumsum(d_group_counts, out=d_group_offsets[1:])
            _sync_hotpath()
            with hotpath_stage("constructive.intersection.multipart_union.fallback", category="refine"):
                compact = segmented_union_all(valid_parts, cp.asnumpy(d_group_offsets))
            _sync_hotpath()

    if compact is None or compact.row_count != int(d_nonempty_rows.size):
        return None

    return device_concat_owned_scatter(
        _empty_device_constructive_output(output_row_count),
        compact,
        d_nonempty_rows,
    )


def _pack_disjoint_multipart_intersection_parts_gpu(
    valid_parts: OwnedGeometryArray,
    d_valid_source_rows: DeviceArray,
    *,
    output_row_count: int,
) -> OwnedGeometryArray | None:
    """Regroup disjoint polygonal intersection fragments without overlay union.

    ``valid_parts`` comes from intersecting each exploded MultiPolygon part
    against the aligned mask polygon. Source MultiPolygon parts are disjoint by
    geometry semantics, and clipping them cannot introduce overlap, so regrouping
    only needs to pack fragments back into Polygon/MultiPolygon rows. Reopening
    the overlay union planner here adds topology work without changing area.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if valid_parts.row_count == 0:
        return _empty_device_constructive_output(output_row_count)
    if not _is_family_only(valid_parts, _POLYGONAL_FAMILIES):
        return None

    exploded = _explode_polygonal_rows_to_polygons_gpu(valid_parts)
    if exploded is None:
        return None
    polygon_parts, d_part_source_rows = exploded
    if polygon_parts.row_count == 0:
        return _empty_device_constructive_output(output_row_count)

    _sync_hotpath()
    with hotpath_stage("constructive.intersection.multipart_direct_pack.group", category="setup"):
        d_source_rows = d_valid_source_rows[
            cp.asarray(d_part_source_rows, dtype=cp.int64)
        ].astype(cp.int64, copy=False)
        part_count = int(d_source_rows.size)
        if part_count != polygon_parts.row_count:
            return None
        d_order_key = (
            d_source_rows * np.int64(max(part_count, 1))
            + cp.arange(part_count, dtype=cp.int64)
        )
        d_order = cp.argsort(d_order_key).astype(cp.int64, copy=False)
        d_sorted_source_rows = d_source_rows[d_order]
        d_counts = cp.bincount(
            d_sorted_source_rows,
            minlength=output_row_count,
        ).astype(cp.int32, copy=False)
        d_nonempty_rows = cp.flatnonzero(d_counts > 0).astype(cp.int64, copy=False)
        if int(d_nonempty_rows.size) == 0:
            return _empty_device_constructive_output(output_row_count)
        d_group_counts = d_counts[d_nonempty_rows].astype(cp.int32, copy=False)
        d_group_starts = cp.cumsum(d_group_counts, dtype=cp.int64) - d_group_counts
        d_single_compact_rows = cp.flatnonzero(d_group_counts == 1).astype(
            cp.int64, copy=False,
        )
        d_multi_compact_rows = cp.flatnonzero(d_group_counts > 1).astype(
            cp.int64, copy=False,
        )

    _sync_hotpath()
    with hotpath_stage("constructive.intersection.multipart_direct_pack.parts", category="emit"):
        sorted_parts = polygon_parts.take(d_order)
        sorted_state = sorted_parts._ensure_device_state()
        polygon_buffer = sorted_state.families.get(GeometryFamily.POLYGON)
        if polygon_buffer is None:
            return None
        if not _sorted_polygon_parts_have_strictly_disjoint_group_bounds(
            sorted_parts,
            d_sorted_source_rows,
        ):
            return None

        compact_family_row_offsets = cp.full(
            int(d_nonempty_rows.size),
            -1,
            dtype=cp.int32,
        )
        compact_tags = cp.full(
            int(d_nonempty_rows.size),
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            dtype=cp.int8,
        )
        compact_tags[d_single_compact_rows] = FAMILY_TAGS[GeometryFamily.POLYGON]
        device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

        if int(d_single_compact_rows.size) > 0:
            d_single_part_rows = d_group_starts[d_single_compact_rows].astype(
                cp.int64, copy=False,
            )
            single_buffer = _device_take_polygon_family_rows(
                polygon_buffer,
                d_single_part_rows,
            )
            device_families[GeometryFamily.POLYGON] = single_buffer
            compact_family_row_offsets[d_single_compact_rows] = cp.arange(
                int(d_single_compact_rows.size),
                dtype=cp.int32,
            )

        if int(d_multi_compact_rows.size) > 0:
            d_multi_counts = d_group_counts[d_multi_compact_rows].astype(
                cp.int32, copy=False,
            )
            d_multi_offsets = cp.empty(int(d_multi_counts.size) + 1, dtype=cp.int32)
            d_multi_offsets[0] = 0
            cp.cumsum(d_multi_counts, out=d_multi_offsets[1:])
            multi_part_total = int(d_multi_offsets[-1])
            d_multi_positions = cp.arange(multi_part_total, dtype=cp.int64)
            d_multi_group_ids = (
                cp.searchsorted(d_multi_offsets, d_multi_positions, side="right")
                .astype(cp.int64, copy=False)
                - 1
            )
            d_multi_part_rows = (
                d_multi_positions
                - d_multi_offsets[d_multi_group_ids].astype(cp.int64, copy=False)
                + d_group_starts[d_multi_compact_rows][d_multi_group_ids].astype(
                    cp.int64,
                    copy=False,
                )
            )
            multi_piece_buffer = _device_take_polygon_family_rows(
                polygon_buffer,
                d_multi_part_rows.astype(cp.int64, copy=False),
            )
            device_families[GeometryFamily.MULTIPOLYGON] = DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTIPOLYGON,
                x=multi_piece_buffer.x,
                y=multi_piece_buffer.y,
                geometry_offsets=d_multi_offsets,
                empty_mask=cp.zeros(int(d_multi_counts.size), dtype=cp.bool_),
                part_offsets=multi_piece_buffer.geometry_offsets,
                ring_offsets=multi_piece_buffer.ring_offsets,
                bounds=None,
            )
            compact_family_row_offsets[d_multi_compact_rows] = cp.arange(
                int(d_multi_compact_rows.size),
                dtype=cp.int32,
            )

        compact = build_device_resident_owned(
            device_families=device_families,
            row_count=int(d_nonempty_rows.size),
            tags=compact_tags,
            validity=cp.ones(int(d_nonempty_rows.size), dtype=cp.bool_),
            family_row_offsets=compact_family_row_offsets,
            execution_mode="gpu",
        )

    _sync_hotpath()
    with hotpath_stage("constructive.intersection.multipart_direct_pack.scatter", category="emit"):
        result = device_concat_owned_scatter(
            _empty_device_constructive_output(output_row_count),
            compact,
            d_nonempty_rows,
        )
    _sync_hotpath()
    record_dispatch_event(
        surface="vibespatial.constructive.binary_constructive",
        operation="intersection",
        implementation="direct_multipart_intersection_pack_gpu",
        reason=(
            "multipart polygon intersection regrouped disjoint fragments "
            "without reopening overlay union topology"
        ),
        detail=(
            f"parts={valid_parts.row_count}, rows={output_row_count}, "
            f"nonempty_rows={int(d_nonempty_rows.size)}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _sorted_polygon_parts_have_strictly_disjoint_group_bounds(
    sorted_parts: OwnedGeometryArray,
    d_sorted_source_rows: DeviceArray,
) -> bool:
    """Return True when same-source polygon fragments are bbox-disjoint."""
    part_count = sorted_parts.row_count
    if part_count <= 1:
        return True
    if part_count > _DIRECT_MULTIPART_PACK_MAX_PAIR_PROBE:
        return False

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_pair_i, d_pair_j = cp.triu_indices(part_count, k=1)
    d_same_source = d_sorted_source_rows[d_pair_i] == d_sorted_source_rows[d_pair_j]
    if not bool(cp.any(d_same_source)):
        return True

    d_bounds = cp.asarray(compute_geometry_bounds_device(sorted_parts), dtype=cp.float64)
    d_left = d_bounds[d_pair_i]
    d_right = d_bounds[d_pair_j]
    d_strictly_separated = (
        (d_left[:, 2] < d_right[:, 0])
        | (d_right[:, 2] < d_left[:, 0])
        | (d_left[:, 3] < d_right[:, 1])
        | (d_right[:, 3] < d_left[:, 1])
    )
    return not bool(cp.any(d_same_source & ~d_strictly_separated))


def _device_take_polygon_family_rows(
    buffer: DeviceFamilyGeometryBuffer,
    family_rows: DeviceArray,
) -> DeviceFamilyGeometryBuffer:
    """Gather polygon family rows from a device buffer."""
    d_ring_space = cp.arange(buffer.ring_offsets.size, dtype=cp.int32)
    d_ring_indices, d_geom_offsets = _device_gather_offset_slices(
        d_ring_space,
        buffer.geometry_offsets,
        family_rows,
    )
    d_ring_indices = d_ring_indices.astype(cp.int64, copy=False)
    d_coords = (
        cp.column_stack([buffer.x, buffer.y])
        if int(buffer.x.size) > 0
        else cp.empty((0, 2), dtype=cp.float64)
    )
    d_gathered, d_ring_offsets = _device_gather_offset_slices(
        d_coords,
        buffer.ring_offsets,
        d_ring_indices,
    )
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=d_gathered[:, 0].copy() if int(d_gathered.size) else cp.empty(0, dtype=cp.float64),
        y=d_gathered[:, 1].copy() if int(d_gathered.size) else cp.empty(0, dtype=cp.float64),
        geometry_offsets=d_geom_offsets,
        empty_mask=buffer.empty_mask[family_rows],
        ring_offsets=d_ring_offsets,
        bounds=None,
    )


def _assemble_disjoint_polygonal_pieces_gpu(
    pieces: list[OwnedGeometryArray],
) -> OwnedGeometryArray | None:
    """Assemble disjoint single-row polygonal pieces into one union geometry."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    polygon_buffers: list[DeviceFamilyGeometryBuffer] = []
    polygon_count = 0
    for piece in pieces:
        if piece.row_count == 0:
            continue
        state = piece._ensure_device_state()
        if not bool(state.validity[0]):
            continue
        family = TAG_FAMILIES.get(int(state.tags[0]))
        if family is GeometryFamily.POLYGON:
            buffer = state.families.get(GeometryFamily.POLYGON)
            if buffer is None:
                return None
            polygon_buffers.append(buffer)
            polygon_count += 1
            continue
        if family is GeometryFamily.MULTIPOLYGON:
            exploded, _ = _explode_multipolygon_rows_to_polygons_gpu(piece)
            if exploded.row_count == 0:
                continue
            exploded_state = exploded._ensure_device_state()
            buffer = exploded_state.families.get(GeometryFamily.POLYGON)
            if buffer is None:
                return None
            polygon_buffers.append(buffer)
            polygon_count += exploded.row_count
            continue
        return None

    if polygon_count == 0:
        return _empty_device_constructive_output(1)

    merged = _concat_device_family_buffers(GeometryFamily.POLYGON, polygon_buffers)
    if polygon_count == 1:
        return build_device_resident_owned(
            device_families={GeometryFamily.POLYGON: merged},
            row_count=1,
            tags=cp.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=cp.int8),
            validity=cp.asarray([True], dtype=cp.bool_),
            family_row_offsets=cp.asarray([0], dtype=cp.int32),
            execution_mode="gpu",
        )

    multipolygon_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=merged.x,
        y=merged.y,
        geometry_offsets=cp.asarray([0, polygon_count], dtype=cp.int32),
        empty_mask=cp.asarray([False], dtype=cp.bool_),
        part_offsets=merged.geometry_offsets,
        ring_offsets=merged.ring_offsets,
        bounds=None,
    )
    return build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: multipolygon_buffer},
        row_count=1,
        tags=cp.asarray([FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]], dtype=cp.int8),
        validity=cp.asarray([True], dtype=cp.bool_),
        family_row_offsets=cp.asarray([0], dtype=cp.int32),
        execution_mode="gpu",
    )


def _explode_single_polygon_family_row_to_polygons_gpu(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Return polygon rows for a single polygon-family geometry on device."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if owned.row_count != 1 or not bool(owned.validity[0]):
        return None

    state = owned._ensure_device_state()
    family = TAG_FAMILIES.get(int(state.tags[0]))
    if family is GeometryFamily.POLYGON:
        return owned
    if family is GeometryFamily.MULTIPOLYGON:
        exploded, _ = _explode_multipolygon_rows_to_polygons_gpu(owned)
        return exploded
    return None


def _build_atomic_edges_from_boundary_segments_gpu(
    start_x: DeviceArray,
    start_y: DeviceArray,
    end_x: DeviceArray,
    end_y: DeviceArray,
    *,
    row_indices: DeviceArray | None = None,
    runtime_selection,
):
    """Build atomic half-edges from surviving boundary segments."""
    from vibespatial.overlay.types import AtomicEdgeDeviceState, AtomicEdgeTable

    boundary_count = int(start_x.size)
    if boundary_count == 0:
        return None

    d_segment_ids = cp.arange(boundary_count, dtype=cp.int32)
    total_atomic = boundary_count * 2
    d_src_x = cp.empty(total_atomic, dtype=cp.float64)
    d_src_y = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_x = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_y = cp.empty(total_atomic, dtype=cp.float64)
    d_source_ids = cp.empty(total_atomic, dtype=cp.int32)
    d_direction = cp.empty(total_atomic, dtype=cp.int8)
    if row_indices is None:
        d_row_indices = cp.zeros(total_atomic, dtype=cp.int32)
    else:
        d_boundary_rows = cp.asarray(row_indices, dtype=cp.int32)
        if int(d_boundary_rows.size) != boundary_count:
            raise ValueError("boundary row index count must match segment count")
        d_row_indices = cp.empty(total_atomic, dtype=cp.int32)
        d_row_indices[0::2] = d_boundary_rows
        d_row_indices[1::2] = d_boundary_rows
    d_part_indices = cp.zeros(total_atomic, dtype=cp.int32)
    d_ring_indices = cp.empty(total_atomic, dtype=cp.int32)
    d_source_side = cp.ones(total_atomic, dtype=cp.int8)

    d_src_x[0::2] = start_x
    d_src_x[1::2] = end_x
    d_src_y[0::2] = start_y
    d_src_y[1::2] = end_y
    d_dst_x[0::2] = end_x
    d_dst_x[1::2] = start_x
    d_dst_y[0::2] = end_y
    d_dst_y[1::2] = start_y
    d_source_ids[0::2] = d_segment_ids
    d_source_ids[1::2] = d_segment_ids
    d_direction[0::2] = 1
    d_direction[1::2] = -1
    d_ring_indices[0::2] = d_segment_ids
    d_ring_indices[1::2] = d_segment_ids

    return AtomicEdgeTable(
        left_segment_count=boundary_count,
        right_segment_count=0,
        runtime_selection=runtime_selection,
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=d_source_ids,
            direction=d_direction,
            src_x=d_src_x,
            src_y=d_src_y,
            dst_x=d_dst_x,
            dst_y=d_dst_y,
            row_indices=d_row_indices,
            part_indices=d_part_indices,
            ring_indices=d_ring_indices,
            source_side=d_source_side,
        ),
        _count=total_atomic,
    )


def _explode_polygonal_rows_to_polygons_gpu(
    owned: OwnedGeometryArray,
) -> tuple[OwnedGeometryArray, DeviceArray] | None:
    """Explode polygon-family rows to polygon rows and keep source row ids."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    state = owned._ensure_device_state()
    polygon_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    multipolygon_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    valid_polygon_rows = cp.flatnonzero(
        state.validity & (state.tags == polygon_tag)
    ).astype(cp.int64, copy=False)
    valid_multipolygon_rows = cp.flatnonzero(
        state.validity & (state.tags == multipolygon_tag)
    ).astype(cp.int64, copy=False)

    polygon_parts: list[OwnedGeometryArray] = []
    source_rows: list[DeviceArray] = []

    if int(valid_polygon_rows.size) > 0:
        polygon_parts.append(owned.take(valid_polygon_rows))
        source_rows.append(valid_polygon_rows.astype(cp.int32, copy=False))

    if int(valid_multipolygon_rows.size) > 0:
        multipolygon_rows = owned.take(valid_multipolygon_rows)
        exploded, local_source_rows = _explode_multipolygon_rows_to_polygons_gpu(
            multipolygon_rows,
        )
        if exploded.row_count > 0:
            polygon_parts.append(exploded)
            source_rows.append(
                valid_multipolygon_rows[
                    cp.asarray(local_source_rows, dtype=cp.int64)
                ].astype(cp.int32, copy=False)
            )

    if not polygon_parts:
        return None
    if len(polygon_parts) == 1:
        return polygon_parts[0], source_rows[0]
    return OwnedGeometryArray.concat(polygon_parts), cp.concatenate(source_rows)


def _dispatch_row_aligned_polygon_known_coverage_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Coverage-union aligned polygon rows in one row-isolated GPU graph."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count == 0:
        return None
    if not (
        bool(cp.all(left._ensure_device_state().validity).item())
        and bool(cp.all(right._ensure_device_state().validity).item())
    ):
        return None

    from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
    from vibespatial.overlay.graph import _gpu_face_walk, build_gpu_half_edge_graph
    from vibespatial.overlay.types import OverlayFaceDeviceState, OverlayFaceTable

    left_exploded = _explode_polygonal_rows_to_polygons_gpu(left)
    right_exploded = _explode_polygonal_rows_to_polygons_gpu(right)
    if left_exploded is None or right_exploded is None:
        return None
    left_polygons, left_source_rows = left_exploded
    right_polygons, right_source_rows = right_exploded
    if left_polygons.row_count == 0 or right_polygons.row_count == 0:
        return None

    left_state = left_polygons._ensure_device_state()
    right_state = right_polygons._ensure_device_state()
    left_buffer = left_state.families.get(GeometryFamily.POLYGON)
    right_buffer = right_state.families.get(GeometryFamily.POLYGON)
    if left_buffer is None or right_buffer is None:
        return None

    merged = _concat_device_family_buffers(
        GeometryFamily.POLYGON,
        [left_buffer, right_buffer],
    )
    polygon_source_rows = cp.concatenate(
        [
            cp.asarray(left_source_rows, dtype=cp.int32),
            cp.asarray(right_source_rows, dtype=cp.int32),
        ]
    )
    d_x = cp.asarray(merged.x)
    d_y = cp.asarray(merged.y)
    d_geometry_offsets = cp.asarray(merged.geometry_offsets, dtype=cp.int64)
    d_ring_offsets = cp.asarray(merged.ring_offsets, dtype=cp.int64)
    if int(d_geometry_offsets.size) < 2 or int(d_ring_offsets.size) < 2 or int(d_x.size) < 2:
        return None

    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int64, copy=False)
    total_segments = int(cp.sum(d_seg_counts).item())
    if total_segments == 0:
        return None

    d_seg_offsets = cp.empty(int(d_seg_counts.size) + 1, dtype=cp.int64)
    d_seg_offsets[0] = 0
    cp.cumsum(d_seg_counts, out=d_seg_offsets[1:])
    d_seg_ids = cp.arange(total_segments, dtype=cp.int64)
    d_ring_of_seg = cp.searchsorted(
        d_seg_offsets[1:],
        d_seg_ids,
        side="right",
    ).astype(cp.int64, copy=False)
    d_polygon_of_ring = cp.searchsorted(
        d_geometry_offsets[1:],
        d_ring_of_seg,
        side="right",
    ).astype(cp.int64, copy=False)
    d_segment_rows = polygon_source_rows[d_polygon_of_ring]
    d_local_seg = d_seg_ids - d_seg_offsets[d_ring_of_seg]
    d_vertex_ids = d_ring_offsets[d_ring_of_seg] + d_local_seg

    start_x = d_x[d_vertex_ids]
    start_y = d_y[d_vertex_ids]
    end_x = d_x[d_vertex_ids + 1]
    end_y = d_y[d_vertex_ids + 1]

    swap = (
        (start_x > end_x)
        | ((start_x == end_x) & (start_y > end_y))
    )
    key_x0 = cp.where(swap, end_x, start_x)
    key_y0 = cp.where(swap, end_y, start_y)
    key_x1 = cp.where(swap, start_x, end_x)
    key_y1 = cp.where(swap, start_y, end_y)

    order = cp.lexsort(
        cp.stack((key_y1, key_x1, key_y0, key_x0, d_segment_rows))
    ).astype(cp.int64, copy=False)
    if int(order.size) == 0:
        return None
    sorted_rows = d_segment_rows[order]
    sorted_x0 = key_x0[order]
    sorted_y0 = key_y0[order]
    sorted_x1 = key_x1[order]
    sorted_y1 = key_y1[order]
    start_mask = cp.empty(int(order.size), dtype=cp.bool_)
    start_mask[0] = True
    if int(order.size) > 1:
        start_mask[1:] = (
            (sorted_rows[1:] != sorted_rows[:-1])
            | (sorted_x0[1:] != sorted_x0[:-1])
            | (sorted_y0[1:] != sorted_y0[:-1])
            | (sorted_x1[1:] != sorted_x1[:-1])
            | (sorted_y1[1:] != sorted_y1[:-1])
        )
    run_starts = cp.flatnonzero(start_mask).astype(cp.int64, copy=False)
    run_lengths = cp.diff(
        cp.concatenate((run_starts, cp.asarray([int(order.size)], dtype=cp.int64)))
    )
    boundary_orders = order[run_starts[run_lengths % 2 == 1]]
    if int(boundary_orders.size) == 0:
        return None

    selection = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=left.row_count,
        requested_mode=dispatch_mode,
        requested_precision=PrecisionMode.AUTO,
        current_residency=combined_residency(left, right),
    )
    atomic_edges = _build_atomic_edges_from_boundary_segments_gpu(
        start_x[boundary_orders],
        start_y[boundary_orders],
        end_x[boundary_orders],
        end_y[boundary_orders],
        row_indices=d_segment_rows[boundary_orders],
        runtime_selection=selection.runtime_selection,
    )
    if atomic_edges is None:
        return None

    half_edge_graph = build_gpu_half_edge_graph(atomic_edges, isolate_rows=True)
    if half_edge_graph.edge_count == 0:
        return None

    (
        face_offsets,
        face_edge_ids,
        bounded_mask,
        signed_area,
        centroid_x,
        centroid_y,
        label_x,
        label_y,
        face_count,
    ) = _gpu_face_walk(half_edge_graph)
    if face_count == 0:
        return None

    selected_face_indices = cp.flatnonzero(
        (bounded_mask != 0) & (signed_area > 0)
    ).astype(cp.int32, copy=False)
    if int(selected_face_indices.size) == 0:
        return None

    faces = OverlayFaceTable(
        runtime_selection=selection.runtime_selection,
        _face_count=face_count,
        device_state=OverlayFaceDeviceState(
            face_offsets=face_offsets,
            face_edge_ids=face_edge_ids,
            bounded_mask=bounded_mask,
            signed_area=signed_area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            left_covered=cp.ones(face_count, dtype=cp.int8),
            right_covered=cp.zeros(face_count, dtype=cp.int8),
        ),
    )
    result = _build_polygon_output_from_faces_gpu(
        half_edge_graph,
        faces,
        selected_face_indices,
        preserve_row_count=left.row_count,
    )
    if result is None or result.row_count != left.row_count:
        return None
    return result


def _dispatch_single_row_polygon_known_coverage_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact GPU union for a known coverage pair of single-row polygon inputs."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count != 1:
        return None
    if not (bool(left.validity[0]) and bool(right.validity[0])):
        return None

    from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
    from vibespatial.overlay.graph import _gpu_face_walk, build_gpu_half_edge_graph
    from vibespatial.overlay.types import OverlayFaceDeviceState, OverlayFaceTable

    left_polygons = _explode_single_polygon_family_row_to_polygons_gpu(left)
    right_polygons = _explode_single_polygon_family_row_to_polygons_gpu(right)
    if (
        left_polygons is None
        or right_polygons is None
        or left_polygons.row_count == 0
        or right_polygons.row_count == 0
    ):
        return None

    left_state = left_polygons._ensure_device_state()
    right_state = right_polygons._ensure_device_state()
    left_buffer = left_state.families.get(GeometryFamily.POLYGON)
    right_buffer = right_state.families.get(GeometryFamily.POLYGON)
    if left_buffer is None or right_buffer is None:
        return None

    merged = _concat_device_family_buffers(
        GeometryFamily.POLYGON,
        [left_buffer, right_buffer],
    )
    d_x = cp.asarray(merged.x)
    d_y = cp.asarray(merged.y)
    d_ring_offsets = cp.asarray(merged.ring_offsets, dtype=cp.int64)
    if int(d_ring_offsets.size) < 2 or int(d_x.size) < 2:
        return None

    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int64, copy=False)
    total_segments = int(cp.sum(d_seg_counts).item())
    if total_segments == 0:
        return None

    d_seg_offsets = cp.empty(int(d_seg_counts.size) + 1, dtype=cp.int64)
    d_seg_offsets[0] = 0
    cp.cumsum(d_seg_counts, out=d_seg_offsets[1:])
    d_seg_ids = cp.arange(total_segments, dtype=cp.int64)
    d_ring_of_seg = cp.searchsorted(
        d_seg_offsets[1:],
        d_seg_ids,
        side="right",
    ).astype(cp.int64, copy=False)
    d_local_seg = d_seg_ids - d_seg_offsets[d_ring_of_seg]
    d_vertex_ids = d_ring_offsets[d_ring_of_seg] + d_local_seg

    start_x = d_x[d_vertex_ids]
    start_y = d_y[d_vertex_ids]
    end_x = d_x[d_vertex_ids + 1]
    end_y = d_y[d_vertex_ids + 1]

    swap = (
        (start_x > end_x)
        | ((start_x == end_x) & (start_y > end_y))
    )
    key_x0 = cp.where(swap, end_x, start_x)
    key_y0 = cp.where(swap, end_y, start_y)
    key_x1 = cp.where(swap, start_x, end_x)
    key_y1 = cp.where(swap, start_y, end_y)

    order = cp.lexsort(
        cp.stack((key_y1, key_x1, key_y0, key_x0))
    ).astype(cp.int64, copy=False)
    if int(order.size) == 0:
        return None
    sorted_x0 = key_x0[order]
    sorted_y0 = key_y0[order]
    sorted_x1 = key_x1[order]
    sorted_y1 = key_y1[order]
    start_mask = cp.empty(int(order.size), dtype=cp.bool_)
    start_mask[0] = True
    if int(order.size) > 1:
        start_mask[1:] = (
            (sorted_x0[1:] != sorted_x0[:-1])
            | (sorted_y0[1:] != sorted_y0[:-1])
            | (sorted_x1[1:] != sorted_x1[:-1])
            | (sorted_y1[1:] != sorted_y1[:-1])
        )
    run_starts = cp.flatnonzero(start_mask).astype(cp.int64, copy=False)
    run_lengths = cp.diff(
        cp.concatenate((run_starts, cp.asarray([int(order.size)], dtype=cp.int64)))
    )
    boundary_orders = order[run_starts[run_lengths % 2 == 1]]
    if int(boundary_orders.size) == 0:
        return None

    selection = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=1,
        requested_mode=dispatch_mode,
        requested_precision=PrecisionMode.AUTO,
        current_residency=combined_residency(left, right),
    )
    atomic_edges = _build_atomic_edges_from_boundary_segments_gpu(
        start_x[boundary_orders],
        start_y[boundary_orders],
        end_x[boundary_orders],
        end_y[boundary_orders],
        runtime_selection=selection.runtime_selection,
    )
    if atomic_edges is None:
        return None

    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    if half_edge_graph.edge_count == 0:
        return None

    (
        face_offsets,
        face_edge_ids,
        bounded_mask,
        signed_area,
        centroid_x,
        centroid_y,
        label_x,
        label_y,
        face_count,
    ) = _gpu_face_walk(half_edge_graph)
    if face_count == 0:
        return None

    selected_face_indices = cp.flatnonzero(
        (bounded_mask != 0) & (signed_area > 0)
    ).astype(cp.int32, copy=False)
    if int(selected_face_indices.size) == 0:
        return None

    faces = OverlayFaceTable(
        runtime_selection=selection.runtime_selection,
        _face_count=face_count,
        device_state=OverlayFaceDeviceState(
            face_offsets=face_offsets,
            face_edge_ids=face_edge_ids,
            bounded_mask=bounded_mask,
            signed_area=signed_area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            left_covered=cp.ones(face_count, dtype=cp.int8),
            right_covered=cp.zeros(face_count, dtype=cp.int8),
        ),
    )
    result = _build_polygon_output_from_faces_gpu(
        half_edge_graph,
        faces,
        selected_face_indices,
        preserve_row_count=1,
    )
    if result is None or result.row_count != 1:
        return None
    return result


def _dispatch_single_row_polygon_partition_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact GPU union for overlapping single-row polygon-family pairs.

    Build one exact overlay plan, materialize the exact partition pieces
    ``left - right``, ``left ∩ right``, and ``right - left`` from that same
    topology, then coverage-merge those pieces on GPU. This avoids the
    currently broken direct union face-selection path while keeping the
    constructive result device-resident end to end.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count != 1:
        return None
    if not (bool(left.validity[0]) and bool(right.validity[0])):
        return None

    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    plan = _build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    try:
        overlap = None
        try:
            direct_union, _ = _materialize_overlay_execution_plan(
                plan,
                operation="union",
                requested=ExecutionMode.GPU,
                preserve_row_count=1,
            )
            if (
                direct_union.row_count == 1
                and _is_polygon_only(direct_union)
                and bool(np.asarray(direct_union.validity, dtype=bool).all())
            ):
                overlap, _ = _materialize_overlay_execution_plan(
                    plan,
                    operation="intersection",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=1,
                )
                from vibespatial.constructive.measurement import _area_gpu_device_fp64
                from vibespatial.constructive.validity import is_valid_owned

                left_area = _area_gpu_device_fp64(left)[0]
                right_area = _area_gpu_device_fp64(right)[0]
                overlap_area = _area_gpu_device_fp64(overlap)[0]
                direct_area = _area_gpu_device_fp64(direct_union)[0]
                expected_area = left_area + right_area - overlap_area
                area_scale = cp.maximum(cp.maximum(cp.abs(expected_area), cp.abs(direct_area)), 1.0)
                area_matches = bool(
                    (
                        cp.abs(direct_area - expected_area)
                        <= (area_scale * 1.0e-8 + 1.0e-6)
                    ).item()
                )
                valid_direct = is_valid_owned(
                    direct_union,
                    dispatch_mode=ExecutionMode.GPU,
                )
                if area_matches and bool(np.asarray(valid_direct, dtype=bool).all()):
                    return direct_union
        except Exception:
            logger.debug(
                "direct single-row polygon union materialization failed; "
                "falling back to partition union",
                exc_info=True,
            )

        left_only, _ = _materialize_overlay_execution_plan(
            plan,
            operation="difference",
            requested=ExecutionMode.GPU,
            preserve_row_count=1,
        )
        if overlap is None:
            overlap, _ = _materialize_overlay_execution_plan(
                plan,
                operation="intersection",
                requested=ExecutionMode.GPU,
                preserve_row_count=1,
            )
        right_only, _ = _materialize_overlay_execution_plan(
            plan,
            operation="right_difference",
            requested=ExecutionMode.GPU,
            preserve_row_count=1,
        )
    finally:
        del plan

    if (
        left_only.row_count != 1
        or overlap.row_count != 1
        or right_only.row_count != 1
    ):
        return None

    piece_candidates = [piece for piece in (left_only, overlap, right_only) if bool(piece.validity[0])]
    if not piece_candidates:
        dominant_rescue = _dominant_tiny_area_polygon_union_rows_gpu(
            left,
            right,
        )
        if dominant_rescue is not None and bool(dominant_rescue.validity[0]):
            record_dispatch_event(
                surface="geopandas.array.union",
                operation="union",
                implementation="single_row_partition_union_degenerate_rescue_gpu",
                reason=(
                    "exact partition union produced no polygon pieces; "
                    "rescued tiny-area polygon pair by preserving the dominant operand"
                ),
                detail="rows=1",
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return dominant_rescue
        return None
    if len(piece_candidates) == 1:
        return piece_candidates[0]

    if not all(_is_polygon_only(piece) for piece in piece_candidates):
        return None

    merged = piece_candidates[0]
    for next_piece in piece_candidates[1:]:
        merged = _dispatch_single_row_polygon_known_coverage_union_gpu(
            merged,
            next_piece,
            dispatch_mode=dispatch_mode,
        )
        if merged is None or merged.row_count != 1:
            return None
        if not bool(merged.validity[0]):
            dominant_rescue = _dominant_tiny_area_polygon_union_rows_gpu(
                left,
                right,
            )
            if dominant_rescue is not None and bool(dominant_rescue.validity[0]):
                return dominant_rescue
            return None
        if not _is_polygon_only(merged):
            return None

    return merged


def _dispatch_single_row_polygon_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact single-row polygon union, preferring the dedicated GPU plans."""
    candidates = (
        lambda: _dispatch_single_row_polygon_partition_union_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
        ),
        lambda: _dispatch_multipolygon_polygon_union_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
        ),
    )
    for dispatch in candidates:
        dispatched = dispatch()
        if dispatched is None:
            continue
        sub_result = _collapse_polygon_overlay_sub_result(dispatched)
        if sub_result is None or sub_result.row_count != 1:
            continue
        sub_state = sub_result._ensure_device_state()
        if not bool(sub_state.validity[0]):
            continue
        return sub_result
    logger.debug("single-row polygon union could not find a valid GPU result")
    return None


def _dispatch_polygon_union_repair_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray | None:
    """Union aligned polygon rows via batched exact partition plans."""

    try:
        batched = _dispatch_polygon_partition_union_gpu(
            left,
            right,
            dispatch_mode=dispatch_mode,
        )
        if batched is not None and batched.row_count == left.row_count:
            return batched
    except Exception:
        logger.debug(
            "batched exact polygon union path failed",
            exc_info=True,
        )

    if left.row_count <= _POLYGON_UNION_EXACT_ROW_FALLBACK_MAX:
        try:
            exact_rows = np.arange(left.row_count, dtype=np.int64)
            return _dispatch_exact_rowwise_polygon_union_rows_gpu(
                left,
                right,
                exact_rows,
                dispatch_mode=dispatch_mode,
            )
        except Exception:
            logger.debug(
                "tiny exact rowwise polygon union fallback failed",
                exc_info=True,
            )
    return None


def _merge_row_aligned_polygon_piece_batches_gpu(
    left_piece: OwnedGeometryArray,
    right_piece: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Union two row-aligned exact piece batches without leaving the device."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left_piece.row_count != right_piece.row_count:
        return None

    left_valid = np.asarray(left_piece.validity, dtype=bool)
    right_valid = np.asarray(right_piece.validity, dtype=bool)
    out = _empty_device_constructive_output(left_piece.row_count)

    both_rows = np.flatnonzero(left_valid & right_valid).astype(np.int64, copy=False)
    left_only_rows = np.flatnonzero(left_valid & ~right_valid).astype(np.int64, copy=False)
    right_only_rows = np.flatnonzero(~left_valid & right_valid).astype(np.int64, copy=False)

    if both_rows.size > 0:
        merged = _dispatch_row_aligned_polygon_known_coverage_union_gpu(
            left_piece.take(both_rows),
            right_piece.take(both_rows),
            dispatch_mode=dispatch_mode,
        )
        if merged is None or merged.row_count != both_rows.size:
            return None
        out = device_concat_owned_scatter(out, merged, cp.asarray(both_rows, dtype=cp.int64))
    if left_only_rows.size > 0:
        out = device_concat_owned_scatter(
            out,
            left_piece.take(left_only_rows),
            cp.asarray(left_only_rows, dtype=cp.int64),
        )
    if right_only_rows.size > 0:
        out = device_concat_owned_scatter(
            out,
            right_piece.take(right_only_rows),
            cp.asarray(right_only_rows, dtype=cp.int64),
        )
    return out


def _dispatch_exact_rowwise_polygon_union_rows_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    row_indices: np.ndarray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Recompute selected polygon union rows exactly on GPU."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.size == 0:
        return _empty_device_constructive_output(0)

    if row_indices.size == left.row_count:
        row_results: list[OwnedGeometryArray] = []
        for row_index in row_indices.tolist():
            if not (bool(left.validity[row_index]) and bool(right.validity[row_index])):
                return None
            d_row = cp.asarray([row_index], dtype=cp.int64)
            sub_result = _dispatch_single_row_polygon_union_gpu(
                left.take(d_row),
                right.take(d_row),
                dispatch_mode=dispatch_mode,
            )
            if sub_result is None:
                return None
            row_results.append(sub_result)
        return OwnedGeometryArray.concat(row_results)

    replacement_rows: list[OwnedGeometryArray] = []
    for row_index in row_indices.tolist():
        if not (bool(left.validity[row_index]) and bool(right.validity[row_index])):
            return None
        d_row = cp.asarray([row_index], dtype=cp.int64)
        sub_result = _dispatch_single_row_polygon_union_gpu(
            left.take(d_row),
            right.take(d_row),
            dispatch_mode=dispatch_mode,
        )
        if sub_result is None:
            return None
        replacement_rows.append(sub_result)
    return OwnedGeometryArray.concat(replacement_rows)


def _dispatch_polygon_partition_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact row-aligned polygon union using one batched partition plan."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)

    left_valid = np.asarray(left.validity, dtype=bool)
    right_valid = np.asarray(right.validity, dtype=bool)
    both_rows = np.flatnonzero(left_valid & right_valid).astype(np.int64, copy=False)
    left_only_rows = np.flatnonzero(left_valid & ~right_valid).astype(np.int64, copy=False)
    right_only_rows = np.flatnonzero(~left_valid & right_valid).astype(np.int64, copy=False)

    out = _empty_device_constructive_output(left.row_count)
    if both_rows.size > 0:
        from vibespatial.overlay.gpu import (
            _build_overlay_execution_plan,
            _materialize_overlay_execution_plan,
        )

        sub_left = left.take(both_rows)
        sub_right = right.take(both_rows)
        plan = _build_overlay_execution_plan(
            sub_left,
            sub_right,
            dispatch_mode=ExecutionMode.GPU,
            _row_isolated=True,
        )
        try:
            merged = None
            try:
                direct_union, _ = _materialize_overlay_execution_plan(
                    plan,
                    operation="union",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=sub_left.row_count,
                )
                if direct_union.row_count == sub_left.row_count and _is_polygon_only(direct_union):
                    merged = direct_union
            except Exception:
                logger.debug(
                    "direct row-aligned polygon union materialization failed; "
                    "falling back to exact partition union",
                    exc_info=True,
                )

            if merged is None:
                left_only, _ = _materialize_overlay_execution_plan(
                    plan,
                    operation="difference",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=sub_left.row_count,
                )
                overlap, _ = _materialize_overlay_execution_plan(
                    plan,
                    operation="intersection",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=sub_left.row_count,
                )
                right_only, _ = _materialize_overlay_execution_plan(
                    plan,
                    operation="right_difference",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=sub_left.row_count,
                )

                merged = _merge_row_aligned_polygon_piece_batches_gpu(
                    left_only,
                    overlap,
                    dispatch_mode=dispatch_mode,
                )
                if merged is None or merged.row_count != sub_left.row_count:
                    return None
                merged = _merge_row_aligned_polygon_piece_batches_gpu(
                    merged,
                    right_only,
                    dispatch_mode=dispatch_mode,
                )
                if merged is None or merged.row_count != sub_left.row_count:
                    return None

            invalid_local_rows = np.flatnonzero(
                ~np.asarray(merged.validity, dtype=bool)
            ).astype(np.int64, copy=False)
            if invalid_local_rows.size:
                exact_rescue = _dispatch_exact_rowwise_polygon_union_rows_gpu(
                    sub_left,
                    sub_right,
                    invalid_local_rows,
                    dispatch_mode=dispatch_mode,
                )
                if (
                    exact_rescue is None
                    or exact_rescue.row_count != invalid_local_rows.size
                    or not bool(np.asarray(exact_rescue.validity, dtype=bool).all())
                ):
                    return None
                merged = device_concat_owned_scatter(
                    merged,
                    exact_rescue,
                    cp.asarray(invalid_local_rows, dtype=cp.int64),
                )
                record_dispatch_event(
                    surface="geopandas.array.union",
                    operation="union",
                    implementation="row_aligned_union_invalid_row_rescue_gpu",
                    reason=(
                        "row-aligned union produced invalid rows; "
                        "rescued them with exact single-row GPU union"
                    ),
                    detail=f"rows={sub_left.row_count}, rescued={invalid_local_rows.size}",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
        finally:
            del plan

        if merged is None or merged.row_count != sub_left.row_count:
            return None
        out = device_concat_owned_scatter(
            out,
            merged,
            cp.asarray(both_rows, dtype=cp.int64),
        )

    if left_only_rows.size > 0:
        out = device_concat_owned_scatter(
            out,
            left.take(left_only_rows),
            cp.asarray(left_only_rows, dtype=cp.int64),
        )
    if right_only_rows.size > 0:
        out = device_concat_owned_scatter(
            out,
            right.take(right_only_rows),
            cp.asarray(right_only_rows, dtype=cp.int64),
        )
    return out


def _dispatch_multipolygon_polygon_union_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Exact single-row GPU rescue for homogeneous MultiPolygon-Polygon union."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count or left.row_count != 1:
        return None

    left_is_mpoly = _is_family_only(left, frozenset({GeometryFamily.MULTIPOLYGON}))
    right_is_poly = _is_family_only(right, frozenset({GeometryFamily.POLYGON}))
    left_is_poly = _is_family_only(left, frozenset({GeometryFamily.POLYGON}))
    right_is_mpoly = _is_family_only(right, frozenset({GeometryFamily.MULTIPOLYGON}))

    if not ((left_is_mpoly and right_is_poly) or (left_is_poly and right_is_mpoly)):
        return None
    if right_is_mpoly:
        return _dispatch_multipolygon_polygon_union_gpu(
            right,
            left,
            dispatch_mode=dispatch_mode,
        )

    left_minus = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=dispatch_mode,
        _prefer_rowwise_polygon_difference_overlay=True,
    )
    right_minus = binary_constructive_owned(
        "difference",
        right,
        left,
        dispatch_mode=dispatch_mode,
        _prefer_rowwise_polygon_difference_overlay=True,
    )
    overlap = _dispatch_multipolygon_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=dispatch_mode,
    )
    if overlap is None:
        return None
    return _assemble_disjoint_polygonal_pieces_gpu([left_minus, right_minus, overlap])


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
    _sync_hotpath()
    with hotpath_stage("constructive.diff.intersection_plan.build", category="setup"):
        intersection_plan = _build_overlay_execution_plan(
            active_left,
            active_right,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=None,
            _row_isolated=True,
        )
    _sync_hotpath()
    with hotpath_stage("constructive.diff.intersection_plan.materialize", category="refine"):
        area_intersection, _ = _materialize_overlay_execution_plan(
            intersection_plan,
            operation="intersection",
            requested=ExecutionMode.GPU,
            preserve_row_count=active_left.row_count,
        )
    _sync_hotpath()
    if area_intersection.row_count != active_left.row_count:
        return None

    area_state = area_intersection._ensure_device_state()
    d_area_valid = area_state.validity.astype(cp.bool_, copy=False)

    d_preserve_rows = cp.flatnonzero(cp.asarray(left_valid & ~right_valid)).astype(cp.int64)
    d_fallback_local = cp.flatnonzero(~d_area_valid).astype(cp.int64, copy=False)
    d_partial_local = cp.flatnonzero(d_area_valid).astype(cp.int64, copy=False)

    partial_result: OwnedGeometryArray | None = None
    if int(d_partial_local.size) > 0:
        _sync_hotpath()
        with hotpath_stage("constructive.diff.difference_plan.build", category="setup"):
            difference_plan = _build_overlay_execution_plan(
                active_left,
                active_right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=None,
                _row_isolated=True,
            )
        _sync_hotpath()
        with hotpath_stage("constructive.diff.difference_plan.materialize", category="refine"):
            partial_result, _ = _materialize_overlay_execution_plan(
                difference_plan,
                operation="difference",
                requested=ExecutionMode.GPU,
                preserve_row_count=active_left.row_count,
            )
        _sync_hotpath()
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
        _sync_hotpath()
        with hotpath_stage("constructive.diff.rowwise_fallback", category="refine"):
            fallback_result = _dispatch_polygon_difference_overlay_rowwise_gpu_legacy(
                fallback_left,
                fallback_right,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=None,
            )
        _sync_hotpath()
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


def _dominant_tiny_area_polygon_union_rows_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Return the dominant operand for rows with a degenerate partner."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return _empty_device_constructive_output(0)
    if not (
        bool(np.asarray(left.validity, dtype=bool).all())
        and bool(np.asarray(right.validity, dtype=bool).all())
    ):
        return None

    from vibespatial.constructive.measurement import area_owned

    left_area = np.abs(np.asarray(area_owned(left), dtype=np.float64))
    right_area = np.abs(np.asarray(area_owned(right), dtype=np.float64))
    if left_area.size != left.row_count or right_area.size != right.row_count:
        return None
    max_area = np.maximum(left_area, right_area)
    min_area = np.minimum(left_area, right_area)
    finite = np.isfinite(max_area) & np.isfinite(min_area)
    tiny_partner = finite & (
        min_area <= np.maximum(max_area * _POLYGON_UNION_DEGENERATE_AREA_RTOL, 0.0)
    )
    if not bool(tiny_partner.all()):
        return None

    dominant_left = left_area >= right_area
    out = _empty_device_constructive_output(left.row_count)
    tiny = _empty_device_constructive_output(left.row_count)
    left_rows = np.flatnonzero(dominant_left).astype(np.int64, copy=False)
    right_rows = np.flatnonzero(~dominant_left).astype(np.int64, copy=False)
    if left_rows.size:
        out = device_concat_owned_scatter(
            out,
            left.take(left_rows),
            cp.asarray(left_rows, dtype=cp.int64),
        )
        tiny = device_concat_owned_scatter(
            tiny,
            right.take(left_rows),
            cp.asarray(left_rows, dtype=cp.int64),
        )
    if right_rows.size:
        out = device_concat_owned_scatter(
            out,
            right.take(right_rows),
            cp.asarray(right_rows, dtype=cp.int64),
        )
        tiny = device_concat_owned_scatter(
            tiny,
            left.take(right_rows),
            cp.asarray(right_rows, dtype=cp.int64),
        )

    try:
        from vibespatial.predicates.binary import (
            NullBehavior,
            evaluate_binary_predicate,
        )

        coverage = evaluate_binary_predicate(
            "covers",
            out,
            tiny,
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.FALSE,
        )
    except Exception:
        logger.debug(
            "tiny-area polygon union rescue coverage proof failed",
            exc_info=True,
        )
        return None
    if coverage.runtime_selection.selected is not ExecutionMode.GPU:
        return None
    coverage_mask = np.asarray(coverage.values, dtype=bool)
    if coverage_mask.size != left.row_count:
        return None
    if bool(coverage_mask.all()):
        return out

    try:
        contact = evaluate_binary_predicate(
            "intersects",
            tiny,
            out,
            dispatch_mode=ExecutionMode.GPU,
            null_behavior=NullBehavior.FALSE,
        )
    except Exception:
        logger.debug(
            "tiny-area polygon union rescue contact proof failed",
            exc_info=True,
        )
        return None
    if contact.runtime_selection.selected is not ExecutionMode.GPU:
        return None
    contact_mask = np.asarray(contact.values, dtype=bool)
    if contact_mask.size != left.row_count:
        return None
    tolerance_mask = _dominant_tiny_area_bounds_within_fp_tolerance(out, tiny)
    if tolerance_mask.size != left.row_count:
        return None
    accepted = coverage_mask | (contact_mask & tolerance_mask)
    if not bool(accepted.all()):
        return None
    return out


def _dominant_tiny_area_bounds_within_fp_tolerance(
    dominant: OwnedGeometryArray,
    tiny: OwnedGeometryArray,
) -> np.ndarray:
    """Return rows where tiny bounds only escape dominant by fp-scale noise."""
    if dominant.row_count != tiny.row_count:
        return np.zeros(0, dtype=bool)
    try:
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

        dominant_bounds = np.asarray(
            compute_geometry_bounds(dominant, dispatch_mode=ExecutionMode.GPU),
            dtype=np.float64,
        )
        tiny_bounds = np.asarray(
            compute_geometry_bounds(tiny, dispatch_mode=ExecutionMode.GPU),
            dtype=np.float64,
        )
    except Exception:
        logger.debug(
            "tiny-area polygon union rescue bounds proof failed",
            exc_info=True,
        )
        return np.zeros(dominant.row_count, dtype=bool)
    if dominant_bounds.shape != (dominant.row_count, 4) or tiny_bounds.shape != (
        tiny.row_count,
        4,
    ):
        return np.zeros(dominant.row_count, dtype=bool)

    coord_scale = np.maximum(
        np.maximum(np.max(np.abs(dominant_bounds), axis=1), np.max(np.abs(tiny_bounds), axis=1)),
        1.0,
    )
    tol = np.maximum(coord_scale * 1.0e-12, 1.0e-12)
    return (
        (tiny_bounds[:, 0] >= dominant_bounds[:, 0] - tol)
        & (tiny_bounds[:, 1] >= dominant_bounds[:, 1] - tol)
        & (tiny_bounds[:, 2] <= dominant_bounds[:, 2] + tol)
        & (tiny_bounds[:, 3] <= dominant_bounds[:, 3] + tol)
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
    _skip_single_row_polygon_difference_exact_correction: bool = False,
    _allow_rectangle_intersection_fast_path: bool = True,
    _skip_polygon_contraction: bool = False,
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
        if op == "union":
            try:
                exact_partition_union = _dispatch_single_row_polygon_partition_union_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                )
                if (
                    exact_partition_union is not None
                    and exact_partition_union.row_count == left.row_count
                    and bool(np.asarray(exact_partition_union.validity, dtype=bool).all())
                ):
                    return exact_partition_union
            except Exception:
                logger.debug(
                    "single-row polygon partition union GPU rescue failed",
                    exc_info=True,
                )
            try:
                multipart_union = _dispatch_multipolygon_polygon_union_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                )
                if multipart_union is not None and multipart_union.row_count == left.row_count:
                    return multipart_union
            except Exception:
                logger.debug(
                    "multipart-polygon exact union GPU rescue failed",
                    exc_info=True,
                )
            if left.row_count == 1:
                try:
                    repaired_union = _dispatch_polygon_union_repair_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                    if (
                        repaired_union is not None
                        and repaired_union.row_count == left.row_count
                        and bool(np.asarray(repaired_union.validity, dtype=bool).all())
                    ):
                        return repaired_union
                except Exception:
                    logger.debug(
                        "single-row polygon union repair GPU rescue failed",
                        exc_info=True,
                    )

        def _try_polygon_rect_intersection():
            try:
                from vibespatial.kernels.constructive.polygon_rect_intersection import (
                    polygon_rect_intersection,
                    polygon_rect_intersection_can_handle,
                )

                rect_left = left
                rect_right = right
                if (
                    (rect_left.is_indexed_view or rect_right.is_indexed_view)
                    and _is_family_only(rect_left, frozenset({GeometryFamily.POLYGON}))
                    and _is_family_only(rect_right, frozenset({GeometryFamily.POLYGON}))
                ):
                    rect_left = _resolve_indexed_polygon_fast_path_candidate(rect_left)
                    rect_right = _resolve_indexed_polygon_fast_path_candidate(rect_right)

                if polygon_rect_intersection_can_handle(rect_left, rect_right):
                    result = polygon_rect_intersection(
                        rect_left,
                        rect_right,
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    if result.row_count == left.row_count:
                        return result
                # Intersection is commutative. Allow rectangle-capable batches on
                # either side so parcel-like rectangle inputs can still use the
                # specialized GPU clip kernel instead of the generic overlay path.
                if polygon_rect_intersection_can_handle(rect_right, rect_left):
                    result = polygon_rect_intersection(
                        rect_right,
                        rect_left,
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    if result.row_count == left.row_count:
                        return result
                mixed_result = _dispatch_mixed_polygon_rect_intersection_gpu(
                    rect_left,
                    rect_right,
                    dispatch_mode=ExecutionMode.GPU,
                )
                if mixed_result is not None and mixed_result.row_count == left.row_count:
                    return mixed_result
            except Exception:
                logger.debug(
                    "polygon-rectangle GPU intersection fast path failed",
                    exc_info=True,
                )
            return None

        valid_mask = left.validity & right.validity
        polygon_pair_count = (
            len(unique_tag_pairs(left.tags[valid_mask], right.tags[valid_mask]))
            if valid_mask.any()
            else 0
        )
        if op == "intersection" and polygon_pair_count > 1:
            if left.row_count <= _MIXED_POLYGON_INTERSECTION_ROWWISE_MAX:
                try:
                    tiny_mixed_rowwise = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                        left,
                        right,
                        dispatch_mode=dispatch_mode,
                        _cached_right_segments=_cached_right_segments,
                    )
                    if (
                        tiny_mixed_rowwise is not None
                        and tiny_mixed_rowwise.row_count == left.row_count
                    ):
                        return tiny_mixed_rowwise
                except Exception:
                    logger.debug(
                        "tiny mixed polygonal GPU intersection rowwise shortcut failed",
                        exc_info=True,
                    )

        if op == "intersection" and _allow_rectangle_intersection_fast_path:
            rect_result = _try_polygon_rect_intersection()
            if rect_result is not None:
                return rect_result

        if op == "intersection" and polygon_pair_count > 1:
            try:
                mixed_polygonal = _dispatch_mixed_binary_constructive_gpu(
                    op,
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=_cached_right_segments,
                )
                if mixed_polygonal is not None and mixed_polygonal.row_count == left.row_count:
                    return mixed_polygonal
            except Exception:
                logger.debug(
                    "mixed polygonal GPU intersection dispatch failed",
                    exc_info=True,
                )

        if op == "intersection":
            try:
                multipart_result = _dispatch_multipolygon_polygon_intersection_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                )
                if multipart_result is not None and multipart_result.row_count == left.row_count:
                    return multipart_result
            except Exception:
                logger.debug(
                    "multipart-polygon exact intersection GPU rescue failed",
                    exc_info=True,
                )

        if op == "intersection" and _prefer_exact_polygon_intersection:
            sh_result = _dispatch_polygon_intersection_sh_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if sh_result is not None and sh_result.row_count == left.row_count:
                return sh_result
            partitioned_sh = _dispatch_partitioned_polygon_intersection_sh_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if partitioned_sh is not None and partitioned_sh.row_count == left.row_count:
                return partitioned_sh
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

        if (
            op == "difference"
            and left.row_count == 1
            and not _skip_single_row_polygon_difference_exact_correction
        ):
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
            sh_result = _dispatch_polygon_intersection_sh_gpu(
                left,
                right,
                dispatch_mode=dispatch_mode,
            )
            if sh_result is not None and sh_result.row_count == left.row_count:
                return sh_result
            if sh_result is None:
                partitioned_sh = _dispatch_partitioned_polygon_intersection_sh_gpu(
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                )
                if partitioned_sh is not None and partitioned_sh.row_count == left.row_count:
                    return partitioned_sh
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
                if not _skip_polygon_contraction and not (
                    op == "difference" and _prefer_rowwise_polygon_difference_overlay
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

        if op == "union":
            return None

        # Fall through to the general overlay pipeline for
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

        if valid_mask.any() and len(unique_tag_pairs(left.tags[valid_mask], right.tags[valid_mask])) > 1:
            try:
                mixed_polygonal = _dispatch_mixed_binary_constructive_gpu(
                    op,
                    left,
                    right,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=_cached_right_segments,
                )
                if mixed_polygonal is not None:
                    return mixed_polygonal
            except Exception:
                logger.debug(
                    "mixed polygonal GPU fallback failed for %s",
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
    _skip_single_row_polygon_difference_exact_correction: bool = False,
    _allow_rectangle_intersection_fast_path: bool = True,
    _skip_polygon_contraction: bool = False,
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
        current_residency=combined_residency(left, right),
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
            _skip_single_row_polygon_difference_exact_correction=(
                _skip_single_row_polygon_difference_exact_correction
            ),
            _allow_rectangle_intersection_fast_path=_allow_rectangle_intersection_fast_path,
            _skip_polygon_contraction=_skip_polygon_contraction,
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
