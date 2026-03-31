"""Containment bypass and Sutherland-Hodgman fast-path for the overlay pipeline.

These functions identify polygons that can skip the full overlay computation:

- **Containment bypass** (lyy.16): polygons fully inside or fully disjoint
  from the corridor are routed directly, avoiding overlay entirely.
- **Batched SH clip** (lyy.18): boundary-crossing simple polygons eligible
  for Sutherland-Hodgman clipping are batched into a single
  ``polygon_intersection`` kernel launch instead of N separate overlay
  invocations.
- **Batch point-in-ring GPU** (lyy.22): GPU-accelerated point-in-polygon
  testing used by the face assembly host fallback path.

Extracted from ``gpu.py`` to reduce file size and clarify module boundaries.
"""
from __future__ import annotations

import logging

import numpy as np
import shapely  # hygiene:ok -- only used for empty-geometry sentinel construction

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.residency import Residency

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)

# Minimum number of candidate pairs before using the GPU kernel.
# Below this threshold the Python fallback is used to avoid kernel
# launch overhead dominating.
_BATCH_PIP_GPU_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Containment bypass: GPU-accelerated identification of polygons fully
# inside the corridor, skipping overlay computation for those polygons.
# ---------------------------------------------------------------------------


def _containment_bypass_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    how: str,
) -> tuple[OwnedGeometryArray | None, cp.ndarray | None]:
    """Identify polygons in *left* that can bypass full overlay via containment/disjointness.

    Returns ``(bypass_result, remainder_mask)`` where:

    - *bypass_result* is an OGA of bypassed polygons, or ``None`` if none qualify.
    - *remainder_mask* is a device boolean mask (True = needs overlay) over
      the rows of *left*, or ``None`` if all polygons were bypassed.

    Semantics by operation
    ----------------------
    **intersection**: contained -> pass-through (L_i intersect R_j = L_i).
        Disjoint -> excluded (empty result).
    **difference**: contained -> excluded (L_i - R_j = empty).
        Disjoint -> pass-through (L_i - R_j = L_i).

    Algorithm
    ---------
    Stage 1a: GPU bounds containment (cheapest filter).
        Compare per-polygon MBR against corridor MBR -- CuPy element-wise.
    Stage 1b: GPU bounds disjointness (for difference).
        Polygons whose MBR does not overlap the corridor MBR are definitely
        disjoint -- conservative but correct.
    Stage 2: GPU vertex-in-polygon (correct containment for bbox candidates).
        For polygons passing the bbox-contained test, check that ALL vertices
        are inside the corridor via thread-per-polygon NVRTC kernels that read
        vertices directly from the source family buffers (no scatter).
    Stage 3: Route results per operation semantics.
    """
    if cp is None:
        return None, None

    # Lazy import: kernel compile function stays in gpu.py.
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )
    from vibespatial.overlay.gpu import _containment_bypass_kernels

    runtime = get_cuda_runtime()
    n_left = left.row_count

    # Right must be a single row.
    if right.row_count != 1:
        return None, None

    # Supported operations: intersection, difference.
    # - intersection: contained -> pass-through, disjoint -> skip (empty)
    # - difference: contained -> empty (L-R=nothing), disjoint -> pass-through (L-R=L)
    # symmetric_difference is NOT eligible: disjoint pairs produce L_i UNION R_j,
    # and the bypass cannot correctly reconstruct R_j's contribution without
    # duplicating R_j per disjoint pair.
    if how not in ("intersection", "difference"):
        return None, None

    # Ensure both sides are on device.
    from vibespatial.runtime.residency import TransferTrigger

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="containment_bypass: move left to device",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="containment_bypass: move right to device",
    )
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()

    # Determine corridor geometry family (Polygon or MultiPolygon).
    corr_family = None
    corr_buffer = None
    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        if family in right_state.families:
            d_buf = right_state.families[family]
            if int(d_buf.geometry_offsets.size) >= 2:
                corr_family = family
                corr_buffer = d_buf
                break

    if corr_family is None:
        # Corridor is not polygonal -- cannot do containment bypass.
        return None, None

    # ------------------------------------------------------------------
    # Stage 1a: GPU bounds containment
    # ------------------------------------------------------------------
    from vibespatial.kernels.core.geometry_analysis import (
        _compute_geometry_bounds_gpu_impl,
    )

    # Ensure left has per-row device bounds.
    if left_state.row_bounds is None:
        _compute_geometry_bounds_gpu_impl(left, compute_type="double")
    d_left_bounds = left_state.row_bounds  # shape (n_left, 4), fp64

    # Compute corridor bounds (single row).
    if right_state.row_bounds is None:
        _compute_geometry_bounds_gpu_impl(right, compute_type="double")
    d_right_bounds = right_state.row_bounds  # shape (1, 4), fp64

    # CuPy element-wise bbox containment (Tier 2).
    # Extract corridor bounds as device scalars -- stays on device.
    d_corr_bounds = cp.asarray(d_right_bounds).ravel()
    d_lb = cp.asarray(d_left_bounds).reshape(n_left, 4)
    d_bbox_inside = (
        (d_lb[:, 0] >= d_corr_bounds[0])
        & (d_lb[:, 1] >= d_corr_bounds[1])
        & (d_lb[:, 2] <= d_corr_bounds[2])
        & (d_lb[:, 3] <= d_corr_bounds[3])
    )

    # ------------------------------------------------------------------
    # Stage 1b: GPU bounds disjointness (difference only)
    # ------------------------------------------------------------------
    # Two MBRs are disjoint when one is entirely left/right/above/below the
    # other.  This is a conservative test: bbox-disjoint => definitely
    # disjoint.  Polygons that are geometrically disjoint but bbox-overlapping
    # will fall through to overlay (correct, just not bypassed).
    d_bbox_disjoint: cp.ndarray | None = None  # type: ignore[name-defined]
    if how == "difference":
        d_bbox_disjoint = (
            (d_lb[:, 2] < d_corr_bounds[0])   # L.xmax < R.xmin
            | (d_lb[:, 0] > d_corr_bounds[2])  # L.xmin > R.xmax
            | (d_lb[:, 3] < d_corr_bounds[1])  # L.ymax < R.ymin
            | (d_lb[:, 1] > d_corr_bounds[3])  # L.ymin > R.ymax
        )

    n_bbox_candidates = int(cp.sum(d_bbox_inside))
    n_bbox_disjoint = int(cp.sum(d_bbox_disjoint)) if d_bbox_disjoint is not None else 0

    if n_bbox_candidates == 0 and n_bbox_disjoint == 0:
        d_remainder_mask = cp.ones(n_left, dtype=cp.bool_)
        return None, d_remainder_mask

    # ------------------------------------------------------------------
    # Stage 2: GPU vertex-in-polygon (thread-per-polygon)
    # ------------------------------------------------------------------
    # Only needed when there are bbox-inside candidates (potential containment).
    d_cand_result: cp.ndarray | None = None  # type: ignore[name-defined]
    d_bbox_indices: cp.ndarray | None = None  # type: ignore[name-defined]

    if n_bbox_candidates > 0:
        d_bbox_indices = cp.flatnonzero(d_bbox_inside).astype(cp.int64)

        # Gather tags and family_row_offsets for bbox-candidate rows.
        d_tags = cp.asarray(left_state.tags)
        d_fro = cp.asarray(left_state.family_row_offsets)
        d_cand_tags = d_tags[d_bbox_indices]
        d_cand_fro = d_fro[d_bbox_indices]

        # Per-candidate result: 1 = fully inside, 0 = not.
        d_cand_result = cp.zeros(n_bbox_candidates, dtype=cp.int32)

        kernels = _containment_bypass_kernels()
        ptr = runtime.pointer
        corr_row = 0  # corridor is always row 0 of its family buffer

        # Process each left polygonal family against the corridor.
        for left_family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if left_family not in left_state.families:
                continue
            tag_val = FAMILY_TAGS[left_family]
            d_family_mask = d_cand_tags == tag_val
            n_family = int(cp.sum(d_family_mask))
            if n_family == 0:
                continue

            d_family_rows = d_cand_fro[d_family_mask].astype(cp.int32)
            left_buf = left_state.families[left_family]

            # Select the kernel variant based on left family x corridor family.
            if left_family is GeometryFamily.POLYGON and corr_family is GeometryFamily.POLYGON:
                kernel = kernels["containment_poly_vs_poly"]
                params = (
                    (
                        ptr(d_family_rows),
                        n_family,
                        ptr(left_buf.x),
                        ptr(left_buf.y),
                        ptr(left_buf.geometry_offsets),
                        ptr(left_buf.ring_offsets),
                        ptr(corr_buffer.x),
                        ptr(corr_buffer.y),
                        ptr(corr_buffer.geometry_offsets),
                        ptr(corr_buffer.ring_offsets),
                        corr_row,
                        ptr(d_cand_result),  # temporary -- write to family slice below
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                    ),
                )
            elif left_family is GeometryFamily.POLYGON and corr_family is GeometryFamily.MULTIPOLYGON:
                kernel = kernels["containment_poly_vs_mpoly"]
                params = (
                    (
                        ptr(d_family_rows),
                        n_family,
                        ptr(left_buf.x),
                        ptr(left_buf.y),
                        ptr(left_buf.geometry_offsets),
                        ptr(left_buf.ring_offsets),
                        ptr(corr_buffer.x),
                        ptr(corr_buffer.y),
                        ptr(corr_buffer.geometry_offsets),
                        ptr(corr_buffer.part_offsets),
                        ptr(corr_buffer.ring_offsets),
                        corr_row,
                        ptr(d_cand_result),
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                    ),
                )
            elif left_family is GeometryFamily.MULTIPOLYGON and corr_family is GeometryFamily.POLYGON:
                kernel = kernels["containment_mpoly_vs_poly"]
                params = (
                    (
                        ptr(d_family_rows),
                        n_family,
                        ptr(left_buf.x),
                        ptr(left_buf.y),
                        ptr(left_buf.geometry_offsets),
                        ptr(left_buf.part_offsets),
                        ptr(left_buf.ring_offsets),
                        ptr(corr_buffer.x),
                        ptr(corr_buffer.y),
                        ptr(corr_buffer.geometry_offsets),
                        ptr(corr_buffer.ring_offsets),
                        corr_row,
                        ptr(d_cand_result),
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                    ),
                )
            else:
                # multipolygon vs multipolygon
                kernel = kernels["containment_mpoly_vs_mpoly"]
                params = (
                    (
                        ptr(d_family_rows),
                        n_family,
                        ptr(left_buf.x),
                        ptr(left_buf.y),
                        ptr(left_buf.geometry_offsets),
                        ptr(left_buf.part_offsets),
                        ptr(left_buf.ring_offsets),
                        ptr(corr_buffer.x),
                        ptr(corr_buffer.y),
                        ptr(corr_buffer.geometry_offsets),
                        ptr(corr_buffer.part_offsets),
                        ptr(corr_buffer.ring_offsets),
                        corr_row,
                        ptr(d_cand_result),
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                    ),
                )

            # The kernel writes to a flat output buffer indexed by thread id
            # (0..n_family-1).  We need to write to the correct positions in
            # d_cand_result.  Allocate a temporary per-family output, launch,
            # then scatter back.
            d_family_out = cp.empty(n_family, dtype=cp.int32)
            # Fix params to point to family-local output.
            params_vals = list(params[0])
            params_vals[-1] = ptr(d_family_out)
            params = (tuple(params_vals), params[1])

            grid, block = runtime.launch_config(kernel, n_family)
            runtime.launch(kernel, grid=grid, block=block, params=params)

            # Scatter family results back to candidate-wide array.
            d_family_cand_positions = cp.flatnonzero(d_family_mask)
            runtime.synchronize()
            d_cand_result[d_family_cand_positions] = d_family_out

    # ------------------------------------------------------------------
    # Stage 3: Route results by operation semantics
    # ------------------------------------------------------------------
    # Determine which rows are contained (all vertices inside corridor).
    n_contained = 0
    d_contained_rows: cp.ndarray | None = None  # type: ignore[name-defined]
    if d_cand_result is not None and d_bbox_indices is not None:
        d_cand_all_inside = d_cand_result == 1
        n_contained = int(cp.sum(d_cand_all_inside))
        if n_contained > 0:
            d_contained_cand_indices = cp.flatnonzero(d_cand_all_inside)
            d_contained_rows = d_bbox_indices[d_contained_cand_indices].astype(cp.int64)

    # Determine which rows are bbox-disjoint.
    d_disjoint_rows: cp.ndarray | None = None  # type: ignore[name-defined]
    if d_bbox_disjoint is not None and n_bbox_disjoint > 0:
        d_disjoint_rows = cp.flatnonzero(d_bbox_disjoint).astype(cp.int64)

    # --- Apply operation-specific routing ---
    if how == "intersection":
        # Contained -> pass-through; disjoint -> excluded (empty).
        if n_contained == 0:
            d_remainder_mask = cp.ones(n_left, dtype=cp.bool_)
            return None, d_remainder_mask

        bypass_oga = left.take(d_contained_rows)
        d_remainder_mask = cp.ones(n_left, dtype=cp.bool_)
        d_remainder_mask[d_contained_rows] = False
        n_remainder = int(cp.sum(d_remainder_mask))
        n_bypassed = n_contained

    elif how == "difference":
        # Contained -> excluded (L-R = empty); disjoint -> pass-through (L-R = L).
        # Both contained and disjoint rows are removed from the remainder.
        d_remainder_mask = cp.ones(n_left, dtype=cp.bool_)

        # Exclude contained rows (result is empty -- don't include in output).
        if d_contained_rows is not None:
            d_remainder_mask[d_contained_rows] = False

        # Disjoint rows pass through as-is (L_i - R_j = L_i).
        if d_disjoint_rows is not None:
            d_remainder_mask[d_disjoint_rows] = False

        n_bypassed = n_contained + n_bbox_disjoint

        if n_bypassed == 0:
            return None, d_remainder_mask

        # Build bypass OGA: only disjoint rows (contained produce empty).
        if d_disjoint_rows is not None and n_bbox_disjoint > 0:
            bypass_oga = left.take(d_disjoint_rows)
        else:
            bypass_oga = None  # all bypassed rows were contained (empty result)

        n_remainder = int(cp.sum(d_remainder_mask))

    else:
        # Should not reach here due to gate above.
        return None, None

    record_dispatch_event(
        surface="geopandas.spatial_overlay",
        operation="containment_bypass",
        implementation="gpu_nvrtc_containment_bypass",
        reason=(
            f"lyy.16 containment/disjointness bypass ({how}): "
            f"{n_bypassed}/{n_left} polygons bypassed, "
            f"{n_remainder} need overlay"
        ),
        detail=(
            f"bbox_contained={n_bbox_candidates}, contained={n_contained}, "
            f"bbox_disjoint={n_bbox_disjoint}, remainder={n_remainder}"
        ),
        selected=ExecutionMode.GPU,
    )

    if n_remainder == 0:
        if bypass_oga is None:
            # All rows were contained in a difference -- empty result.
            empty = from_shapely_geometries([shapely.Point()])
            return empty.take(np.asarray([], dtype=np.int64)), None
        return bypass_oga, None

    return bypass_oga, d_remainder_mask


# ---------------------------------------------------------------------------
# Batched Sutherland-Hodgman clip for boundary-crossing simple polygons
# (lyy.18)
# ---------------------------------------------------------------------------
# After containment bypass identifies remainder polygons (boundary-crossing),
# this tier batches all SH-eligible remainder polygons into a SINGLE
# polygon_intersection kernel launch, instead of N separate overlay pipeline
# invocations.
#
# SH eligibility of the CLIP polygon (the corridor / right side):
#   - Must be Polygon family (not MultiPolygon)
#   - Must have exactly 1 ring (no holes)
#   - Exterior ring vertex count recorded for workspace budget
#
# SH eligibility of each SUBJECT polygon (left side):
#   - Must be Polygon family (not MultiPolygon)
#   - Must have exactly 1 ring (no holes)
#   - Exterior ring vertex count + clip vertex count <= MAX_CLIP_VERTS
#
# Non-eligible remainder polygons are routed to the per-group overlay.
# ---------------------------------------------------------------------------


def _is_clip_polygon_sh_eligible(
    right: OwnedGeometryArray,
) -> tuple[bool, int]:
    """Check whether the single clip polygon is SH-eligible.

    Returns ``(eligible, clip_vert_count)`` where *clip_vert_count* is the
    number of exterior ring vertices (excluding the closing duplicate) when
    eligible, or 0 when not eligible.

    The check uses host-side metadata (offset arrays) which are small O(1)
    for a single-row geometry.  No device transfer of coordinate data.
    """
    from vibespatial.kernels.constructive.polygon_intersection import _MAX_CLIP_VERTS

    if right.row_count != 1:
        return False, 0

    # Must be Polygon family (not MultiPolygon).
    right._ensure_host_state()
    if GeometryFamily.POLYGON not in right.families:
        return False, 0
    poly_buf = right.families[GeometryFamily.POLYGON]
    if poly_buf.row_count == 0:
        return False, 0

    # Check single ring (no holes).
    geom_offsets = poly_buf.geometry_offsets
    ring_count = int(geom_offsets[1] - geom_offsets[0])
    if ring_count != 1:
        logger.debug(
            "SH batch clip: clip polygon has %d rings (holes) -- skipping SH tier",
            ring_count,
        )
        return False, 0

    # Count exterior ring vertices.
    ring_offsets = poly_buf.ring_offsets
    first_ring = int(geom_offsets[0])
    n_verts = int(ring_offsets[first_ring + 1] - ring_offsets[first_ring])

    # The kernel strips the closing vertex if last == first, so effective
    # vertex count is n_verts - 1 for closed rings.  But this is a budget
    # check -- be conservative and use the raw count.
    if n_verts > _MAX_CLIP_VERTS:
        logger.debug(
            "SH batch clip: clip polygon has %d vertices (limit %d) -- skipping SH tier",
            n_verts, _MAX_CLIP_VERTS,
        )
        return False, 0

    return True, n_verts


def _classify_remainder_sh_eligible(
    left: OwnedGeometryArray,
    clip_vert_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify remainder polygons into SH-eligible and complex sets.

    Returns ``(sh_eligible_mask, complex_mask)`` as boolean numpy arrays
    over the rows of *left*.  Uses host-side offset arrays (small metadata)
    to classify without touching coordinate data.

    A remainder polygon is SH-eligible if:
    - It is Polygon family (not MultiPolygon)
    - It has exactly 1 ring (no holes)
    - Its exterior ring vertex count + clip_vert_count <= MAX_CLIP_VERTS

    Parameters
    ----------
    left : OwnedGeometryArray
        The remainder polygons (already filtered to boundary-crossing only).
    clip_vert_count : int
        Vertex count of the clip polygon's exterior ring.
    """
    from vibespatial.kernels.constructive.polygon_intersection import _MAX_CLIP_VERTS

    n = left.row_count
    sh_eligible = np.zeros(n, dtype=bool)

    left._ensure_host_state()

    # Only Polygon family rows can be SH-eligible (not MultiPolygon).
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    is_poly = left.tags == poly_tag

    if not np.any(is_poly):
        return sh_eligible, ~sh_eligible

    poly_buf = left.families.get(GeometryFamily.POLYGON)
    if poly_buf is None or poly_buf.row_count == 0:
        return sh_eligible, ~sh_eligible

    geom_offsets = poly_buf.geometry_offsets
    ring_offsets = poly_buf.ring_offsets

    # Vectorized: compute rings_per_row and vertex_count for all poly rows.
    rings_per_row = np.diff(geom_offsets)  # shape (poly_buf.row_count,)
    single_ring = rings_per_row == 1

    # For single-ring polygons, compute exterior ring vertex count.
    first_ring_idx = geom_offsets[:-1]
    ext_verts = np.where(
        single_ring,
        ring_offsets[first_ring_idx + 1] - ring_offsets[first_ring_idx],
        _MAX_CLIP_VERTS + 1,  # sentinel: exceeds limit
    )

    # SH-eligible: single ring AND combined verts fit in workspace.
    poly_sh_ok = single_ring & (ext_verts + clip_vert_count <= _MAX_CLIP_VERTS)

    # Map back from family-row space to global-row space.
    poly_indices = np.flatnonzero(is_poly)
    fro = left.family_row_offsets[poly_indices]  # family row offset per global row
    sh_eligible[poly_indices] = poly_sh_ok[fro]

    # Also require the row to be valid.
    sh_eligible &= left.validity

    return sh_eligible, ~sh_eligible


def _batched_sh_clip(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    sh_eligible_mask: np.ndarray,
) -> OwnedGeometryArray | None:
    """Batch SH-eligible remainder polygons into a single polygon_intersection call.

    Replicates *right* (the clip polygon, 1 row) to match the count of
    SH-eligible *left* rows, then calls ``polygon_intersection`` for all
    pairs in a single kernel launch.

    Returns the clipped result OwnedGeometryArray, or ``None`` if the
    kernel fails.

    Parameters
    ----------
    left : OwnedGeometryArray
        The remainder polygons (boundary-crossing).
    right : OwnedGeometryArray
        The single clip polygon (SH-eligible, verified by caller).
    sh_eligible_mask : np.ndarray
        Boolean mask over *left* rows, True = SH-eligible.
    """
    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    n_eligible = int(sh_eligible_mask.sum())
    if n_eligible == 0:
        return None

    # Subset left to SH-eligible rows only.
    # Use device indices when available for zero-copy take().
    if cp is not None:
        d_eligible_indices = cp.asarray(np.flatnonzero(sh_eligible_mask)).astype(cp.int64)
        left_eligible = left.take(d_eligible_indices)
    else:
        h_eligible_indices = np.flatnonzero(sh_eligible_mask).astype(np.int64)
        left_eligible = left.take(h_eligible_indices)

    # Replicate right (1 row) to match n_eligible rows.
    if cp is not None:
        d_rep_indices = cp.zeros(n_eligible, dtype=cp.int64)
        right_replicated = right.take(d_rep_indices)
    else:
        h_rep_indices = np.zeros(n_eligible, dtype=np.int64)
        right_replicated = right.take(h_rep_indices)

    # Single batched kernel launch for all SH-eligible pairs.
    result = polygon_intersection(
        left_eligible,
        right_replicated,
        dispatch_mode=ExecutionMode.GPU,
    )

    record_dispatch_event(
        surface="geopandas.spatial_overlay",
        operation="batched_sh_clip",
        implementation="gpu_polygon_intersection_batched",
        reason=(
            f"lyy.18 batched SH clip: {n_eligible} boundary-crossing polygons "
            f"clipped in single kernel launch"
        ),
        detail=f"sh_eligible={n_eligible}, total_remainder={left.row_count}",
        selected=ExecutionMode.GPU,
    )

    return result


def _combine_bypass_results(
    containment_result: OwnedGeometryArray | None,
    sh_clip_result: OwnedGeometryArray | None,
    overlay_result: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Combine containment-bypass, SH-clip, and overlay results into one OGA.

    Order: containment-bypass, SH-clip, overlay remainder.  If only one part
    is non-empty, returns it directly (no copy).
    """
    parts: list[OwnedGeometryArray] = []
    if containment_result is not None:
        parts.append(containment_result)
    if sh_clip_result is not None:
        parts.append(sh_clip_result)
    if overlay_result.row_count > 0:
        parts.append(overlay_result)
    if len(parts) > 1:
        return OwnedGeometryArray.concat(parts)
    elif len(parts) == 1:
        return parts[0]
    # No parts -- return the (empty) overlay result as-is.
    return overlay_result


def _batch_point_in_ring_gpu(
    pairs: list[tuple[int, int]],
    cycle_samples: dict[int, tuple[float, float]],
    cycle_rings: dict[int, np.ndarray],
) -> np.ndarray:
    """Test multiple (sample_point, ring) pairs on the GPU.

    Parameters
    ----------
    pairs : list of (cycle_index, container_index) tuples
        Each pair says "test the sample point of cycle_index against the
        ring of container_index".
    cycle_samples : dict mapping cycle_index -> (sample_x, sample_y)
    cycle_rings : dict mapping cycle_index -> ring coordinates (N x 2 closed)

    Returns
    -------
    np.ndarray of int32, shape (len(pairs),)
        1 if the sample point is inside the ring, 0 otherwise.
    """
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )

    # Lazy import: kernel compile function stays in gpu.py.
    from vibespatial.overlay.gpu import _batch_pip_kernels

    pair_count = len(pairs)
    if pair_count == 0:
        return np.empty(0, dtype=np.int32)

    # --- Build host-side input arrays ---
    h_sample_x = np.empty(pair_count, dtype=np.float64)
    h_sample_y = np.empty(pair_count, dtype=np.float64)
    h_pair_ring_idx = np.empty(pair_count, dtype=np.int32)

    # Collect unique container rings and assign contiguous indices
    container_ids_seen: dict[int, int] = {}
    ring_list: list[int] = []
    for _, container_index in pairs:
        if container_index not in container_ids_seen:
            container_ids_seen[container_index] = len(ring_list)
            ring_list.append(container_index)

    for i, (cycle_index, container_index) in enumerate(pairs):
        sx, sy = cycle_samples[cycle_index]
        h_sample_x[i] = sx
        h_sample_y[i] = sy
        h_pair_ring_idx[i] = container_ids_seen[container_index]

    # Build concatenated ring coordinate arrays and offset table
    ring_coords_x: list[np.ndarray] = []
    ring_coords_y: list[np.ndarray] = []
    offsets = [0]
    for container_index in ring_list:
        ring = cycle_rings[container_index]
        ring_coords_x.append(np.ascontiguousarray(ring[:, 0]))
        ring_coords_y.append(np.ascontiguousarray(ring[:, 1]))
        offsets.append(offsets[-1] + ring.shape[0])

    h_ring_x = np.concatenate(ring_coords_x).astype(np.float64, copy=False)
    h_ring_y = np.concatenate(ring_coords_y).astype(np.float64, copy=False)
    h_ring_offsets = np.asarray(offsets, dtype=np.int32)

    # --- Upload to GPU ---
    runtime = get_cuda_runtime()
    d_sample_x = runtime.from_host(h_sample_x)
    d_sample_y = runtime.from_host(h_sample_y)
    d_ring_x = runtime.from_host(h_ring_x)
    d_ring_y = runtime.from_host(h_ring_y)
    d_ring_offsets = runtime.from_host(h_ring_offsets)
    d_pair_ring_idx = runtime.from_host(h_pair_ring_idx)
    d_results = runtime.allocate((pair_count,), np.int32, zero=True)

    # --- Compile and launch ---
    kernels = _batch_pip_kernels()
    kernel = kernels["batch_point_in_ring"]
    grid, block = runtime.launch_config(kernel, pair_count)

    ptr = runtime.pointer
    params = (
        (ptr(d_sample_x), ptr(d_sample_y),
         ptr(d_ring_x), ptr(d_ring_y),
         ptr(d_ring_offsets), ptr(d_pair_ring_idx),
         ptr(d_results), pair_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # --- Download results ---
    return np.asarray(runtime.copy_device_to_host(d_results), dtype=np.int32)
