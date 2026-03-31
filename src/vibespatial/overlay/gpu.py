from __future__ import annotations

import contextlib
import logging

import numpy as np
import shapely

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    exclusive_sum,
    segmented_reduce_sum,
    sort_pairs,
    unique_sorted_pairs,
)

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "radix_sort_i32_i32", "radix_sort_i64_i32", "radix_sort_u64_i32",
    "unique_by_key_i32_i32", "unique_by_key_u64_i32",
    "segmented_reduce_sum_f64",
    "segmented_reduce_min_f64", "segmented_reduce_max_f64",
    "select_i32",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.dispatch import record_dispatch_event  # noqa: E402
from vibespatial.runtime.fallbacks import record_fallback_event  # noqa: E402
from vibespatial.runtime.residency import Residency  # noqa: E402
from vibespatial.spatial.segment_primitives import (  # noqa: E402
    DeviceSegmentTable,
    SegmentIntersectionDeviceState,
    SegmentIntersectionResult,
    SegmentTable,
    _extract_segments_gpu,
    classify_segment_intersections,
)

from .types import (  # noqa: E402  # Re-exported for backward compatibility
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    HalfEdgeGraph,
    HalfEdgeGraphDeviceState,
    OverlayFaceDeviceState,
    OverlayFaceTable,
    SplitEventDeviceState,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)

# lyy.22: Number of CUDA streams in the per-group overlay stream pool.
# Each stream processes one overlay pair at a time.  Streams enable
# deferred synchronization: results are collected after all pairs are
# dispatched rather than synchronizing per pair.
#
# NOTE: non_blocking=False is required for correctness.  The overlay
# pipeline interleaves CuPy operations (which respect the CuPy stream
# context) with NVRTC kernel launches via runtime.launch() (which use
# the null/default CUDA stream).  Non-blocking streams do NOT
# synchronize with the null stream, which would allow CuPy ops on the
# stream to race with NVRTC kernels on the null stream within a single
# _overlay_owned call.  Blocking streams (the default) synchronize with
# the null stream on every operation, maintaining correct ordering.
#
# The primary benefit of this pattern is structural: it provides a
# framework for future stream-parallel execution once runtime.launch()
# is plumbed with per-stream support.  In the current architecture,
# GPU work is serialized by the null-stream NVRTC kernels regardless
# of the CuPy stream assignment.
_OVERLAY_STREAM_POOL_SIZE = 2


from vibespatial.overlay.gpu_kernels import (  # noqa: E402
    _BATCH_POINT_IN_RING_KERNEL_NAMES,
    _BATCH_POINT_IN_RING_KERNEL_SOURCE,
    _CONTAINMENT_BYPASS_KERNEL_NAMES,
    _CONTAINMENT_BYPASS_KERNEL_SOURCE,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE,
    _OVERLAY_FACE_LABEL_KERNEL_NAMES,
    _OVERLAY_FACE_LABEL_KERNEL_SOURCE,
    _OVERLAY_FACE_WALK_KERNEL_NAMES,
    _OVERLAY_FACE_WALK_KERNEL_SOURCE,
    _OVERLAY_SPLIT_KERNEL_NAMES,
    _OVERLAY_SPLIT_KERNEL_SOURCE,
)

_OVERLAY_COORDINATE_SCALE = 1_000_000_000.0

# Minimum number of candidate pairs before using the GPU kernel.
# Below this threshold the Python fallback is used to avoid kernel
# launch overhead dominating.
_BATCH_PIP_GPU_THRESHOLD = 100


from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES),
    ("overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES),
    ("overlay-face-label", _OVERLAY_FACE_LABEL_KERNEL_SOURCE, _OVERLAY_FACE_LABEL_KERNEL_NAMES),
    ("overlay-face-assembly", _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE, _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES),
    ("overlay-batch-pip", _BATCH_POINT_IN_RING_KERNEL_SOURCE, _BATCH_POINT_IN_RING_KERNEL_NAMES),
    ("overlay-containment-bypass", _CONTAINMENT_BYPASS_KERNEL_SOURCE, _CONTAINMENT_BYPASS_KERNEL_NAMES),
])


def _overlay_split_kernels():
    return compile_kernel_group("overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES)


def _overlay_face_walk_kernels():
    return compile_kernel_group("overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES)


def _overlay_face_label_kernels():
    return compile_kernel_group("overlay-face-label", _OVERLAY_FACE_LABEL_KERNEL_SOURCE, _OVERLAY_FACE_LABEL_KERNEL_NAMES)


def _overlay_face_assembly_kernels():
    return compile_kernel_group("overlay-face-assembly", _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE, _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES)


def _batch_pip_kernels():
    return compile_kernel_group("overlay-batch-pip", _BATCH_POINT_IN_RING_KERNEL_SOURCE, _BATCH_POINT_IN_RING_KERNEL_NAMES)


def _containment_bypass_kernels():
    return compile_kernel_group(
        "overlay-containment-bypass",
        _CONTAINMENT_BYPASS_KERNEL_SOURCE,
        _CONTAINMENT_BYPASS_KERNEL_NAMES,
    )


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


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _free_split_event_device_state(split_events: SplitEventTable) -> None:
    """Release SplitEventTable device arrays that are no longer needed.

    After build_gpu_atomic_edges has consumed the split events, the large
    float64 buffers (x, y, t, packed_keys) and int32 metadata arrays on
    device are dead.  Freeing them promptly reduces peak GPU memory by
    ~40-60% of the split event footprint.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = split_events.device_state
    if ds is None:
        return
    for arr in (ds.source_segment_ids, ds.packed_keys, ds.t, ds.x, ds.y,
                ds.source_side, ds.row_indices, ds.part_indices, ds.ring_indices):
        runtime.free(arr)


def _free_atomic_edge_excess(atomic_edges: AtomicEdgeTable) -> None:
    """Release AtomicEdgeTable device arrays NOT shared with HalfEdgeGraph.

    The HalfEdgeGraph holds references to src_x, src_y and per-edge metadata
    (source_segment_ids, source_side, row_indices, part_indices, ring_indices,
    direction) from the AtomicEdgeDeviceState.  Only dst_x and dst_y are
    exclusively consumed during half-edge graph construction and are safe to
    free.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = atomic_edges.device_state
    if ds is None:
        return
    runtime.free(ds.dst_x)
    runtime.free(ds.dst_y)


def _segment_metadata(
    source_segment_ids: np.ndarray,
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_count = left_segments.count
    source_side = np.where(source_segment_ids < left_count, 1, 2).astype(np.int8, copy=False)
    row_indices = np.empty(source_segment_ids.size, dtype=np.int32)
    part_indices = np.empty(source_segment_ids.size, dtype=np.int32)
    ring_indices = np.empty(source_segment_ids.size, dtype=np.int32)

    left_mask = source_side == 1
    if np.any(left_mask):
        left_ids = source_segment_ids[left_mask]
        row_indices[left_mask] = left_segments.row_indices[left_ids]
        part_indices[left_mask] = left_segments.part_indices[left_ids]
        ring_indices[left_mask] = left_segments.ring_indices[left_ids]

    right_mask = ~left_mask
    if np.any(right_mask):
        right_ids = source_segment_ids[right_mask] - left_count
        row_indices[right_mask] = right_segments.row_indices[right_ids]
        part_indices[right_mask] = right_segments.part_indices[right_ids]
        ring_indices[right_mask] = right_segments.ring_indices[right_ids]

    return source_side, row_indices, part_indices, ring_indices


def _segment_metadata_gpu(
    d_source_segment_ids,
    *,
    left_count: int,
    left_segments: SegmentTable | DeviceSegmentTable,
    right_segments: SegmentTable | DeviceSegmentTable,
):
    """Derive source_side / row / part / ring indices entirely on GPU.

    When *left_segments* and *right_segments* are ``DeviceSegmentTable``
    instances (GPU-resident), the lookup tables are used directly on
    device with zero host-device transfers.  When they are CPU-resident
    ``SegmentTable`` instances, the metadata arrays are uploaded once.
    """
    d_ids = cp.asarray(d_source_segment_ids)

    # source_side: 1 for left, 2 for right
    d_source_side = cp.where(d_ids < left_count, cp.int8(1), cp.int8(2))

    # Build combined lookup tables (left then right) so a single
    # gather with the raw source_segment_id works directly.
    # Use device arrays directly when available (DeviceSegmentTable),
    # upload from host only for legacy SegmentTable.
    def _to_device(arr):
        """Wrap a host or device array as a CuPy array."""
        return cp.asarray(arr)

    d_all_row = cp.concatenate((
        _to_device(left_segments.row_indices),
        _to_device(right_segments.row_indices),
    ))

    left_has_parts = (
        left_segments.part_indices is not None
        if isinstance(left_segments, DeviceSegmentTable)
        else hasattr(left_segments, "part_indices")
    )
    right_has_parts = (
        right_segments.part_indices is not None
        if isinstance(right_segments, DeviceSegmentTable)
        else hasattr(right_segments, "part_indices")
    )

    if left_has_parts and right_has_parts:
        d_all_part = cp.concatenate((
            _to_device(left_segments.part_indices),
            _to_device(right_segments.part_indices),
        ))
        d_all_ring = cp.concatenate((
            _to_device(left_segments.ring_indices),
            _to_device(right_segments.ring_indices),
        ))
    else:
        # Fallback: zero-fill part/ring indices when not available
        total = left_count + right_segments.count
        d_all_part = cp.zeros(total, dtype=cp.int32)
        d_all_ring = cp.zeros(total, dtype=cp.int32)

    # Right-side IDs are offset by left_count in the combined table,
    # which matches the segment numbering convention already.
    d_row_indices = d_all_row[d_ids]
    d_part_indices = d_all_part[d_ids]
    d_ring_indices = d_all_ring[d_ids]

    return d_source_side, d_row_indices, d_part_indices, d_ring_indices


def _quantize_coordinate(values):
    return cp.rint(values * _OVERLAY_COORDINATE_SCALE).astype(cp.int64, copy=False)


def _empty_half_edge_graph(
    atomic_edges: AtomicEdgeTable,
) -> HalfEdgeGraph:
    runtime = get_cuda_runtime()
    empty_i32 = np.asarray([], dtype=np.int32)
    empty_i8 = np.asarray([], dtype=np.int8)
    empty_f64 = np.asarray([], dtype=np.float64)
    empty_device_i32 = runtime.allocate((0,), np.int32)
    empty_device_i8 = runtime.allocate((0,), np.int8)
    empty_device_f64 = runtime.allocate((0,), np.float64)
    return HalfEdgeGraph(
        left_segment_count=atomic_edges.left_segment_count,
        right_segment_count=atomic_edges.right_segment_count,
        runtime_selection=atomic_edges.runtime_selection,
        _edge_count=0,
        _source_segment_ids=empty_i32,
        _source_side=empty_i8,
        _row_indices=empty_i32,
        _part_indices=empty_i32,
        _ring_indices=empty_i32,
        _direction=empty_i8,
        _src_x=empty_f64,
        _src_y=empty_f64,
        _dst_x=empty_f64,
        _dst_y=empty_f64,
        _node_x=empty_f64,
        _node_y=empty_f64,
        _src_node_ids=empty_i32,
        _dst_node_ids=empty_i32,
        _angle=empty_f64,
        _sorted_edge_ids=empty_i32,
        _edge_positions=empty_i32,
        _next_edge_ids=empty_i32,
        device_state=HalfEdgeGraphDeviceState(
            node_x=empty_device_f64,
            node_y=empty_device_f64,
            src_node_ids=empty_device_i32,
            dst_node_ids=empty_device_i32,
            angle=empty_device_f64,
            sorted_edge_ids=empty_device_i32,
            edge_positions=empty_device_i32,
            next_edge_ids=empty_device_i32,
            src_x=empty_device_f64,
            src_y=empty_device_f64,
            source_segment_ids=empty_device_i32,
            source_side=empty_device_i8,
            row_indices=empty_device_i32,
            part_indices=empty_device_i32,
            ring_indices=empty_device_i32,
            direction=empty_device_i8,
        ),
    )


from vibespatial.overlay.host_fallback import (  # noqa: E402
    _build_polygon_output_from_faces,
    _face_sample_point,
    _label_face_coverage,
    _signed_area_and_centroid,
)


def _has_polygonal_families(geom: OwnedGeometryArray) -> bool:
    """Return True if the geometry array has POLYGON or MULTIPOLYGON families."""
    return (
        GeometryFamily.POLYGON in geom.families
        or GeometryFamily.MULTIPOLYGON in geom.families
    )


def _empty_polygon_output(runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    residency = Residency.DEVICE if cp is not None else Residency.HOST
    empty_validity = np.asarray([], dtype=bool)
    empty_tags = np.asarray([], dtype=np.int8)
    empty_offsets = np.asarray([], dtype=np.int32)
    device_state = None
    if residency is Residency.DEVICE:
        try:
            rt = get_cuda_runtime()
            device_state = OwnedGeometryDeviceState(
                validity=rt.from_host(empty_validity),
                tags=rt.from_host(empty_tags),
                family_row_offsets=rt.from_host(empty_offsets),
                families={},
            )
        except Exception:
            residency = Residency.HOST
            device_state = None
    return OwnedGeometryArray(
        validity=empty_validity,
        tags=empty_tags,
        family_row_offsets=empty_offsets,
        families={},
        residency=residency,
        runtime_history=[runtime_selection],
        device_state=device_state,
    )


def _build_device_backed_fixed_polygon_output(
    device_x: DeviceArray,
    device_y: DeviceArray,
    *,
    row_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    runtime = get_cuda_runtime()
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    ring_offsets = np.arange(0, (row_count + 1) * 5, 5, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=None,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        runtime_history=[runtime_selection],
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None,
                )
            },
        ),
    )


def _axis_aligned_box_bounds(values: OwnedGeometryArray) -> np.ndarray | None:
    if set(values.families) != {GeometryFamily.POLYGON}:
        return None
    polygon_buffer = values.families[GeometryFamily.POLYGON]
    row_count = polygon_buffer.row_count
    if row_count == 0 or row_count != values.row_count:
        return None
    if polygon_buffer.ring_offsets is None:
        return None
    if not np.array_equal(polygon_buffer.geometry_offsets, np.arange(row_count + 1, dtype=np.int32)):
        return None
    if not np.array_equal(polygon_buffer.ring_offsets, np.arange(0, (row_count + 1) * 5, 5, dtype=np.int32)):
        return None
    if np.any(polygon_buffer.empty_mask):
        return None

    x = polygon_buffer.x.reshape(row_count, 5)
    y = polygon_buffer.y.reshape(row_count, 5)
    if not (np.allclose(x[:, 0], x[:, 4]) and np.allclose(y[:, 0], y[:, 4])):
        return None

    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    axis_aligned = ((np.abs(dx) < 1e-12) ^ (np.abs(dy) < 1e-12))
    if not np.all(axis_aligned):
        return None
    return np.column_stack(
        (
            np.min(x[:, :4], axis=1),
            np.min(y[:, :4], axis=1),
            np.max(x[:, :4], axis=1),
            np.max(y[:, :4], axis=1),
        )
    ).astype(np.float64, copy=False)


def _build_polygon_output_from_faces_gpu(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    selected_face_indices: np.ndarray | cp.ndarray,
) -> OwnedGeometryArray | None:
    """GPU face-to-polygon assembly (Phase 11: GPU boundary cycle detection).

    Full GPU pipeline:
      Steps 1-2: Edge-to-face mapping and face selection via CuPy scatter.
      Step 3: Boundary edge identification via NVRTC kernel.
      Step 4: Boundary next-edge computation via NVRTC kernel.
      Step 5: Cycle detection via GPU pointer jumping + list ranking;
              per-cycle area/centroid via segmented reduction.
      Steps 6-7: Coordinate offset computation and ring scatter via GPU.
      Steps 7b-8: Hole ring extraction and merge on device.
      Step 8b: GPU nesting depth for boundary rings (even depth = exterior).
      Step 9: Hole-to-exterior assignment via GPU PIP kernel.
      Step 9b: GPU sibling hole nesting depth (even = valid hole, odd = skip).
      Step 10: GPU output assembly with device-side sorting, grouping,
              and host-side row_polygons construction; D->H transfer at the
              ADR-0005 materialization boundary.

    Returns None if GPU is unavailable (caller falls back to CPU path).
    """
    if cp is None or half_edge_graph.device_state is None or faces.device_state is None:
        return None
    if selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)

    runtime = get_cuda_runtime()
    kernels = _overlay_face_assembly_kernels()
    walk_kernels = _overlay_face_walk_kernels()
    kernels.update(walk_kernels)
    ptr = runtime.pointer
    edge_count = half_edge_graph.edge_count
    face_count = faces.face_count
    block = (256, 1, 1)
    edge_grid = (max(1, (edge_count + 255) // 256), 1, 1)

    device = half_edge_graph.device_state
    face_device = faces.device_state

    # --- Step 1: Map edges to faces (Tier 2: CuPy vectorised scatter) ---
    # Build edge_face_ids on device: for each edge, which face does it belong to?
    d_edge_face_ids = cp.full(edge_count, -1, dtype=cp.int32)
    d_face_offsets = cp.asarray(face_device.face_offsets)
    d_face_edge_ids = cp.asarray(face_device.face_edge_ids)
    total_face_edges = int(d_face_edge_ids.size)
    if total_face_edges > 0:
        # For each slot in face_edge_ids, find which face it belongs to
        slot_ids = cp.arange(total_face_edges, dtype=cp.int32)
        slot_face = cp.searchsorted(d_face_offsets[1:], slot_ids, side='right').astype(cp.int32)
        d_edge_face_ids[d_face_edge_ids] = slot_face

    # --- Step 2: Build face selection mask on device ---
    d_face_selected = cp.zeros(face_count, dtype=cp.int8)
    d_face_selected[cp.asarray(selected_face_indices)] = 1

    # --- Step 3: Identify boundary edges via GPU kernel ---
    d_is_boundary = cp.empty(edge_count, dtype=cp.int8)
    runtime.launch(
        kernels["compute_boundary_edges"],
        grid=edge_grid, block=block,
        params=(
            (ptr(d_edge_face_ids), ptr(d_face_selected), ptr(device.next_edge_ids),
             ptr(d_is_boundary), edge_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    boundary_count = int(cp.sum(d_is_boundary != 0))
    if boundary_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # --- Step 4: Compute boundary next pointers via GPU kernel ---
    d_boundary_next = cp.full(edge_count, -1, dtype=cp.int32)
    max_steps = edge_count
    runtime.launch(
        kernels["compute_boundary_next"],
        grid=edge_grid, block=block,
        params=(
            (ptr(d_edge_face_ids), ptr(d_face_selected), ptr(device.next_edge_ids),
             ptr(d_is_boundary), ptr(d_boundary_next), edge_count, max_steps),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    # --- Step 5: Detect boundary cycles via GPU pointer jumping ---
    boundary_edge_indices = cp.flatnonzero(d_is_boundary != 0).astype(cp.int32, copy=False)
    boundary_count = int(boundary_edge_indices.size)
    # Phase 25 memory: d_is_boundary, d_edge_face_ids, d_face_selected,
    # and the Step 1 face_offsets/edge_ids copies are dead.
    del d_is_boundary, d_edge_face_ids, d_face_selected
    del d_face_offsets, d_face_edge_ids

    # Build compact boundary-local next array
    edge_to_compact = cp.full(edge_count, -1, dtype=cp.int32)
    edge_to_compact[boundary_edge_indices] = cp.arange(boundary_count, dtype=cp.int32)
    compact_next = edge_to_compact[d_boundary_next[boundary_edge_indices]]
    # Phase 25 memory: edge_to_compact is dead after compact_next is built.
    del edge_to_compact

    # Pointer jumping to find cycle labels (minimum compact index in cycle)
    cycle_label = cp.arange(boundary_count, dtype=cp.int32)
    jump_b = compact_next.copy()
    max_iter_b = max(1, int(np.ceil(np.log2(max(1, boundary_count)))))
    for _ in range(max_iter_b):
        cycle_label = cp.minimum(cycle_label, cycle_label[jump_b])
        jump_b = jump_b[jump_b]
    # Phase 25 memory: jump_b is dead after pointer jumping.
    del jump_b

    # List ranking: compute position within each cycle using NVRTC kernel
    d_rank_b = cp.empty(boundary_count, dtype=cp.int32)
    d_compact_next_i64 = compact_next.astype(cp.int64)
    boundary_grid = (max(1, (boundary_count + 255) // 256), 1, 1)
    rank_params = (
        (ptr(cycle_label), ptr(d_compact_next_i64),
         ptr(d_rank_b), boundary_count, boundary_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    runtime.launch(
        kernels["list_rank_within_cycle"],
        grid=boundary_grid, block=block, params=rank_params,
    )

    # Sort by (cycle_label, rank) to get cycle-ordered boundary edges
    packed_b = cycle_label.astype(cp.int64) * int(boundary_count) + d_rank_b.astype(cp.int64)
    # Phase 25 memory: d_rank_b and d_compact_next_i64 are dead.
    del d_rank_b, d_compact_next_i64
    b_sort = sort_pairs(packed_b, cp.arange(boundary_count, dtype=cp.int32), synchronize=False)
    del packed_b
    sorted_compact_ids = b_sort.values
    sorted_labels = cycle_label[sorted_compact_ids]
    # Phase 25 memory: cycle_label consumed; compact_next dead.
    del cycle_label, compact_next

    # Find unique cycles and their segment boundaries
    b_start_mask = cp.empty(boundary_count, dtype=cp.bool_)
    b_start_mask[0] = True
    if boundary_count > 1:
        b_start_mask[1:] = sorted_labels[1:] != sorted_labels[:-1]
    cycle_starts = cp.flatnonzero(b_start_mask).astype(cp.int32, copy=False)
    cycle_ends = cp.concatenate((cycle_starts[1:], cp.asarray([boundary_count], dtype=cp.int32)))
    cycle_lengths = cycle_ends - cycle_starts
    del b_start_mask, sorted_labels

    # Filter cycles with >= 3 edges
    valid_cycle_mask = cycle_lengths >= 3
    valid_cycle_indices = cp.flatnonzero(valid_cycle_mask).astype(cp.int32, copy=False)
    ring_count = int(valid_cycle_indices.size)
    del valid_cycle_mask

    if ring_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    valid_cycle_starts = cycle_starts[valid_cycle_indices]
    valid_cycle_ends = cycle_ends[valid_cycle_indices]
    valid_cycle_lengths = cycle_lengths[valid_cycle_indices]
    del cycle_starts, cycle_ends, cycle_lengths, valid_cycle_indices

    # Map sorted compact ids back to full edge ids for the valid cycles
    sorted_full_edge_ids = boundary_edge_indices[sorted_compact_ids]
    del sorted_compact_ids

    # Compute per-boundary-edge shoelace contributions on device
    d_src_x_b = cp.asarray(device.src_x)
    d_src_y_b = cp.asarray(device.src_y)
    b_x0 = d_src_x_b[sorted_full_edge_ids]
    b_y0 = d_src_y_b[sorted_full_edge_ids]
    b_next_edges = d_boundary_next[sorted_full_edge_ids]
    b_x1 = d_src_x_b[b_next_edges]
    b_y1 = d_src_y_b[b_next_edges]
    del b_next_edges
    b_cross = b_x0 * b_y1 - b_x1 * b_y0
    b_cx_contrib = (b_x0 + b_x1) * b_cross
    b_cy_contrib = (b_y0 + b_y1) * b_cross
    # Phase 25 memory: boundary coordinate arrays consumed by cross products.
    del b_x0, b_y0, b_x1, b_y1

    # Segmented reduce for per-cycle area and centroid
    cross_sums_b = segmented_reduce_sum(b_cross, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    cx_sums_b = segmented_reduce_sum(b_cx_contrib, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    cy_sums_b = segmented_reduce_sum(b_cy_contrib, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    del b_cross, b_cx_contrib, b_cy_contrib

    d_ring_area = cross_sums_b * 0.5
    safe_twice_b = cp.where(cross_sums_b == 0.0, 1.0, cross_sums_b)
    factor_b = 1.0 / (3.0 * safe_twice_b)
    d_ring_centroid_x = cx_sums_b * factor_b
    d_ring_centroid_y = cy_sums_b * factor_b
    del cross_sums_b, cx_sums_b, cy_sums_b, safe_twice_b, factor_b

    # Ring edge starts (first full edge id of each valid cycle) and counts
    d_ring_edge_starts = sorted_full_edge_ids[valid_cycle_starts]
    d_ring_edge_counts = valid_cycle_lengths
    del sorted_full_edge_ids, valid_cycle_starts, valid_cycle_ends, valid_cycle_lengths

    # Source row per cycle: take the row_index of the first edge (device-resident
    # until Step 10 materialization boundary per ADR-0005).
    # Read from device_state directly to avoid D->H->D round-trip.
    d_row_indices = cp.asarray(device.row_indices)
    d_cycle_source_rows = d_row_indices[d_ring_edge_starts].astype(cp.int32)

    # --- Step 6: Compute ring coordinate offsets (Tier 3a: exclusive_scan) ---
    # Each ring needs edge_count + 1 coordinates (for closure)
    ring_coord_counts = d_ring_edge_counts + 1
    d_ring_coord_offsets = exclusive_sum(ring_coord_counts.astype(cp.int32, copy=False))
    # Single scalar D->H read: total_coords = last_offset + last_count.
    total_coords = int(cp.asnumpy(d_ring_coord_offsets[-1:] + ring_coord_counts[-1:])[0])  # zcopy:ok(allocation-fence: need total_coords to size d_out_x/d_out_y output buffers)

    # --- Step 7: Scatter ring coordinates via GPU kernel ---
    # Zero-fill to prevent denormalized garbage in unwritten positions if the
    # scatter kernel skips any coordinate slot due to an out-of-bounds
    # boundary_next index.  cp.empty() would recycle a pool block containing
    # stale int32 metadata, which reinterprets as denormalized float64 values
    # (e.g. 4e-316) that crash GEOS with TopologyException.
    d_out_x = cp.zeros(total_coords, dtype=cp.float64)
    d_out_y = cp.zeros(total_coords, dtype=cp.float64)
    ring_grid = (max(1, (ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["scatter_ring_coordinates"],
        grid=ring_grid, block=block,
        params=(
            (ptr(device.src_x), ptr(device.src_y),
             ptr(d_ring_edge_starts), ptr(d_ring_coord_offsets),
             ptr(d_ring_edge_counts), ptr(d_boundary_next),
             ptr(d_out_x), ptr(d_out_y), ring_count,
             edge_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
             KERNEL_PARAM_I32),
        ),
    )

    # Phase 25 memory: d_ring_edge_starts, d_ring_coord_offsets, and
    # d_ring_edge_counts are still needed for Step 8 merge.

    # --- Step 7b: Extract hole rings from unselected bounded faces ---
    # Per ADR-0016, holes are unselected bounded faces whose ring coordinates
    # form interior rings of the output polygons.
    # GPU filter: hole faces are unselected bounded faces (Tier 2: CuPy).
    d_bounded_mask_dev = cp.asarray(face_device.bounded_mask)
    d_face_selected_bool = cp.zeros(face_count, dtype=cp.bool_)
    d_face_selected_bool[cp.asarray(selected_face_indices)] = True
    d_hole_mask = (d_bounded_mask_dev != 0) & (~d_face_selected_bool)
    del d_bounded_mask_dev, d_face_selected_bool
    d_hole_fi = cp.flatnonzero(d_hole_mask).astype(cp.int32)
    del d_hole_mask

    # Extract hole ring coordinates using scatter_ring_coordinates kernel.
    # Hole faces use next_edge_ids (not boundary_next) for edge traversal.
    d_face_offsets_dev = cp.asarray(face_device.face_offsets)
    d_face_edge_ids_dev = cp.asarray(face_device.face_edge_ids)
    d_next_i32 = cp.asarray(device.next_edge_ids).astype(cp.int32)

    if d_hole_fi.size > 0:
        d_hole_starts_in_face = d_face_offsets_dev[d_hole_fi]
        d_hole_ends_in_face = d_face_offsets_dev[d_hole_fi + 1]
        d_hole_lengths = d_hole_ends_in_face - d_hole_starts_in_face

        # Filter to holes with >= 3 edges
        d_valid_hole_mask = d_hole_lengths >= 3
        d_valid_hole_idx = cp.flatnonzero(d_valid_hole_mask).astype(cp.int32)
        n_valid_holes = int(d_valid_hole_idx.size)

        if n_valid_holes > 0:
            d_vh_starts = d_hole_starts_in_face[d_valid_hole_idx]
            d_vh_lengths = d_hole_lengths[d_valid_hole_idx]

            # First edge of each hole face (the canonical cycle start)
            d_hole_edge_starts = d_face_edge_ids_dev[d_vh_starts]

            # Scatter coordinates via GPU kernel
            d_hole_coord_counts = d_vh_lengths + 1  # +1 for ring closure
            d_hole_coord_offsets_partial = exclusive_sum(d_hole_coord_counts.astype(cp.int32))
            # exclusive_sum returns [0, c0, c0+c1, ...] of size n_valid_holes.
            # Append the total to form a proper (n+1) offset array so that
            # offsets[:-1] gives n start indices and offsets[1:] gives n ends.
            _last_offset = d_hole_coord_offsets_partial[-1:] + d_hole_coord_counts[-1:]
            d_hole_coord_offsets = cp.concatenate([d_hole_coord_offsets_partial, _last_offset])
            total_hole_coords = int(cp.asnumpy(_last_offset)[0])

            # Zero-fill: same rationale as boundary ring output above.
            d_hole_x = cp.zeros(total_hole_coords, dtype=cp.float64)
            d_hole_y = cp.zeros(total_hole_coords, dtype=cp.float64)
            hole_grid = (max(1, (n_valid_holes + 255) // 256), 1, 1)
            runtime.launch(
                kernels["scatter_ring_coordinates"],
                grid=hole_grid, block=block,
                params=(
                    (ptr(device.src_x), ptr(device.src_y),
                     ptr(d_hole_edge_starts), ptr(d_hole_coord_offsets),
                     ptr(d_vh_lengths), ptr(d_next_i32),
                     ptr(d_hole_x), ptr(d_hole_y), n_valid_holes,
                     edge_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                     KERNEL_PARAM_I32),
                ),
            )

            # Compute per-hole-ring shoelace area/centroid on device
            # Phase 25 memory: removed unused hole_slot_ids searchsorted result.
            # Shoelace: cross product per edge (skip last coord = closure)
            h_x0 = d_hole_x[:-1] if total_hole_coords > 1 else d_hole_x
            h_y0 = d_hole_y[:-1] if total_hole_coords > 1 else d_hole_y
            h_x1 = d_hole_x[1:] if total_hole_coords > 1 else d_hole_x
            h_y1 = d_hole_y[1:] if total_hole_coords > 1 else d_hole_y

            # Per-ring starts/ends for segmented reduce (use coord offsets)
            h_cross = h_x0 * h_y1 - h_x1 * h_y0
            h_cx_c = (h_x0 + h_x1) * h_cross
            h_cy_c = (h_y0 + h_y1) * h_cross
            del h_x0, h_y0, h_x1, h_y1

            # Segmented reduce -- use coord-based segments (each ring's coords
            # are contiguous, last coord is closure vertex)
            hole_seg_starts = d_hole_coord_offsets[:-1]
            hole_seg_ends = d_hole_coord_offsets[1:] - 1  # exclude closure vertex from cross products
            h_area = segmented_reduce_sum(h_cross, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values * 0.5
            h_cx_sum = segmented_reduce_sum(h_cx_c, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values
            h_cy_sum = segmented_reduce_sum(h_cy_c, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values
            del h_cross, h_cx_c, h_cy_c, hole_seg_starts, hole_seg_ends
            h_safe_twice = cp.where(h_area == 0.0, 0.5, h_area * 2.0)
            h_factor = 1.0 / (3.0 * h_safe_twice)
            h_centroid_x = h_cx_sum * h_factor
            h_centroid_y = h_cy_sum * h_factor
            del h_cx_sum, h_cy_sum, h_safe_twice, h_factor

            # Filter degenerate holes on device (|area| < 1e-12)
            d_nondegenerate = cp.abs(h_area) >= 1e-12
            d_nondegen_idx = cp.flatnonzero(d_nondegenerate).astype(cp.int32)
            n_valid_nondegenerate = int(d_nondegen_idx.size)
            del d_nondegenerate

            if n_valid_nondegenerate > 0:
                # Keep only non-degenerate hole rings (device-resident)
                _d_hole_area = h_area[d_nondegen_idx]
                _d_hole_cx = h_centroid_x[d_nondegen_idx]
                _d_hole_cy = h_centroid_y[d_nondegen_idx]
                _d_hole_lengths = d_vh_lengths[d_nondegen_idx]
                _d_hole_starts = d_hole_coord_offsets[:-1][d_nondegen_idx]
                # These device arrays are passed directly to Step 8
                _hole_device_data = (_d_hole_area, _d_hole_cx, _d_hole_cy,
                                     _d_hole_lengths, _d_hole_starts,
                                     d_hole_x, d_hole_y, d_hole_coord_offsets,
                                     n_valid_nondegenerate)
            else:
                _hole_device_data = None
        else:
            _hole_device_data = None
    else:
        _hole_device_data = None

    # --- Step 8: Merge boundary + hole ring data on device ---
    boundary_ring_count = ring_count

    if _hole_device_data is not None:
        (_d_hole_area, _d_hole_cx, _d_hole_cy, _d_hole_lengths,
         _d_hole_starts, _d_hole_all_x, _d_hole_all_y,
         _d_hole_coord_offsets_full, n_holes) = _hole_device_data

        # Build compact hole coordinate buffer: gather non-degenerate hole
        # ring coords into a contiguous buffer. Use per-hole coord counts.
        d_hole_coord_counts = _d_hole_lengths + 1  # +1 for ring closure
        d_hole_offsets_compact = exclusive_sum(d_hole_coord_counts.astype(cp.int32))
        # exclusive_sum returns [0, c0, c0+c1, ...] of size n_holes; total
        # is last_offset + last_count (not just last_offset).
        total_hole_compact = int(cp.asnumpy(d_hole_offsets_compact[-1:])[0]) + int(cp.asnumpy(d_hole_coord_counts[-1:])[0])

        # Gather coords from the original hole buffer using starts + lengths
        if total_hole_compact > 0:
            slot_ids_h = cp.arange(total_hole_compact, dtype=cp.int32)
            slot_ring_h = cp.searchsorted(d_hole_offsets_compact[1:], slot_ids_h, side='right').astype(cp.int32)
            slot_local_h = slot_ids_h - d_hole_offsets_compact[slot_ring_h]
            slot_src_h = _d_hole_starts[slot_ring_h] + slot_local_h
            d_hole_compact_x = _d_hole_all_x[slot_src_h]
            d_hole_compact_y = _d_hole_all_y[slot_src_h]
        else:
            d_hole_compact_x = cp.empty(0, dtype=cp.float64)
            d_hole_compact_y = cp.empty(0, dtype=cp.float64)

        d_hole_edge_counts = _d_hole_lengths

        # Merge: boundary rings [0..ring_count), hole rings [ring_count..total)
        # Hole faces are traversed counterclockwise (positive area) in the
        # overlay half-edge graph, but they represent interior rings that must
        # have negative (clockwise) area so assign_holes_to_exteriors treats
        # them as holes rather than self-assigned exteriors.
        d_all_area = cp.concatenate((d_ring_area, -cp.abs(_d_hole_area)))
        d_all_cx = cp.concatenate((d_ring_centroid_x, _d_hole_cx))
        d_all_cy = cp.concatenate((d_ring_centroid_y, _d_hole_cy))
        d_all_x = cp.concatenate((d_out_x, d_hole_compact_x))
        d_all_y = cp.concatenate((d_out_y, d_hole_compact_y))
        # Merged coordinate offsets: boundary offsets + compact hole offsets shifted.
        # Use total_coords (not boundary_total) because total_coords is the
        # actual byte count of boundary ring coordinates in d_out_x/d_out_y,
        # i.e. where the compact hole coords start in d_all_x/d_all_y.
        d_hole_offsets_shifted = d_hole_offsets_compact + total_coords
        d_all_coord_offsets = cp.concatenate((d_ring_coord_offsets, d_hole_offsets_shifted))
        d_all_edge_counts = cp.concatenate((d_ring_edge_counts, d_hole_edge_counts))
        # Phase 25 memory: pre-merge ring arrays consumed by concatenation.
        del d_ring_area, d_ring_centroid_x, d_ring_centroid_y
        del d_out_x, d_out_y, d_ring_coord_offsets, d_ring_edge_counts
        del d_hole_compact_x, d_hole_compact_y, d_hole_offsets_compact
        del _d_hole_area, _d_hole_cx, _d_hole_cy, d_hole_edge_counts
        del d_hole_offsets_shifted
    else:
        n_holes = 0
        d_all_area = d_ring_area
        d_all_cx = d_ring_centroid_x
        d_all_cy = d_ring_centroid_y
        d_all_x = d_out_x
        d_all_y = d_out_y
        d_all_coord_offsets = d_ring_coord_offsets
        d_all_edge_counts = d_ring_edge_counts

    # Phase 25 memory: boundary_next, face data copies, and hole-detection
    # arrays are dead after the merge.
    del d_boundary_next, d_face_offsets_dev, d_face_edge_ids_dev, d_next_i32
    del d_hole_fi, d_src_x_b, d_src_y_b
    total_ring_count = boundary_ring_count + n_holes

    # --- Source rows for ALL rings on device ---
    # Boundary rings: from cycle walk; holes: inherit from exterior later.
    d_all_source_rows = cp.full(total_ring_count, -1, dtype=cp.int32)
    d_all_source_rows[:boundary_ring_count] = d_cycle_source_rows

    # --- Step 8b: GPU nesting depth for boundary rings ---
    # Boundary rings with positive area might be nested inside other
    # positive-area boundary rings from the same source row. Count
    # containment depth: even → true exterior, odd → nested interior.
    d_is_boundary_flag = cp.zeros(total_ring_count, dtype=cp.bool_)
    d_is_boundary_flag[:boundary_ring_count] = True
    d_pos_area_boundary = d_is_boundary_flag & (d_all_area > 0.0)
    n_pos_boundary = int(cp.sum(d_pos_area_boundary))

    if n_pos_boundary > 1:
        # Launch nesting depth kernel only when there are multiple
        # positive-area boundary rings (single ring is trivially exterior)
        d_boundary_depth = cp.zeros(boundary_ring_count, dtype=cp.int32)
        boundary_depth_grid = (max(1, (boundary_ring_count + 255) // 256), 1, 1)
        runtime.launch(
            kernels["count_boundary_nesting_depth"],
            grid=boundary_depth_grid, block=block,
            params=(
                (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
                 ptr(d_all_source_rows), ptr(d_all_coord_offsets),
                 ptr(d_all_edge_counts), ptr(d_all_x), ptr(d_all_y),
                 ptr(d_boundary_depth), boundary_ring_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )
        # True exteriors: positive area + even nesting depth
        d_even_depth = (d_boundary_depth % 2) == 0
        d_exterior_mask = (d_all_area[:boundary_ring_count] > 0.0) & d_even_depth
        # Extend to full ring array (hole rings are never exteriors)
        d_exterior_mask_full = cp.zeros(total_ring_count, dtype=cp.bool_)
        d_exterior_mask_full[:boundary_ring_count] = d_exterior_mask
    else:
        # Single or zero positive-area boundary rings: no nesting possible
        d_exterior_mask_full = cp.zeros(total_ring_count, dtype=cp.bool_)
        d_exterior_mask_full[:boundary_ring_count] = (
            d_all_area[:boundary_ring_count] > 0.0
        )

    d_exterior_indices = cp.flatnonzero(d_exterior_mask_full).astype(cp.int32)
    exterior_count = int(d_exterior_indices.size)

    if exterior_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # --- Step 9: Assign holes to exteriors via GPU kernel ---
    d_exterior_id = cp.full(total_ring_count, -1, dtype=cp.int32)
    ring_grid_all = (max(1, (total_ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["assign_holes_to_exteriors"],
        grid=ring_grid_all, block=block,
        params=(
            (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
             ptr(d_all_coord_offsets), ptr(d_all_edge_counts),
             ptr(d_all_x), ptr(d_all_y),
             ptr(d_exterior_indices), exterior_count,
             ptr(d_exterior_id), total_ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # --- Step 9b: GPU sibling hole nesting depth ---
    # For each hole assigned to an exterior, count how many sibling holes
    # (same exterior, larger |area|) contain its centroid. Even local
    # depth -> valid direct hole; odd -> nested inside another hole (skip).
    d_sibling_depth = cp.zeros(total_ring_count, dtype=cp.int32)

    # Check if there are any holes assigned
    d_is_hole = (d_exterior_id >= 0) & (d_exterior_id != cp.arange(total_ring_count, dtype=cp.int32))
    n_assigned_holes = int(cp.sum(d_is_hole))

    if n_assigned_holes > 1:
        runtime.launch(
            kernels["count_sibling_hole_depth"],
            grid=ring_grid_all, block=block,
            params=(
                (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
                 ptr(d_exterior_id), ptr(d_all_coord_offsets),
                 ptr(d_all_edge_counts), ptr(d_all_x), ptr(d_all_y),
                 ptr(d_sibling_depth), total_ring_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # --- Step 10: Device-resident output assembly ---
    # Build GeoArrow-format polygon output entirely on device.
    # Valid holes: assigned to an exterior AND even sibling depth.
    d_valid_hole_mask = d_is_hole & ((d_sibling_depth % 2) == 0)

    # Also filter: only keep holes whose assigned exterior is a true exterior
    d_ext_is_valid = d_exterior_mask_full[d_exterior_id.clip(0, total_ring_count - 1)]
    d_valid_hole_mask = d_valid_hole_mask & d_ext_is_valid

    # Propagate source rows from exterior to holes on device
    d_hole_ext_ids = d_exterior_id.clip(0, total_ring_count - 1)
    d_all_source_rows = cp.where(
        d_valid_hole_mask,
        d_all_source_rows[d_hole_ext_ids],
        d_all_source_rows,
    )

    # Build the ordered ring list: for each exterior, gather its holes.
    # Create a sort key: (source_row, exterior_id, is_exterior_flag desc,
    # ring_id) so rings within a polygon group sort as
    # [exterior, hole_0, hole_1, ...].
    d_is_output_ring = d_exterior_mask_full | d_valid_hole_mask
    d_output_ring_ids = cp.flatnonzero(d_is_output_ring).astype(cp.int32)
    n_output_rings = int(d_output_ring_ids.size)

    if n_output_rings == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # For each output ring, its exterior id (exteriors map to themselves)
    d_out_ext_id = cp.where(
        d_exterior_mask_full[d_output_ring_ids],
        d_output_ring_ids,
        d_exterior_id[d_output_ring_ids],
    )
    d_out_source_row = d_all_source_rows[d_output_ring_ids]
    d_out_is_ext = d_exterior_mask_full[d_output_ring_ids].astype(cp.int32)

    # Sort by (source_row, exterior_id, NOT is_exterior, ring_id) so
    # exteriors come first within each polygon group.
    # Use CuPy lexsort (Tier 2) for arbitrary value ranges.
    d_sort_order = cp.lexsort(cp.stack((
        d_output_ring_ids,           # last key = least significant
        1 - d_out_is_ext,            # exteriors (0) before holes (1)
        d_out_ext_id,                # group by exterior
        d_out_source_row,            # primary key: source row
    )))
    d_sorted_output_ids = d_output_ring_ids[d_sort_order]

    # Identify polygon boundaries: a new polygon starts at each exterior ring
    d_sorted_is_ext = d_exterior_mask_full[d_sorted_output_ids]
    d_sorted_source_row = d_all_source_rows[d_sorted_output_ids]

    # Each exterior starts a new polygon group
    # Count rings per polygon (for ring_offsets)
    d_poly_start_mask = d_sorted_is_ext
    d_poly_starts = cp.flatnonzero(d_poly_start_mask).astype(cp.int32)
    n_polygons = int(d_poly_starts.size)

    if n_polygons == 0:
        return _empty_polygon_output(faces.runtime_selection)

    d_poly_ends = cp.concatenate((d_poly_starts[1:], cp.asarray([n_output_rings], dtype=cp.int32)))
    d_rings_per_poly = d_poly_ends - d_poly_starts

    # Source row per polygon (from the exterior ring)
    d_poly_source_row = d_sorted_source_row[d_poly_starts]

    # Polygons are already sorted by source row (primary sort key of the
    # ring sort). Detect source-row group boundaries directly.
    d_row_change = cp.empty(n_polygons, dtype=cp.bool_)
    d_row_change[0] = True
    if n_polygons > 1:
        d_row_change[1:] = d_poly_source_row[1:] != d_poly_source_row[:-1]
    d_row_starts = cp.flatnonzero(d_row_change).astype(cp.int32)
    d_row_ends = cp.concatenate((d_row_starts[1:], cp.asarray([n_polygons], dtype=cp.int32)))
    d_polys_per_row = d_row_ends - d_row_starts
    n_output_rows = int(d_row_starts.size)

    # Transfer only small O(ring_count) structural metadata to host.
    # Coordinate arrays d_all_x/d_all_y stay on device (Phase 14 zero-copy).
    h_sorted_output_ids = cp.asnumpy(d_sorted_output_ids)
    h_rings_per_poly = cp.asnumpy(d_rings_per_poly)
    h_polys_per_row = cp.asnumpy(d_polys_per_row)
    h_row_source_ids = cp.asnumpy(d_poly_source_row[d_row_starts])
    h_poly_starts = cp.asnumpy(d_poly_starts)
    h_all_coord_offsets = cp.asnumpy(d_all_coord_offsets)
    h_all_edge_counts = cp.asnumpy(d_all_edge_counts)

    return _build_device_resident_polygon_output(
        d_all_x=d_all_x,
        d_all_y=d_all_y,
        h_all_coord_offsets=h_all_coord_offsets,
        h_all_edge_counts=h_all_edge_counts,
        h_sorted_output_ids=h_sorted_output_ids,
        h_rings_per_poly=h_rings_per_poly,
        h_polys_per_row=h_polys_per_row,
        h_row_source_ids=h_row_source_ids,
        h_poly_starts=h_poly_starts,
        n_output_rows=n_output_rows,
        runtime_selection=faces.runtime_selection,
    )


def _build_device_resident_polygon_output(
    *,
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    h_all_coord_offsets: np.ndarray,
    h_all_edge_counts: np.ndarray,
    h_sorted_output_ids: np.ndarray,
    h_rings_per_poly: np.ndarray,
    h_polys_per_row: np.ndarray,
    h_row_source_ids: np.ndarray,
    h_poly_starts: np.ndarray,
    n_output_rows: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    """Build device-resident OwnedGeometryArray from GPU face assembly results.

    Accepts GPU-computed ring grouping (Phase 12 sibling hole nesting) and
    builds GeoArrow offset arrays on host via vectorised NumPy, then gathers
    coordinates on device via CuPy fancy indexing.

    Phase 14 (ADR-0005): eliminates dominant D->H coordinate transfer.
    Phase 26: Vectorised output assembly -- eliminates per-row and per-ring
    Python loops that dominated wall time for large polygon counts.
    """
    runtime = get_cuda_runtime()

    # --- Vectorised classification: polygon vs multipolygon per row ---
    # A row with exactly 1 polygon is a POLYGON; otherwise MULTIPOLYGON.
    is_polygon_row = h_polys_per_row == 1
    output_row_count = n_output_rows
    if output_row_count == 0:
        return _empty_polygon_output(runtime_selection)

    validity = np.ones(output_row_count, dtype=bool)
    tags = np.full(output_row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(output_row_count, -1, dtype=np.int32)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]

    # Total polygon count: sum of polys_per_row for polygon-type rows (== 1).
    # Total multipolygon count: number of multipolygon rows.
    n_total_polys = int(h_rings_per_poly.size)  # total output polygons

    # Expand polygon index -> its output rings (all sorted_output_ids entries
    # are already in [exterior, hole0, hole1, ...] order from GPU sort).
    # Flat ring order for ALL polygons (both POLYGON and MULTIPOLYGON families):
    # ring_indices[i] = the i-th ring in output order across all polygons.
    # Use vectorised expansion: repeat each polygon's ring range.
    ring_counts_per_poly = h_rings_per_poly.astype(np.int32)
    total_output_rings = int(ring_counts_per_poly.sum())
    if total_output_rings == 0:
        return _empty_polygon_output(runtime_selection)

    # Build per-polygon -> row mapping: polygon p belongs to the row
    # determined by cumulative polys_per_row.
    # poly_to_row[p] = which output row polygon p belongs to.
    poly_to_row = np.repeat(np.arange(output_row_count, dtype=np.int32), h_polys_per_row)

    # For each polygon, the rings in sorted_output_ids
    # are at positions [poly_starts[p], poly_starts[p]+rings_per_poly[p]).
    # Vectorised expansion: for each ring slot, compute which polygon it
    # belongs to and its offset within that polygon's ring list.
    ring_poly_ids = np.repeat(np.arange(n_total_polys, dtype=np.int32), ring_counts_per_poly)
    poly_ring_offsets_prefix = np.zeros(n_total_polys + 1, dtype=np.int32)
    np.cumsum(ring_counts_per_poly, out=poly_ring_offsets_prefix[1:])
    ring_local_offsets = np.arange(total_output_rings, dtype=np.int32) - poly_ring_offsets_prefix[ring_poly_ids]
    # Global ring indices in the all-rings coordinate arrays
    all_ring_order = h_sorted_output_ids[h_poly_starts[ring_poly_ids] + ring_local_offsets]

    # Coord counts per ring: edge_count + 1
    all_ring_coord_counts = h_all_edge_counts[all_ring_order].astype(np.int32) + 1

    # --- Separate POLYGON and MULTIPOLYGON families ---
    # Polygon family: rows with exactly 1 polygon
    poly_row_mask = is_polygon_row
    mpoly_row_mask = ~is_polygon_row

    # Map rows to polygon indices (within poly_to_row)
    # A polygon belongs to the polygon family if its row is a polygon row.
    poly_mask_per_polygon = poly_row_mask[poly_to_row]
    mpoly_mask_per_polygon = mpoly_row_mask[poly_to_row]

    # Ring-level masks: a ring belongs to the polygon family if its polygon does.
    ring_is_poly = poly_mask_per_polygon[ring_poly_ids]
    ring_is_mpoly = mpoly_mask_per_polygon[ring_poly_ids]

    # Assign family tags and family_row_offsets (vectorised, no Python loop)
    tags[poly_row_mask] = poly_tag
    tags[mpoly_row_mask] = mpoly_tag
    polygon_count = int(poly_row_mask.sum())
    multipolygon_count = int(mpoly_row_mask.sum())
    # Family-local sequential index: polygon rows get 0..polygon_count-1,
    # multipolygon rows get 0..multipolygon_count-1.
    if polygon_count > 0:
        family_row_offsets[poly_row_mask] = np.arange(polygon_count, dtype=np.int32)
    if multipolygon_count > 0:
        family_row_offsets[mpoly_row_mask] = np.arange(multipolygon_count, dtype=np.int32)

    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    if polygon_count > 0:
        # Extract polygon rings
        poly_ring_indices = np.flatnonzero(ring_is_poly)
        poly_ring_order = all_ring_order[poly_ring_indices]
        poly_coord_counts = all_ring_coord_counts[poly_ring_indices]
        # Ring offsets = cumulative coord counts
        h_poly_ring_offsets = np.zeros(len(poly_ring_indices) + 1, dtype=np.int32)
        np.cumsum(poly_coord_counts, out=h_poly_ring_offsets[1:])
        # Geometry offsets: each polygon has rings_per_poly rings.
        # Since polygon rows have exactly 1 polygon each, geometry offsets
        # are just the cumulative ring-per-polygon counts for polygon family.
        rings_per_family_poly = ring_counts_per_poly[poly_mask_per_polygon]
        h_poly_geom_offsets = np.zeros(polygon_count + 1, dtype=np.int32)
        np.cumsum(rings_per_family_poly, out=h_poly_geom_offsets[1:])
        # Vectorised coordinate gather
        d_poly_x, d_poly_y = _gather_coords_vectorised(
            d_all_x, d_all_y, h_all_coord_offsets, poly_ring_order, poly_coord_counts,
        )
        device_families[GeometryFamily.POLYGON] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=d_poly_x,
            y=d_poly_y,
            geometry_offsets=runtime.from_host(h_poly_geom_offsets),
            empty_mask=runtime.from_host(np.zeros(polygon_count, dtype=np.bool_)),
            ring_offsets=runtime.from_host(h_poly_ring_offsets),
            bounds=None,
        )

    if multipolygon_count > 0:
        # Extract multipolygon rings
        mpoly_ring_indices = np.flatnonzero(ring_is_mpoly)
        mpoly_ring_order = all_ring_order[mpoly_ring_indices]
        mpoly_coord_counts = all_ring_coord_counts[mpoly_ring_indices]
        # Ring offsets
        h_mpoly_ring_offsets = np.zeros(len(mpoly_ring_indices) + 1, dtype=np.int32)
        np.cumsum(mpoly_coord_counts, out=h_mpoly_ring_offsets[1:])
        # Part offsets: each polygon (part) within a multipolygon
        # has rings_per_poly rings. Cumulative within each multipolygon row.
        mpoly_polygons = np.flatnonzero(mpoly_mask_per_polygon)
        rings_per_mpoly_part = ring_counts_per_poly[mpoly_polygons]
        h_mpoly_part_offsets = np.zeros(len(mpoly_polygons) + 1, dtype=np.int32)
        np.cumsum(rings_per_mpoly_part, out=h_mpoly_part_offsets[1:])
        # Geometry offsets: parts per multipolygon row
        mpoly_row_indices = poly_to_row[mpoly_polygons]
        # Map row indices to family-local multipolygon index
        mpoly_row_to_family = np.full(output_row_count, -1, dtype=np.int32)
        mpoly_rows = np.flatnonzero(mpoly_row_mask)
        mpoly_row_to_family[mpoly_rows] = np.arange(multipolygon_count, dtype=np.int32)
        mpoly_family_ids = mpoly_row_to_family[mpoly_row_indices]
        parts_per_geom = np.bincount(mpoly_family_ids, minlength=multipolygon_count).astype(np.int32)
        h_mpoly_geom_offsets = np.zeros(multipolygon_count + 1, dtype=np.int32)
        np.cumsum(parts_per_geom, out=h_mpoly_geom_offsets[1:])
        # Vectorised coordinate gather
        d_mpoly_x, d_mpoly_y = _gather_coords_vectorised(
            d_all_x, d_all_y, h_all_coord_offsets, mpoly_ring_order, mpoly_coord_counts,
        )
        device_families[GeometryFamily.MULTIPOLYGON] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            x=d_mpoly_x,
            y=d_mpoly_y,
            geometry_offsets=runtime.from_host(h_mpoly_geom_offsets),
            empty_mask=runtime.from_host(
                np.zeros(multipolygon_count, dtype=np.bool_),
            ),
            part_offsets=runtime.from_host(h_mpoly_part_offsets),
            ring_offsets=runtime.from_host(h_mpoly_ring_offsets),
            bounds=None,
        )

    result = build_device_resident_owned(
        device_families=device_families,
        row_count=output_row_count,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
    )
    result.runtime_history.append(runtime_selection)
    return result


def _gather_coords_vectorised(
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    h_all_coord_offsets: np.ndarray,
    ring_order: np.ndarray,
    coord_counts: np.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Vectorised coordinate gather: build flat index array without Python loops.

    For each ring in ring_order, gathers coord_counts[i] consecutive
    coordinates starting at h_all_coord_offsets[ring_order[i]].
    """
    if ring_order.size == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    total_coords = int(coord_counts.sum())
    # Build per-ring start positions in source coordinate array
    ring_starts = h_all_coord_offsets[ring_order].astype(np.int64)
    # Expand: for each coordinate slot, compute source index.
    # slot_ring[i] = which ring does coordinate slot i belong to?
    ring_offsets = np.zeros(len(ring_order) + 1, dtype=np.int64)
    np.cumsum(coord_counts, out=ring_offsets[1:])
    slot_ring = np.repeat(np.arange(len(ring_order), dtype=np.int32), coord_counts)
    slot_local = np.arange(total_coords, dtype=np.int64) - ring_offsets[slot_ring]
    h_gather = ring_starts[slot_ring] + slot_local
    d_gather = cp.asarray(h_gather)
    return d_all_x[d_gather], d_all_y[d_gather]


def _select_overlay_face_indices_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
) -> cp.ndarray:
    """Device-resident face selection -- no _ensure_host() D->H triggers.

    Reads bounded_mask, left_covered, and right_covered directly from the
    OverlayFaceTable's device_state, applies the overlay operation mask
    entirely on device, and returns the selected face indices as a
    device-resident CuPy int32 array.
    """
    ds = faces.device_state
    d_bounded = cp.asarray(ds.bounded_mask)
    d_left = cp.asarray(ds.left_covered)
    d_right = cp.asarray(ds.right_covered)

    if operation == "intersection":
        d_mask = (d_bounded != 0) & (d_left != 0) & (d_right != 0)
    elif operation == "union":
        d_mask = (d_bounded != 0) & ((d_left != 0) | (d_right != 0))
    elif operation == "difference":
        d_mask = (d_bounded != 0) & (d_left != 0) & (d_right == 0)
    elif operation == "symmetric_difference":
        d_mask = (d_bounded != 0) & (d_left != d_right)
    elif operation == "identity":
        d_mask = (d_bounded != 0) & (d_left != 0)
    else:
        raise ValueError(f"unsupported overlay operation: {operation}")

    return cp.flatnonzero(d_mask).astype(cp.int32)


def _assemble_faces_from_device_indices(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    d_selected_face_indices: cp.ndarray,
) -> OwnedGeometryArray:
    """Try CPU face assembly first, fall back to GPU.

    Accepts device-resident (CuPy) face indices from
    ``_select_overlay_face_indices_gpu`` and handles the D->H conversion
    only for the CPU assembly path.  GPU assembly receives the CuPy array
    directly (zero-copy).

    The CPU-first ordering is intentional: CPU face boundary walking is
    faster for most cases.  GPU assembly handles the "spans multiple
    source rows" edge case (ADR-0016 Stage 8).
    """
    if d_selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)
    try:
        selected_face_indices = cp.asnumpy(d_selected_face_indices)
        return _build_polygon_output_from_faces(half_edge_graph, faces, selected_face_indices)
    except RuntimeError:
        result = _build_polygon_output_from_faces_gpu(
            half_edge_graph, faces, d_selected_face_indices,
        )
        if result is None:
            raise
        return result


def _overlay_intersection_rectangles_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    requested: ExecutionMode,
) -> OwnedGeometryArray | None:
    if cp is None or left.row_count != right.row_count:
        return None
    left_bounds = _axis_aligned_box_bounds(left)
    right_bounds = _axis_aligned_box_bounds(right)
    if left_bounds is None or right_bounds is None:
        return None

    runtime_selection = RuntimeSelection(
        requested=requested,
        selected=ExecutionMode.GPU,
        reason="GPU rectangle intersection fast path selected",
    )
    left_device = cp.asarray(left_bounds)
    right_device = cp.asarray(right_bounds)
    xmin = cp.maximum(left_device[:, 0], right_device[:, 0])
    ymin = cp.maximum(left_device[:, 1], right_device[:, 1])
    xmax = cp.minimum(left_device[:, 2], right_device[:, 2])
    ymax = cp.minimum(left_device[:, 3], right_device[:, 3])
    keep = (xmin < xmax) & (ymin < ymax)
    keep_rows = cp.flatnonzero(keep).astype(cp.int32, copy=False)
    if int(keep_rows.size) == 0:
        return _empty_polygon_output(runtime_selection)

    xmin = xmin[keep_rows]
    ymin = ymin[keep_rows]
    xmax = xmax[keep_rows]
    ymax = ymax[keep_rows]
    row_count = int(keep_rows.size)
    out_x = cp.empty((row_count * 5,), dtype=cp.float64)
    out_y = cp.empty((row_count * 5,), dtype=cp.float64)
    out_x[0::5] = xmin
    out_y[0::5] = ymin
    out_x[1::5] = xmax
    out_y[1::5] = ymin
    out_x[2::5] = xmax
    out_y[2::5] = ymax
    out_x[3::5] = xmin
    out_y[3::5] = ymax
    out_x[4::5] = xmin
    out_y[4::5] = ymin
    return _build_device_backed_fixed_polygon_output(
        out_x,
        out_y,
        row_count=row_count,
        runtime_selection=runtime_selection,
    )


def build_gpu_half_edge_graph(atomic_edges: AtomicEdgeTable) -> HalfEdgeGraph:
    _require_gpu_arrays()
    _ = get_cuda_runtime()
    if atomic_edges.count == 0:
        return _empty_half_edge_graph(atomic_edges)

    device = atomic_edges.device_state
    edge_count = int(atomic_edges.count)
    all_x = cp.concatenate((device.src_x, device.dst_x))
    all_y = cp.concatenate((device.src_y, device.dst_y))
    qx = _quantize_coordinate(all_x)
    qy = _quantize_coordinate(all_y)

    point_order = cp.lexsort(cp.stack((qy, qx)))
    sorted_qx = qx[point_order]
    sorted_qy = qy[point_order]
    del qx, qy  # Phase 25 memory: quantized coords consumed
    point_start_mask = cp.empty((int(all_x.size),), dtype=cp.bool_)
    point_start_mask[0] = True
    if int(all_x.size) > 1:
        point_start_mask[1:] = (sorted_qx[1:] != sorted_qx[:-1]) | (sorted_qy[1:] != sorted_qy[:-1])
    del sorted_qx, sorted_qy  # Phase 25 memory: sorted quantized coords consumed
    point_node_ids_sorted = cp.cumsum(point_start_mask.astype(cp.int32), dtype=cp.int32) - 1
    point_node_ids = cp.empty((int(all_x.size),), dtype=cp.int32)
    point_node_ids[point_order] = point_node_ids_sorted
    del point_node_ids_sorted  # Phase 25 memory

    src_node_ids = point_node_ids[:edge_count]
    dst_node_ids = point_node_ids[edge_count:]
    node_x = all_x[point_order][point_start_mask]
    node_y = all_y[point_order][point_start_mask]
    del all_x, all_y, point_order, point_start_mask  # Phase 25 memory

    angle = cp.arctan2(device.dst_y - device.src_y, device.dst_x - device.src_x)
    angle_key = _quantize_coordinate(angle + np.pi)
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    sorted_edge_ids = cp.lexsort(cp.stack((edge_ids, angle_key, src_node_ids)))
    del angle_key  # Phase 25 memory
    sorted_src_nodes = src_node_ids[sorted_edge_ids]

    span_start_mask = cp.empty((edge_count,), dtype=cp.bool_)
    span_start_mask[0] = True
    if edge_count > 1:
        span_start_mask[1:] = sorted_src_nodes[1:] != sorted_src_nodes[:-1]
    del sorted_src_nodes  # Phase 25 memory
    span_group_ids = cp.cumsum(span_start_mask.astype(cp.int32), dtype=cp.int32) - 1
    span_starts = cp.flatnonzero(span_start_mask).astype(cp.int32, copy=False)
    span_ends = cp.concatenate((span_starts[1:], cp.asarray([edge_count], dtype=cp.int32)))
    del span_start_mask  # Phase 25 memory
    edge_positions = cp.empty((edge_count,), dtype=cp.int32)
    edge_positions[sorted_edge_ids] = cp.arange(edge_count, dtype=cp.int32)

    twin_edge_ids = edge_ids ^ 1
    del edge_ids  # Phase 25 memory
    twin_positions = edge_positions[twin_edge_ids]
    del twin_edge_ids  # Phase 25 memory
    twin_group_ids = span_group_ids[twin_positions]
    del span_group_ids  # Phase 25 memory
    twin_group_starts = span_starts[twin_group_ids]
    twin_group_ends = span_ends[twin_group_ids]
    del twin_group_ids, span_starts, span_ends  # Phase 25 memory
    previous_positions = twin_positions - 1
    del twin_positions  # Phase 25 memory
    previous_positions = cp.where(
        previous_positions < twin_group_starts,
        twin_group_ends - 1,
        previous_positions,
    )
    del twin_group_starts, twin_group_ends  # Phase 25 memory
    next_edge_ids = sorted_edge_ids[previous_positions]
    del previous_positions  # Phase 25 memory

    # Carry per-edge metadata from AtomicEdgeTable device state directly
    # -- no D->H transfer.  GPU consumers read device_state.row_indices etc.
    ae_ds = atomic_edges.device_state
    return HalfEdgeGraph(
        left_segment_count=atomic_edges.left_segment_count,
        right_segment_count=atomic_edges.right_segment_count,
        runtime_selection=atomic_edges.runtime_selection,
        _edge_count=edge_count,
        device_state=HalfEdgeGraphDeviceState(
            node_x=node_x,
            node_y=node_y,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            angle=angle,
            sorted_edge_ids=sorted_edge_ids,
            edge_positions=edge_positions,
            next_edge_ids=next_edge_ids,
            src_x=device.src_x,
            src_y=device.src_y,
            source_segment_ids=ae_ds.source_segment_ids,
            source_side=ae_ds.source_side,
            row_indices=ae_ds.row_indices,
            part_indices=ae_ds.part_indices,
            ring_indices=ae_ds.ring_indices,
            direction=ae_ds.direction,
        ),
    )


def _gpu_label_face_coverage(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    label_x: cp.ndarray,
    label_y: cp.ndarray,
    face_count: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU face labeling: test face sample points against input geometries.

    Returns (left_covered, right_covered) as CuPy int8 arrays.
    """
    runtime = get_cuda_runtime()
    kernels = _overlay_face_label_kernels()
    ptr = runtime.pointer

    left_covered = cp.zeros(face_count, dtype=cp.int8)
    right_covered = cp.zeros(face_count, dtype=cp.int8)

    if face_count == 0:
        return left_covered, right_covered

    grid, block = runtime.launch_config(
        kernels["label_face_coverage_polygon"], face_count,
    )

    for side_input, out_covered in [(left, left_covered), (right, right_covered)]:
        device_state = side_input._ensure_device_state()

        has_poly = GeometryFamily.POLYGON in device_state.families
        has_mpoly = GeometryFamily.MULTIPOLYGON in device_state.families

        poly_count = 0
        mp_count = 0
        poly_buf = None
        mp_buf = None

        if has_poly:
            poly_buf = device_state.families[GeometryFamily.POLYGON]
            poly_count = side_input.families[GeometryFamily.POLYGON].row_count
        if has_mpoly:
            mp_buf = device_state.families[GeometryFamily.MULTIPOLYGON]
            mp_count = side_input.families[GeometryFamily.MULTIPOLYGON].row_count

        launch_poly = has_poly and poly_count > 0
        launch_mpoly = (has_mpoly and mp_count > 0
                        and mp_buf is not None and mp_buf.part_offsets is not None)

        if launch_poly and launch_mpoly:
            # Both families present — launch on separate CUDA streams so
            # the kernels can overlap.  They write to non-overlapping (or
            # idempotent) positions in out_covered.
            s_poly = runtime.create_stream()
            s_mpoly = runtime.create_stream()
            try:
                poly_params = (
                    (ptr(label_x), ptr(label_y),
                     ptr(poly_buf.x), ptr(poly_buf.y),
                     ptr(poly_buf.geometry_offsets), ptr(poly_buf.ring_offsets),
                     poly_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                runtime.launch(kernels["label_face_coverage_polygon"],
                               grid=grid, block=block, params=poly_params,
                               stream=s_poly)

                mp_params = (
                    (ptr(label_x), ptr(label_y),
                     ptr(mp_buf.x), ptr(mp_buf.y),
                     ptr(mp_buf.geometry_offsets), ptr(mp_buf.part_offsets),
                     ptr(mp_buf.ring_offsets),
                     mp_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                runtime.launch(kernels["label_face_coverage_multipolygon"],
                               grid=grid, block=block, params=mp_params,
                               stream=s_mpoly)

                s_poly.synchronize()
                s_mpoly.synchronize()
            finally:
                runtime.destroy_stream(s_poly)
                runtime.destroy_stream(s_mpoly)
        else:
            # Single family — launch on the default (null) stream.
            if launch_poly:
                params = (
                    (ptr(label_x), ptr(label_y),
                     ptr(poly_buf.x), ptr(poly_buf.y),
                     ptr(poly_buf.geometry_offsets), ptr(poly_buf.ring_offsets),
                     poly_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                runtime.launch(kernels["label_face_coverage_polygon"],
                               grid=grid, block=block, params=params)
            if launch_mpoly:
                params = (
                    (ptr(label_x), ptr(label_y),
                     ptr(mp_buf.x), ptr(mp_buf.y),
                     ptr(mp_buf.geometry_offsets), ptr(mp_buf.part_offsets),
                     ptr(mp_buf.ring_offsets),
                     mp_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                runtime.launch(kernels["label_face_coverage_multipolygon"],
                               grid=grid, block=block, params=params)
    return left_covered, right_covered


def _gpu_face_walk(half_edge_graph: HalfEdgeGraph) -> tuple[
    cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray,
    cp.ndarray, cp.ndarray, cp.ndarray, int,
]:
    """GPU face walk via pointer jumping + shoelace aggregation.

    Returns (face_offsets, face_edge_ids, bounded_mask, signed_area,
             centroid_x, centroid_y, label_x, label_y, face_count)
    as CuPy device arrays (except face_count which is int).
    """
    runtime = get_cuda_runtime()
    device = half_edge_graph.device_state
    edge_count = half_edge_graph.edge_count
    kernels = _overlay_face_walk_kernels()
    ptr = runtime.pointer

    # --- Step 1: Pointer jumping to find cycles (Tier 2 CuPy) ---
    face_id = cp.arange(edge_count, dtype=cp.int32)
    jump = cp.asarray(device.next_edge_ids).copy()
    max_iterations = max(1, int(np.ceil(np.log2(edge_count))))

    for _ in range(max_iterations):
        face_id = cp.minimum(face_id, face_id[jump])
        jump = jump[jump]

    # --- Step 2: Per-edge shoelace contributions ---
    d_cross = cp.empty(edge_count, dtype=cp.float64)
    d_cx_contrib = cp.empty(edge_count, dtype=cp.float64)
    d_cy_contrib = cp.empty(edge_count, dtype=cp.float64)
    shoelace_params = (
        (ptr(device.src_x), ptr(device.src_y), ptr(device.next_edge_ids),
         ptr(d_cross), ptr(d_cx_contrib), ptr(d_cy_contrib), edge_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernels["compute_shoelace_contributions"], edge_count)
    runtime.launch(kernels["compute_shoelace_contributions"], grid=grid, block=block, params=shoelace_params)

    # Phase 25 memory: jump is dead after pointer jumping.
    del jump

    # --- Step 3: Group edges by face_id via sort, then aggregate ---
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    sort_result = sort_pairs(face_id, edge_ids, synchronize=False)
    sorted_face_ids = sort_result.keys
    sorted_edge_ids = sort_result.values

    # Find unique face_ids and segment boundaries
    face_start_mask = cp.empty(edge_count, dtype=cp.bool_)
    face_start_mask[0] = True
    if edge_count > 1:
        face_start_mask[1:] = sorted_face_ids[1:] != sorted_face_ids[:-1]
    del sorted_face_ids  # Phase 25 memory
    starts = cp.flatnonzero(face_start_mask).astype(cp.int32, copy=False)
    ends = cp.concatenate((starts[1:], cp.asarray([edge_count], dtype=cp.int32)))
    del face_start_mask  # Phase 25 memory

    # Per-face edge counts
    face_lengths = ends - starts

    # Filter faces with < 3 edges
    valid_face_indices = cp.flatnonzero(face_lengths >= 3).astype(cp.int32, copy=False)
    face_count = int(valid_face_indices.size)

    if face_count == 0:
        empty_i32 = cp.asarray([0], dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, cp.empty(0, dtype=cp.int32), empty_i8, empty_f64, empty_f64, empty_f64, empty_f64, empty_f64, 0

    valid_starts = starts[valid_face_indices]
    valid_ends = ends[valid_face_indices]
    valid_lengths = face_lengths[valid_face_indices]
    del starts, ends, face_lengths, valid_face_indices  # Phase 25 memory

    # Segmented reduce for cross, cx_contrib, cy_contrib
    # Reorder contributions to match sorted edge order
    sorted_cross = d_cross[sorted_edge_ids]
    sorted_cx = d_cx_contrib[sorted_edge_ids]
    sorted_cy = d_cy_contrib[sorted_edge_ids]
    # Phase 25 memory: raw contribution arrays consumed after gather.
    del d_cross, d_cx_contrib, d_cy_contrib

    cross_sums = segmented_reduce_sum(sorted_cross, valid_starts, valid_ends, num_segments=face_count).values
    cx_sums = segmented_reduce_sum(sorted_cx, valid_starts, valid_ends, num_segments=face_count).values
    cy_sums = segmented_reduce_sum(sorted_cy, valid_starts, valid_ends, num_segments=face_count).values
    del sorted_cross, sorted_cx, sorted_cy  # Phase 25 memory

    # signed_area = twice_area * 0.5
    signed_area = cross_sums * 0.5
    # centroid: factor = 1 / (3 * twice_area), cx = sum_cx * factor
    twice_area = cross_sums
    safe_twice_area = cp.where(twice_area == 0.0, 1.0, twice_area)
    factor = 1.0 / (3.0 * safe_twice_area)
    centroid_x = cx_sums * factor
    centroid_y = cy_sums * factor
    # For zero-area faces, use mean of coordinates.  Compute the fallback
    # unconditionally on device (cheap segmented reduce + cp.where) to avoid
    # the implicit D2H sync that cp.any(zero_area_mask) would trigger.
    zero_area_mask = twice_area == 0.0
    sorted_src_x = cp.asarray(device.src_x)[sorted_edge_ids]
    sorted_src_y = cp.asarray(device.src_y)[sorted_edge_ids]
    mean_x = segmented_reduce_sum(sorted_src_x, valid_starts, valid_ends, num_segments=face_count).values
    mean_y = segmented_reduce_sum(sorted_src_y, valid_starts, valid_ends, num_segments=face_count).values
    safe_lengths = cp.where(valid_lengths == 0, 1, valid_lengths).astype(cp.float64)
    centroid_x = cp.where(zero_area_mask, mean_x / safe_lengths, centroid_x)
    centroid_y = cp.where(zero_area_mask, mean_y / safe_lengths, centroid_y)
    del zero_area_mask, sorted_src_x, sorted_src_y, mean_x, mean_y, safe_lengths  # Phase 25 memory
    del cross_sums, cx_sums, cy_sums, twice_area, safe_twice_area, factor  # Phase 25 memory

    # Build compact face_offsets and face_edge_ids in cycle traversal order.
    # GPU list ranking: each edge gets its position within its face cycle,
    # then sort by (face_id, rank) to produce contiguous cycle-ordered layout.
    # This replaces the prior serial host walk of next_edge_ids.
    d_rank = cp.empty(edge_count, dtype=cp.int32)
    max_walk = edge_count  # upper bound on cycle length
    rank_params = (
        (ptr(face_id), ptr(device.next_edge_ids),
         ptr(d_rank), edge_count, max_walk),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    rank_grid, rank_block = runtime.launch_config(kernels["list_rank_within_cycle"], edge_count)
    runtime.launch(
        kernels["list_rank_within_cycle"],
        grid=rank_grid, block=rank_block, params=rank_params,
    )

    # Pack (face_id, rank) into a single sort key so sort_pairs gives us
    # edges grouped by face and ordered within each cycle.
    packed_key = face_id.astype(cp.int64) * int(edge_count) + d_rank.astype(cp.int64)
    cycle_sort = sort_pairs(packed_key, edge_ids, synchronize=False)
    cycle_sorted_edge_ids = cycle_sort.values

    # Build face_offsets from valid_lengths on device (CCCL exclusive_sum).
    # exclusive_sum produces face_count elements; append the total to form
    # a proper CSR offset array with face_count+1 entries so that
    # face_offsets[face_count] gives the total edge count.
    _prefix = exclusive_sum(valid_lengths.astype(cp.int32, copy=False))
    _total = valid_lengths.sum().reshape(1).astype(cp.int32)
    face_offsets = cp.concatenate((_prefix, _total))

    # Extract cycle-ordered edges for valid faces only.  The cycle-sorted
    # array has the same segment boundaries (valid_starts, valid_ends) as
    # the face_id-sorted array since the packed key preserves face_id order.
    # Gather all valid-face edges into a contiguous output using vectorised
    # CuPy index arithmetic — no per-face host loop.
    total_edges = int(_total.item())

    # Build a flat gather index: for each slot in the output, compute the
    # source position in cycle_sorted_edge_ids.
    # slot_face = which valid face does this slot belong to?
    slot_ids = cp.arange(total_edges, dtype=cp.int32)
    slot_face = cp.searchsorted(face_offsets[1:], slot_ids, side='right').astype(cp.int32)
    # local offset within that face's segment
    slot_local = slot_ids - face_offsets[slot_face]
    # source position in the cycle-sorted full array
    src_pos = valid_starts[slot_face] + slot_local
    face_edge_ids = cycle_sorted_edge_ids[src_pos]
    # Phase 25 memory: face walk intermediates consumed.
    del slot_ids, slot_face, slot_local, src_pos, cycle_sorted_edge_ids
    del sorted_edge_ids, valid_starts, valid_ends, valid_lengths
    del packed_key, d_rank

    # --- Step 4: Face sample points via kernel ---
    label_x = cp.empty(face_count, dtype=cp.float64)
    label_y = cp.empty(face_count, dtype=cp.float64)
    bounded_mask = cp.empty(face_count, dtype=cp.int8)
    sample_params = (
        (ptr(device.src_x), ptr(device.src_y), ptr(device.next_edge_ids),
         ptr(face_offsets), ptr(face_edge_ids), ptr(signed_area),
         ptr(label_x), ptr(label_y), ptr(bounded_mask), face_count,
         edge_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
         KERNEL_PARAM_I32),
    )
    sample_grid, sample_block = runtime.launch_config(kernels["compute_face_sample_points"], face_count)
    runtime.launch(kernels["compute_face_sample_points"], grid=sample_grid, block=sample_block, params=sample_params)

    return face_offsets, face_edge_ids, bounded_mask, signed_area, centroid_x, centroid_y, label_x, label_y, face_count


def build_gpu_overlay_faces(
    left,
    right,
    *,
    half_edge_graph: HalfEdgeGraph | None = None,
    atomic_edges: AtomicEdgeTable | None = None,
    split_events: SplitEventTable | None = None,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OverlayFaceTable:
    runtime = get_cuda_runtime()
    if half_edge_graph is None:
        if atomic_edges is None:
            if split_events is None:
                split_events = build_gpu_split_events(
                    left,
                    right,
                    intersection_result=intersection_result,
                    dispatch_mode=dispatch_mode,
                )
            atomic_edges = build_gpu_atomic_edges(split_events)
        half_edge_graph = build_gpu_half_edge_graph(atomic_edges)

    edge_count = half_edge_graph.edge_count
    if edge_count == 0:
        empty_device_i32 = runtime.allocate((1,), np.int32)
        empty_device_i32_flat = runtime.allocate((0,), np.int32)
        empty_device_i8 = runtime.allocate((0,), np.int8)
        empty_device_f64 = runtime.allocate((0,), np.float64)
        # Device-primary empty face table -- host arrays are None and
        # will be lazily materialized via _ensure_host if accessed.
        return OverlayFaceTable(
            runtime_selection=half_edge_graph.runtime_selection,
            _face_count=0,
            device_state=OverlayFaceDeviceState(
                face_offsets=empty_device_i32,
                face_edge_ids=empty_device_i32_flat,
                bounded_mask=empty_device_i8,
                signed_area=empty_device_f64,
                centroid_x=empty_device_f64,
                centroid_y=empty_device_f64,
                left_covered=empty_device_i8,
                right_covered=empty_device_i8,
            ),
        )

    # GPU face walk path: pointer jumping + shoelace aggregation
    if cp is not None and half_edge_graph.device_state is not None:
        (d_face_offsets, d_face_edge_ids, d_bounded_mask, d_signed_area,
         d_centroid_x, d_centroid_y, d_label_x, d_label_y, face_count) = _gpu_face_walk(half_edge_graph)

        if face_count > 0:
            # GPU face labeling: test sample points against input geometries
            d_left_covered, d_right_covered = _gpu_label_face_coverage(
                left, right, d_label_x, d_label_y, face_count)
            # Mask out unbounded faces (keep on device -- ADR-0005)
            d_left_covered = cp.where(d_bounded_mask != 0, d_left_covered, 0).astype(cp.int8)
            d_right_covered = cp.where(d_bounded_mask != 0, d_right_covered, 0).astype(cp.int8)
        else:
            # _gpu_face_walk already returned device arrays for the zero case
            d_left_covered = cp.empty(0, dtype=cp.int8)
            d_right_covered = cp.empty(0, dtype=cp.int8)

        # Device-primary: host arrays are None, lazily materialized on demand
        return OverlayFaceTable(
            runtime_selection=half_edge_graph.runtime_selection,
            _face_count=face_count,
            device_state=OverlayFaceDeviceState(
                face_offsets=d_face_offsets,
                face_edge_ids=d_face_edge_ids,
                bounded_mask=d_bounded_mask,
                signed_area=d_signed_area,
                centroid_x=d_centroid_x,
                centroid_y=d_centroid_y,
                left_covered=d_left_covered,
                right_covered=d_right_covered,
            ),
        )

    # CPU fallback path
    visited = np.zeros(edge_count, dtype=bool)
    face_edge_groups: list[np.ndarray] = []
    signed_area_values: list[float] = []
    centroid_x_values: list[float] = []
    centroid_y_values: list[float] = []
    label_x_values: list[float] = []
    label_y_values: list[float] = []
    bounded_mask_values: list[int] = []

    for edge_id in range(edge_count):
        if visited[edge_id]:
            continue
        cycle_edges: list[int] = []
        current = edge_id
        while not visited[current]:
            visited[current] = True
            cycle_edges.append(current)
            current = int(half_edge_graph.next_edge_ids[current])
        if current != edge_id or len(cycle_edges) < 3:
            continue
        points = np.column_stack(
            (
                half_edge_graph.src_x[np.asarray(cycle_edges, dtype=np.int32)],
                half_edge_graph.src_y[np.asarray(cycle_edges, dtype=np.int32)],
            )
        )
        signed_area, centroid_x, centroid_y = _signed_area_and_centroid(points)
        face_edge_groups.append(np.asarray(cycle_edges, dtype=np.int32))
        signed_area_values.append(signed_area)
        centroid_x_values.append(centroid_x)
        centroid_y_values.append(centroid_y)
        sample_x, sample_y = _face_sample_point(points)
        label_x_values.append(sample_x)
        label_y_values.append(sample_y)
        bounded_mask_values.append(1 if signed_area > 1e-12 else 0)

    # Track whether coverage was computed on device (avoids D->H->D roundtrip).
    _gpu_coverage = False
    if not face_edge_groups:
        face_offsets = np.asarray([0], dtype=np.int32)
        face_edge_ids = np.asarray([], dtype=np.int32)
        bounded_mask = np.asarray([], dtype=np.int8)
        signed_area = np.asarray([], dtype=np.float64)
        centroid_x = np.asarray([], dtype=np.float64)
        centroid_y = np.asarray([], dtype=np.float64)
        left_covered = np.asarray([], dtype=np.int8)
        right_covered = np.asarray([], dtype=np.int8)
    else:
        face_lengths = np.asarray([group.size for group in face_edge_groups], dtype=np.int32)
        face_offsets = np.empty((face_lengths.size + 1,), dtype=np.int32)
        face_offsets[0] = 0
        face_offsets[1:] = np.cumsum(face_lengths, dtype=np.int32)
        face_edge_ids = np.concatenate(face_edge_groups).astype(np.int32, copy=False)
        bounded_mask = np.asarray(bounded_mask_values, dtype=np.int8)
        signed_area = np.asarray(signed_area_values, dtype=np.float64)
        centroid_x = np.asarray(centroid_x_values, dtype=np.float64)
        centroid_y = np.asarray(centroid_y_values, dtype=np.float64)
        label_x = np.asarray(label_x_values, dtype=np.float64)
        label_y = np.asarray(label_y_values, dtype=np.float64)
        # Prefer GPU labeling even when the face walk was done on CPU.
        # This avoids the Shapely roundtrip (to_shapely -> union_all -> covers).
        cpu_face_count_for_label = label_x.size
        if (
            cp is not None
            and cpu_face_count_for_label > 0
            and (_has_polygonal_families(left) or _has_polygonal_families(right))
        ):
            d_label_x = runtime.from_host(label_x)
            d_label_y = runtime.from_host(label_y)
            d_lc, d_rc = _gpu_label_face_coverage(
                left, right, d_label_x, d_label_y, cpu_face_count_for_label,
            )
            runtime.synchronize()
            # Apply bounded_mask on device to avoid D->H->D roundtrip.
            d_bounded_mask = runtime.from_host(bounded_mask)
            d_mask = d_bounded_mask != 0
            d_lc = cp.where(d_mask, d_lc, cp.int8(0)).astype(cp.int8, copy=False)
            d_rc = cp.where(d_mask, d_rc, cp.int8(0)).astype(cp.int8, copy=False)
            # Host copies deferred -- lazily materialised by property accessor.
            left_covered = None
            right_covered = None
            _gpu_coverage = True
        else:
            left_covered, right_covered = _label_face_coverage(left, right, label_x, label_y)
            left_covered = np.where(bounded_mask != 0, left_covered, 0).astype(np.int8, copy=False)
            right_covered = np.where(bounded_mask != 0, right_covered, 0).astype(np.int8, copy=False)

    cpu_face_count = max(0, int(face_offsets.size) - 1)
    # Build device state; reuse arrays already on device when GPU coverage ran.
    d_face_offsets = runtime.from_host(face_offsets)
    d_face_edge_ids = runtime.from_host(face_edge_ids)
    if not _gpu_coverage:
        d_bounded_mask = runtime.from_host(bounded_mask)
    d_signed_area = runtime.from_host(signed_area)
    d_centroid_x = runtime.from_host(centroid_x)
    d_centroid_y = runtime.from_host(centroid_y)
    d_left_covered = d_lc if _gpu_coverage else runtime.from_host(left_covered)
    d_right_covered = d_rc if _gpu_coverage else runtime.from_host(right_covered)
    return OverlayFaceTable(
        runtime_selection=half_edge_graph.runtime_selection,
        _face_count=cpu_face_count,
        _face_offsets=face_offsets,
        _face_edge_ids=face_edge_ids,
        _bounded_mask=bounded_mask,
        _signed_area=signed_area,
        _centroid_x=centroid_x,
        _centroid_y=centroid_y,
        _left_covered=left_covered,
        _right_covered=right_covered,
        device_state=OverlayFaceDeviceState(
            face_offsets=d_face_offsets,
            face_edge_ids=d_face_edge_ids,
            bounded_mask=d_bounded_mask,
            signed_area=d_signed_area,
            centroid_x=d_centroid_x,
            centroid_y=d_centroid_y,
            left_covered=d_left_covered,
            right_covered=d_right_covered,
        ),
    )


def build_gpu_split_events(
    left,
    right,
    *,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> SplitEventTable:
    _require_gpu_arrays()
    runtime = get_cuda_runtime()

    # GPU-native segment extraction -- no CPU loop, no host round-trip.
    # lyy.15: reuse pre-extracted right-side segments when provided
    # (N-vs-1 overlay caches the corridor segments once).
    left_segments = _extract_segments_gpu(left)
    _owns_right_segments = _cached_right_segments is None
    right_segments = (
        _cached_right_segments
        if _cached_right_segments is not None
        else _extract_segments_gpu(right)
    )

    result = intersection_result or classify_segment_intersections(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_device_segments=_cached_right_segments,
    )
    if result.runtime_selection.selected is not ExecutionMode.GPU:
        raise RuntimeError("build_gpu_split_events requires a GPU segment-intersection result")
    owns_intersection_state = False
    if result.device_state is None:
        device_state = SegmentIntersectionDeviceState(
            left_rows=runtime.from_host(result.left_rows),
            left_segments=runtime.from_host(result.left_segments),
            left_lookup=runtime.from_host(result.left_lookup),
            right_rows=runtime.from_host(result.right_rows),
            right_segments=runtime.from_host(result.right_segments),
            right_lookup=runtime.from_host(result.right_lookup),
            kinds=runtime.from_host(result.kinds.astype(np.int8, copy=False)),
            point_x=runtime.from_host(result.point_x.astype(np.float64, copy=False)),
            point_y=runtime.from_host(result.point_y.astype(np.float64, copy=False)),
            overlap_x0=runtime.from_host(result.overlap_x0.astype(np.float64, copy=False)),
            overlap_y0=runtime.from_host(result.overlap_y0.astype(np.float64, copy=False)),
            overlap_x1=runtime.from_host(result.overlap_x1.astype(np.float64, copy=False)),
            overlap_y1=runtime.from_host(result.overlap_y1.astype(np.float64, copy=False)),
            ambiguous_rows=runtime.allocate((0,), np.int32),
        )
        owns_intersection_state = True
    else:
        device_state = result.device_state

    left_count = int(left_segments.count)
    right_count = int(right_segments.count)
    segment_total = left_count + right_count
    base_event_count = segment_total * 2
    kernels = _overlay_split_kernels()

    # Segment coordinate arrays are already device-resident from
    # _extract_segments_gpu -- use them directly, no from_host.
    left_x0 = left_segments.x0
    left_y0 = left_segments.y0
    left_x1 = left_segments.x1
    left_y1 = left_segments.y1
    right_x0 = right_segments.x0
    right_y0 = right_segments.y0
    right_x1 = right_segments.x1
    right_y1 = right_segments.y1

    endpoint_source_ids = runtime.allocate((base_event_count,), np.int32)
    endpoint_t = runtime.allocate((base_event_count,), np.float64)
    endpoint_x = runtime.allocate((base_event_count,), np.float64)
    endpoint_y = runtime.allocate((base_event_count,), np.float64)
    endpoint_keys = runtime.allocate((base_event_count,), np.uint64)

    try:
        ptr = runtime.pointer
        endpoint_params = (
            (
                ptr(left_x0),
                ptr(left_y0),
                ptr(left_x1),
                ptr(left_y1),
                ptr(right_x0),
                ptr(right_y0),
                ptr(right_x1),
                ptr(right_y1),
                left_count,
                right_count,
                ptr(endpoint_source_ids),
                ptr(endpoint_t),
                ptr(endpoint_x),
                ptr(endpoint_y),
                ptr(endpoint_keys),
                base_event_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        endpoint_grid, endpoint_block = runtime.launch_config(kernels["emit_endpoint_split_events"], base_event_count)
        runtime.launch(
            kernels["emit_endpoint_split_events"],
            grid=endpoint_grid,
            block=endpoint_block,
            params=endpoint_params,
        )

        pair_counts = runtime.allocate((result.count,), np.int32)
        count_params = (
            (ptr(device_state.kinds), ptr(pair_counts), result.count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        count_grid, count_block = runtime.launch_config(kernels["count_pair_split_events"], result.count)
        runtime.launch(
            kernels["count_pair_split_events"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        pair_offsets = exclusive_sum(pair_counts)
        pair_event_count = (
            int(cp.asnumpy(pair_offsets[-1] + pair_counts[-1])) if int(result.count) else 0
        )
        extra_source_ids = runtime.allocate((pair_event_count,), np.int32)
        extra_t = runtime.allocate((pair_event_count,), np.float64)
        extra_x = runtime.allocate((pair_event_count,), np.float64)
        extra_y = runtime.allocate((pair_event_count,), np.float64)
        extra_keys = runtime.allocate((pair_event_count,), np.uint64)

        try:
            scatter_params = (
                (
                    ptr(device_state.left_lookup),
                    ptr(device_state.right_lookup),
                    ptr(device_state.kinds),
                    ptr(device_state.point_x),
                    ptr(device_state.point_y),
                    ptr(device_state.overlap_x0),
                    ptr(device_state.overlap_y0),
                    ptr(device_state.overlap_x1),
                    ptr(device_state.overlap_y1),
                    ptr(left_x0),
                    ptr(left_y0),
                    ptr(left_x1),
                    ptr(left_y1),
                    ptr(right_x0),
                    ptr(right_y0),
                    ptr(right_x1),
                    ptr(right_y1),
                    ptr(pair_offsets),
                    left_count,
                    ptr(extra_source_ids),
                    ptr(extra_t),
                    ptr(extra_x),
                    ptr(extra_y),
                    ptr(extra_keys),
                    result.count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
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
                ),
            )
            scatter_grid, scatter_block = runtime.launch_config(kernels["scatter_pair_split_events"], result.count)
            runtime.launch(
                kernels["scatter_pair_split_events"],
                grid=scatter_grid,
                block=scatter_block,
                params=scatter_params,
            )

            all_source_ids = cp.concatenate((endpoint_source_ids, extra_source_ids))
            all_t = cp.concatenate((endpoint_t, extra_t))
            all_x = cp.concatenate((endpoint_x, extra_x))
            all_y = cp.concatenate((endpoint_y, extra_y))
            all_keys = cp.concatenate((endpoint_keys, extra_keys))

            event_indices = cp.arange(int(all_keys.size), dtype=cp.int32)
            sorted_pairs = sort_pairs(all_keys, event_indices, synchronize=False)
            unique_pairs = unique_sorted_pairs(sorted_pairs.keys, sorted_pairs.values)

            unique_indices = unique_pairs.values
            unique_source_ids = all_source_ids[unique_indices]
            unique_t = all_t[unique_indices]
            unique_x = all_x[unique_indices]
            unique_y = all_y[unique_indices]
            unique_keys = unique_pairs.keys

            # Derive source_side / row / part / ring indices on GPU.
            d_source_side, d_row_indices, d_part_indices, d_ring_indices = (
                _segment_metadata_gpu(
                    unique_source_ids,
                    left_count=left_count,
                    left_segments=left_segments,
                    right_segments=right_segments,
                )
            )
            event_count = int(unique_source_ids.size)

            return SplitEventTable(
                left_segment_count=left_count,
                right_segment_count=right_count,
                runtime_selection=result.runtime_selection,
                device_state=SplitEventDeviceState(
                    source_segment_ids=unique_source_ids,
                    packed_keys=unique_keys,
                    t=unique_t,
                    x=unique_x,
                    y=unique_y,
                    source_side=d_source_side,
                    row_indices=d_row_indices,
                    part_indices=d_part_indices,
                    ring_indices=d_ring_indices,
                ),
                _count=event_count,
            )
        finally:
            runtime.free(pair_counts)
            runtime.free(pair_offsets)
            runtime.free(extra_source_ids)
            runtime.free(extra_t)
            runtime.free(extra_x)
            runtime.free(extra_y)
            runtime.free(extra_keys)
    finally:
        # Free DeviceSegmentTable arrays (x0/y0/x1/y1 are aliases of
        # left_x0 etc., plus row/segment/part/ring metadata).
        # lyy.15: skip freeing right_segments when they are cached
        # (caller owns the lifetime of the cached segments).
        _segs_to_free = [left_segments]
        if _owns_right_segments:
            _segs_to_free.append(right_segments)
        for _dst in _segs_to_free:
            runtime.free(_dst.x0)
            runtime.free(_dst.y0)
            runtime.free(_dst.x1)
            runtime.free(_dst.y1)
            runtime.free(_dst.row_indices)
            runtime.free(_dst.segment_indices)
            if _dst.part_indices is not None:
                runtime.free(_dst.part_indices)
            if _dst.ring_indices is not None:
                runtime.free(_dst.ring_indices)
        runtime.free(endpoint_source_ids)
        runtime.free(endpoint_t)
        runtime.free(endpoint_x)
        runtime.free(endpoint_y)
        runtime.free(endpoint_keys)
        if owns_intersection_state:
            runtime.free(device_state.left_rows)
            runtime.free(device_state.left_segments)
            runtime.free(device_state.left_lookup)
            runtime.free(device_state.right_rows)
            runtime.free(device_state.right_segments)
            runtime.free(device_state.right_lookup)
            runtime.free(device_state.kinds)
            runtime.free(device_state.point_x)
            runtime.free(device_state.point_y)
            runtime.free(device_state.overlap_x0)
            runtime.free(device_state.overlap_y0)
            runtime.free(device_state.overlap_x1)
            runtime.free(device_state.overlap_y1)
            runtime.free(device_state.ambiguous_rows)


def build_gpu_atomic_edges(split_events: SplitEventTable) -> AtomicEdgeTable:
    _require_gpu_arrays()
    runtime = get_cuda_runtime()
    kernels = _overlay_split_kernels()
    device = split_events.device_state
    if split_events.count < 2:
        empty_device_i32 = runtime.allocate((0,), np.int32)
        empty_device_i8 = runtime.allocate((0,), np.int8)
        empty_device_f64 = runtime.allocate((0,), np.float64)
        return AtomicEdgeTable(
            left_segment_count=split_events.left_segment_count,
            right_segment_count=split_events.right_segment_count,
            runtime_selection=split_events.runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=empty_device_i32,
                direction=empty_device_i8,
                src_x=empty_device_f64,
                src_y=empty_device_f64,
                dst_x=empty_device_f64,
                dst_y=empty_device_f64,
                row_indices=empty_device_i32,
                part_indices=empty_device_i32,
                ring_indices=empty_device_i32,
                source_side=empty_device_i8,
            ),
            _count=0,
        )

    adjacency_mask = (
        device.source_segment_ids[:-1] == device.source_segment_ids[1:]
    ).astype(cp.uint8, copy=False)
    adjacency_counts = adjacency_mask.astype(cp.int32, copy=False)
    adjacency_offsets = exclusive_sum(adjacency_counts)
    pair_count = int(cp.asnumpy(adjacency_offsets[-1] + adjacency_counts[-1])) if int(adjacency_counts.size) else 0  # zcopy:ok(allocation-fence: need pair_count to size 6 atomic-edge output buffers)

    out_source_ids = runtime.allocate((pair_count * 2,), np.int32)
    out_direction = runtime.allocate((pair_count * 2,), np.int8)
    out_src_x = runtime.allocate((pair_count * 2,), np.float64)
    out_src_y = runtime.allocate((pair_count * 2,), np.float64)
    out_dst_x = runtime.allocate((pair_count * 2,), np.float64)
    out_dst_y = runtime.allocate((pair_count * 2,), np.float64)
    try:
        ptr = runtime.pointer
        row_count = max(0, split_events.count - 1)
        params = (
            (
                ptr(device.source_segment_ids),
                ptr(device.x),
                ptr(device.y),
                ptr(adjacency_mask),
                ptr(adjacency_offsets),
                ptr(out_source_ids),
                ptr(out_direction),
                ptr(out_src_x),
                ptr(out_src_y),
                ptr(out_dst_x),
                ptr(out_dst_y),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernels["emit_atomic_edges"], row_count)
        runtime.launch(
            kernels["emit_atomic_edges"],
            grid=grid,
            block=block,
            params=params,
        )
        runtime.synchronize()

        # Derive source_side and row / part / ring indices on GPU via
        # searchsorted against split_events device metadata, avoiding
        # host round-trip.
        d_out_ids = cp.asarray(out_source_ids)
        left_count = split_events.left_segment_count
        d_source_side = cp.where(d_out_ids < left_count, cp.int8(1), cp.int8(2))

        se_device = split_events.device_state
        d_se_source_ids = cp.asarray(se_device.source_segment_ids)
        clip_idx = cp.clip(
            cp.searchsorted(d_se_source_ids, d_out_ids),
            0, max(split_events.count - 1, 0),
        )
        d_se_row = cp.asarray(se_device.row_indices)
        d_se_part = cp.asarray(se_device.part_indices)
        d_se_ring = cp.asarray(se_device.ring_indices)
        d_row_indices = d_se_row[clip_idx]
        d_part_indices = d_se_part[clip_idx]
        d_ring_indices = d_se_ring[clip_idx]

        # Row/part/ring stay on device; downstream build_gpu_half_edge_graph
        # reads device_state directly.  Host copies are lazily materialized
        # via AtomicEdgeTable.row_indices / part_indices / ring_indices
        # properties on first access.
        return AtomicEdgeTable(
            left_segment_count=split_events.left_segment_count,
            right_segment_count=split_events.right_segment_count,
            runtime_selection=split_events.runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=out_source_ids,
                direction=out_direction,
                src_x=out_src_x,
                src_y=out_src_y,
                dst_x=out_dst_x,
                dst_y=out_dst_y,
                row_indices=d_row_indices,
                part_indices=d_part_indices,
                ring_indices=d_ring_indices,
                source_side=d_source_side,
            ),
            _count=pair_count * 2,
        )
    finally:
        runtime.free(adjacency_mask)
        runtime.free(adjacency_offsets)


def overlay_intersection_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="intersection", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_union_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="union", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="difference", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_symmetric_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="symmetric_difference", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_identity_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="identity", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def _overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    operation: str,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    requested = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    _polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    polygon_only = (
        set(left.families) <= _polygonal_families
        and set(right.families) <= _polygonal_families
    )
    if not polygon_only:
        raise NotImplementedError("GPU overlay owned operations currently support polygon/multipolygon inputs")

    if requested is ExecutionMode.CPU:
        # CPU-only mode: explicit CPU request, Shapely operations
        left_values = np.asarray(left.to_shapely(), dtype=object)
        right_values = np.asarray(right.to_shapely(), dtype=object)
        if operation == "intersection":
            values = shapely.intersection(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "union":
            values = shapely.union(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "difference":
            values = shapely.difference(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "symmetric_difference":
            values = shapely.symmetric_difference(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "identity":
            values = [geometry for geometry in left_values.tolist() if geometry is not None and not geometry.is_empty]
        else:
            raise ValueError(f"unsupported overlay operation: {operation}")
        result = from_shapely_geometries(
            values,
            residency=Residency.HOST,
        )
        result.runtime_history.append(
            RuntimeSelection(
                requested=requested,
                selected=ExecutionMode.CPU,
                reason=f"CPU requested for overlay {operation}",
            )
        )
        return result

    if requested is ExecutionMode.GPU and cp is None:
        raise RuntimeError("GPU execution was requested, but CuPy is not available")

    selected = ExecutionMode.GPU if cp is not None else ExecutionMode.CPU
    if requested is ExecutionMode.GPU and selected is not ExecutionMode.GPU:
        raise RuntimeError("GPU execution was requested, but no CUDA runtime is available")
    if requested is ExecutionMode.AUTO and selected is ExecutionMode.CPU:
        # Phase 24: AUTO mode, no GPU available — CPU fallback is expected.
        record_fallback_event(
            surface=f"geopandas.overlay.{operation}",
            reason="AUTO mode: no GPU runtime available",
            detail=f"operation={operation}, left_rows={left.row_count}, right_rows={right.row_count}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_overlay_owned",
            d2h_transfer=False,
        )
        return _overlay_owned(left, right, operation=operation, dispatch_mode=ExecutionMode.CPU)

    # Ensure host-side coordinate buffers are materialised.  Device-resident
    # inputs from pylibcudf I/O have structural metadata on host but empty
    # x/y stubs (host_materialized=False).  The overlay pipeline accesses
    # polygon_buffer.x/y on host in multiple places (rectangle detection,
    # _extract_segments_gpu via device state).
    left._ensure_host_state()
    right._ensure_host_state()

    if operation == "intersection":
        rectangle_fast_path = _overlay_intersection_rectangles_gpu(left, right, requested=requested)
        if rectangle_fast_path is not None:
            return rectangle_fast_path

    # Phase 20: The 10K row CPU threshold (_GPU_OVERLAY_MAX_ROWS) has been
    # removed.  Phases 7-15 eliminated the serial bottlenecks that made GPU
    # overlay slower than Shapely at high row counts.  For AUTO mode the GPU
    # path is now selected whenever a CUDA runtime is available; input
    # residency is already on-device when the caller used the zero-copy
    # pipeline, so no additional transfer heuristic is needed here — the
    # adaptive runtime handles crossover decisions upstream via
    # plan_dispatch_selection().

    split_events = build_gpu_split_events(
        left, right,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=_cached_right_segments,
    )
    atomic_edges = build_gpu_atomic_edges(split_events)
    # Phase 25 memory: split_events device state is fully consumed by
    # build_gpu_atomic_edges.  Free its device arrays to release ~5 large
    # float64 buffers (x, y, t, packed_keys, source_segment_ids) before
    # the half-edge graph and face table allocate more.
    _free_split_event_device_state(split_events)
    # Return freed split-event blocks to the device so they can be reused
    # by the half-edge graph and face table allocations that follow.
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    # Phase 25 memory: atomic_edges device state arrays that are NOT
    # shared with half_edge_graph can be freed.  The HalfEdgeGraph keeps
    # references to src_x, src_y, source_segment_ids, source_side,
    # row_indices, part_indices, ring_indices, direction from the
    # AtomicEdgeDeviceState — but dst_x and dst_y are only needed during
    # build_gpu_half_edge_graph and can be freed now.
    _free_atomic_edge_excess(atomic_edges)
    # Return freed atomic-edge excess blocks to the device before face
    # table allocation.
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup
    faces = build_gpu_overlay_faces(left, right, half_edge_graph=half_edge_graph)
    # Phase 13: Device-resident face selection — avoids triggering
    # _ensure_host() D->H on bounded_mask/left_covered/right_covered.
    d_selected_face_indices = _select_overlay_face_indices_gpu(faces, operation=operation)
    # GPU face assembly is the primary path.  The GPU pipeline (Phases 7-14)
    # returns device-resident results, so it is strongly preferred.
    #
    # - GPU-explicit mode: GPU only, raise on failure.
    # - AUTO mode: GPU first, CPU fallback when GPU is unavailable or errors.
    if requested is ExecutionMode.GPU:
        # Strict GPU: no fallback.
        result = _build_polygon_output_from_faces_gpu(
            half_edge_graph, faces, d_selected_face_indices,
        )
        if result is None:
            raise RuntimeError(
                "GPU face assembly returned None (device state unavailable) "
                "despite GPU execution mode being requested"
            )
        face_assembly_mode = ExecutionMode.GPU
    else:
        # AUTO: GPU first, CPU fallback on failure.
        gpu_result: OwnedGeometryArray | None = None
        gpu_failed = False
        gpu_fail_reason = ""
        try:
            gpu_result = _build_polygon_output_from_faces_gpu(
                half_edge_graph, faces, d_selected_face_indices,
            )
            if gpu_result is None:
                gpu_failed = True
                gpu_fail_reason = "GPU face assembly unavailable (no device state)"
        except Exception as exc:
            gpu_failed = True
            gpu_fail_reason = f"GPU face assembly raised {type(exc).__name__}: {exc}"

        if gpu_failed:
            record_fallback_event(
                surface="overlay.gpu._overlay_owned",
                reason=gpu_fail_reason,
                detail=f"operation={operation}",
                requested=requested,
                selected=ExecutionMode.CPU,
                pipeline="overlay",
                d2h_transfer=True,
            )
            # CPU fallback needs host indices
            selected_face_indices = cp.asnumpy(d_selected_face_indices)
            result = _build_polygon_output_from_faces(
                half_edge_graph, faces, selected_face_indices,
            )
            face_assembly_mode = ExecutionMode.CPU
        else:
            result = gpu_result  # type: ignore[assignment]
            face_assembly_mode = ExecutionMode.GPU
    # Phase 25 memory: free intermediate overlay structures once the final
    # OwnedGeometryArray has been built.  The result holds its own device
    # coordinate arrays; the half-edge graph, face table, and face indices
    # are no longer needed.
    del half_edge_graph, faces, d_selected_face_indices
    # Return freed overlay structures to the device.  When _overlay_owned
    # is called repeatedly (per-group loop in spatial_overlay_owned), this
    # prevents pool-cached blocks from accumulating across iterations.
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup
    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=face_assembly_mode,
            reason=f"GPU overlay {operation}: face assembly on {face_assembly_mode.value}",
        )
    )
    return result


def spatial_overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    how: str = "intersection",
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Spatial overlay: intersect each left geometry with spatially overlapping right geometries.

    Unlike :func:`overlay_intersection_owned` which does row-matched pairwise overlay
    (left[0] vs right[0], left[1] vs right[1], ...), this function performs a spatial
    join to discover which (left_i, right_j) pairs overlap, then runs pairwise overlay
    on each discovered pair, and assembles the results.

    This implements the GeoPandas ``overlay(left, right, how=...)`` semantics on GPU,
    following ADR-0016 (8-stage overlay pipeline) and ADR-0012 (two-layer architecture).

    Parameters
    ----------
    left : OwnedGeometryArray
        Left geometry array (e.g. 10K vegetation polygons).
    right : OwnedGeometryArray
        Right geometry array (e.g. 1 dissolved corridor, or 1K zoning polygons).
    how : str
        Overlay operation: "intersection", "union", "difference", "symmetric_difference".
    dispatch_mode : ExecutionMode
        Execution mode for the pairwise overlay step.

    Returns
    -------
    OwnedGeometryArray
        Result geometries from all overlapping pairs, with empty/null results filtered out.
    """
    from vibespatial.spatial.indexing import generate_bounds_pairs

    requested = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)

    # Stage 1: Spatial join — find candidate (left_i, right_j) pairs via MBR overlap.
    # Uses the GPU sweep-plane kernel when geometry count exceeds threshold (ADR-0033 Tier 1).
    candidate_pairs = generate_bounds_pairs(left, right)
    if candidate_pairs.count == 0:
        result = from_shapely_geometries([shapely.Point()])
        result = result.take(np.asarray([], dtype=np.int64))
        result.runtime_history.append(
            RuntimeSelection(
                requested=requested,
                selected=ExecutionMode.CPU,
                reason=f"spatial_overlay {how}: no candidate pairs found",
            )
        )
        return result

    # lyy.32: Keep spatial join pair indices on GPU when device-resident.
    # When candidate_pairs has device indices (GPU spatial join), all sorting,
    # grouping, and subset selection is done with CuPy on device.  Device
    # index arrays are passed directly to take() which routes to device_take(),
    # avoiding D->H->D round-trips.  When device indices are not available
    # (CPU spatial join), falls back to the existing numpy path.
    _pairs_on_device = (
        cp is not None
        and candidate_pairs.device_left_indices is not None
        and candidate_pairs.device_right_indices is not None
    )

    if _pairs_on_device:
        # Device path: all grouping operations stay on GPU.
        d_left_indices = candidate_pairs.device_left_indices
        d_right_indices = candidate_pairs.device_right_indices

        # Ensure int64 for CuPy argsort stability and take() compatibility.
        d_left_indices = d_left_indices.astype(cp.int64, copy=False)
        d_right_indices = d_right_indices.astype(cp.int64, copy=False)

        # Sort pairs by left index for grouping.  CuPy argsort is
        # stable by default (mergesort), preserving relative order
        # within each group.
        d_sort_order = cp.argsort(d_left_indices)
        d_left_indices = d_left_indices[d_sort_order]
        d_right_indices = d_right_indices[d_sort_order]

        d_unique_left, d_group_starts = cp.unique(d_left_indices, return_index=True)
        d_group_ends = cp.append(d_group_starts[1:], cp.array([d_left_indices.size], dtype=d_group_starts.dtype))
        d_unique_right = cp.unique(d_right_indices)

        # take() with CuPy indices routes to device_take() — zero-copy.
        left_subset = left.take(d_unique_left) if int(d_unique_left.size) < left.row_count else left
        right_subset = right.take(d_unique_right) if int(d_unique_right.size) < right.row_count else right

        # Remap right indices from original row space to right_subset row space.
        d_right_remap = cp.empty(right.row_count, dtype=cp.int64)
        d_right_remap[d_unique_right] = cp.arange(int(d_unique_right.size), dtype=cp.int64)
        d_right_subset_indices = d_right_remap[d_right_indices]

        # Keep group boundaries on device — they are small O(unique_left)
        # arrays used for filtering in the containment/SH bypass paths.
        # They are materialised to host lazily right before the per-group
        # Python loop or segmented_union_all call (FIX-09: ZCOPY001).
        group_starts = d_group_starts  # device CuPy array
        group_ends = d_group_ends  # device CuPy array
        unique_left = d_unique_left  # device array for take()
        right_subset_indices = d_right_subset_indices  # device array for take()

        # Host copies for Shapely fallback path (lazy — only materialised if
        # owned-dispatch fails).  These are set after the owned-dispatch
        # try/except block when needed.
        left_indices = None  # deferred; materialised in fallback
        right_indices = None
    else:
        # CPU path: pair indices are already on host.
        left_indices = candidate_pairs.left_indices
        right_indices = candidate_pairs.right_indices

        # Sort pairs by left index for grouping.
        sort_order = np.argsort(left_indices, kind="mergesort")
        left_indices = left_indices[sort_order]
        right_indices = right_indices[sort_order]

        unique_left, group_starts = np.unique(left_indices, return_index=True)
        group_ends = np.append(group_starts[1:], len(left_indices))

        unique_right = np.unique(right_indices)

        # take() operates at buffer level — no Shapely materialization.
        left_subset = left.take(unique_left) if len(unique_left) < left.row_count else left
        right_subset = right.take(unique_right) if len(unique_right) < right.row_count else right

        # Remap right indices from original row space to right_subset row space.
        right_remap = np.empty(right.row_count, dtype=np.intp)
        right_remap[unique_right] = np.arange(len(unique_right))
        right_subset_indices = right_remap[right_indices]

    # Strategy detection: detect workload shape and select the overlay strategy.
    # broadcast_right + intersection: containment bypass (lyy.16) + batched SH
    # clip (lyy.18) reduce work before per-group overlay.  Other strategies
    # fall through to the existing per-group path.
    from vibespatial.overlay.strategies import select_overlay_strategy

    strategy = select_overlay_strategy(
        left, right, how,
        candidate_pair_count=candidate_pairs.count,
    )

    _ws_label = strategy.workload_shape.value if strategy.workload_shape is not None else strategy.name
    record_dispatch_event(
        surface="geopandas.spatial_overlay",
        operation=f"spatial_overlay_{how}",
        implementation=f"spatial_overlay_{strategy.name}",
        reason=strategy.reason,
        detail=(
            f"left={left.row_count}, right={right.row_count}, "
            f"pairs={candidate_pairs.count}, strategy={strategy.name}, "
            f"workload_shape={_ws_label}"
        ),
        requested=requested,
        selected=ExecutionMode.GPU if cp is not None else ExecutionMode.CPU,
    )

    # Strategy-specific implementations.
    # lyy.16/lyy.11: containment/disjointness bypass for broadcast_right.
    #   intersection: contained -> pass-through, disjoint -> skip.
    #   difference: contained -> empty, disjoint -> pass-through.
    # lyy.18: batched SH clip for boundary-crossing simple polygons.
    _containment_result: OwnedGeometryArray | None = None
    _containment_remainder_mask: cp.ndarray | None = None  # type: ignore[name-defined]
    _sh_clip_result: OwnedGeometryArray | None = None

    _bypass_eligible_ops = ("intersection", "difference")
    if strategy.name == "broadcast_right" and how in _bypass_eligible_ops and cp is not None:
        try:
            _containment_result, _containment_remainder_mask = (
                _containment_bypass_gpu(left_subset, right_subset, how)
            )
        except Exception:
            # If containment bypass fails, fall through to per-group.
            logger.debug(
                "lyy.16 containment bypass failed, falling through to overlay",
                exc_info=True,
            )
            _containment_result = None
            _containment_remainder_mask = None

        if _containment_remainder_mask is not None:
            # Some polygons need overlay — filter left_subset to remainder only.
            # Rebuild group structures for the remainder polygons.
            d_remainder_indices = cp.flatnonzero(_containment_remainder_mask).astype(cp.int64)
            left_subset = left_subset.take(d_remainder_indices)

            # Rebuild pair grouping: filter pairs to only reference remainder rows.
            # The remainder_mask is over the original left_subset row space.
            # We need to remap unique_left, group_starts, group_ends to the
            # filtered left_subset.
            if _pairs_on_device:
                # d_left_indices are in original left row space.  d_unique_left
                # maps to left_subset row space (0..len(unique_left)-1).
                # We need to filter to rows where remainder_mask is True.
                #
                # Approach: rebuild grouping from scratch for the filtered pairs.
                # Only pairs whose left_subset row is in the remainder survive.

                # Build a mapping: old left_subset row -> new left_subset row.
                # _containment_remainder_mask is a bool mask over old left_subset.
                d_old_to_new = cp.full(int(_containment_remainder_mask.size), -1, dtype=cp.int64)
                d_old_to_new[d_remainder_indices] = cp.arange(int(d_remainder_indices.size), dtype=cp.int64)

                # Filter groups: each group corresponds to one unique_left row
                # in the old left_subset.  If that row is in the remainder, keep
                # the group; remap its grp_idx to the new left_subset space.
                # unique_left[grp_idx] is the original left row; grp_idx is
                # the old left_subset row.
                d_grp_mask = _containment_remainder_mask[
                    cp.arange(len(group_starts), dtype=cp.int64)
                ]
                new_group_indices = cp.flatnonzero(d_grp_mask)

                # FIX-09: index device arrays directly — no D2H for
                # group boundary filtering.
                group_starts = group_starts[new_group_indices]
                group_ends = group_ends[new_group_indices]
                unique_left = unique_left[new_group_indices]

                # right_subset_indices are unchanged — they reference right_subset
                # rows which are not affected by left filtering.
            else:
                # CPU path: rebuild grouping.
                h_remainder_mask = cp.asnumpy(_containment_remainder_mask) if cp is not None else _containment_remainder_mask  # zcopy:ok(CPU-path branch; H2D at 5450 is in device branch — mutually exclusive)
                old_to_new = np.full(len(h_remainder_mask), -1, dtype=np.int64)
                h_remainder_indices = np.flatnonzero(h_remainder_mask)
                old_to_new[h_remainder_indices] = np.arange(len(h_remainder_indices), dtype=np.int64)

                grp_mask = h_remainder_mask[np.arange(len(group_starts))]
                new_grp_indices = np.flatnonzero(grp_mask)
                group_starts = group_starts[new_grp_indices]
                group_ends = group_ends[new_grp_indices]
                unique_left = unique_left[new_grp_indices]

        elif _containment_result is not None and _containment_remainder_mask is None:
            # ALL polygons were bypassed — no overlay needed.
            # Skip the entire per-group processing.
            result = _containment_result
            result.runtime_history.append(
                RuntimeSelection(
                    requested=requested,
                    selected=ExecutionMode.GPU,
                    reason=(
                        f"spatial_overlay {how}: all polygons bypassed "
                        f"(containment/disjointness bypass)"
                    ),
                )
            )
            return result

        # lyy.18: Batched SH clip for boundary-crossing simple polygons.
        # After containment bypass, left_subset contains only remainder
        # polygons.  Check if the clip polygon (right_subset) is SH-eligible.
        # If so, batch-clip all SH-eligible remainder polygons in a single
        # polygon_intersection kernel launch, further reducing the number of
        # polygons that fall through to the expensive per-group overlay.
        # SH clip is only applicable to intersection (clipping semantics).
        if how == "intersection" and left_subset.row_count > 0:
            try:
                clip_eligible, clip_vert_count = _is_clip_polygon_sh_eligible(right_subset)
                if clip_eligible:
                    sh_eligible_mask, _complex_mask = _classify_remainder_sh_eligible(
                        left_subset, clip_vert_count,
                    )
                    n_sh = int(sh_eligible_mask.sum())

                    if n_sh > 0:
                        _sh_clip_result = _batched_sh_clip(
                            left_subset, right_subset, sh_eligible_mask,
                        )

                        if _sh_clip_result is not None and n_sh < left_subset.row_count:
                            # Some polygons were SH-clipped; filter left_subset
                            # and grouping structures to only the complex remainder.
                            d_complex_indices = cp.asarray(
                                np.flatnonzero(~sh_eligible_mask)
                            ).astype(cp.int64)
                            left_subset = left_subset.take(d_complex_indices)

                            # Rebuild grouping for the reduced left_subset.
                            if _pairs_on_device:
                                # complex_mask[i] is True for old left_subset
                                # rows that are NOT SH-eligible.  Rebuild the
                                # old-to-new mapping for group filtering.
                                d_old_to_new_sh = cp.full(
                                    len(sh_eligible_mask), -1, dtype=cp.int64,
                                )
                                d_old_to_new_sh[d_complex_indices] = cp.arange(
                                    int(d_complex_indices.size), dtype=cp.int64,
                                )
                                d_sh_grp_mask = cp.asarray(~sh_eligible_mask)[
                                    cp.arange(len(group_starts), dtype=cp.int64)
                                ]
                                new_grp_sh = cp.flatnonzero(d_sh_grp_mask)
                                # FIX-09: index device arrays directly.
                                group_starts = group_starts[new_grp_sh]
                                group_ends = group_ends[new_grp_sh]
                                unique_left = unique_left[new_grp_sh]
                            else:
                                h_complex_mask = ~sh_eligible_mask
                                old_to_new_sh = np.full(
                                    len(sh_eligible_mask), -1, dtype=np.int64,
                                )
                                h_complex_indices = np.flatnonzero(h_complex_mask)
                                old_to_new_sh[h_complex_indices] = np.arange(
                                    len(h_complex_indices), dtype=np.int64,
                                )
                                grp_mask_sh = h_complex_mask[
                                    np.arange(len(group_starts))
                                ]
                                new_grp_sh = np.flatnonzero(grp_mask_sh)
                                group_starts = group_starts[new_grp_sh]
                                group_ends = group_ends[new_grp_sh]
                                unique_left = unique_left[new_grp_sh]

                        elif _sh_clip_result is not None:
                            # ALL remainder polygons were SH-clipped.
                            # No overlay needed for any polygon.
                            left_subset = left_subset.take(
                                np.asarray([], dtype=np.int64)
                            )
                            group_starts = group_starts[:0]
                            group_ends = group_ends[:0]
                            if _pairs_on_device:
                                unique_left = unique_left[:0]
                            else:
                                unique_left = unique_left[:0]

                    else:
                        logger.debug(
                            "lyy.18 SH batch clip: clip polygon eligible but "
                            "no remainder polygons qualify (all have holes or "
                            "too many vertices)"
                        )
                else:
                    logger.debug(
                        "lyy.18 SH batch clip: clip polygon not SH-eligible "
                        "(holes or >%d vertices) -- skipping SH tier",
                        64,
                    )
            except Exception:
                logger.debug(
                    "lyy.18 SH batch clip failed, falling through to overlay",
                    exc_info=True,
                )
                _sh_clip_result = None

    elif strategy.name == "broadcast_left":
        pass  # fall through to per_group

    # Release GPU pool memory after containment bypass and SH batch clip:
    # bounds check, PIP, and clip kernels produce large intermediates that
    # are no longer needed before the per-group overlay loop.
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup

    # Stage 2: Per-left-group processing.
    #
    # Previous approach gathered ALL pairs into a single batch and ran one
    # global binary_constructive_owned call.  This caused two bugs:
    #   Bug 1 (O(n**2) scaling): segment candidate generation used a GLOBAL
    #     sort-sweep so segments from independent geometry pairs
    #     cross-contaminated, producing O(n**2) segment comparisons.
    #   Bug 2 (incorrect difference): computed L_i - R_j per pair instead of
    #     L_i - union(R_j for all j overlapping L_i), yielding multiple
    #     partial-difference fragments per left geometry.
    #
    # Fix: group pairs by left index and process each left geometry
    # independently.  For difference/symmetric_difference, union all right
    # neighbours first via segmented_union_all (matching the approach in
    # _overlay_difference in api/tools/overlay.py).  For intersection/union,
    # process per-pair within each group to keep segment sets isolated.
    #
    # Performance: selective materialization (ADR-0005).
    # take() subsets participating rows at buffer level — no Shapely
    # round-trip.  Per-group calls each handle O(1) segment pairs, giving
    # overall O(N) scaling instead of O(N**2).

    # Remap left indices: unique_left[i] -> i (left_subset row space).
    # left_subset row i corresponds to unique_left[i] in the original array.
    # For per-group processing we iterate over groups 0..len(unique_left)-1.

    # ------------------------------------------------------------------
    # Owned-dispatch path: per-left-group processing via binary_constructive_owned.
    # For difference/symmetric_difference, uses segmented_union_all to union
    # right neighbours before computing the set operation — matching the
    # correct semantics of L_i - union(R_j for all j).
    # ------------------------------------------------------------------
    _used_owned_dispatch = False
    try:
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        # Force GPU dispatch when a GPU runtime is available: the spatial
        # overlay pipeline has already committed to GPU for spatial join
        # and pair generation.  The pairwise constructive step must also
        # use GPU to avoid the 50K CONSTRUCTIVE crossover threshold
        # routing small pair batches to CPU (which triggers a fallback
        # event per batch and forces a D->H->D round-trip through Shapely).
        from vibespatial.runtime import has_gpu_runtime
        _pairwise_mode = ExecutionMode.GPU if has_gpu_runtime() else requested

        if how in ("difference", "symmetric_difference"):
            # Union all right neighbours per left group, then compute one
            # set operation per unique left geometry.
            # This mirrors the approach in _overlay_difference (api/tools/overlay.py).
            from vibespatial.kernels.constructive.segmented_union import (
                segmented_union_all,
            )

            # Build CSR-style group offsets for segmented_union_all.
            # segmented_union_all expects host numpy offsets.
            # FIX-09: when pairs are on device, materialise group_starts
            # to host here (single bulk D2H for small metadata).
            if _pairs_on_device:
                h_group_starts = cp.asnumpy(group_starts)  # zcopy:ok(small metadata for segmented_union_all)
                n_rsi = int(right_subset_indices.shape[0])
            else:
                h_group_starts = group_starts  # already host numpy
                n_rsi = len(right_subset_indices)
            group_offsets = np.empty(len(h_group_starts) + 1, dtype=np.int64)
            group_offsets[:-1] = h_group_starts
            group_offsets[-1] = n_rsi

            right_gathered = right_subset.take(right_subset_indices)
            right_unions = segmented_union_all(right_gathered, group_offsets)

            # left_subset has one row per unique left geometry, aligned with
            # right_unions (one unioned geometry per group).
            result_owned = binary_constructive_owned(
                how, left_subset, right_unions,
                dispatch_mode=_pairwise_mode,
            )

        else:
            # intersection / union: process per-pair within each group.
            # Each (L_i, R_j) pair produces an independent result fragment.
            # Processing per-group keeps segment sets isolated, avoiding
            # global O(n**2) cross-contamination.
            result_parts: list[OwnedGeometryArray] = []
            _xp = cp if _pairs_on_device else np  # array module for index construction

            # lyy.15: Cache right-side segment extraction for broadcast_right.
            # In the N-vs-1 pattern, the right geometry (corridor) is identical
            # for every pair.  Pre-extract its segments ONCE and reuse across
            # all iterations, avoiding redundant _extract_segments_gpu calls
            # (2 per iteration: once in build_gpu_split_events, once in
            # classify_segment_intersections).
            _cached_right_segs: DeviceSegmentTable | None = None
            _is_broadcast_right_overlay = (
                strategy.name == "broadcast_right"
                and right_subset.row_count == 1
                and cp is not None
            )
            if _is_broadcast_right_overlay and len(unique_left) > 1:
                try:
                    _cached_right_segs = _extract_segments_gpu(right_subset)
                    logger.debug(
                        "lyy.15: cached right-side segments (%d segments) "
                        "for %d per-group iterations",
                        _cached_right_segs.count, len(unique_left),
                    )
                except Exception:
                    logger.debug(
                        "lyy.15: right-side segment caching failed, "
                        "falling through to per-iteration extraction",
                        exc_info=True,
                    )
                    _cached_right_segs = None

            # lyy.21: Batched overlay for broadcast_right many-vs-one.
            #
            # When all pairs share the same right polygon (broadcast_right),
            # gather all complex left polygons and dispatch them in a single
            # binary_constructive_owned call.  Each row is still processed
            # independently (per-row isolation) because
            # binary_constructive_owned routes to _overlay_owned which runs
            # the 8-stage pipeline per-pair.
            #
            # Optimization: for broadcast_right with a single right polygon,
            # call _overlay_owned directly per pair, bypassing the per-
            # iteration overhead of binary_constructive_owned dispatch
            # (workload detection, SH kernel pre-flight, dispatch selection,
            # precision planning).  Host state for right_subset is
            # materialised once before the loop.
            #
            # Cross-contamination guard: per-pair isolation is maintained
            # because each _overlay_owned call operates on exactly one
            # (L_i, R_0) pair.  Segments from pair i never interact with
            # segments from pair j.
            try:
                # lyy.22: Stream pool for per-group overlay.
                #
                # Create a bounded pool of CUDA streams and assign each
                # overlay pair to a stream in round-robin order.  Each
                # pair's GPU work (CuPy operations) is issued on its
                # assigned stream.  Results are collected after all pairs
                # are dispatched, with per-stream synchronization.
                #
                # IMPORTANT: non_blocking=False (default) is required.
                # See _OVERLAY_STREAM_POOL_SIZE comment for rationale.
                _n_groups = len(unique_left)
                _pool_size = min(_OVERLAY_STREAM_POOL_SIZE, _n_groups)
                _use_stream_pool = cp is not None and _n_groups > 1 and _pool_size > 0
                if _use_stream_pool:
                    _stream_pool = [
                        cp.cuda.Stream(non_blocking=False)
                        for _ in range(_pool_size)
                    ]
                    logger.debug(
                        "lyy.22: created stream pool with %d streams "
                        "for %d per-group iterations",
                        len(_stream_pool), _n_groups,
                    )
                else:
                    _stream_pool = None

                # FIX-09: materialise group boundaries to host once
                # right before the per-group Python loop.  This is the
                # only D2H for these arrays; all upstream filtering
                # (containment bypass, SH bypass) operated on device.
                if _pairs_on_device and _n_groups > 0:
                    _h_group_starts = cp.asnumpy(group_starts)
                    _h_group_ends = cp.asnumpy(group_ends)
                else:
                    _h_group_starts = group_starts
                    _h_group_ends = group_ends

                if _is_broadcast_right_overlay and _n_groups > 0:
                    # Materialise right host state once outside the loop.
                    # _overlay_owned calls _ensure_host_state on both
                    # inputs; doing it here for right avoids N redundant
                    # no-op calls.
                    right_subset._ensure_host_state()

                    # lyy.22: Dispatch all pairs across the stream pool,
                    # collecting (stream, result) futures for deferred
                    # synchronization.
                    _futures: list[tuple] = []
                    for grp_idx in range(_n_groups):
                        start, end = int(_h_group_starts[grp_idx]), int(_h_group_ends[grp_idx])
                        n_pairs = end - start
                        _stream = _stream_pool[grp_idx % len(_stream_pool)] if _stream_pool else None
                        with (_stream if _stream is not None else contextlib.nullcontext()):
                            left_row = left_subset.take(
                                _xp.array([grp_idx], dtype=_xp.int64)
                            )
                            if n_pairs == 1:
                                # Common case: one pair (L_i, R_0).
                                # Call _overlay_owned directly, bypassing
                                # binary_constructive_owned dispatch overhead.
                                grp_result = _overlay_owned(
                                    left_row, right_subset,
                                    operation=how,
                                    dispatch_mode=_pairwise_mode,
                                    _cached_right_segments=_cached_right_segs,
                                )
                            else:
                                # Multiple right neighbours: replicate left
                                # row and use binary_constructive_owned for
                                # the full dispatch (rare in broadcast_right).
                                right_rows = right_subset.take(
                                    right_subset_indices[start:end]
                                )
                                left_replicated = left_row.take(
                                    _xp.zeros(n_pairs, dtype=_xp.int64)
                                )
                                grp_result = binary_constructive_owned(
                                    how, left_replicated, right_rows,
                                    dispatch_mode=_pairwise_mode,
                                    _cached_right_segments=_cached_right_segs,
                                )
                            _futures.append((_stream, grp_result))

                    # lyy.22: Synchronize per stream and collect results.
                    # Each stream is synced at most once (the first time
                    # we encounter a result from it), which guarantees
                    # all prior work on that stream has completed.
                    _synced_streams: set[int] = set()
                    for _stream, grp_result in _futures:
                        if _stream is not None:
                            _sid = id(_stream)
                            if _sid not in _synced_streams:
                                _stream.synchronize()
                                _synced_streams.add(_sid)
                        result_parts.append(grp_result)

                else:
                    # General case: multiple right neighbours per group,
                    # or non-broadcast_right strategy.
                    _futures_gen: list[tuple] = []
                    for grp_idx in range(_n_groups):
                        start, end = int(_h_group_starts[grp_idx]), int(_h_group_ends[grp_idx])
                        n_pairs = end - start
                        _stream = _stream_pool[grp_idx % len(_stream_pool)] if _stream_pool else None
                        with (_stream if _stream is not None else contextlib.nullcontext()):
                            left_row = left_subset.take(
                                _xp.array([grp_idx], dtype=_xp.int64)
                            )
                            right_rows = right_subset.take(
                                right_subset_indices[start:end]
                            )
                            # Replicate left row to match the number of
                            # right neighbours.
                            if n_pairs > 1:
                                left_replicated = left_row.take(
                                    _xp.zeros(n_pairs, dtype=_xp.int64)
                                )
                            else:
                                left_replicated = left_row
                            grp_result = binary_constructive_owned(
                                how, left_replicated, right_rows,
                                dispatch_mode=_pairwise_mode,
                                _cached_right_segments=_cached_right_segs,
                            )
                            _futures_gen.append((_stream, grp_result))

                    # lyy.22: Synchronize per stream and collect results.
                    _synced_streams_gen: set[int] = set()
                    for _stream, grp_result in _futures_gen:
                        if _stream is not None:
                            _sid = id(_stream)
                            if _sid not in _synced_streams_gen:
                                _stream.synchronize()
                                _synced_streams_gen.add(_sid)
                        result_parts.append(grp_result)
            finally:
                # lyy.15: Free cached right-side segments after all
                # iterations (or on exception).
                if _cached_right_segs is not None:
                    _rt = get_cuda_runtime()
                    _rt.free(_cached_right_segs.x0)
                    _rt.free(_cached_right_segs.y0)
                    _rt.free(_cached_right_segs.x1)
                    _rt.free(_cached_right_segs.y1)
                    _rt.free(_cached_right_segs.row_indices)
                    _rt.free(_cached_right_segs.segment_indices)
                    if _cached_right_segs.part_indices is not None:
                        _rt.free(_cached_right_segs.part_indices)
                    if _cached_right_segs.ring_indices is not None:
                        _rt.free(_cached_right_segs.ring_indices)
                    _cached_right_segs = None
                # lyy.22: Destroy stream pool to avoid resource leaks.
                # CuPy streams are lightweight (~1-2us each) but should
                # still be cleaned up.
                if _stream_pool is not None:
                    for _s in _stream_pool:
                        # CuPy streams do not require explicit destruction;
                        # they are released when garbage collected.  We
                        # synchronize each one to ensure all dispatched work
                        # is complete before leaving the finally block.
                        try:
                            _s.synchronize()
                        except Exception:
                            pass
                    _stream_pool = None

            if result_parts:
                result_owned = OwnedGeometryArray.concat(result_parts)
            else:
                result_owned = from_shapely_geometries([shapely.Point()])
                result_owned = result_owned.take(np.asarray([], dtype=np.int64))

        # Release GPU pool memory after per-group overlay loop: each
        # iteration's split events, half-edge graphs, and face tables
        # leave freed-but-cached blocks in the CuPy pool.
        try:
            get_cuda_runtime().free_pool_memory()
        except Exception:
            pass  # best-effort cleanup

        # Filter empty/null using owned-level metadata (validity + empty_mask)
        # instead of to_shapely() — avoids D->H->D ping-pong.
        # binary_constructive_owned returns polygon-family results (no
        # GeometryCollections), so collection flattening is unnecessary.
        result_owned._ensure_host_state()
        non_empty = result_owned.validity.copy()
        for family, buf in result_owned.families.items():
            family_rows = (result_owned.tags == FAMILY_TAGS[family])
            non_empty[family_rows] &= ~buf.empty_mask[
                result_owned.family_row_offsets[family_rows]
            ]
        keep_indices = np.flatnonzero(non_empty)
        if keep_indices.size == 0:
            result = from_shapely_geometries([shapely.Point()])
            result = result.take(np.asarray([], dtype=np.int64))
        else:
            result = result_owned.take(keep_indices)

        # lyy.16 + lyy.18: Combine bypass results with overlay results.
        result = _combine_bypass_results(
            _containment_result, _sh_clip_result, result,
        )

        _used_owned_dispatch = True
    except (NotImplementedError, ImportError, ValueError):
        _used_owned_dispatch = False

    if not _used_owned_dispatch:
        # lyy.32: When owned dispatch failed and pairs were on device,
        # materialise grouping arrays to host for the Shapely fallback path.
        # This D->H transfer is acceptable because the Shapely path already
        # materialises the full geometries to host below.
        if _pairs_on_device:
            left_indices = cp.asnumpy(d_left_indices)
            right_indices = cp.asnumpy(d_right_indices)
            group_starts = cp.asnumpy(d_group_starts)
            group_ends = cp.asnumpy(d_group_ends)
            unique_left = cp.asnumpy(d_unique_left)
            right_subset_indices = cp.asnumpy(d_right_subset_indices)

        # Phase 24: Record fallback event for spatial overlay CPU path.
        record_fallback_event(
            surface=f"geopandas.spatial_overlay.{how}",
            reason="owned-path dispatch failed, falling back to Shapely",
            detail=f"how={how}, pairs={len(left_indices)}",
            requested=requested,
            selected=ExecutionMode.CPU,
            pipeline="spatial_overlay_owned",
            d2h_transfer=True,
        )
        # Shapely fallback: materialize participating rows for validation + clipping.
        left_shapely_orig = np.asarray(left_subset.to_shapely(), dtype=object)
        right_shapely_orig = np.asarray(right_subset.to_shapely(), dtype=object)

        # Validate input geometries ONCE before replication (ADR-0019).
        # This avoids running make_valid on 10K replicated copies of the same
        # invalid geometry. Validate on the (smaller) subset, then gather.
        left_invalid_mask = ~np.asarray(shapely.is_valid(left_shapely_orig), dtype=bool)
        right_invalid_mask = ~np.asarray(shapely.is_valid(right_shapely_orig), dtype=bool)
        if np.any(left_invalid_mask):
            left_shapely_orig[left_invalid_mask] = shapely.make_valid(left_shapely_orig[left_invalid_mask])
        if np.any(right_invalid_mask):
            right_shapely_orig[right_invalid_mask] = shapely.make_valid(right_shapely_orig[right_invalid_mask])

    if not _used_owned_dispatch:
        # Shapely fast paths and general case (only when owned dispatch failed).

        # Fast path: many-vs-one intersection with clip_by_rect (ADR-0033 Tier 2).
        _used_clip_by_rect = False
        if how == "intersection" and right.row_count == 1:
            right_geom = right_shapely_orig[0]
            if right_geom is not None and right_geom.geom_type == "Polygon" and not right_geom.is_empty:
                coords = np.asarray(right_geom.exterior.coords)
                if len(coords) == 5:
                    xs, ys = coords[:4, 0], coords[:4, 1]
                    x_vals = np.unique(xs)
                    y_vals = np.unique(ys)
                    if len(x_vals) == 2 and len(y_vals) == 2 and len(right_geom.interiors) == 0:
                        xmin, xmax = float(x_vals[0]), float(x_vals[1])
                        ymin, ymax = float(y_vals[0]), float(y_vals[1])
                        # Clip all participating left geometries against the rectangle.
                        result_geoms = shapely.clip_by_rect(left_shapely_orig, xmin, ymin, xmax, ymax)
                        _used_clip_by_rect = True

        # Fast path: many-vs-one intersection with GPU centroid pre-filter.
        _used_centroid_filter = False
        if (not _used_clip_by_rect
                and how == "intersection"
                and right.row_count == 1
                and left.row_count >= 100
                and _has_polygonal_families(left)):
            right_geom = right_shapely_orig[0]
            if right_geom is not None and not right_geom.is_empty:
                from vibespatial.constructive.polygon import polygon_centroids_owned
                from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
                try:
                    cx, cy = polygon_centroids_owned(left_subset)
                    centroids = shapely.points(cx, cy)
                    inside_mask = np.asarray(shapely.within(centroids, right_geom), dtype=bool)
                    mask_bounds = right_geom.bounds
                    left_bounds = compute_geometry_bounds(left_subset)
                    inside_idx = np.flatnonzero(inside_mask)
                    if inside_idx.size > 0:
                        bbox_fully_inside = (
                            (left_bounds[inside_idx, 0] >= mask_bounds[0])
                            & (left_bounds[inside_idx, 1] >= mask_bounds[1])
                            & (left_bounds[inside_idx, 2] <= mask_bounds[2])
                            & (left_bounds[inside_idx, 3] >= mask_bounds[1])
                        )
                        fully_inside_rows = inside_idx[bbox_fully_inside]
                    else:
                        fully_inside_rows = np.asarray([], dtype=np.intp)
                    all_rows = np.arange(left_subset.row_count)
                    need_clip_rows = np.setdiff1d(all_rows, fully_inside_rows)
                    if need_clip_rows.size < left_subset.row_count:
                        result_parts_shapely: list = []
                        if fully_inside_rows.size > 0:
                            result_parts_shapely.extend(left_shapely_orig[fully_inside_rows].tolist())
                        if need_clip_rows.size > 0:
                            clip_left = left_shapely_orig[need_clip_rows]
                            clip_right = np.full(len(need_clip_rows), right_geom, dtype=object)
                            clipped = shapely.intersection(clip_left, clip_right)
                            result_parts_shapely.extend(clipped.tolist())
                        result_geoms = np.asarray(result_parts_shapely, dtype=object)
                        _used_centroid_filter = True
                except Exception:
                    pass  # fall through to general path

        if not _used_clip_by_rect and not _used_centroid_filter:
            # Per-left-group Shapely path: process each left geometry against
            # its overlapping right neighbours independently.  For difference,
            # this computes L_i - union(R_j) to produce correct results.
            if how == "difference":
                result_list: list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    if len(right_neighbors) == 1:
                        right_union = right_neighbors[0]
                    else:
                        right_union = shapely.union_all(right_neighbors)
                    diff = shapely.difference(np.array([left_geom], dtype=object),
                                              np.array([right_union], dtype=object))
                    result_list.append(diff[0])
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "symmetric_difference":
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    if len(right_neighbors) == 1:
                        right_union = right_neighbors[0]
                    else:
                        right_union = shapely.union_all(right_neighbors)
                    sd = shapely.symmetric_difference(
                        np.array([left_geom], dtype=object),
                        np.array([right_union], dtype=object),
                    )
                    result_list.append(sd[0])
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "intersection":
                # Per-pair intersection: L_i intersect R_j for each pair.
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    left_arr = np.full(len(right_neighbors), left_geom, dtype=object)
                    inter = shapely.intersection(left_arr, right_neighbors)
                    result_list.extend(inter.tolist())
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "union":
                # Per-pair union: L_i union R_j for each pair.
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    left_arr = np.full(len(right_neighbors), left_geom, dtype=object)
                    unions = shapely.union(left_arr, right_neighbors)
                    result_list.extend(unions.tolist())
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            else:
                raise ValueError(f"unsupported spatial overlay operation: {how}")

    if not _used_owned_dispatch:
        # Stage 3: Filter out empty/null results (Shapely path only).
        # The owned-dispatch path does its own filtering above.
        result_arr = np.asarray(result_geoms, dtype=object) if not isinstance(result_geoms, np.ndarray) else result_geoms
        non_null = result_arr != None  # noqa: E711 — intentional identity check for numpy
        non_empty_mask = np.zeros(len(result_arr), dtype=bool)
        if np.any(non_null):
            non_empty_mask[non_null] = ~shapely.is_empty(result_arr[non_null])
        candidates = result_arr[non_empty_mask]

        # Check for GeometryCollections that need flattening
        valid_geoms = []
        has_collections = False
        for g in candidates:
            if g.geom_type == "GeometryCollection":
                has_collections = True
                for part in g.geoms:
                    if part.geom_type in (
                        "Point", "LineString", "Polygon",
                        "MultiPoint", "MultiLineString", "MultiPolygon",
                    ) and not part.is_empty:
                        valid_geoms.append(part)
            else:
                valid_geoms.append(g)

        if not has_collections:
            valid_geoms = list(candidates)

        if not valid_geoms:
            result = from_shapely_geometries([shapely.Point()])
            result = result.take(np.asarray([], dtype=np.int64))
        else:
            result = from_shapely_geometries(valid_geoms)

        # lyy.16 + lyy.18: Combine bypass results with Shapely fallback results.
        result = _combine_bypass_results(
            _containment_result, _sh_clip_result, result,
        )

    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=ExecutionMode.GPU if candidate_pairs.pairs_examined > 0 and cp is not None else ExecutionMode.CPU,
            reason=(
                f"spatial_overlay {how}: {candidate_pairs.count} candidate pairs from "
                f"{left.row_count}x{right.row_count} inputs"
            ),
        )
    )
    return result
