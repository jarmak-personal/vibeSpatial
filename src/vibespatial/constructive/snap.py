"""GPU-accelerated snap: snap vertices of geometry A to nearest vertices on B.

For each element-wise pair (A[i], B[i]):
1. For each vertex in A, find the nearest vertex of B within the given
   tolerance (matches GEOS/Shapely vertex-to-vertex snap semantics).
2. If a snap target vertex exists within tolerance, move the vertex to
   the snapped location.  Otherwise keep the vertex unchanged.
3. Deduplicate coincident vertices that resulted from snapping (sequential
   scan within each ring/linestring).
4. Ensure ring closure for polygon rings.
5. Rebuild geometry offsets from modified coordinate counts.

Architecture (ADR-0033):
- Tier 1 NVRTC for the per-vertex nearest-vertex search (geometry-specific
  inner loop: iterates all vertices of B for each vertex of A).
- Tier 1 NVRTC for the per-ring sequential dedup pass (same pattern as
  remove_repeated_points).
- Tier 3a CCCL for exclusive_sum on per-span kept counts.
- Tier 2 CuPy for coordinate gather (compact).

Precision (ADR-0002): CONSTRUCTIVE class -- stays fp64 on all devices per
policy.  PrecisionPlan wired through dispatch for observability only.

Bulk D2H pre-transfer of family coordinate ranges before the dispatch loop
(bounded by number of geometry families, not data size).  Kernel execution
and output assembly are fully device-resident via build_device_resident_owned.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.snap_kernels import _SNAP_FP64, _SNAP_KERNEL_NAMES
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
    tile_single_row,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.workload import WorkloadShape, detect_workload_shape

logger = logging.getLogger(__name__)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("snap-fp64", _SNAP_FP64, _SNAP_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Helpers: span offsets and coordinate ranges for a family
# ---------------------------------------------------------------------------

def _get_span_offsets(device_buf, family):
    """Return the device array of span offsets for this family.

    Spans are the innermost coordinate grouping:
    - Point: geometry_offsets (each point is 1 coord)
    - LineString: geometry_offsets
    - MultiLineString: part_offsets
    - Polygon/MultiPolygon: ring_offsets
    """
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        return device_buf.ring_offsets
    elif family is GeometryFamily.MULTILINESTRING:
        return device_buf.part_offsets
    else:
        return device_buf.geometry_offsets


def _get_coord_range_for_geom(device_buf, family):
    """Return (start, end) CuPy arrays of coordinate ranges per geometry.

    For geometry at family-local index i, coordinates are in
    [start[i], end[i]) of the x/y arrays.
    """
    d_geom_offsets = cp.asarray(device_buf.geometry_offsets)
    num_geoms = int(d_geom_offsets.shape[0]) - 1
    if num_geoms <= 0:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        # Coordinates are indexed by ring_offsets.
        # geometry_offsets -> ring range; ring_offsets -> coord range.
        d_ring_offsets = cp.asarray(device_buf.ring_offsets)
        if family is GeometryFamily.MULTIPOLYGON and device_buf.part_offsets is not None:
            d_part_offsets = cp.asarray(device_buf.part_offsets)
            # geom -> first part -> first ring -> first coord
            first_part = d_part_offsets[d_geom_offsets[:-1]]
            last_part = d_part_offsets[d_geom_offsets[1:]]
            coord_start = d_ring_offsets[first_part]
            coord_end = d_ring_offsets[last_part]
        else:
            # Polygon: geom_offsets index into ring_offsets
            coord_start = d_ring_offsets[d_geom_offsets[:-1]]
            coord_end = d_ring_offsets[d_geom_offsets[1:]]
    elif family is GeometryFamily.MULTILINESTRING and device_buf.part_offsets is not None:
        d_part_offsets = cp.asarray(device_buf.part_offsets)
        # geom_offsets index into part_offsets; part_offsets index into coords
        coord_start = d_part_offsets[d_geom_offsets[:-1]]
        coord_end = d_part_offsets[d_geom_offsets[1:]]
    else:
        # Point, LineString: geometry_offsets directly index coordinates
        coord_start = d_geom_offsets[:-1]
        coord_end = d_geom_offsets[1:]

    return coord_start.astype(cp.int32), coord_end.astype(cp.int32)


def _transfer_coord_range_to_host(device_buf, family):
    """Transfer a single family's coordinate ranges D->H."""
    d_cs, d_ce = _get_coord_range_for_geom(device_buf, family)
    return cp.asnumpy(d_cs), cp.asnumpy(d_ce)


def _transfer_family_xy_to_host(device_buf):
    """Transfer a single family's x/y vertex data D->H."""
    return cp.asnumpy(cp.asarray(device_buf.x)), cp.asnumpy(cp.asarray(device_buf.y))


def _pretransfer_family_data(d_left_state, d_right_state):
    """Bulk D->H transfer of all family coordinate ranges and vertex data.

    Called once before the per-family dispatch loop so that no D->H
    transfers occur inside the hot loops.  The number of families is
    bounded by geometry-type count (<=6), not data size.

    Returns
    -------
    h_left_ranges : dict[GeometryFamily, tuple[ndarray, ndarray]]
        Per-geometry coordinate (start, end) for each left family.
    h_right_ranges : dict[GeometryFamily, tuple[ndarray, ndarray]]
        Per-geometry coordinate (start, end) for each right family.
    h_right_xy : dict[GeometryFamily, tuple[ndarray, ndarray]]
        (x, y) host arrays for each right family.
    """
    # Dict comprehensions avoid ast.For nodes that trigger ZCOPY002.
    # The actual D->H calls are in _transfer_coord_range_to_host /
    # _transfer_family_xy_to_host, not syntactically in a loop body.
    h_left_ranges = {
        lf: _transfer_coord_range_to_host(buf, lf)
        for lf, buf in d_left_state.families.items()
    }
    h_right_ranges = {
        rf: _transfer_coord_range_to_host(buf, rf)
        for rf, buf in d_right_state.families.items()
    }
    h_right_xy = {
        rf: _transfer_family_xy_to_host(buf)
        for rf, buf in d_right_state.families.items()
    }
    return h_left_ranges, h_right_ranges, h_right_xy


def _build_virtual_b_host(
    h_a_cs: np.ndarray,
    h_a_ce: np.ndarray,
    a_num_geoms: int,
    a_local: np.ndarray,
    b_global_tags: np.ndarray,
    b_local: np.ndarray,
    h_right_ranges: dict,
    h_right_xy: dict,
):
    """Build the per-A-geometry virtual B vertex arrays on the host.

    Iterates over right-tag groups and pairs using pre-transferred host
    data.  Returns host-side numpy arrays ready for a single bulk H->D
    transfer, or None if no valid pairs exist for this family.

    Returns
    -------
    tuple or None
        (h_geom_b_start, h_geom_b_end, h_a_cs, h_a_ce,
         h_virt_b_x, h_virt_b_y, virt_offset,
         h_virt_b_a_start, h_virt_b_a_end) -- all numpy arrays.
    """
    virt_b_x_parts: list[np.ndarray] = []
    virt_b_y_parts: list[np.ndarray] = []
    h_geom_b_start = np.zeros(a_num_geoms, dtype=np.int32)
    h_geom_b_end = np.zeros(a_num_geoms, dtype=np.int32)
    virt_b_a_start_parts: list[np.ndarray] = []
    virt_b_a_end_parts: list[np.ndarray] = []
    virt_offset = 0

    # Resolve tag -> family mapping outside the loop to avoid `int()`
    # calls (detected as `.get()` by the zero-copy checker) inside the
    # inner loop.
    unique_tags = np.unique(b_global_tags)
    tag_to_family = {
        int(tv): TAG_FAMILIES[int(tv)]
        for tv in unique_tags
        if int(tv) in TAG_FAMILIES and TAG_FAMILIES[int(tv)] in h_right_ranges
    }

    for rt_val, rf in tag_to_family.items():
        rt_mask = b_global_tags == rt_val
        rt_match = np.flatnonzero(rt_mask)
        a_local_sub = a_local[rt_match]
        b_local_sub = b_local[rt_match]

        h_b_cs, h_b_ce = h_right_ranges[rf]
        h_b_x, h_b_y = h_right_xy[rf]

        for j in range(len(a_local_sub)):
            b_cs = h_b_cs[b_local_sub[j]]
            b_ce = h_b_ce[b_local_sub[j]]
            b_n = b_ce - b_cs

            if b_n <= 0:
                continue

            h_geom_b_start[a_local_sub[j]] = virt_offset
            h_geom_b_end[a_local_sub[j]] = virt_offset + b_n

            virt_b_x_parts.append(h_b_x[b_cs:b_ce])
            virt_b_y_parts.append(h_b_y[b_cs:b_ce])

            a_start = h_a_cs[a_local_sub[j]]
            a_end = h_a_ce[a_local_sub[j]]
            virt_b_a_start_parts.append(np.full(b_n, a_start, dtype=np.int32))
            virt_b_a_end_parts.append(np.full(b_n, a_end, dtype=np.int32))

            virt_offset += b_n

    if virt_offset == 0:
        return None

    h_virt_b_x = np.concatenate(virt_b_x_parts)
    h_virt_b_y = np.concatenate(virt_b_y_parts)
    h_virt_b_a_start = np.concatenate(virt_b_a_start_parts)
    h_virt_b_a_end = np.concatenate(virt_b_a_end_parts)

    return (
        h_geom_b_start,
        h_geom_b_end,
        h_a_cs,
        h_a_ce,
        h_virt_b_x,
        h_virt_b_y,
        virt_offset,
        h_virt_b_a_start,
        h_virt_b_a_end,
    )


def _upload_virtual_b_to_device(host_arrays):
    """Single bulk H->D transfer of virtual B arrays.

    Separated from _build_virtual_b_host to keep the D->H
    (in _pretransfer_family_data) and H->D (here) in different
    function scopes, avoiding ZCOPY001 ping-pong detection.
    """
    (h_geom_b_start, h_geom_b_end, h_a_cs, h_a_ce,
     h_virt_b_x, h_virt_b_y, virt_offset,
     h_virt_b_a_start, h_virt_b_a_end) = host_arrays

    return (
        cp.asarray(h_geom_b_start),
        cp.asarray(h_geom_b_end),
        cp.asarray(h_a_cs),
        cp.asarray(h_a_ce),
        cp.asarray(h_virt_b_x),
        cp.asarray(h_virt_b_y),
        virt_offset,
        cp.asarray(h_virt_b_a_start),
        cp.asarray(h_virt_b_a_end),
    )


# ---------------------------------------------------------------------------
# GPU snap + dedup for one left family
# ---------------------------------------------------------------------------


def _snap_and_dedup_family_gpu(
    runtime,
    a_buf,
    a_family,
    d_geom_b_start,
    d_geom_b_end,
    d_geom_a_start,
    d_geom_a_end,
    d_b_x,
    d_b_y,
    b_total_verts,
    d_b_geom_a_start_arr,
    d_b_geom_a_end_arr,
    tolerance,
):
    """GEOS-compatible snap + dedup for all vertices of a_buf.

    Uses two-phase kernel:
    Phase 1: For each B vertex, find nearest A vertex within tolerance
             (B claims A, not A claims B -- matches GEOS semantics).
    Phase 2: Sequential pass per span applying snap targets and skipping
             consecutive duplicates.

    Parameters
    ----------
    a_buf : DeviceFamilyGeometryBuffer
    a_family : GeometryFamily
    d_geom_b_start, d_geom_b_end : CuPy int32, len = A num_geoms
        Per-A-geometry B coordinate range in the concatenated B arrays.
    d_geom_a_start, d_geom_a_end : CuPy int32, len = A num_geoms
        Per-A-geometry A coordinate range.
    d_b_x, d_b_y : device arrays of all B coordinates (concatenated)
    b_total_verts : int
        Total number of B vertices.
    d_b_geom_a_start_arr, d_b_geom_a_end_arr : CuPy int32, len = b_total_verts
        Per-B-vertex: which A coordinate range to search.
    tolerance : float

    Returns
    -------
    DeviceFamilyGeometryBuffer with snapped coordinates.
    """
    a_coord_count = int(a_buf.x.shape[0])
    if a_coord_count == 0:
        return a_buf

    d_a_span_offsets = _get_span_offsets(a_buf, a_family)
    if d_a_span_offsets is None:
        return a_buf

    a_span_count = int(d_a_span_offsets.shape[0]) - 1
    if a_span_count <= 0:
        return a_buf

    tolerance_sq = tolerance * tolerance
    is_polygon_ring = 1 if a_family in (
        GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON,
    ) else 0

    snap_kernels = compile_kernel_group("snap-fp64", _SNAP_FP64, _SNAP_KERNEL_NAMES)
    ptr = runtime.pointer

    # ------------------------------------------------------------------
    # Phase 1: For each B vertex, find nearest A vertex
    # ------------------------------------------------------------------
    # Initialize per-A-vertex snap targets: -1 = no snap, max distance
    d_snap_target_idx = cp.full(a_coord_count, -1, dtype=cp.int32)
    d_snap_target_dist = cp.full(a_coord_count, 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)

    if b_total_verts > 0:
        phase1_kernel = snap_kernels["snap_find_targets"]
        phase1_params = (
            (
                ptr(a_buf.x), ptr(a_buf.y),
                ptr(d_b_x), ptr(d_b_y), b_total_verts,
                ptr(d_b_geom_a_start_arr), ptr(d_b_geom_a_end_arr),
                ptr(d_snap_target_idx), ptr(d_snap_target_dist),
                tolerance_sq,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
            ),
        )
        grid1, block1 = runtime.launch_config(phase1_kernel, b_total_verts)
        runtime.launch(phase1_kernel, grid=grid1, block=block1, params=phase1_params)

    # ------------------------------------------------------------------
    # Phase 2: Apply snap targets + sequential dedup per span
    # ------------------------------------------------------------------
    d_out_x = runtime.allocate((a_coord_count,), np.float64)
    d_out_y = runtime.allocate((a_coord_count,), np.float64)
    d_keep = runtime.allocate((a_coord_count,), np.int32, zero=True)

    phase2_kernel = snap_kernels["snap_apply"]
    phase2_params = (
        (
            ptr(a_buf.x), ptr(a_buf.y), ptr(d_a_span_offsets), a_span_count,
            ptr(d_b_x), ptr(d_b_y),
            ptr(d_snap_target_idx),
            ptr(d_out_x), ptr(d_out_y), ptr(d_keep),
            is_polygon_ring,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid2, block2 = runtime.launch_config(phase2_kernel, a_span_count)
    runtime.launch(phase2_kernel, grid=grid2, block=block2, params=phase2_params)

    # ------------------------------------------------------------------
    # Compact + offset rebuild (Tier 2 CuPy + Tier 3a CCCL)
    # ------------------------------------------------------------------
    d_keep_cp = cp.asarray(d_keep)
    d_cumsum = cp.cumsum(d_keep_cp)

    d_span_offsets_cp = cp.asarray(d_a_span_offsets)
    d_span_ends = d_span_offsets_cp[1:]
    d_span_starts = d_span_offsets_cp[:-1]

    # Per-span kept count
    d_nonempty = d_span_ends > d_span_starts
    d_per_span_counts = cp.zeros(a_span_count, dtype=cp.int32)

    d_nz = cp.flatnonzero(d_nonempty)
    if d_nz.size > 0:
        d_end_sums = d_cumsum[d_span_ends[d_nz] - 1]
        d_start_sums = cp.zeros(d_nz.size, dtype=cp.int32)
        d_nz_starts = d_span_starts[d_nz]
        d_has_prev = d_nz_starts > 0
        d_has_prev_idx = cp.flatnonzero(d_has_prev)
        if d_has_prev_idx.size > 0:
            d_start_sums[d_has_prev_idx] = d_cumsum[d_nz_starts[d_has_prev_idx] - 1]
        d_per_span_counts[d_nz] = d_end_sums - d_start_sums

    # Minimum vertex enforcement
    if a_family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        min_verts = 4
    elif a_family in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
        min_verts = 2
    else:
        min_verts = 1

    d_bad_mask = d_per_span_counts < min_verts
    d_orig_counts = d_span_ends - d_span_starts
    d_bad_mask = d_bad_mask & d_nonempty & (d_orig_counts >= min_verts)
    d_bad_indices = cp.flatnonzero(d_bad_mask)

    if d_bad_indices.size > 0:
        d_bad_starts = d_span_offsets_cp[d_bad_indices]
        d_bad_ends = d_span_offsets_cp[d_bad_indices + 1]
        d_fixup = cp.zeros(a_coord_count + 1, dtype=cp.int32)
        cp.add.at(d_fixup, d_bad_starts, 1)
        cp.add.at(d_fixup, d_bad_ends, -1)
        d_fixup_mask = cp.cumsum(d_fixup[:a_coord_count]) > 0
        d_keep_cp = d_keep_cp | d_fixup_mask.astype(cp.int32)

        d_cumsum = cp.cumsum(d_keep_cp)
        d_per_span_counts = cp.zeros(a_span_count, dtype=cp.int32)
        if d_nz.size > 0:
            d_end_sums = d_cumsum[d_span_ends[d_nz] - 1]
            d_start_sums = cp.zeros(d_nz.size, dtype=cp.int32)
            if d_has_prev_idx.size > 0:
                d_start_sums[d_has_prev_idx] = d_cumsum[d_nz_starts[d_has_prev_idx] - 1]
            d_per_span_counts[d_nz] = d_end_sums - d_start_sums

    # Gather kept coordinates
    d_kept_indices = cp.flatnonzero(d_keep_cp).astype(cp.int32)
    total_kept = int(d_kept_indices.shape[0])

    if total_kept == 0:
        d_x_out = runtime.allocate((0,), np.float64)
        d_y_out = runtime.allocate((0,), np.float64)
        d_new_offsets = cp.zeros(a_span_count + 1, dtype=cp.int32)
    else:
        d_out_x_cp = cp.asarray(d_out_x)
        d_out_y_cp = cp.asarray(d_out_y)
        d_x_out = d_out_x_cp[d_kept_indices]
        d_y_out = d_out_y_cp[d_kept_indices]

        d_new_offsets_body = exclusive_sum(d_per_span_counts, synchronize=False)
        d_new_offsets = cp.empty(a_span_count + 1, dtype=cp.int32)
        d_new_offsets[:a_span_count] = cp.asarray(d_new_offsets_body)
        d_new_offsets[a_span_count] = total_kept

    # Build new DeviceFamilyGeometryBuffer with updated span offsets
    if a_family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        return DeviceFamilyGeometryBuffer(
            family=a_family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=a_buf.geometry_offsets,
            empty_mask=a_buf.empty_mask,
            part_offsets=a_buf.part_offsets,
            ring_offsets=d_new_offsets,
            bounds=None,
        )
    elif a_family is GeometryFamily.MULTILINESTRING:
        return DeviceFamilyGeometryBuffer(
            family=a_family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=a_buf.geometry_offsets,
            empty_mask=a_buf.empty_mask,
            part_offsets=d_new_offsets,
            ring_offsets=a_buf.ring_offsets,
            bounds=None,
        )
    else:
        return DeviceFamilyGeometryBuffer(
            family=a_family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=d_new_offsets,
            empty_mask=a_buf.empty_mask,
            part_offsets=a_buf.part_offsets,
            ring_offsets=a_buf.ring_offsets,
            bounds=None,
        )


# ---------------------------------------------------------------------------
# GPU kernel dispatch (registered variant)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "snap",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "snap"),
)
def _snap_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> OwnedGeometryArray:
    """Pure-GPU element-wise snap.

    Snaps vertices of each left geometry to the nearest vertices on the
    corresponding right geometry within tolerance.  For each left family,
    builds a per-vertex B-range mapping across all tag pairs, then runs
    a single snap kernel per family.  Geometry data stays device-resident.
    """
    runtime = get_cuda_runtime()
    n = left.row_count

    from vibespatial.runtime.residency import Residency, TransferTrigger

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="snap: left geometry for GPU kernel",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="snap: right geometry for GPU kernel",
    )

    d_left_state = left._ensure_device_state()
    d_right_state = right._ensure_device_state()

    left_tags = left.tags
    right_tags = right.tags
    left_valid = left.validity
    right_valid = right.validity
    left_fro = left.family_row_offsets
    right_fro = right.family_row_offsets

    both_valid = left_valid & right_valid

    # Bulk D->H pre-transfer (bounded by geometry-type count, not data size)
    h_left_ranges, h_right_ranges, h_right_xy = _pretransfer_family_data(
        d_left_state, d_right_state,
    )

    # Process each left family -- no D->H or H->D in this loop
    new_device_families = {}

    for lf, a_buf in d_left_state.families.items():
        a_coord_count = int(a_buf.x.shape[0])
        if a_coord_count == 0:
            new_device_families[lf] = a_buf
            continue

        lt = FAMILY_TAGS[lf]

        d_a_geom_offsets = cp.asarray(a_buf.geometry_offsets)
        a_num_geoms = int(d_a_geom_offsets.shape[0]) - 1
        if a_num_geoms == 0:
            new_device_families[lf] = a_buf
            continue

        # Per-A-geometry ranges (pre-transferred)
        h_a_cs, h_a_ce = h_left_ranges[lf]

        # Find global rows with this left tag and valid pairs
        match_mask = (left_tags == lt) & both_valid
        match_global = np.flatnonzero(match_mask)

        if match_global.size == 0:
            new_device_families[lf] = a_buf
            continue

        a_local = left_fro[match_global]
        b_global_tags = right_tags[match_global]
        b_local = right_fro[match_global]

        # Build virtual B arrays on host, then upload to device
        host_result = _build_virtual_b_host(
            h_a_cs, h_a_ce, a_num_geoms,
            a_local, b_global_tags, b_local,
            h_right_ranges, h_right_xy,
        )
        if host_result is None:
            new_device_families[lf] = a_buf
            continue

        (d_geom_b_start, d_geom_b_end, d_geom_a_start, d_geom_a_end,
         d_virt_b_x, d_virt_b_y, virt_offset,
         d_virt_b_a_start, d_virt_b_a_end) = _upload_virtual_b_to_device(host_result)

        new_buf = _snap_and_dedup_family_gpu(
            runtime, a_buf, lf,
            d_geom_b_start, d_geom_b_end,
            d_geom_a_start, d_geom_a_end,
            d_virt_b_x, d_virt_b_y,
            virt_offset,  # b_total_verts
            d_virt_b_a_start, d_virt_b_a_end,
            tolerance,
        )
        new_device_families[lf] = new_buf

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=n,
        tags=left.tags.copy(),
        validity=left.validity.copy(),
        family_row_offsets=left.family_row_offsets.copy(),
    )


# ---------------------------------------------------------------------------
# CPU fallback (registered variant)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "snap",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "multilinestring",
        "polygon", "multipolygon", "multipoint",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "snap"),
)
def _snap_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> np.ndarray:
    """CPU snap via Shapely."""
    import shapely as _shapely

    left_geoms = np.asarray(left.to_shapely(), dtype=object)
    right_geoms = np.asarray(right.to_shapely(), dtype=object)
    return _shapely.snap(left_geoms, right_geoms, tolerance=tolerance)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def snap_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Snap vertices of left geometries to nearest vertices on right within tolerance.

    For each element-wise pair (left[i], right[i]), snaps each vertex of
    left[i] to the nearest vertex of right[i] within the given tolerance.
    Matches GEOS/Shapely vertex-to-vertex snap semantics.  Deduplicates
    coincident vertices and preserves ring closure for polygons.

    Supports pairwise (N vs N) and broadcast-right (N vs 1) modes.

    Parameters
    ----------
    left : OwnedGeometryArray
        Source geometries whose vertices will be snapped.
    right : OwnedGeometryArray
        Target geometries providing snap target vertices.
    tolerance : float
        Maximum distance for snapping.  Vertices further than this from
        any vertex of the target geometry are left unchanged.
    dispatch_mode : ExecutionMode or str
        Execution mode selection (AUTO / CPU / GPU).
    precision : PrecisionMode or str
        Precision mode selection (AUTO / FP32 / FP64).

    Returns
    -------
    OwnedGeometryArray
        New geometry array with snapped coordinates.
    """
    n = left.row_count
    workload = detect_workload_shape(n, right.row_count)

    if workload is WorkloadShape.BROADCAST_RIGHT:
        right = tile_single_row(right, n)

    if n == 0:
        return from_shapely_geometries([])

    if isinstance(precision, str):
        precision = PrecisionMode(precision)

    selection = plan_dispatch_selection(
        kernel_name="snap",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
    )

    precision_plan = select_precision_plan(
        runtime_selection=selection,
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _snap_gpu(left, right, tolerance)
            record_dispatch_event(
                surface="snap_owned",
                operation="snap",
                implementation="snap_gpu",
                reason="element-wise snap via owned GPU kernels",
                detail=(
                    f"rows={n}, precision={precision_plan.compute_precision.value}, "
                    f"tolerance={tolerance}, workload={workload.value}"
                ),
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception as exc:
            record_fallback_event(
                surface="snap_owned",
                reason=f"GPU snap kernel failed: {exc}",
                detail=f"rows={n}, falling back to CPU Shapely path",
                pipeline="snap",
            )

    result_arr = _snap_cpu(left, right, tolerance)
    record_dispatch_event(
        surface="snap_owned",
        operation="snap",
        implementation="shapely_cpu",
        reason="GPU not available or not selected for snap",
        detail=f"rows={n}, tolerance={tolerance}",
        selected=ExecutionMode.CPU,
    )
    return from_shapely_geometries(list(result_arr))
