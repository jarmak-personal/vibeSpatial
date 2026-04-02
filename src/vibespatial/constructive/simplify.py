"""GPU-accelerated geometry simplification (Visvalingam-Whyatt).

Visvalingam-Whyatt is more GPU-friendly than Douglas-Peucker because it
avoids recursion. Per-coordinate effective area computation, iterative
elimination of vertices below the tolerance.

For LineString/MultiLineString: simplify coordinate sequences.
For Polygon/MultiPolygon: simplify each ring, preserving closure.
Points/MultiPoints pass through unchanged.

ADR-0033: Tier 1 NVRTC for per-coordinate area computation.
ADR-0002: COARSE class — simplification tolerance is user-specified.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.simplify_cpu import _simplify_cpu
from vibespatial.constructive.simplify_kernels import _VW_AREA_FP64, _VW_AREA_KERNEL_NAMES
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    build_device_resident_owned,
    build_updated_device_family_buffer,
    build_updated_host_family_buffer,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

logger = logging.getLogger(__name__)

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("vw-area-fp64", _VW_AREA_FP64, _VW_AREA_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# GPU implementation: single-pass area-threshold simplification
# ---------------------------------------------------------------------------

def _simplify_family_gpu(runtime, device_buf, family, tolerance):
    """GPU single-pass simplify for one family.

    Unlike the iterative CPU VW algorithm (remove min-area vertex, recompute,
    repeat), the GPU path uses a single-pass area threshold approximation:
    compute per-vertex effective area once, then keep all vertices whose area
    exceeds tolerance**2.  This is correct for ``preserve_topology=False``.

    Returns a new :class:`DeviceFamilyGeometryBuffer` with simplified
    coordinates and updated span-level offsets.
    """
    from vibespatial.cuda.cccl_primitives import exclusive_sum

    # 1. Determine span offsets for this family
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        d_span_offsets = device_buf.ring_offsets
    elif family is GeometryFamily.MULTILINESTRING:
        d_span_offsets = device_buf.part_offsets
    else:  # LineString
        d_span_offsets = device_buf.geometry_offsets

    if d_span_offsets is None:
        return device_buf

    span_count = int(d_span_offsets.shape[0]) - 1
    if span_count <= 0:
        return device_buf

    coord_count = int(device_buf.x.shape[0])
    if coord_count == 0:
        return device_buf

    # 2. Launch VW effective area kernel (Tier 1 NVRTC, 1 thread per span)
    kernels = compile_kernel_group(
        "vw-area-fp64", _VW_AREA_FP64, _VW_AREA_KERNEL_NAMES,
    )
    kernel = kernels["vw_effective_area"]

    d_areas = runtime.allocate((coord_count,), np.float64)

    ptr = runtime.pointer
    params = (
        (
            ptr(device_buf.x), ptr(device_buf.y), ptr(d_span_offsets),
            ptr(d_areas), 0.0, 0.0, span_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, span_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # 3. Build keep mask: area >= tolerance**2  (Tier 2 CuPy boolean)
    area_threshold = tolerance * tolerance
    d_areas_cp = cp.asarray(d_areas)
    d_keep = d_areas_cp >= area_threshold  # boolean CuPy array

    # 4. Force first/last of each span to keep=True  (all on device, no D→H)
    d_span_offsets_cp = cp.asarray(d_span_offsets)
    d_starts = d_span_offsets_cp[:-1]
    d_ends = d_span_offsets_cp[1:] - 1
    valid = d_ends >= d_starts
    d_keep[d_starts[valid]] = True
    d_keep[d_ends[valid]] = True

    # 5. Compute per-span keep counts in bulk (Tier 2 CuPy cumsum + diff)
    #    All operations stay on device -- no host round-trips.
    d_keep_int = d_keep.astype(cp.int32)
    d_cumsum = cp.cumsum(d_keep_int)

    d_span_ends_cp = d_span_offsets_cp[1:]
    d_span_starts_cp = d_starts  # reuse from step 4

    d_end_sums = d_cumsum[d_span_ends_cp - 1]

    # For start sums: where start > 0, read cumsum[start - 1]; else 0
    d_start_sums = cp.zeros(span_count, dtype=cp.int32)
    d_nonzero_mask = d_span_starts_cp > 0
    d_nz_indices = cp.flatnonzero(d_nonzero_mask)
    if d_nz_indices.size > 0:
        d_start_sums[d_nz_indices] = d_cumsum[
            d_span_starts_cp[d_nz_indices] - 1
        ]
    d_per_span_counts = d_end_sums - d_start_sums

    # 6. For polygon rings: ensure >= 4 kept vertices per span
    #    Device-side check and fixup via cumsum scatter (no Python loops).
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        d_bad_mask = d_per_span_counts < 4
        d_bad_indices = cp.flatnonzero(d_bad_mask)

        if d_bad_indices.size > 0:
            # Mark all coordinates within bad spans using cumsum scatter:
            # +1 at each bad span start, -1 at each bad span end, cumsum > 0 = keep.
            d_bad_starts = d_span_offsets_cp[d_bad_indices]
            d_bad_ends = d_span_offsets_cp[d_bad_indices + 1]
            d_fixup = cp.zeros(coord_count + 1, dtype=cp.int32)
            cp.add.at(d_fixup, d_bad_starts, 1)
            cp.add.at(d_fixup, d_bad_ends, -1)
            d_fixup_mask = cp.cumsum(d_fixup[:coord_count]) > 0
            d_keep = d_keep | d_fixup_mask

            # Recompute per-span counts since mask changed
            d_keep_int = d_keep.astype(cp.int32)
            d_cumsum = cp.cumsum(d_keep_int)
            d_end_sums = d_cumsum[d_span_ends_cp - 1]
            d_start_sums = cp.zeros(span_count, dtype=cp.int32)
            if d_nz_indices.size > 0:
                d_start_sums[d_nz_indices] = d_cumsum[
                    d_span_starts_cp[d_nz_indices] - 1
                ]
            d_per_span_counts = d_end_sums - d_start_sums

    # 7. Gather kept coordinates (Tier 2 CuPy fancy indexing)
    d_kept_indices = cp.flatnonzero(d_keep).astype(cp.int32)
    total_kept = int(d_kept_indices.shape[0])

    if total_kept == 0:
        d_x_out = runtime.allocate((0,), np.float64)
        d_y_out = runtime.allocate((0,), np.float64)
        d_new_offsets = cp.zeros(span_count + 1, dtype=cp.int32)
    else:
        d_x_cp = cp.asarray(device_buf.x)
        d_y_cp = cp.asarray(device_buf.y)
        d_x_out = d_x_cp[d_kept_indices]
        d_y_out = d_y_cp[d_kept_indices]

        # 8. Recompute span offsets via exclusive_sum of per-span keep counts
        d_new_offsets_body = exclusive_sum(d_per_span_counts, synchronize=False)
        d_new_offsets = cp.empty(span_count + 1, dtype=cp.int32)
        d_new_offsets[:span_count] = cp.asarray(d_new_offsets_body)
        d_new_offsets[span_count] = total_kept

    return build_updated_device_family_buffer(
        family,
        device_buf,
        d_x_out,
        d_y_out,
        d_new_offsets,
    )


@register_kernel_variant(
    "geometry_simplify",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "simplify"),
)
def _simplify_gpu(owned, tolerance):
    """GPU simplify -- returns device-resident OwnedGeometryArray.

    Uses a single-pass area threshold (NOT iterative VW) which is the
    GPU-friendly approximation for preserve_topology=False.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            new_device_families[family] = device_buf
            continue
        new_device_families[family] = _simplify_family_gpu(
            runtime, device_buf, family, tolerance,
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=owned.tags.copy(),
        validity=owned.validity.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
    )


# ---------------------------------------------------------------------------
# CPU Visvalingam-Whyatt implementation
# ---------------------------------------------------------------------------

def _vw_simplify_span(x, y, start, end, tolerance, preserve_topology=True):
    """Visvalingam-Whyatt simplification for one coordinate span.

    Returns (x_out, y_out) arrays of kept coordinates.
    """
    n = end - start
    if n <= 2:
        return x[start:end].copy(), y[start:end].copy()

    area_threshold = tolerance * tolerance  # squared tolerance as area proxy

    # Compute initial effective areas
    coords_x = x[start:end].copy()
    coords_y = y[start:end].copy()
    keep = np.ones(n, dtype=bool)

    # Iterative elimination
    while True:
        indices = np.flatnonzero(keep)
        if len(indices) <= 2:
            break

        # Compute areas for interior vertices
        min_area = np.inf
        min_idx = -1
        for j in range(1, len(indices) - 1):
            i = indices[j]
            prev_i = indices[j - 1]
            next_i = indices[j + 1]
            ax = coords_x[i] - coords_x[prev_i]
            ay = coords_y[i] - coords_y[prev_i]
            bx = coords_x[next_i] - coords_x[prev_i]
            by = coords_y[next_i] - coords_y[prev_i]
            area = abs(ax * by - ay * bx) * 0.5
            if area < min_area:
                min_area = area
                min_idx = i

        if min_area >= area_threshold:
            break

        keep[min_idx] = False

    return coords_x[keep], coords_y[keep]


def _simplify_family_cpu(buf, family, tolerance, preserve_topology=True):
    """Simplify one family's geometries on CPU."""
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        return buf  # Points don't simplify

    # Determine which offsets define the spans to simplify
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        span_offsets = buf.ring_offsets
    elif family is GeometryFamily.MULTILINESTRING:
        span_offsets = buf.part_offsets
    else:
        span_offsets = buf.geometry_offsets

    if span_offsets is None:
        return buf

    span_count = len(span_offsets) - 1
    new_x_parts = []
    new_y_parts = []
    new_span_offsets = [0]

    for s in range(span_count):
        start = int(span_offsets[s])
        end = int(span_offsets[s + 1])
        sx, sy = _vw_simplify_span(buf.x, buf.y, start, end, tolerance, preserve_topology)

        # For polygons, ensure at least 4 coordinates (closed ring)
        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON) and len(sx) < 4:
            sx = buf.x[start:end].copy()
            sy = buf.y[start:end].copy()

        new_x_parts.append(sx)
        new_y_parts.append(sy)
        new_span_offsets.append(new_span_offsets[-1] + len(sx))

    new_x = np.concatenate(new_x_parts) if new_x_parts else np.empty(0, dtype=np.float64)
    new_y = np.concatenate(new_y_parts) if new_y_parts else np.empty(0, dtype=np.float64)
    new_span_offsets = np.asarray(new_span_offsets, dtype=np.int32)

    return build_updated_host_family_buffer(
        family=family,
        host_buf=buf,
        x_out=new_x,
        y_out=new_y,
        new_offsets=new_span_offsets,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def simplify_owned(
    owned: OwnedGeometryArray,
    tolerance: float,
    *,
    preserve_topology: bool = True,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Simplify geometries using Visvalingam-Whyatt algorithm.

    Parameters
    ----------
    tolerance : float
        Simplification tolerance. Vertices with effective area below
        tolerance^2 are removed.
    preserve_topology : bool
        If True, ensures simplified geometries remain valid.
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO / CPU / GPU).
    precision : PrecisionMode
        Precision mode selection (AUTO / FP32 / FP64).

    Returns
    -------
    OwnedGeometryArray
        New geometry array with simplified coordinates.
    """
    row_count = owned.row_count
    if row_count == 0:
        return owned

    if tolerance <= 0:
        return owned  # R7 provenance rewrite: simplify(0) = identity

    selection = plan_dispatch_selection(
        kernel_name="geometry_simplify",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        # Precision plan computed for future fp32 kernel wiring (ADR-0002 COARSE).
        # The GPU kernel currently runs fp64-only; the plan is not yet consumed.
        _precision_plan = selection.precision_plan
        try:
            result = _simplify_gpu(owned, tolerance)
        except Exception:
            logger.warning("GPU simplify failed, falling back to CPU", exc_info=True)
        else:
            record_dispatch_event(
                surface="geopandas.array.simplify",
                operation="simplify",
                implementation="gpu_nvrtc_vw_simplify",
                reason="GPU NVRTC Visvalingam-Whyatt simplify kernel",
                detail=f"rows={row_count}, precision=fp64",
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _simplify_cpu(owned, tolerance, preserve_topology)
    record_dispatch_event(
        surface="geopandas.array.simplify",
        operation="simplify",
        implementation="cpu_vw_simplify",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return result
