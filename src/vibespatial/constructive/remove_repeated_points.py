"""GPU-accelerated removal of repeated (consecutive duplicate) points.

Per ring/linestring (sequential within, parallel across):
1. For each coordinate, compute squared distance to the previous *kept* coordinate
2. Mark as "keep" if distance >= tolerance^2 (or if it is the first point)
3. Special case: for Polygon rings, always keep the last point if it closes
   the ring (matches first point)
4. Compact kept coordinates using CuPy fancy indexing
5. Rebuild geometry offsets from per-ring/per-linestring kept counts

ADR-0033: Tier 1 NVRTC for per-ring scan (sequential dependency: compare to
  previous *kept* point, so one thread per ring/linestring).
  Tier 3a CCCL for exclusive_sum on per-span kept counts.
  Tier 2 CuPy for gather (compact).
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
  PrecisionPlan wired through for observability only.
ADR-0034: NVRTC precompilation via request_nvrtc_warmup at module scope.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

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
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC kernel source: mark keep/remove flags per ring
# ---------------------------------------------------------------------------
# One thread per ring/linestring.  Sequential scan within each span:
# compare each coordinate to the previous *kept* coordinate.
#
# Ring closure: the last point is always kept if it equals the first
# point (preserves polygon ring closure and linestring minimum count).

_REMOVE_REPEATED_KERNEL_SOURCE = r"""
extern "C" __global__ void remove_repeated_points_mark(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    int* __restrict__ keep_flags,
    double tolerance_sq,
    int span_count
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n = end - start;

    if (n <= 0) return;

    /* Always keep the first point */
    keep_flags[start] = 1;
    double last_kept_x = x[start];
    double last_kept_y = y[start];

    if (n == 1) return;

    /* Sequential scan: compare to previous *kept* point */
    for (int i = start + 1; i < end; i++) {{
        const double dx = x[i] - last_kept_x;
        const double dy = y[i] - last_kept_y;
        if (dx * dx + dy * dy > tolerance_sq) {{
            keep_flags[i] = 1;
            last_kept_x = x[i];
            last_kept_y = y[i];
        }} else {{
            keep_flags[i] = 0;
        }}
    }}

    /* For polygon rings: always keep the last point if it equals the
       first point (ring closure preservation). For linestrings, the
       minimum vertex count (>= 2) is enforced by the host-side fixup. */
    if (n >= 2) {{
        const double fx = x[start];
        const double fy = y[start];
        const double lx = x[end - 1];
        const double ly = y[end - 1];
        if (fx == lx && fy == ly) {{
            keep_flags[end - 1] = 1;
        }}
    }}
}}
"""

_REMOVE_REPEATED_KERNEL_NAMES = ("remove_repeated_points_mark",)
_REMOVE_REPEATED_FP64 = _REMOVE_REPEATED_KERNEL_SOURCE.format()

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("remove-repeated-fp64", _REMOVE_REPEATED_FP64, _REMOVE_REPEATED_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# GPU implementation: per-family remove_repeated_points
# ---------------------------------------------------------------------------

def _remove_repeated_family_gpu(runtime, device_buf, family, tolerance):
    """GPU remove_repeated_points for one family.

    One NVRTC kernel marks keep/remove flags per span (one thread per span),
    then CuPy gathers kept coordinates and CCCL exclusive_sum rebuilds offsets.

    Returns a new DeviceFamilyGeometryBuffer with deduplicated coordinates
    and updated span-level offsets.
    """
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

    # 2. Launch mark kernel (Tier 1 NVRTC, 1 thread per span)
    kernels = compile_kernel_group(
        "remove-repeated-fp64", _REMOVE_REPEATED_FP64, _REMOVE_REPEATED_KERNEL_NAMES,
    )
    kernel = kernels["remove_repeated_points_mark"]

    d_keep = runtime.allocate((coord_count,), np.int32, zero=True)

    tolerance_sq = tolerance * tolerance

    ptr = runtime.pointer
    params = (
        (
            ptr(device_buf.x), ptr(device_buf.y), ptr(d_span_offsets),
            ptr(d_keep), tolerance_sq, span_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, span_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # 3. Compute per-span keep counts in bulk (Tier 2 CuPy cumsum + diff)
    #    All operations stay on device -- no host round-trips.
    d_keep_cp = cp.asarray(d_keep)
    d_cumsum = cp.cumsum(d_keep_cp)

    d_span_offsets_cp = cp.asarray(d_span_offsets)
    d_span_ends_cp = d_span_offsets_cp[1:]
    d_span_starts_cp = d_span_offsets_cp[:-1]

    # Per-span kept count = cumsum[end-1] - cumsum[start-1]  (with start=0 special case)
    # Handle empty spans: where start == end, count is 0
    d_nonempty = d_span_ends_cp > d_span_starts_cp
    d_per_span_counts = cp.zeros(span_count, dtype=cp.int32)

    d_nz = cp.flatnonzero(d_nonempty)
    if d_nz.size > 0:
        d_end_sums = d_cumsum[d_span_ends_cp[d_nz] - 1]
        d_start_sums = cp.zeros(d_nz.size, dtype=cp.int32)
        d_nz_starts = d_span_starts_cp[d_nz]
        d_has_prev = d_nz_starts > 0
        d_has_prev_idx = cp.flatnonzero(d_has_prev)
        if d_has_prev_idx.size > 0:
            d_start_sums[d_has_prev_idx] = d_cumsum[d_nz_starts[d_has_prev_idx] - 1]
        d_per_span_counts[d_nz] = d_end_sums - d_start_sums

    # 4. Minimum vertex enforcement:
    #    - Polygon rings: >= 4 kept vertices (triangle + closure)
    #    - LineString / MultiLineString spans: >= 2 kept vertices
    #    If a span has fewer than the minimum, keep all its original points.
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        min_verts = 4
    else:
        min_verts = 2

    d_bad_mask = d_per_span_counts < min_verts
    # Only apply to non-empty spans that originally had enough points
    d_orig_counts = d_span_ends_cp - d_span_starts_cp
    d_bad_mask = d_bad_mask & d_nonempty & (d_orig_counts >= min_verts)
    d_bad_indices = cp.flatnonzero(d_bad_mask)

    if d_bad_indices.size > 0:
        # Mark all coordinates within bad spans as keep
        d_bad_starts = d_span_offsets_cp[d_bad_indices]
        d_bad_ends = d_span_offsets_cp[d_bad_indices + 1]
        d_fixup = cp.zeros(coord_count + 1, dtype=cp.int32)
        cp.add.at(d_fixup, d_bad_starts, 1)
        cp.add.at(d_fixup, d_bad_ends, -1)
        d_fixup_mask = cp.cumsum(d_fixup[:coord_count]) > 0
        d_keep_cp = d_keep_cp | d_fixup_mask.astype(cp.int32)

        # Recompute per-span counts since mask changed
        d_cumsum = cp.cumsum(d_keep_cp)
        d_per_span_counts = cp.zeros(span_count, dtype=cp.int32)
        if d_nz.size > 0:
            d_end_sums = d_cumsum[d_span_ends_cp[d_nz] - 1]
            d_start_sums = cp.zeros(d_nz.size, dtype=cp.int32)
            if d_has_prev_idx.size > 0:
                d_start_sums[d_has_prev_idx] = d_cumsum[d_nz_starts[d_has_prev_idx] - 1]
            d_per_span_counts[d_nz] = d_end_sums - d_start_sums

    # 5. Gather kept coordinates (Tier 2 CuPy fancy indexing)
    d_kept_indices = cp.flatnonzero(d_keep_cp).astype(cp.int32)
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

        # 6. Recompute span offsets via exclusive_sum of per-span keep counts
        d_new_offsets_body = exclusive_sum(d_per_span_counts, synchronize=False)
        d_new_offsets = cp.empty(span_count + 1, dtype=cp.int32)
        d_new_offsets[:span_count] = cp.asarray(d_new_offsets_body)
        d_new_offsets[span_count] = total_kept

    # 7. Build DeviceFamilyGeometryBuffer with updated span offsets
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=d_new_offsets,
            bounds=None,
        )
    elif family is GeometryFamily.MULTILINESTRING:
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=d_new_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=None,
        )
    else:  # LineString
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=d_new_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=None,
        )


# ---------------------------------------------------------------------------
# GPU dispatch variant (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "remove_repeated_points",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "remove_repeated_points"),
)
def _remove_repeated_points_gpu(owned, tolerance):
    """GPU remove_repeated_points -- returns device-resident OwnedGeometryArray.

    Uses NVRTC kernel for per-ring/linestring sequential duplicate marking,
    CuPy for coordinate compaction, CCCL for offset rebuild.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            # Points: single coordinate, nothing to remove
            new_device_families[family] = device_buf
            continue
        new_device_families[family] = _remove_repeated_family_gpu(
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
# CPU dispatch variant (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "remove_repeated_points",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "linestring", "multilinestring",
        "polygon", "multipolygon", "multipoint",
    ),
    supports_mixed=True,
    tags=("cpu", "shapely", "remove_repeated_points"),
)
def _remove_repeated_points_cpu(owned, tolerance):
    """CPU remove_repeated_points via Shapely."""
    import shapely as _shapely

    geoms = owned.to_shapely()
    geom_array = np.asarray(geoms, dtype=object)
    results = _shapely.remove_repeated_points(geom_array, tolerance=tolerance)
    return from_shapely_geometries(list(results))


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def remove_repeated_points_owned(
    owned: OwnedGeometryArray,
    tolerance: float = 0.0,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Remove repeated (consecutive duplicate) points from geometries.

    For each ring/linestring, sequentially scans coordinates and removes
    any point within ``tolerance`` distance of the previous *kept* point.
    Always preserves ring closure for polygons.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    tolerance : float
        Distance threshold. Points closer than this to the previous kept
        point are removed. Default 0.0 removes exact duplicates.
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO / CPU / GPU).
    precision : PrecisionMode
        Precision mode selection (AUTO / FP32 / FP64).

    Returns
    -------
    OwnedGeometryArray
        New geometry array with repeated points removed.
    """
    row_count = owned.row_count
    if row_count == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="remove_repeated_points",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        # Precision plan computed for observability (ADR-0002 CONSTRUCTIVE -> fp64).
        _precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        try:
            result = _remove_repeated_points_gpu(owned, tolerance)
        except Exception:
            logger.warning(
                "GPU remove_repeated_points failed, falling back to CPU",
                exc_info=True,
            )
        else:
            record_dispatch_event(
                surface="geopandas.array.remove_repeated_points",
                operation="remove_repeated_points",
                implementation="gpu_nvrtc_remove_repeated",
                reason="GPU NVRTC per-ring sequential dedup kernel",
                detail=f"rows={row_count}, tolerance={tolerance}, precision=fp64",
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _remove_repeated_points_cpu(owned, tolerance)
    record_dispatch_event(
        surface="geopandas.array.remove_repeated_points",
        operation="remove_repeated_points",
        implementation="cpu_shapely",
        reason="CPU fallback via shapely.remove_repeated_points",
        detail=f"rows={row_count}, tolerance={tolerance}",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return result
