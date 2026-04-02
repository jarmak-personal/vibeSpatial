"""GPU-accelerated convex hull computation for all geometry types.

Uses Andrew's monotone chain algorithm to compute the convex hull of each
geometry.  Output is always Polygon family: each row produces a closed polygon
whose exterior ring is the convex hull of the input geometry's coordinates.

Degenerate cases:
  - Single point -> degenerate polygon (point repeated 4 times to form a
    closed ring).
  - Collinear / <3 unique points -> degenerate polygon from the convex hull
    of those points (line segment repeated to close, or single point).

ADR-0033: Tier 1 NVRTC for monotone chain (geometry-specific inner loop),
          Tier 3a CCCL segmented_sort for per-geometry x-sort,
          Tier 3a CCCL exclusive_sum for output offset computation.
ADR-0002: COARSE class -- hull vertices are exact coordinate subsets,
          no new coordinates created.  Stays fp64 on all devices since
          no compute-heavy accumulation occurs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.convex_hull_kernels import (
    _CONVEX_HULL_FP64,
    _CONVEX_HULL_KERNEL_NAMES,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034): register kernels for background precompilation
# ---------------------------------------------------------------------------
from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("convex-hull-fp64", _CONVEX_HULL_FP64, _CONVEX_HULL_KERNEL_NAMES),
])
request_warmup(["exclusive_scan_i32", "segmented_sort_asc_f64"])


# ---------------------------------------------------------------------------
# CPU implementation: Andrew's monotone chain convex hull
# ---------------------------------------------------------------------------

def _cross(ox, oy, ax, ay, bx, by):
    """2D cross product of vectors OA and OB."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _convex_hull_geometry_cpu(x, y, start, end):
    """Compute convex hull of coordinates x[start:end], y[start:end].

    Uses Andrew's monotone chain algorithm.

    Returns
    -------
    hull_x, hull_y : np.ndarray
        Coordinates of the convex hull as a closed ring (first == last).
        For degenerate inputs (0-2 unique points), returns a degenerate
        closed polygon.
    """
    n = end - start
    if n == 0:
        # Empty geometry -> degenerate single-point polygon at origin.
        return np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])

    px = x[start:end]
    py = y[start:end]

    # De-duplicate and sort by (x, y) lexicographically.
    coords = np.column_stack((px, py))
    unique_coords = np.unique(coords, axis=0)
    nu = len(unique_coords)

    if nu == 1:
        # Single unique point -> degenerate polygon (point repeated).
        cx, cy = unique_coords[0]
        return (
            np.array([cx, cx, cx, cx], dtype=np.float64),
            np.array([cy, cy, cy, cy], dtype=np.float64),
        )

    if nu == 2:
        # Two unique points -> degenerate polygon (line segment, closed).
        x0, y0 = unique_coords[0]
        x1, y1 = unique_coords[1]
        return (
            np.array([x0, x1, x1, x0, x0], dtype=np.float64),
            np.array([y0, y1, y1, y0, y0], dtype=np.float64),
        )

    # Sort by x, then by y (lexicographic).
    order = np.lexsort((unique_coords[:, 1], unique_coords[:, 0]))
    pts = unique_coords[order]

    # Build lower hull.
    lower = []
    for i in range(nu):
        while len(lower) >= 2 and _cross(
            lower[-2][0], lower[-2][1],
            lower[-1][0], lower[-1][1],
            pts[i, 0], pts[i, 1],
        ) <= 0:
            lower.pop()
        lower.append(pts[i])

    # Build upper hull.
    upper = []
    for i in range(nu - 1, -1, -1):
        while len(upper) >= 2 and _cross(
            upper[-2][0], upper[-2][1],
            upper[-1][0], upper[-1][1],
            pts[i, 0], pts[i, 1],
        ) <= 0:
            upper.pop()
        upper.append(pts[i])

    # Concatenate lower and upper hulls, removing duplicate endpoints.
    hull = lower[:-1] + upper[:-1]

    # Close the ring.
    hull.append(hull[0])

    hull_arr = np.array(hull, dtype=np.float64)
    return hull_arr[:, 0].copy(), hull_arr[:, 1].copy()


def _convex_hull_family_cpu(buf, family):
    """Compute convex hull for one family's geometries on CPU.

    Parameters
    ----------
    buf : FamilyGeometryBuffer
        Family buffer containing the source geometry coordinates.
    family : GeometryFamily
        The geometry family of the source buffer.

    Returns
    -------
    FamilyGeometryBuffer
        Polygon-family buffer containing one convex hull polygon per row.
    """
    row_count = buf.row_count
    if row_count == 0:
        return FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=0,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.array([0], dtype=np.int32),
            empty_mask=np.zeros(0, dtype=bool),
            ring_offsets=np.array([0], dtype=np.int32),
            bounds=None,
        )

    # Determine coordinate spans for each geometry row.
    # For all families, we gather all coordinates belonging to each row.
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        # Polygons: gather all coords across all rings.
        # geometry_offsets -> ring indices, ring_offsets -> coord indices.
        geom_off = buf.geometry_offsets
        ring_off = buf.ring_offsets
        coord_spans = []
        for r in range(row_count):
            first_ring = geom_off[r]
            last_ring = geom_off[r + 1]
            if first_ring == last_ring:
                coord_spans.append((0, 0))
            else:
                cs = ring_off[first_ring]
                ce = ring_off[last_ring]
                coord_spans.append((int(cs), int(ce)))
    elif family in (GeometryFamily.MULTILINESTRING,):
        # MultiLineString: gather all coords across all parts.
        geom_off = buf.geometry_offsets
        part_off = buf.part_offsets
        coord_spans = []
        for r in range(row_count):
            first_part = geom_off[r]
            last_part = geom_off[r + 1]
            if first_part == last_part:
                coord_spans.append((0, 0))
            else:
                cs = part_off[first_part]
                ce = part_off[last_part]
                coord_spans.append((int(cs), int(ce)))
    else:
        # Point, MultiPoint, LineString: geometry_offsets -> coord indices.
        geom_off = buf.geometry_offsets
        coord_spans = []
        for r in range(row_count):
            cs = geom_off[r]
            ce = geom_off[r + 1]
            coord_spans.append((int(cs), int(ce)))

    # Compute hull for each geometry.
    hull_x_parts = []
    hull_y_parts = []
    ring_offsets = [0]
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)

    for r in range(row_count):
        cs, ce = coord_spans[r]
        hx, hy = _convex_hull_geometry_cpu(buf.x, buf.y, cs, ce)
        hull_x_parts.append(hx)
        hull_y_parts.append(hy)
        ring_offsets.append(ring_offsets[-1] + len(hx))

    all_x = np.concatenate(hull_x_parts) if hull_x_parts else np.empty(0, dtype=np.float64)
    all_y = np.concatenate(hull_y_parts) if hull_y_parts else np.empty(0, dtype=np.float64)
    ring_offsets_arr = np.asarray(ring_offsets, dtype=np.int32)

    return FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=all_x,
        y=all_y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.zeros(row_count, dtype=bool),
        ring_offsets=ring_offsets_arr,
        bounds=None,
    )


# ---------------------------------------------------------------------------
# CPU kernel variant (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "convex_hull",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("numpy", "constructive", "convex_hull"),
)
def _convex_hull_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU convex hull for all geometry families using NumPy.

    Returns a host-resident Polygon OwnedGeometryArray.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries
        return from_shapely_geometries([])

    # Materialize device-resident coordinate buffers to host before
    # accessing family buffers.  Without this, device-only arrays have
    # stub host buffers with empty x/y arrays, causing IndexError.
    if owned.device_state is not None:
        owned._ensure_host_state()
        record_fallback_event(
            surface="convex_hull_cpu",
            reason="CPU fallback requires host-resident coordinates",
            d2h_transfer=True,
        )

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # Collect per-row hull data indexed by global row, then rebuild
    # in global row order for the output Polygon buffer.
    ordered_x = []
    ordered_y = []
    ring_offsets = [0]
    per_row_x = [None] * row_count
    per_row_y = [None] * row_count

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        global_rows = np.flatnonzero(mask)
        fam_rows = family_row_offsets[global_rows]

        poly_buf = _convex_hull_family_cpu(buf, family)
        for gi, fr in zip(global_rows, fam_rows):
            ring_start = poly_buf.ring_offsets[fr]
            ring_end = poly_buf.ring_offsets[fr + 1]
            per_row_x[gi] = poly_buf.x[ring_start:ring_end]
            per_row_y[gi] = poly_buf.y[ring_start:ring_end]

    for r in range(row_count):
        if per_row_x[r] is not None:
            ordered_x.append(per_row_x[r])
            ordered_y.append(per_row_y[r])
            ring_offsets.append(ring_offsets[-1] + len(per_row_x[r]))
        else:
            # Null/invalid row: emit a degenerate point polygon.
            ordered_x.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            ordered_y.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            ring_offsets.append(ring_offsets[-1] + 4)

    final_x = np.concatenate(ordered_x) if ordered_x else np.empty(0, dtype=np.float64)
    final_y = np.concatenate(ordered_y) if ordered_y else np.empty(0, dtype=np.float64)
    ring_offsets_arr = np.asarray(ring_offsets, dtype=np.int32)
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)

    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=final_x,
        y=final_y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.zeros(row_count, dtype=bool),
        ring_offsets=ring_offsets_arr,
        bounds=None,
    )

    new_tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    new_family_row_offsets = np.arange(row_count, dtype=np.int32)

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=new_tags,
        family_row_offsets=new_family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.HOST,
    )


# ---------------------------------------------------------------------------
# GPU implementation: 3-stage pipeline
#   1. CCCL segmented_sort (per-geometry x-sort)
#   2. NVRTC 2-pass monotone chain (count -> exclusive_sum -> write)
#   3. Build Polygon DeviceFamilyGeometryBuffer
#
# ADR-0033: Tier 3a (segmented_sort, exclusive_sum) + Tier 1 (NVRTC chain)
# ADR-0002: COARSE class, stays fp64 (hull = coordinate subset)
# ---------------------------------------------------------------------------

def _compute_flat_coord_offsets(device_buf, family):
    """Compute per-geometry flat coordinate offset array on device.

    For all families, returns a CuPy int32 array of shape (row_count+1,)
    where offsets[g] .. offsets[g+1] spans all coordinates of geometry g.

    For families with multi-level offset indirection (Polygon, MultiLineString),
    the indirection is resolved into a flat coordinate range.

    Returns None if the family is not supported for GPU path.
    """
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT,
                  GeometryFamily.LINESTRING):
        # geometry_offsets directly index into coords
        return cp.asarray(device_buf.geometry_offsets)

    if family is GeometryFamily.POLYGON:
        # Polygon: geometry_offsets -> ring indices, ring_offsets -> coord indices
        # coord_start[g] = ring_offsets[geometry_offsets[g]]
        # coord_end[g]   = ring_offsets[geometry_offsets[g+1]]
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_ring_off = cp.asarray(device_buf.ring_offsets)
        # Fancy-index ring_offsets at the geometry boundary positions
        d_flat_offsets = d_ring_off[d_geom_off]
        return d_flat_offsets

    if family is GeometryFamily.MULTILINESTRING:
        # MultiLineString: geometry_offsets -> part indices, part_offsets -> coord indices
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_part_off = cp.asarray(device_buf.part_offsets)
        d_flat_offsets = d_part_off[d_geom_off]
        return d_flat_offsets

    # MultiPolygon: complex 3-level indirection, not supported on GPU path
    return None


def _convex_hull_family_gpu(runtime, device_buf, family, row_count):
    """GPU convex hull for a single family.

    Pipeline:
      1. Compute flat coord offsets per geometry
      2. segmented_sort coordinates by x within each geometry
      3. NVRTC count pass (1 thread/geometry) -> hull_counts
      4. exclusive_sum(hull_counts) -> hull_offsets
      5. NVRTC write pass (1 thread/geometry) -> output coords
      6. Build DeviceFamilyGeometryBuffer for Polygon output

    Returns a DeviceFamilyGeometryBuffer (Polygon family).
    """
    from vibespatial.cuda.cccl_primitives import exclusive_sum, segmented_sort

    # --- Step 1: flat coord offsets ---
    d_flat_offsets = _compute_flat_coord_offsets(device_buf, family)
    if d_flat_offsets is None:
        return None  # unsupported family

    d_flat_offsets_i32 = d_flat_offsets.astype(cp.int32)

    # Segment start/end arrays for segmented_sort
    d_seg_starts = d_flat_offsets_i32[:-1].copy()
    d_seg_ends = d_flat_offsets_i32[1:].copy()

    # --- Step 2: segmented sort by x ---
    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)
    coord_count = int(d_x.shape[0])

    if coord_count == 0:
        # All geometries are empty -- fall through to kernel launch which handles n=0
        pass

    # Sort x within each geometry segment, carrying y along via index permutation.
    # Create index array so we can reorder y by the same permutation.
    d_indices = cp.arange(coord_count, dtype=cp.int32)

    if coord_count > 0 and row_count > 0:
        sort_result = segmented_sort(
            keys=d_x.copy(),
            values=d_indices,
            starts=d_seg_starts,
            ends=d_seg_ends,
        )
        d_sorted_x = sort_result.keys
        d_sorted_y = d_y[sort_result.values]
    else:
        d_sorted_x = d_x.copy()
        d_sorted_y = d_y.copy()

    # --- Step 3: NVRTC count pass ---
    kernels = compile_kernel_group(
        "convex-hull-fp64", _CONVEX_HULL_FP64, _CONVEX_HULL_KERNEL_NAMES,
    )

    d_hull_counts = runtime.allocate((row_count,), np.int32, zero=True)
    ptr = runtime.pointer

    count_params = (
        (
            ptr(d_sorted_x), ptr(d_sorted_y), ptr(d_flat_offsets_i32),
            ptr(d_hull_counts), row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
        ),
    )
    count_kernel = kernels["convex_hull_count"]
    grid, block = runtime.launch_config(count_kernel, row_count)
    runtime.launch(count_kernel, grid=grid, block=block, params=count_params)

    # --- Step 4: exclusive_sum for output offsets ---
    # No sync needed between count kernel and exclusive_sum (same null stream).
    d_hull_offsets = exclusive_sum(d_hull_counts, synchronize=False)

    # Get total output coordinate count via single async transfer
    total_coords = count_scatter_total(runtime, d_hull_counts, d_hull_offsets)

    # --- Step 5: NVRTC write pass ---
    d_ox = runtime.allocate((max(total_coords, 1),), np.float64)
    d_oy = runtime.allocate((max(total_coords, 1),), np.float64)

    write_params = (
        (
            ptr(d_sorted_x), ptr(d_sorted_y), ptr(d_flat_offsets_i32),
            ptr(d_hull_offsets), ptr(d_ox), ptr(d_oy), row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    write_kernel = kernels["convex_hull_write"]
    grid, block = runtime.launch_config(write_kernel, row_count)
    runtime.launch(write_kernel, grid=grid, block=block, params=write_params)

    # --- Step 6: build Polygon output ---
    # Polygon output: 1 ring per geometry, so:
    #   geometry_offsets = [0, 1, 2, ..., N]   (1 ring per polygon)
    #   ring_offsets = hull_offsets with total appended
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)

    # ring_offsets = hull_offsets ++ [total_coords]
    d_ring_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    d_ring_offsets[:row_count] = cp.asarray(d_hull_offsets)
    d_ring_offsets[row_count] = total_coords

    d_empty_mask = cp.zeros(row_count, dtype=cp.bool_)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=d_ox[:total_coords] if total_coords > 0 else d_ox[:0],
        y=d_oy[:total_coords] if total_coords > 0 else d_oy[:0],
        geometry_offsets=d_geom_offsets,
        empty_mask=d_empty_mask,
        ring_offsets=d_ring_offsets,
        bounds=None,
    )


@register_kernel_variant(
    "convex_hull",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon",
    ),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "convex_hull"),
)
def _convex_hull_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU convex hull using CCCL segmented_sort + NVRTC monotone chain.

    Returns a device-resident Polygon OwnedGeometryArray.
    Handles single-family inputs of Point, MultiPoint, LineString,
    MultiLineString, and Polygon.  Mixed families and MultiPolygon
    fall back to the CPU path via the dispatcher.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count

    # For GPU path, process each family separately, then merge into
    # a single Polygon output ordered by global row.
    # Since supports_mixed=False, the dispatcher only routes here for
    # single-family inputs.  For multi-family, it falls back to CPU.

    # Find the single active family
    active_families = [
        (fam, dbuf) for fam, dbuf in d_state.families.items()
        if int(dbuf.geometry_offsets.size) >= 2
    ]

    if len(active_families) != 1:
        # Defensive: should not happen since supports_mixed=False
        logger.warning("GPU convex_hull received %d families, falling back to CPU",
                        len(active_families))
        return _convex_hull_cpu(owned)

    family, device_buf = active_families[0]

    if family is GeometryFamily.MULTIPOLYGON:
        # MultiPolygon has 3-level offset indirection, fall back to CPU
        return _convex_hull_cpu(owned)

    result_buf = _convex_hull_family_gpu(runtime, device_buf, family, row_count)
    if result_buf is None:
        return _convex_hull_cpu(owned)

    # Build device-resident OwnedGeometryArray
    new_tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    new_family_row_offsets = np.arange(row_count, dtype=np.int32)

    return build_device_resident_owned(
        device_families={GeometryFamily.POLYGON: result_buf},
        row_count=row_count,
        tags=new_tags,
        validity=owned.validity.copy(),
        family_row_offsets=new_family_row_offsets,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def convex_hull_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the convex hull of each geometry.

    Uses Andrew's monotone chain algorithm.  Output is always Polygon
    family: each row produces a closed polygon whose exterior ring is
    the convex hull of the input geometry's coordinates.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str
        Execution mode selection.  Defaults to AUTO.
    precision : PrecisionMode or str
        Precision mode selection.  Defaults to AUTO.
        COARSE class: hull vertices are exact coordinate subsets,
        stays fp64 on all devices.

    Returns
    -------
    OwnedGeometryArray
        Polygon OwnedGeometryArray with one convex hull per input row.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries

        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="convex_hull",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    # Check if GPU path is viable
    if selection.selected is ExecutionMode.GPU and cp is not None:
        # COARSE class: always fp64, precision_plan used only for event metadata.
        precision_plan = selection.precision_plan
        # GPU path supports single-family non-MultiPolygon inputs.
        # Use family_has_rows() which checks device buffers when present,
        # avoiding false negatives from unmaterialized host stubs.
        families_with_rows = [
            fam for fam in owned.families
            if owned.family_has_rows(fam)
        ]
        is_single_family = len(families_with_rows) == 1
        has_multipolygon = GeometryFamily.MULTIPOLYGON in families_with_rows

        if is_single_family and not has_multipolygon:
            try:
                result = _convex_hull_gpu(owned)
            except Exception:
                logger.debug("GPU convex_hull failed, falling back to CPU",
                            exc_info=True)
            else:
                record_dispatch_event(
                    surface="geopandas.array.convex_hull",
                    operation="convex_hull",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                    implementation="convex_hull_gpu_nvrtc",
                    reason=selection.reason,
                    detail=f"precision={precision_plan.compute_precision}",
                )
                return result

    result = _convex_hull_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.convex_hull",
        operation="convex_hull",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
        implementation="convex_hull_cpu_numpy",
        reason=selection.reason,
    )
    return result
