"""GPU-accelerated minimum bounding circle and radius computation.

Uses Ritter's bounding sphere algorithm (adapted to 2D) for an O(n)
per-geometry approximation of the minimum enclosing circle.  One thread
per geometry computes (center_x, center_y, radius).

Two public entry points:
  - minimum_bounding_circle_owned:  Returns Polygon OwnedGeometryArray
    (circle tessellated to N-gon).
  - minimum_bounding_radius_owned:  Returns float64 array of radii.

Both share the same core NVRTC kernel.  The polygon output adds a
circle-to-polygon tessellation pass (also NVRTC).

Architecture (ADR-0033 tier classification):
  - Ritter's inner loop: Tier 1 NVRTC (geometry-specific iteration)
  - Tessellation: Tier 1 NVRTC (per-geometry circle -> polygon)
  - Offset computation: Tier 2 CuPy (element-wise arange)

Precision (ADR-0002):
  - minimum_bounding_circle: CONSTRUCTIVE class -- fp64 by design.
    Polygon vertices must be exact; no fp32 path until robustness proves safe.
  - minimum_bounding_radius: METRIC class -- fp64 by design.
    Ritter's distance comparisons need fp64 to guarantee containment.

Ritter's is an approximation: the returned circle always CONTAINS all
geometry points, but the radius may be up to ~5% larger than the exact
minimum enclosing circle (Welzl).  Shapely uses Welzl, so radius values
will not match exactly -- the key invariant is containment.

Algorithm (Ritter's 2D bounding circle):
  1. Compute centroid of all coordinates.
  2. Find the point P furthest from centroid.
  3. Find the point Q furthest from P.
  4. Initial circle: center = midpoint(P, Q), radius = dist(P, Q) / 2.
  5. Scan all points: for any point outside the circle, expand:
     - new_radius = (old_radius + dist(center, outside_point)) / 2
     - shift center toward outside_point by (new_radius - old_radius)
  6. Two expansion passes for convergence.
"""

from __future__ import annotations

import logging
import math

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.minimum_bounding_circle_kernels import (
    _N_TESSELLATION_VERTS,
    _RITTER_KERNEL_NAMES,
    _RITTER_KERNEL_SOURCE,
    _TESSELLATE_KERNEL_NAMES,
    _TESSELLATE_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
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
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034): register kernels for background precompilation
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("ritter-bounding-circle", _RITTER_KERNEL_SOURCE, _RITTER_KERNEL_NAMES),
    ("tessellate-circle", _TESSELLATE_KERNEL_SOURCE, _TESSELLATE_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Helper: flat coordinate offsets per geometry (reused from convex_hull)
# ---------------------------------------------------------------------------

def _compute_flat_coord_offsets(device_buf, family):
    """Compute per-geometry flat coordinate offset array on device.

    Returns a CuPy int32 array of shape (row_count+1,) where
    offsets[g] .. offsets[g+1] spans all coordinates of geometry g.

    For families with multi-level offset indirection (Polygon,
    MultiPolygon, MultiLineString), the indirection is resolved.
    """
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT,
                  GeometryFamily.LINESTRING):
        return cp.asarray(device_buf.geometry_offsets)

    if family is GeometryFamily.POLYGON:
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_ring_off = cp.asarray(device_buf.ring_offsets)
        return d_ring_off[d_geom_off]

    if family is GeometryFamily.MULTILINESTRING:
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_part_off = cp.asarray(device_buf.part_offsets)
        return d_part_off[d_geom_off]

    if family is GeometryFamily.MULTIPOLYGON:
        # 3-level indirection: geometry -> part -> ring -> coord
        # coord_start[g] = ring_offsets[part_offsets[geometry_offsets[g]]]
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_part_off = cp.asarray(device_buf.part_offsets)
        d_ring_off = cp.asarray(device_buf.ring_offsets)
        return d_ring_off[d_part_off[d_geom_off]]

    return None


# ---------------------------------------------------------------------------
# GPU: core Ritter's kernel launcher
# ---------------------------------------------------------------------------

def _ritter_family_gpu(runtime, device_buf, family, row_count):
    """Run Ritter's bounding circle for one family on GPU.

    Returns (d_cx, d_cy, d_radius) -- three device arrays of shape (row_count,).
    """
    d_flat_offsets = _compute_flat_coord_offsets(device_buf, family)
    if d_flat_offsets is None:
        return None

    d_flat_offsets_i32 = d_flat_offsets.astype(cp.int32)

    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    kernels = compile_kernel_group(
        "ritter-bounding-circle", _RITTER_KERNEL_SOURCE, _RITTER_KERNEL_NAMES,
    )
    kernel = kernels["ritter_bounding_circle"]

    d_cx = runtime.allocate((row_count,), np.float64)
    d_cy = runtime.allocate((row_count,), np.float64)
    d_radius = runtime.allocate((row_count,), np.float64)

    ptr = runtime.pointer
    params = (
        (
            ptr(d_x), ptr(d_y), ptr(d_flat_offsets_i32),
            ptr(d_cx), ptr(d_cy), ptr(d_radius),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    return d_cx, d_cy, d_radius


def _tessellate_circles_gpu(runtime, d_cx, d_cy, d_radius, row_count):
    """Tessellate circles to polygon coordinate buffers on GPU.

    Returns DeviceFamilyGeometryBuffer for Polygon output.
    """
    n_verts = _N_TESSELLATION_VERTS
    coords_per_geom = n_verts + 1  # closed ring
    total_coords = row_count * coords_per_geom

    d_ox = runtime.allocate((max(total_coords, 1),), np.float64)
    d_oy = runtime.allocate((max(total_coords, 1),), np.float64)

    kernels = compile_kernel_group(
        "tessellate-circle", _TESSELLATE_KERNEL_SOURCE, _TESSELLATE_KERNEL_NAMES,
    )
    kernel = kernels["tessellate_circle"]

    ptr = runtime.pointer
    params = (
        (
            ptr(d_cx), ptr(d_cy), ptr(d_radius),
            ptr(d_ox), ptr(d_oy),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # Build Polygon structure: 1 ring per geometry, each ring = coords_per_geom
    # geometry_offsets = [0, 1, 2, ..., N]
    # ring_offsets = [0, coords_per_geom, 2*coords_per_geom, ..., N*coords_per_geom]
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_ring_offsets = cp.arange(row_count + 1, dtype=cp.int32) * coords_per_geom
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


# ---------------------------------------------------------------------------
# GPU kernel variants (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_bounding_circle",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "minimum_bounding_circle"),
)
def _minimum_bounding_circle_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU minimum bounding circle using Ritter's + tessellation.

    Returns a device-resident Polygon OwnedGeometryArray.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count

    # Find the single active family (supports_mixed=False)
    active_families = [
        (fam, dbuf) for fam, dbuf in d_state.families.items()
        if int(dbuf.geometry_offsets.size) >= 2
    ]

    if len(active_families) != 1:
        logger.warning(
            "GPU minimum_bounding_circle received %d families, falling back to CPU",
            len(active_families),
        )
        return _minimum_bounding_circle_cpu(owned)

    family, device_buf = active_families[0]

    result = _ritter_family_gpu(runtime, device_buf, family, row_count)
    if result is None:
        return _minimum_bounding_circle_cpu(owned)

    d_cx, d_cy, d_radius = result

    # Tessellate circles to polygons
    result_buf = _tessellate_circles_gpu(runtime, d_cx, d_cy, d_radius, row_count)

    # Build device-resident OwnedGeometryArray
    d_validity = cp.asarray(d_state.validity)
    new_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8)
    new_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    d_null = ~d_validity
    if int(d_null.any()) != 0:
        new_tags[d_null] = -1
        new_family_row_offsets[d_null] = -1

    return build_device_resident_owned(
        device_families={GeometryFamily.POLYGON: result_buf},
        row_count=row_count,
        tags=new_tags,
        validity=d_validity,
        family_row_offsets=new_family_row_offsets,
        execution_mode="gpu",
    )


@register_kernel_variant(
    "minimum_bounding_radius",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=False,
    tags=("cuda-python", "metric", "minimum_bounding_radius"),
)
def _minimum_bounding_radius_gpu(owned: OwnedGeometryArray) -> np.ndarray:
    """GPU minimum bounding radius using Ritter's algorithm.

    Returns float64 array of shape (row_count,).
    Single D2H transfer at the end for the radius array.
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count

    # Find the single active family (supports_mixed=False)
    active_families = [
        (fam, dbuf) for fam, dbuf in d_state.families.items()
        if int(dbuf.geometry_offsets.size) >= 2
    ]

    if len(active_families) != 1:
        logger.warning(
            "GPU minimum_bounding_radius received %d families, falling back to CPU",
            len(active_families),
        )
        return _minimum_bounding_radius_cpu(owned)

    family, device_buf = active_families[0]

    result = _ritter_family_gpu(runtime, device_buf, family, row_count)
    if result is None:
        return _minimum_bounding_radius_cpu(owned)

    _d_cx, _d_cy, d_radius = result

    # Single D2H transfer at pipeline end
    runtime.synchronize()
    return runtime.copy_device_to_host(d_radius)


# ---------------------------------------------------------------------------
# CPU implementations
# ---------------------------------------------------------------------------

def _ritter_cpu(x, y, start, end):
    """Ritter's bounding circle for coordinates x[start:end], y[start:end].

    Returns (cx, cy, radius).
    """
    n = end - start
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(x[start]), float(y[start]), 0.0

    # Step 1: centroid
    px = x[start:end]
    py = y[start:end]
    mean_x = px.mean()
    mean_y = py.mean()

    # Step 2: point furthest from centroid
    dists_sq = (px - mean_x) ** 2 + (py - mean_y) ** 2
    p_idx = np.argmax(dists_sq)

    # Step 3: point furthest from P
    dists_sq = (px - px[p_idx]) ** 2 + (py - py[p_idx]) ** 2
    q_idx = np.argmax(dists_sq)

    # Step 4: initial circle from P-Q diameter
    cx = (px[p_idx] + px[q_idx]) / 2.0
    cy = (py[p_idx] + py[q_idx]) / 2.0
    dx = px[q_idx] - px[p_idx]
    dy = py[q_idx] - py[p_idx]
    radius = math.sqrt(dx * dx + dy * dy) / 2.0

    # Step 5-6: expand (2 passes)
    for _pass in range(2):
        for i in range(n):
            dx = px[i] - cx
            dy = py[i] - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > radius:
                new_radius = (radius + dist) / 2.0
                shift = new_radius - radius
                inv_dist = 1.0 / dist
                cx += dx * inv_dist * shift
                cy += dy * inv_dist * shift
                radius = new_radius

    return float(cx), float(cy), float(radius)


def _get_coord_spans_cpu(buf, family, row_count):
    """Get (start, end) coordinate spans for each geometry row on CPU."""
    coord_spans = []
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        geom_off = buf.geometry_offsets
        if family is GeometryFamily.MULTIPOLYGON:
            part_off = buf.part_offsets
            ring_off = buf.ring_offsets
            for r in range(row_count):
                first_part = geom_off[r]
                last_part = geom_off[r + 1]
                if first_part == last_part:
                    coord_spans.append((0, 0))
                else:
                    first_ring = part_off[first_part]
                    last_ring = part_off[last_part]
                    cs = ring_off[first_ring]
                    ce = ring_off[last_ring]
                    coord_spans.append((int(cs), int(ce)))
        else:
            ring_off = buf.ring_offsets
            for r in range(row_count):
                first_ring = geom_off[r]
                last_ring = geom_off[r + 1]
                if first_ring == last_ring:
                    coord_spans.append((0, 0))
                else:
                    cs = ring_off[first_ring]
                    ce = ring_off[last_ring]
                    coord_spans.append((int(cs), int(ce)))
    elif family is GeometryFamily.MULTILINESTRING:
        geom_off = buf.geometry_offsets
        part_off = buf.part_offsets
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
        # Point, MultiPoint, LineString
        geom_off = buf.geometry_offsets
        for r in range(row_count):
            cs = geom_off[r]
            ce = geom_off[r + 1]
            coord_spans.append((int(cs), int(ce)))
    return coord_spans


def _tessellate_circle_cpu(cx, cy, radius, n_verts=_N_TESSELLATION_VERTS):
    """Tessellate a circle to a closed polygon ring on CPU."""
    angles = np.linspace(0, 2 * math.pi, n_verts, endpoint=False)
    ring_x = cx + radius * np.cos(angles)
    ring_y = cy + radius * np.sin(angles)
    # Close the ring
    ring_x = np.append(ring_x, ring_x[0])
    ring_y = np.append(ring_y, ring_y[0])
    return ring_x.astype(np.float64), ring_y.astype(np.float64)


@register_kernel_variant(
    "minimum_bounding_circle",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("numpy", "constructive", "minimum_bounding_circle"),
)
def _minimum_bounding_circle_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU minimum bounding circle using Ritter's + tessellation."""
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries
        return from_shapely_geometries([])

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    per_row_x = [None] * row_count
    per_row_y = [None] * row_count

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        global_rows = np.flatnonzero(mask)
        fam_rows = family_row_offsets[global_rows]

        coord_spans = _get_coord_spans_cpu(buf, family, buf.row_count)

        for gi, fr in zip(global_rows, fam_rows):
            cs, ce = coord_spans[fr]
            cx, cy, radius = _ritter_cpu(buf.x, buf.y, cs, ce)
            rx, ry = _tessellate_circle_cpu(cx, cy, radius)
            per_row_x[gi] = rx
            per_row_y[gi] = ry

    # Build polygon output
    ordered_x = []
    ordered_y = []
    ring_offsets = [0]
    for r in range(row_count):
        if per_row_x[r] is not None:
            ordered_x.append(per_row_x[r])
            ordered_y.append(per_row_y[r])
            ring_offsets.append(ring_offsets[-1] + len(per_row_x[r]))
        else:
            # Null row: degenerate point polygon
            degen = np.zeros(4, dtype=np.float64)
            ordered_x.append(degen)
            ordered_y.append(degen)
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


@register_kernel_variant(
    "minimum_bounding_radius",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("numpy", "metric", "minimum_bounding_radius"),
)
def _minimum_bounding_radius_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU minimum bounding radius using Ritter's algorithm."""
    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    result = np.zeros(row_count, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        global_rows = np.flatnonzero(mask)
        fam_rows = family_row_offsets[global_rows]

        coord_spans = _get_coord_spans_cpu(buf, family, buf.row_count)

        for gi, fr in zip(global_rows, fam_rows):
            cs, ce = coord_spans[fr]
            _cx, _cy, radius = _ritter_cpu(buf.x, buf.y, cs, ce)
            result[gi] = radius

    return result


# ---------------------------------------------------------------------------
# Public dispatch APIs
# ---------------------------------------------------------------------------

def minimum_bounding_circle_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the minimum bounding circle of each geometry (as polygon).

    Uses Ritter's algorithm for an O(n) per-geometry approximation.
    The returned circle always CONTAINS all geometry points but may be
    slightly larger than the exact minimum enclosing circle.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str
        Execution mode selection.  Defaults to AUTO.
    precision : PrecisionMode or str
        Precision mode selection.  CONSTRUCTIVE class: stays fp64.

    Returns
    -------
    OwnedGeometryArray
        Polygon OwnedGeometryArray with one circle polygon per input row.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="minimum_bounding_circle",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = selection.precision_plan
        families_with_rows = [
            fam for fam, buf in owned.families.items()
            if buf.row_count > 0
        ]
        is_single_family = len(families_with_rows) == 1

        if is_single_family:
            try:
                result = _minimum_bounding_circle_gpu(owned)
            except Exception:
                logger.debug(
                    "GPU minimum_bounding_circle failed, falling back to CPU",
                    exc_info=True,
                )
            else:
                record_dispatch_event(
                    surface="geopandas.array.minimum_bounding_circle",
                    operation="minimum_bounding_circle",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                    implementation="ritter_gpu_nvrtc",
                    reason=selection.reason,
                    detail=f"precision={precision_plan.compute_precision}",
                )
                return result

    result = _minimum_bounding_circle_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.minimum_bounding_circle",
        operation="minimum_bounding_circle",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
        implementation="ritter_cpu_numpy",
        reason=selection.reason,
    )
    return result


def minimum_bounding_radius_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Compute the minimum bounding radius of each geometry.

    Uses Ritter's algorithm for an O(n) per-geometry approximation.
    Returns float64 array of radii.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str
        Execution mode selection.  Defaults to AUTO.
    precision : PrecisionMode or str
        Precision mode selection.  METRIC class: stays fp64.

    Returns
    -------
    np.ndarray
        float64 array of shape (row_count,) with bounding radius per geometry.
    """
    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="minimum_bounding_radius",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = selection.precision_plan
        families_with_rows = [
            fam for fam, buf in owned.families.items()
            if buf.row_count > 0
        ]
        is_single_family = len(families_with_rows) == 1

        if is_single_family:
            try:
                result = _minimum_bounding_radius_gpu(owned)
                # Mark invalid rows as NaN
                result[~owned.validity] = np.nan
            except Exception:
                logger.debug(
                    "GPU minimum_bounding_radius failed, falling back to CPU",
                    exc_info=True,
                )
            else:
                record_dispatch_event(
                    surface="geopandas.array.minimum_bounding_radius",
                    operation="minimum_bounding_radius",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                    implementation="ritter_gpu_nvrtc",
                    reason=selection.reason,
                    detail=f"precision={precision_plan.compute_precision}",
                )
                return result

    result = _minimum_bounding_radius_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.minimum_bounding_radius",
        operation="minimum_bounding_radius",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
        implementation="ritter_cpu_numpy",
        reason=selection.reason,
    )
    return result
