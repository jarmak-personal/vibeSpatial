"""GPU-accelerated minimum rotated rectangle (oriented envelope) computation.

Uses rotating calipers on the convex hull to find the minimum-area bounding
rectangle for each geometry.  Output is always Polygon family: each row
produces a closed 5-vertex polygon (the minimum-area rotated rectangle).

Algorithm per geometry:
  1. Compute convex hull (reuses existing GPU convex_hull kernel).
  2. For each edge of the convex hull, compute the perpendicular bounding
     rectangle by rotating coordinates into an edge-aligned frame.
  3. Track the rectangle with minimum area.
  4. Output the 4 corners (+ closure) of the minimum-area rectangle.

Degenerate cases:
  - Point -> degenerate polygon (point repeated 5 times).
  - Collinear / <3 hull vertices -> degenerate rectangle from the line
    segment endpoints.

ADR-0033: Tier 1 NVRTC for rotating calipers (geometry-specific inner loop).
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC kernel source: rotating calipers minimum-area bounding rectangle
#
# One thread per geometry.  Input is the convex hull (closed ring, Polygon
# family).  For each edge of the hull, the kernel rotates all hull vertices
# into the edge-aligned coordinate system, computes the axis-aligned
# bounding box, and tracks the minimum-area rectangle.  The 4 corners of
# the best rectangle are un-rotated back and written as a 5-vertex closed
# polygon.
#
# ADR-0033 Tier 1: geometry-specific inner loop (rotating calipers).
# ADR-0002: CONSTRUCTIVE class, always fp64.
# ---------------------------------------------------------------------------

_MIN_RECT_KERNEL_SOURCE = r"""
extern "C" __global__ void minimum_rotated_rectangle(
    const double* __restrict__ hull_x,
    const double* __restrict__ hull_y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geom_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    /* Determine the hull ring span for this geometry.
       Polygon layout: geom_offsets[row]..geom_offsets[row+1] are ring indices,
       ring_offsets[ring_idx]..ring_offsets[ring_idx+1] are coord indices.
       Convex hull output always has exactly 1 ring per polygon. */
    const int ring_idx = geom_offsets[row];
    const int cs = ring_offsets[ring_idx];
    const int ce = ring_offsets[ring_idx + 1];
    const int n_coords = ce - cs;  /* includes closing vertex */

    /* Output base: 5 vertices per geometry */
    const int out_base = row * 5;

    /* Degenerate: 0 or 1 coords -> point at origin or single coord */
    if (n_coords <= 1) {
        double px = (n_coords == 1) ? hull_x[cs] : 0.0;
        double py = (n_coords == 1) ? hull_y[cs] : 0.0;
        for (int i = 0; i < 5; i++) {
            out_x[out_base + i] = px;
            out_y[out_base + i] = py;
        }
        return;
    }

    /* Number of unique hull vertices (closed ring: n_coords - 1) */
    const int n_verts = n_coords - 1;

    /* Degenerate: 2 unique vertices -> line segment, rectangle is the segment */
    if (n_verts <= 2) {
        double x0 = hull_x[cs];
        double y0 = hull_y[cs];
        double x1 = hull_x[cs + 1];
        double y1 = hull_y[cs + 1];
        out_x[out_base + 0] = x0;  out_y[out_base + 0] = y0;
        out_x[out_base + 1] = x1;  out_y[out_base + 1] = y1;
        out_x[out_base + 2] = x1;  out_y[out_base + 2] = y1;
        out_x[out_base + 3] = x0;  out_y[out_base + 3] = y0;
        out_x[out_base + 4] = x0;  out_y[out_base + 4] = y0;
        return;
    }

    /* Rotating calipers: for each edge, rotate into edge-aligned frame,
       compute AABB, track minimum area. */
    double best_area = 1.0e300;  /* sentinel: larger than any real area */
    double best_c0x, best_c0y, best_c1x, best_c1y;
    double best_c2x, best_c2y, best_c3x, best_c3y;

    for (int e = 0; e < n_verts; e++) {
        /* Edge from hull[e] to hull[(e+1) % n_verts] */
        const int i0 = cs + e;
        const int i1 = cs + ((e + 1) % n_verts);
        const double ex = hull_x[i1] - hull_x[i0];
        const double ey = hull_y[i1] - hull_y[i0];

        /* Edge length and normalized direction */
        const double edge_len = sqrt(ex * ex + ey * ey);
        if (edge_len < 1.0e-15) continue;  /* degenerate zero-length edge */

        const double dx = ex / edge_len;
        const double dy = ey / edge_len;

        /* Rotate all hull vertices into edge-aligned frame:
           u = x*dx + y*dy   (along edge)
           v = -x*dy + y*dx  (perpendicular to edge) */
        double min_u = 1.0e300, max_u = -1.0e300;
        double min_v = 1.0e300, max_v = -1.0e300;

        for (int j = 0; j < n_verts; j++) {
            const double vx = hull_x[cs + j];
            const double vy = hull_y[cs + j];
            const double u = vx * dx + vy * dy;
            const double v = -vx * dy + vy * dx;

            if (u < min_u) min_u = u;
            if (u > max_u) max_u = u;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }

        const double area = (max_u - min_u) * (max_v - min_v);

        if (area < best_area) {
            best_area = area;

            /* Un-rotate the 4 corners back to original space:
               x = u*dx - v*dy
               y = u*dy + v*dx */
            best_c0x = min_u * dx - min_v * dy;
            best_c0y = min_u * dy + min_v * dx;

            best_c1x = max_u * dx - min_v * dy;
            best_c1y = max_u * dy + min_v * dx;

            best_c2x = max_u * dx - max_v * dy;
            best_c2y = max_u * dy + max_v * dx;

            best_c3x = min_u * dx - max_v * dy;
            best_c3y = min_u * dy + max_v * dx;
        }
    }

    /* Write the 5-vertex closed polygon */
    out_x[out_base + 0] = best_c0x;  out_y[out_base + 0] = best_c0y;
    out_x[out_base + 1] = best_c1x;  out_y[out_base + 1] = best_c1y;
    out_x[out_base + 2] = best_c2x;  out_y[out_base + 2] = best_c2y;
    out_x[out_base + 3] = best_c3x;  out_y[out_base + 3] = best_c3y;
    out_x[out_base + 4] = best_c0x;  out_y[out_base + 4] = best_c0y;  /* close ring */
}
"""

_MIN_RECT_KERNEL_NAMES = ("minimum_rotated_rectangle",)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034): register kernels for background precompilation
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("min-rotated-rect", _MIN_RECT_KERNEL_SOURCE, _MIN_RECT_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# CPU fallback implementation
# ---------------------------------------------------------------------------

def _minimum_rotated_rectangle_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU minimum rotated rectangle via Shapely oriented_envelope.

    Returns a host-resident Polygon OwnedGeometryArray.
    """
    import shapely

    from vibespatial.geometry.owned import (
        from_shapely_geometries,
    )

    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    # Materialize to Shapely geometries and use oriented_envelope
    geoms = owned.to_shapely()
    result_geoms = shapely.oriented_envelope(geoms)

    return from_shapely_geometries(result_geoms)


# ---------------------------------------------------------------------------
# CPU kernel variant (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_rotated_rectangle",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("shapely", "constructive", "minimum_rotated_rectangle"),
)
def _min_rect_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU minimum rotated rectangle using Shapely oriented_envelope."""
    return _minimum_rotated_rectangle_cpu(owned)


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_rotated_rectangle",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon",
    ),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "minimum_rotated_rectangle"),
)
def _min_rect_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU minimum rotated rectangle using convex hull + rotating calipers.

    Pipeline:
      1. Compute convex hull on GPU (reuses existing convex_hull_owned).
      2. Launch rotating calipers NVRTC kernel on hull output.
      3. Assemble output as 5-vertex closed Polygon.

    Returns a device-resident Polygon OwnedGeometryArray.
    """
    from .convex_hull import convex_hull_owned

    runtime = get_cuda_runtime()
    row_count = owned.row_count

    # Step 1: Compute convex hull on GPU -- stays device-resident
    hull_owned = convex_hull_owned(owned, dispatch_mode=ExecutionMode.GPU)

    # The hull output is a Polygon OwnedGeometryArray with one ring per row.
    # We need the device buffers for the hull coordinates.
    d_state = hull_owned._ensure_device_state()

    # The hull is always single-family Polygon
    hull_device_buf = d_state.families[GeometryFamily.POLYGON]
    d_hull_x = cp.asarray(hull_device_buf.x)
    d_hull_y = cp.asarray(hull_device_buf.y)
    d_ring_offsets = cp.asarray(hull_device_buf.ring_offsets)
    d_geom_offsets = cp.asarray(hull_device_buf.geometry_offsets)

    # Step 2: Allocate output -- 5 vertices per geometry
    d_out_x = runtime.allocate((row_count * 5,), np.float64)
    d_out_y = runtime.allocate((row_count * 5,), np.float64)

    # Step 3: Launch rotating calipers kernel
    kernels = compile_kernel_group(
        "min-rotated-rect", _MIN_RECT_KERNEL_SOURCE, _MIN_RECT_KERNEL_NAMES,
    )
    kernel = kernels["minimum_rotated_rectangle"]

    ptr = runtime.pointer
    params = (
        (
            ptr(d_hull_x), ptr(d_hull_y),
            ptr(d_ring_offsets), ptr(d_geom_offsets),
            ptr(d_out_x), ptr(d_out_y),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # Step 4: Build output Polygon OwnedGeometryArray (5 verts per ring)
    from .point import _build_device_backed_polygon_output

    return _build_device_backed_polygon_output(
        d_out_x,
        d_out_y,
        row_count=row_count,
        bounds=None,
        verts_per_ring=5,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def minimum_rotated_rectangle_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the minimum rotated rectangle (oriented envelope) of each geometry.

    Uses rotating calipers on the convex hull.  Output is always Polygon
    family: each row produces a closed 5-vertex polygon (the minimum-area
    bounding rectangle).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str
        Execution mode selection.  Defaults to AUTO.
    precision : PrecisionMode or str
        Precision mode selection.  Defaults to AUTO.
        CONSTRUCTIVE class: stays fp64 on all devices per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Polygon OwnedGeometryArray with one minimum rotated rectangle per
        input row.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries

        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="minimum_rotated_rectangle",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU and cp is not None:
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        # GPU path supports single-family non-MultiPolygon inputs
        families_with_rows = [
            fam for fam, buf in owned.families.items()
            if buf.row_count > 0
        ]
        is_single_family = len(families_with_rows) == 1
        has_multipolygon = GeometryFamily.MULTIPOLYGON in families_with_rows

        if is_single_family and not has_multipolygon:
            try:
                result = _min_rect_gpu(owned)
            except Exception:
                logger.debug(
                    "GPU minimum_rotated_rectangle failed, falling back to CPU",
                    exc_info=True,
                )
            else:
                record_dispatch_event(
                    surface="geopandas.array.minimum_rotated_rectangle",
                    operation="minimum_rotated_rectangle",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                    implementation="min_rotated_rect_gpu_nvrtc",
                    reason=selection.reason,
                    detail=f"precision={precision_plan.compute_precision}",
                )
                return result

    result = _minimum_rotated_rectangle_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.minimum_rotated_rectangle",
        operation="minimum_rotated_rectangle",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
        implementation="min_rotated_rect_cpu_shapely",
        reason=selection.reason,
    )
    return result
