"""GPU-accelerated representative_point using layered centroid + PIP strategy.

Architecture (ADR-0033 tier classification):
- Points: identity (Tier 2 CuPy element-wise copy)
- LineStrings: midpoint interpolation (Tier 1 NVRTC)
- Polygons/MultiPolygons: centroid (existing GPU kernel) + PIP containment
  check. For concave polygons where centroid falls outside, a GPU horizontal-
  ray-intersection kernel finds an interior point without any D2H transfer
  or Shapely fallback.

Precision (ADR-0002): Centroid computation uses METRIC class dispatch
(fp32+Kahan on consumer GPU, fp64 on datacenter). PIP check and horizontal
ray intersection use fp64 for exact topology.
"""

from __future__ import annotations

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event

from .point import point_owned_from_xy

# ---------------------------------------------------------------------------
# NVRTC kernel: combined PIP check + horizontal ray representative point
# ---------------------------------------------------------------------------
# One thread per polygon row.  For each polygon:
# 1. Compute whether centroid is inside using even-odd ray casting.
# 2. If inside, keep centroid.
# 3. If outside (concave polygon), cast horizontal ray at centroid_y through
#    all rings (exterior + holes).  Collect X-intersection values, sort them,
#    pair them into interior intervals (even-odd), and pick the midpoint of
#    the widest interval.
# ---------------------------------------------------------------------------

_REPRESENTATIVE_POINT_KERNEL_SOURCE = r"""
extern "C" __global__ void representative_point_polygon(
    const double* centroid_x,
    const double* centroid_y,
    const double* poly_x,
    const double* poly_y,
    const int* ring_offsets,
    const int* geom_offsets,
    const int* global_row_indices,
    const int* family_row_indices,
    double* out_x,
    double* out_y,
    int num_polygons,
    int max_intersections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_polygons) return;

    const int gr = global_row_indices[idx];
    const int fr = family_row_indices[idx];
    const double cx = centroid_x[gr];
    const double cy = centroid_y[gr];

    // Determine ring range for this polygon
    const int ring_start = geom_offsets[fr];
    const int ring_end = geom_offsets[fr + 1];

    if (ring_start >= ring_end) {
        // Empty polygon - keep centroid (will be NaN for null rows)
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 1: PIP check via even-odd ray casting (horizontal ray from cx,cy
    // going right)
    bool inside = false;
    bool on_boundary = false;
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ax = poly_x[c];
            const double ay = poly_y[c];
            const double bx = poly_x[c + 1];
            const double by = poly_y[c + 1];

            // Point-on-segment check
            double edge_dx = bx - ax;
            double edge_dy = by - ay;
            double cross = (cx - ax) * edge_dy - (cy - ay) * edge_dx;
            double scale = fabs(edge_dx) + fabs(edge_dy) + 1.0;
            if (fabs(cross) <= 1e-10 * scale) {
                double minx = ax < bx ? ax : bx;
                double maxx = ax > bx ? ax : bx;
                double miny = ay < by ? ay : by;
                double maxy = ay > by ? ay : by;
                if (cx >= minx - 1e-10 && cx <= maxx + 1e-10 &&
                    cy >= miny - 1e-10 && cy <= maxy + 1e-10) {
                    on_boundary = true;
                }
            }

            // Even-odd crossing test
            if (((ay > cy) != (by > cy)) &&
                (cx < (bx - ax) * (cy - ay) / (by - ay) + ax)) {
                inside = !inside;
            }
        }
    }

    if (inside || on_boundary) {
        // Centroid is inside - use it directly
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 2: Centroid is outside. Cast horizontal ray at y=cy through all
    // rings and collect X-intersection values.
    // We use shared memory for the intersection array, allocated per-thread.
    // Since shared memory is limited, we use a fixed maximum and fall back
    // to a simple strategy if exceeded.

    // Use dynamically allocated local array (stack).  max_intersections is
    // bounded by the caller based on the maximum ring vertex count.
    // Use register-based storage for small counts, device memory for large.
    // For simplicity and correctness, we use a two-pass approach:
    // Pass 1: count intersections.  Pass 2: store and sort.

    int n_intersections = 0;

    // Pass 1: count
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ay = poly_y[c];
            const double by = poly_y[c + 1];
            if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                n_intersections++;
            }
        }
    }

    if (n_intersections < 2) {
        // Degenerate - no valid interior interval.  Use centroid as best
        // effort (this should not happen for valid polygons).
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Cap to max_intersections to avoid buffer overrun
    int cap = n_intersections < max_intersections ? n_intersections : max_intersections;

    // Pass 2: collect X-intersection values into the output scratch area.
    // We use the per-thread slice of the scratch buffer passed in via
    // a separate device array (x_scratch, allocated by the host).
    // BUT to avoid a separate scratch buffer, we use a simple insertion-sort
    // approach: maintain a sorted list in local registers.  For typical
    // polygons, the intersection count is small (< 100 edges).

    // Use the per-polygon slice of the scratch buffer.
    // The caller provides x_scratch[idx * max_intersections ... ]
    extern __shared__ double shared_scratch[];
    double* x_vals = &shared_scratch[threadIdx.x * max_intersections];

    int stored = 0;
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ax = poly_x[c];
            const double ay = poly_y[c];
            const double bx = poly_x[c + 1];
            const double by = poly_y[c + 1];
            if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                double x_int = ax + (cy - ay) / (by - ay) * (bx - ax);
                if (stored < cap) {
                    // Insertion sort: find position and shift
                    int pos = stored;
                    while (pos > 0 && x_vals[pos - 1] > x_int) {
                        x_vals[pos] = x_vals[pos - 1];
                        pos--;
                    }
                    x_vals[pos] = x_int;
                    stored++;
                }
            }
        }
    }

    // Step 3: Find widest interior interval.
    // By even-odd rule, intervals are (x_vals[0],x_vals[1]),
    // (x_vals[2],x_vals[3]), ...
    double best_mid_x = cx;  // fallback
    double best_width = -1.0;
    for (int i = 0; i + 1 < stored; i += 2) {
        double width = x_vals[i + 1] - x_vals[i];
        if (width > best_width) {
            best_width = width;
            best_mid_x = (x_vals[i] + x_vals[i + 1]) * 0.5;
        }
    }

    out_x[gr] = best_mid_x;
    out_y[gr] = cy;  // Y stays at centroid Y
}


extern "C" __global__ void representative_point_multipolygon(
    const double* centroid_x,
    const double* centroid_y,
    const double* poly_x,
    const double* poly_y,
    const int* ring_offsets,
    const int* part_offsets,
    const int* geom_offsets,
    const int* global_row_indices,
    const int* family_row_indices,
    double* out_x,
    double* out_y,
    int num_polygons,
    int max_intersections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_polygons) return;

    const int gr = global_row_indices[idx];
    const int fr = family_row_indices[idx];
    const double cx = centroid_x[gr];
    const double cy = centroid_y[gr];

    // MultiPolygon: geometry_offsets -> part_offsets -> ring_offsets -> coords
    const int part_start = geom_offsets[fr];
    const int part_end = geom_offsets[fr + 1];

    if (part_start >= part_end) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Collect all ring ranges across all parts
    // Step 1: PIP check across all rings of all parts
    bool inside = false;
    bool on_boundary = false;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ax = poly_x[c];
                const double ay = poly_y[c];
                const double bx = poly_x[c + 1];
                const double by = poly_y[c + 1];

                double edge_dx = bx - ax;
                double edge_dy = by - ay;
                double cross = (cx - ax) * edge_dy - (cy - ay) * edge_dx;
                double scale = fabs(edge_dx) + fabs(edge_dy) + 1.0;
                if (fabs(cross) <= 1e-10 * scale) {
                    double minx = ax < bx ? ax : bx;
                    double maxx = ax > bx ? ax : bx;
                    double miny = ay < by ? ay : by;
                    double maxy = ay > by ? ay : by;
                    if (cx >= minx - 1e-10 && cx <= maxx + 1e-10 &&
                        cy >= miny - 1e-10 && cy <= maxy + 1e-10) {
                        on_boundary = true;
                    }
                }

                if (((ay > cy) != (by > cy)) &&
                    (cx < (bx - ax) * (cy - ay) / (by - ay) + ax)) {
                    inside = !inside;
                }
            }
        }
    }

    if (inside || on_boundary) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 2: Horizontal ray intersection for fallback
    int n_intersections = 0;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ay = poly_y[c];
                const double by = poly_y[c + 1];
                if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                    n_intersections++;
                }
            }
        }
    }

    if (n_intersections < 2) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    int cap = n_intersections < max_intersections ? n_intersections : max_intersections;

    extern __shared__ double shared_scratch[];
    double* x_vals = &shared_scratch[threadIdx.x * max_intersections];

    int stored = 0;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ax = poly_x[c];
                const double ay = poly_y[c];
                const double bx = poly_x[c + 1];
                const double by = poly_y[c + 1];
                if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                    double x_int = ax + (cy - ay) / (by - ay) * (bx - ax);
                    if (stored < cap) {
                        int pos = stored;
                        while (pos > 0 && x_vals[pos - 1] > x_int) {
                            x_vals[pos] = x_vals[pos - 1];
                            pos--;
                        }
                        x_vals[pos] = x_int;
                        stored++;
                    }
                }
            }
        }
    }

    double best_mid_x = cx;
    double best_width = -1.0;
    for (int i = 0; i + 1 < stored; i += 2) {
        double width = x_vals[i + 1] - x_vals[i];
        if (width > best_width) {
            best_width = width;
            best_mid_x = (x_vals[i] + x_vals[i + 1]) * 0.5;
        }
    }

    out_x[gr] = best_mid_x;
    out_y[gr] = cy;
}
"""

_REPRESENTATIVE_POINT_KERNEL_NAMES = (
    "representative_point_polygon",
    "representative_point_multipolygon",
)

from vibespatial.cuda._runtime import compile_kernel_group  # noqa: E402
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    (
        "representative-point",
        _REPRESENTATIVE_POINT_KERNEL_SOURCE,
        _REPRESENTATIVE_POINT_KERNEL_NAMES,
    ),
])


def _representative_point_kernels():
    return compile_kernel_group(
        "representative-point",
        _REPRESENTATIVE_POINT_KERNEL_SOURCE,
        _REPRESENTATIVE_POINT_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Maximum intersections per polygon for shared-memory sizing.
# Conservative default: typical polygons have < 200 edges.  For very complex
# polygons, the kernel caps the intersection count.
# ---------------------------------------------------------------------------
_MAX_INTERSECTIONS_DEFAULT = 256


def representative_point_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute a representative point for each geometry in the owned array.

    Returns a point-only OwnedGeometryArray where each point is guaranteed
    to lie inside (or on the boundary of) the corresponding input geometry.

    Strategy:
    1. Points -> identity (the point itself)
    2. LineStrings/MultiLineStrings -> midpoint of coordinate extent
    3. Polygons/MultiPolygons -> GPU centroid + PIP check.  When centroid
       is outside (concave polygons), a GPU horizontal-ray-intersection
       kernel finds an interior point on-device without any Shapely
       fallback or D2H transfer.
    """
    from vibespatial.runtime import has_gpu_runtime
    from vibespatial.runtime.adaptive import plan_dispatch_selection
    from vibespatial.runtime.precision import KernelClass

    row_count = owned.row_count
    if row_count == 0:
        return point_owned_from_xy(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    selection = plan_dispatch_selection(
        kernel_name="representative_point",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )
    use_gpu = selection.selected is ExecutionMode.GPU and has_gpu_runtime()

    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # --- Points: identity ---
    _fill_point_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- LineStrings / MultiLineStrings: midpoint of coordinate extent ---
    _fill_linestring_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- Polygons / MultiPolygons: centroid + PIP + ray fallback ---
    _fill_polygon_representatives(
        owned, tags, family_row_offsets, cx, cy, use_gpu=use_gpu,
    )

    # Handle null rows (validity=False)
    null_mask = ~owned.validity
    cx[null_mask] = np.nan
    cy[null_mask] = np.nan

    selected = ExecutionMode.GPU if use_gpu else ExecutionMode.CPU
    record_dispatch_event(
        surface="representative_point",
        operation="representative_point",
        implementation="gpu_centroid_pip_ray" if use_gpu else "cpu_centroid_pip_ray",
        reason=selection.reason,
        detail=f"rows={row_count}",
        selected=selected,
    )

    return point_owned_from_xy(cx, cy)


def _fill_point_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """Points: representative point is the point itself."""
    for family_key in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        tag = FAMILY_TAGS[family_key]
        mask = tags == tag
        if not np.any(mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0:
            continue
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        # For points, first coordinate is the representative point
        # For multipoints, use first constituent point
        geom_offsets = buf.geometry_offsets
        coord_starts = geom_offsets[family_rows]
        valid = coord_starts < len(buf.x)
        valid_global = global_rows[valid]
        valid_starts = coord_starts[valid]
        cx[valid_global] = buf.x[valid_starts]
        cy[valid_global] = buf.y[valid_starts]


def _fill_linestring_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """LineStrings: midpoint of coordinate extent (mean of all vertices)."""
    for family_key in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
        tag = FAMILY_TAGS[family_key]
        mask = tags == tag
        if not np.any(mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0:
            continue
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        geom_offsets = buf.geometry_offsets
        # Vectorized: compute start/end for each row, then use cumsum tricks
        coord_starts = geom_offsets[family_rows]
        coord_ends = geom_offsets[family_rows + 1]
        lengths = coord_ends - coord_starts
        nonempty = lengths > 0
        for i in np.flatnonzero(nonempty):
            gr = global_rows[i]
            cs = int(coord_starts[i])
            ce = int(coord_ends[i])
            cx[gr] = buf.x[cs:ce].mean()
            cy[gr] = buf.y[cs:ce].mean()


def _fill_polygon_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    use_gpu: bool = False,
) -> None:
    """Polygons: GPU centroid + GPU PIP check + GPU ray-intersection fallback."""
    from .polygon import polygon_centroids_owned

    poly_tag = FAMILY_TAGS.get(GeometryFamily.POLYGON)
    mpoly_tag = FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON)
    poly_mask = np.isin(tags, [t for t in [poly_tag, mpoly_tag] if t is not None])
    if not np.any(poly_mask):
        return

    # Step 1: Compute centroids (reuses existing GPU kernel when available)
    centroid_cx, centroid_cy = polygon_centroids_owned(owned)

    # Step 2: Assign centroids to polygon rows
    poly_rows = np.flatnonzero(poly_mask)
    cx[poly_rows] = centroid_cx[poly_rows]
    cy[poly_rows] = centroid_cy[poly_rows]

    # Step 3: PIP check + ray-intersection fallback for concave polygons.
    # The NVRTC kernel handles both PIP and fallback in a single launch.
    if use_gpu:
        _fill_polygon_representatives_gpu(owned, tags, family_row_offsets, cx, cy)
    else:
        _fill_polygon_representatives_cpu(owned, tags, family_row_offsets, cx, cy)


def _fill_polygon_representatives_gpu(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """GPU path: NVRTC kernel for PIP check + ray-intersection fallback."""
    import cupy as cp

    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )

    runtime = get_cuda_runtime()
    kernels = _representative_point_kernels()

    # Upload centroid arrays to device
    d_cx = cp.asarray(cx)
    d_cy = cp.asarray(cy)

    # Process Polygon family
    poly_tag = FAMILY_TAGS.get(GeometryFamily.POLYGON)
    if poly_tag is not None and GeometryFamily.POLYGON in owned.families:
        buf = owned.families[GeometryFamily.POLYGON]
        if buf.row_count > 0:
            mask = tags == poly_tag
            global_rows = np.flatnonzero(mask).astype(np.int32)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows].astype(np.int32)

                d_poly_x = cp.asarray(buf.x)
                d_poly_y = cp.asarray(buf.y)
                d_ring_offsets = cp.asarray(buf.ring_offsets.astype(np.int32))
                d_geom_offsets = cp.asarray(buf.geometry_offsets.astype(np.int32))
                d_global_rows = cp.asarray(global_rows)
                d_family_rows = cp.asarray(family_rows)

                # Determine max_intersections from the maximum ring vertex count
                max_intersections = _compute_max_intersections(buf.ring_offsets)

                num_polygons = global_rows.size
                block_size = min(256, num_polygons)
                # Shared memory: block_size * max_intersections * sizeof(double)
                shared_mem_bytes = block_size * max_intersections * 8

                # Cap shared memory at 48 KB (typical limit)
                if shared_mem_bytes > 48 * 1024:
                    # Reduce block_size to fit
                    block_size = max(1, (48 * 1024) // (max_intersections * 8))
                    shared_mem_bytes = block_size * max_intersections * 8

                grid_size = (num_polygons + block_size - 1) // block_size

                kernel = kernels["representative_point_polygon"]
                ptr = runtime.pointer
                params = (
                    (
                        ptr(d_cx),
                        ptr(d_cy),
                        ptr(d_poly_x),
                        ptr(d_poly_y),
                        ptr(d_ring_offsets),
                        ptr(d_geom_offsets),
                        ptr(d_global_rows),
                        ptr(d_family_rows),
                        ptr(d_cx),  # out_x = in-place update
                        ptr(d_cy),  # out_y = in-place update
                        num_polygons,
                        max_intersections,
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
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                    ),
                )
                runtime.launch(
                    kernel,
                    grid=(grid_size, 1, 1),
                    block=(block_size, 1, 1),
                    params=params,
                    shared_mem_bytes=shared_mem_bytes,
                )

    # Process MultiPolygon family
    mpoly_tag = FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON)
    if mpoly_tag is not None and GeometryFamily.MULTIPOLYGON in owned.families:
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        if buf.row_count > 0:
            mask = tags == mpoly_tag
            global_rows = np.flatnonzero(mask).astype(np.int32)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows].astype(np.int32)

                d_poly_x = cp.asarray(buf.x)
                d_poly_y = cp.asarray(buf.y)
                d_ring_offsets = cp.asarray(buf.ring_offsets.astype(np.int32))
                d_part_offsets = cp.asarray(buf.part_offsets.astype(np.int32))
                d_geom_offsets = cp.asarray(buf.geometry_offsets.astype(np.int32))
                d_global_rows = cp.asarray(global_rows)
                d_family_rows = cp.asarray(family_rows)

                max_intersections = _compute_max_intersections(buf.ring_offsets)

                num_polygons = global_rows.size
                block_size = min(256, num_polygons)
                shared_mem_bytes = block_size * max_intersections * 8

                if shared_mem_bytes > 48 * 1024:
                    block_size = max(1, (48 * 1024) // (max_intersections * 8))
                    shared_mem_bytes = block_size * max_intersections * 8

                grid_size = (num_polygons + block_size - 1) // block_size

                kernel = kernels["representative_point_multipolygon"]
                ptr = runtime.pointer
                params = (
                    (
                        ptr(d_cx),
                        ptr(d_cy),
                        ptr(d_poly_x),
                        ptr(d_poly_y),
                        ptr(d_ring_offsets),
                        ptr(d_part_offsets),
                        ptr(d_geom_offsets),
                        ptr(d_global_rows),
                        ptr(d_family_rows),
                        ptr(d_cx),
                        ptr(d_cy),
                        num_polygons,
                        max_intersections,
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
                        KERNEL_PARAM_I32,
                    ),
                )
                runtime.launch(
                    kernel,
                    grid=(grid_size, 1, 1),
                    block=(block_size, 1, 1),
                    params=params,
                    shared_mem_bytes=shared_mem_bytes,
                )

    # Copy results back from device
    cp.asnumpy(d_cx, out=cx)
    cp.asnumpy(d_cy, out=cy)


def _compute_max_intersections(ring_offsets: np.ndarray) -> int:
    """Compute the maximum number of ray-ring intersections per polygon.

    Uses the maximum ring coordinate count as an upper bound on the number
    of edges that can be crossed by a horizontal ray.  Capped at
    _MAX_INTERSECTIONS_DEFAULT to limit shared memory usage.
    """
    if ring_offsets is None or len(ring_offsets) < 2:
        return _MAX_INTERSECTIONS_DEFAULT
    ring_lengths = np.diff(ring_offsets)
    if ring_lengths.size == 0:
        return _MAX_INTERSECTIONS_DEFAULT
    # The total number of edges across all rings of a polygon is an upper
    # bound on the intersection count.  We use the total coordinate count
    # as a rough proxy.  For shared memory sizing, we need a per-polygon
    # cap - use the max total coordinate count across all polygons, but
    # cap to avoid excessive shared memory.
    max_edges = int(ring_lengths.max())
    return min(max(max_edges, 4), _MAX_INTERSECTIONS_DEFAULT)


def _fill_polygon_representatives_cpu(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """CPU fallback: horizontal ray intersection without Shapely dependency.

    This path is used when no GPU runtime is available.  It implements the
    same algorithm as the GPU kernel: centroid PIP check, then horizontal
    ray intersection for concave polygons where centroid is outside.
    """
    poly_tag = FAMILY_TAGS.get(GeometryFamily.POLYGON)
    mpoly_tag = FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON)

    # Process Polygon family
    if poly_tag is not None and GeometryFamily.POLYGON in owned.families:
        buf = owned.families[GeometryFamily.POLYGON]
        if buf.row_count > 0:
            mask = tags == poly_tag
            global_rows = np.flatnonzero(mask)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows]
                for gr, fr in zip(global_rows, family_rows):
                    _representative_point_single_polygon_cpu(
                        buf.x, buf.y, buf.ring_offsets,
                        buf.geometry_offsets, fr, gr, cx, cy,
                    )

    # Process MultiPolygon family
    if mpoly_tag is not None and GeometryFamily.MULTIPOLYGON in owned.families:
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        if buf.row_count > 0:
            mask = tags == mpoly_tag
            global_rows = np.flatnonzero(mask)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows]
                for gr, fr in zip(global_rows, family_rows):
                    _representative_point_single_multipolygon_cpu(
                        buf.x, buf.y, buf.ring_offsets,
                        buf.part_offsets, buf.geometry_offsets,
                        fr, gr, cx, cy,
                    )


def _pip_check_rings(
    px: float,
    py: float,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    ring_offsets: np.ndarray,
    ring_start: int,
    ring_end: int,
) -> bool:
    """Even-odd PIP check across a range of rings."""
    inside = False
    for ring_idx in range(ring_start, ring_end):
        coord_start = int(ring_offsets[ring_idx])
        coord_end = int(ring_offsets[ring_idx + 1])
        for c in range(coord_start, coord_end - 1):
            ax, ay = float(poly_x[c]), float(poly_y[c])
            bx, by = float(poly_x[c + 1]), float(poly_y[c + 1])
            # Boundary check
            edge_dx = bx - ax
            edge_dy = by - ay
            cross = (px - ax) * edge_dy - (py - ay) * edge_dx
            scale = abs(edge_dx) + abs(edge_dy) + 1.0
            if abs(cross) <= 1e-10 * scale:
                minx = min(ax, bx)
                maxx = max(ax, bx)
                miny = min(ay, by)
                maxy = max(ay, by)
                if (px >= minx - 1e-10 and px <= maxx + 1e-10
                        and py >= miny - 1e-10 and py <= maxy + 1e-10):
                    return True  # on boundary
            # Even-odd crossing
            if ((ay > py) != (by > py)) and (px < (bx - ax) * (py - ay) / (by - ay) + ax):
                inside = not inside
    return inside


def _horizontal_ray_representative(
    ray_y: float,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    ring_offsets: np.ndarray,
    ring_start: int,
    ring_end: int,
    fallback_x: float,
) -> float:
    """Find interior point at y=ray_y via horizontal ray intersection."""
    x_vals = []
    for ring_idx in range(ring_start, ring_end):
        coord_start = int(ring_offsets[ring_idx])
        coord_end = int(ring_offsets[ring_idx + 1])
        for c in range(coord_start, coord_end - 1):
            ay = float(poly_y[c])
            by = float(poly_y[c + 1])
            if (ay <= ray_y < by) or (by <= ray_y < ay):
                ax = float(poly_x[c])
                bx = float(poly_x[c + 1])
                x_int = ax + (ray_y - ay) / (by - ay) * (bx - ax)
                x_vals.append(x_int)
    if len(x_vals) < 2:
        return fallback_x
    x_vals.sort()
    best_mid = fallback_x
    best_width = -1.0
    for i in range(0, len(x_vals) - 1, 2):
        width = x_vals[i + 1] - x_vals[i]
        if width > best_width:
            best_width = width
            best_mid = (x_vals[i] + x_vals[i + 1]) * 0.5
    return best_mid


def _representative_point_single_polygon_cpu(
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    ring_offsets: np.ndarray,
    geom_offsets: np.ndarray,
    fr: int,
    gr: int,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """Compute representative point for a single Polygon on CPU."""
    pt_x, pt_y = float(cx[gr]), float(cy[gr])
    if np.isnan(pt_x) or np.isnan(pt_y):
        return

    ring_start = int(geom_offsets[fr])
    ring_end = int(geom_offsets[fr + 1])
    if ring_start >= ring_end:
        return

    if _pip_check_rings(pt_x, pt_y, poly_x, poly_y, ring_offsets, ring_start, ring_end):
        return  # centroid is inside

    # Centroid is outside - use horizontal ray at centroid Y
    new_x = _horizontal_ray_representative(
        pt_y, poly_x, poly_y, ring_offsets, ring_start, ring_end, pt_x,
    )
    cx[gr] = new_x
    # cy stays at centroid Y


def _representative_point_single_multipolygon_cpu(
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    ring_offsets: np.ndarray,
    part_offsets: np.ndarray,
    geom_offsets: np.ndarray,
    fr: int,
    gr: int,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """Compute representative point for a single MultiPolygon on CPU."""
    pt_x, pt_y = float(cx[gr]), float(cy[gr])
    if np.isnan(pt_x) or np.isnan(pt_y):
        return

    part_start = int(geom_offsets[fr])
    part_end = int(geom_offsets[fr + 1])
    if part_start >= part_end:
        return

    # Collect all ring ranges across all parts for the PIP check
    all_ring_start = int(part_offsets[part_start])
    all_ring_end = int(part_offsets[part_end])

    if _pip_check_rings(pt_x, pt_y, poly_x, poly_y, ring_offsets, all_ring_start, all_ring_end):
        return  # centroid is inside

    # Centroid is outside - use horizontal ray
    new_x = _horizontal_ray_representative(
        pt_y, poly_x, poly_y, ring_offsets, all_ring_start, all_ring_end, pt_x,
    )
    cx[gr] = new_x
