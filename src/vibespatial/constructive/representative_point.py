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

from vibespatial.constructive.representative_point_kernels import (
    _MAX_INTERSECTIONS_DEFAULT,
    _REPRESENTATIVE_POINT_KERNEL_NAMES,
    _REPRESENTATIVE_POINT_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import compile_kernel_group
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event

from .point import point_owned_from_xy

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


def _build_point_oga_metadata(row_count: int, validity):
    """Build device-resident validity, tags, and family row offsets for Point OGA."""
    import cupy as cp

    d_validity = cp.asarray(validity, dtype=cp.bool_)
    out_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=cp.int8)
    out_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    d_null = ~d_validity
    if int(d_null.any()) != 0:
        out_tags[d_null] = -1
        out_family_row_offsets[d_null] = -1
    return d_validity, out_tags, out_family_row_offsets


def _build_device_resident_point_output_from_device(
    d_cx,  # cupy device array
    d_cy,  # cupy device array
    validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident Point OGA from CuPy device coordinate arrays.

    Zero D2H transfer — device arrays are used directly.
    """
    import cupy as cp

    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    row_count = int(d_cx.size)
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_empty = cp.zeros(row_count, dtype=cp.uint8)

    d_validity, out_tags, out_family_row_offsets = _build_point_oga_metadata(
        row_count, validity,
    )

    device_families = {
        GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=d_cx,
            y=d_cy,
            geometry_offsets=d_geom_offsets,
            empty_mask=d_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _build_device_resident_point_output_from_host(
    cx: np.ndarray,
    cy: np.ndarray,
    validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident Point OGA by uploading host coordinate arrays.

    Used when GPU is selected but the coordinates were computed on host
    (e.g. point-only or linestring-only inputs with no polygon rows).
    """
    import cupy as cp

    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    row_count = len(cx)
    device_x = cp.asarray(cx)
    device_y = cp.asarray(cy)
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_empty = cp.zeros(row_count, dtype=cp.uint8)

    d_validity, out_tags, out_family_row_offsets = _build_point_oga_metadata(
        row_count, validity,
    )

    device_families = {
        GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=device_x,
            y=device_y,
            geometry_offsets=d_geom_offsets,
            empty_mask=d_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


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

    When the GPU path is active, the output OGA is device-resident — no
    D2H transfer occurs for the coordinate data.
    """
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
    use_gpu = selection.selected is ExecutionMode.GPU

    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # --- Points: identity ---
    _fill_point_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- LineStrings / MultiLineStrings: midpoint of coordinate extent ---
    _fill_linestring_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- Polygons / MultiPolygons: centroid + PIP + ray fallback ---
    device_result = _fill_polygon_representatives(
        owned, tags, family_row_offsets, cx, cy, use_gpu=use_gpu,
    )

    selected = ExecutionMode.GPU if use_gpu else ExecutionMode.CPU
    record_dispatch_event(
        surface="representative_point",
        operation="representative_point",
        implementation="gpu_centroid_pip_ray" if use_gpu else "cpu_centroid_pip_ray",
        reason=selection.reason,
        detail=f"rows={row_count}",
        selected=selected,
    )

    if use_gpu and device_result is not None:
        # GPU path produced device arrays that already contain all rows
        # (point/linestring values from the host upload, polygon values
        # from the kernel).  Apply null masking on device and build
        # device-resident output — zero D2H transfer.
        import cupy as cp

        d_cx, d_cy = device_result
        null_mask = ~owned.validity
        if null_mask.any():
            d_null = cp.asarray(null_mask)
            d_cx[d_null] = cp.nan
            d_cy[d_null] = cp.nan
        return _build_device_resident_point_output_from_device(
            d_cx, d_cy, owned.validity,
        )

    # CPU path or GPU path without polygon rows: host arrays are authoritative
    null_mask = ~owned.validity
    cx[null_mask] = np.nan
    cy[null_mask] = np.nan

    if use_gpu:
        # GPU was selected but no polygon rows — still build device-resident
        return _build_device_resident_point_output_from_host(
            cx, cy, owned.validity,
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
) -> tuple | None:
    """Polygons: GPU centroid + GPU PIP check + GPU ray-intersection fallback.

    When *use_gpu* is True and the GPU kernel runs, returns ``(d_cx, d_cy)``
    — CuPy device arrays containing the final representative-point
    coordinates for ALL rows (point/linestring values come from the host
    upload, polygon values from the kernel).  The caller should build the
    output OGA directly from these device arrays to avoid a D2H round-trip.

    Returns ``None`` on the CPU path or when there are no polygon rows.
    """
    from .polygon import polygon_centroids_owned

    poly_tag = FAMILY_TAGS.get(GeometryFamily.POLYGON)
    mpoly_tag = FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON)
    poly_mask = np.isin(tags, [t for t in [poly_tag, mpoly_tag] if t is not None])
    if not np.any(poly_mask):
        return None

    # Step 1: Compute centroids (reuses existing GPU kernel when available)
    centroid_cx, centroid_cy = polygon_centroids_owned(owned)

    # Step 2: Assign centroids to polygon rows
    poly_rows = np.flatnonzero(poly_mask)
    cx[poly_rows] = centroid_cx[poly_rows]
    cy[poly_rows] = centroid_cy[poly_rows]

    # Step 3: PIP check + ray-intersection fallback for concave polygons.
    # The NVRTC kernel handles both PIP and fallback in a single launch.
    if use_gpu:
        return _fill_polygon_representatives_gpu(owned, tags, family_row_offsets, cx, cy)
    else:
        _fill_polygon_representatives_cpu(owned, tags, family_row_offsets, cx, cy)
        return None


def _fill_polygon_representatives_gpu(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple:
    """GPU path: NVRTC kernel for PIP check + ray-intersection fallback.

    Returns ``(d_cx, d_cy)`` — CuPy device arrays with final coordinates.
    The host arrays ``cx``/``cy`` (with point/linestring/centroid values)
    are uploaded to device; the kernel modifies polygon rows in-place.
    The device arrays are returned without copying back to host.
    """
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

    # Return device arrays — caller builds device-resident OGA directly.
    # No D2H transfer: the kernel-modified d_cx/d_cy stay on device.
    return d_cx, d_cy


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
