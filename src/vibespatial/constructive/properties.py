"""GPU-accelerated geometry property computations.

All operations are pure offset arithmetic — no coordinate reads needed
(except is_closed and is_ccw which read coordinates).

Operations:
- num_coordinates: geometry_offsets[row+1] - geometry_offsets[row]
  (for polygon/multipolygon, uses ring_offsets indirection)
- num_geometries: always 1 for simple types, part count for multi types
- num_interior_rings: ring_count - 1 per polygon
- x/y accessors: read coordinate at geometry_offsets[row] for Point family
- is_closed: compare first/last coordinates per span (NVRTC Tier 1)
- is_ccw: shoelace signed area on exterior ring (NVRTC Tier 1)
- get_geometry: extract i-th sub-geometry from Multi* types (Tier 2: CuPy)

ADR-0033:
- num_coordinates, num_geometries, num_interior_rings: Tier 1 (NVRTC offset_diff)
  for simple/multi families; Tier 1 (NVRTC nested-loop) for polygon families
- is_closed, is_ccw: Tier 1 (NVRTC)
- get_geometry: Tier 2 (CuPy offset arithmetic, no custom NVRTC kernel)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.properties_cpu import get_geometry_cpu
from vibespatial.constructive.properties_kernels import (
    _IS_CCW_FP64,
    _IS_CCW_KERNEL_NAMES,
    _IS_CLOSED_FP64,
    _IS_CLOSED_KERNEL_NAMES,
    _NUM_COORDS_FP64,
    _NUM_COORDS_KERNEL_NAMES,
    _OFFSET_DIFF_FP64,
    _OFFSET_DIFF_KERNEL_NAMES,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

request_nvrtc_warmup([
    ("is-closed-fp64", _IS_CLOSED_FP64, _IS_CLOSED_KERNEL_NAMES),
    ("is-ccw-fp64", _IS_CCW_FP64, _IS_CCW_KERNEL_NAMES),
    ("offset-diff", _OFFSET_DIFF_FP64, _OFFSET_DIFF_KERNEL_NAMES),
])


def _offset_diff_gpu(d_offsets, d_family_rows, result, global_rows, *, kernel_name="offset_diff"):
    """Launch the offset_diff NVRTC kernel and scatter results into *result*."""
    runtime = get_cuda_runtime()
    n = int(d_family_rows.size)
    kernel_group = compile_kernel_group(
        "offset-diff", _OFFSET_DIFF_FP64, _OFFSET_DIFF_KERNEL_NAMES,
    )
    kernel = kernel_group[kernel_name]
    d_out = runtime.allocate((n,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_offsets), ptr(d_family_rows), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        result[global_rows] = runtime.copy_device_to_host(d_out)
    finally:
        runtime.free(d_out)


request_nvrtc_warmup([
    ("num-coords-fp64", _NUM_COORDS_FP64, _NUM_COORDS_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# num_coordinates_owned
# ---------------------------------------------------------------------------


def num_coordinates_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Compute per-geometry coordinate count from offset arrays.

    Avoids Shapely materialization by reading offset buffers directly.
    Device-native path uses NVRTC offset_diff kernel for simple families
    and nested-loop NVRTC kernels for families with offset indirection
    (Polygon, MultiLineString, MultiPolygon).
    """
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.int32)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path: NVRTC offset_diff for simple, nested-loop for complex ---
        if (
            cp is not None
            and owned.device_state is not None
            and family in owned.device_state.families
        ):
            d_buf = owned.device_state.families[family]
            d_family_rows = cp.asarray(family_rows)

            if family in (
                GeometryFamily.POINT,
                GeometryFamily.LINESTRING,
                GeometryFamily.MULTIPOINT,
            ):
                _offset_diff_gpu(
                    d_buf.geometry_offsets, d_family_rows, result, global_rows,
                )

            elif family is GeometryFamily.POLYGON:
                # Tier 1 NVRTC: ring offset indirection
                _num_coords_polygon_gpu(d_buf, d_family_rows, result, global_rows)

            elif family is GeometryFamily.MULTILINESTRING:
                # Tier 1 NVRTC: part offset indirection
                _num_coords_multilinestring_gpu(d_buf, d_family_rows, result, global_rows)

            elif family is GeometryFamily.MULTIPOLYGON:
                # Tier 1 NVRTC: part + ring offset indirection
                _num_coords_multipolygon_gpu(d_buf, d_family_rows, result, global_rows)

            continue  # skip host fallback

        # --- Host path (existing logic) ---
        if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
            # Direct: geometry_offsets[row+1] - geometry_offsets[row]
            offsets = buf.geometry_offsets
            counts = offsets[family_rows + 1] - offsets[family_rows]
            result[global_rows] = counts
        elif family is GeometryFamily.POLYGON:
            # Sum coordinates across all rings
            offsets = buf.geometry_offsets
            ring_offsets = buf.ring_offsets
            for gi, fr in zip(global_rows, family_rows):
                ring_start = offsets[fr]
                ring_end = offsets[fr + 1]
                total = 0
                for ri in range(ring_start, ring_end):
                    total += int(ring_offsets[ri + 1] - ring_offsets[ri])
                result[gi] = total
        elif family is GeometryFamily.MULTILINESTRING:
            offsets = buf.geometry_offsets
            part_offsets = buf.part_offsets
            for gi, fr in zip(global_rows, family_rows):
                part_start = offsets[fr]
                part_end = offsets[fr + 1]
                total = 0
                for pi in range(part_start, part_end):
                    total += int(part_offsets[pi + 1] - part_offsets[pi])
                result[gi] = total
        elif family is GeometryFamily.MULTIPOLYGON:
            offsets = buf.geometry_offsets
            part_offsets = buf.part_offsets
            ring_offsets = buf.ring_offsets
            for gi, fr in zip(global_rows, family_rows):
                poly_start = offsets[fr]
                poly_end = offsets[fr + 1]
                total = 0
                for pi in range(poly_start, poly_end):
                    ring_start = part_offsets[pi]
                    ring_end = part_offsets[pi + 1]
                    for ri in range(ring_start, ring_end):
                        total += int(ring_offsets[ri + 1] - ring_offsets[ri])
                result[gi] = total

    # Invalid rows get 0
    result[~owned.validity] = 0
    return result


def _num_coords_polygon_gpu(d_buf, d_family_rows, result, global_rows):
    """Launch NVRTC kernel for polygon coordinate counting (ring indirection)."""
    runtime = get_cuda_runtime()
    n = int(d_family_rows.size)
    kernel_group = compile_kernel_group("num-coords-fp64", _NUM_COORDS_FP64, _NUM_COORDS_KERNEL_NAMES)
    kernel = kernel_group["num_coords_polygon"]
    d_out = runtime.allocate((n,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_buf.ring_offsets), ptr(d_buf.geometry_offsets),
             ptr(d_family_rows), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        result[global_rows] = runtime.copy_device_to_host(d_out)
    finally:
        runtime.free(d_out)


def _num_coords_multilinestring_gpu(d_buf, d_family_rows, result, global_rows):
    """Launch NVRTC kernel for multilinestring coordinate counting (part indirection)."""
    runtime = get_cuda_runtime()
    n = int(d_family_rows.size)
    kernel_group = compile_kernel_group("num-coords-fp64", _NUM_COORDS_FP64, _NUM_COORDS_KERNEL_NAMES)
    kernel = kernel_group["num_coords_multilinestring"]
    d_out = runtime.allocate((n,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_buf.part_offsets), ptr(d_buf.geometry_offsets),
             ptr(d_family_rows), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        result[global_rows] = runtime.copy_device_to_host(d_out)
    finally:
        runtime.free(d_out)


def _num_coords_multipolygon_gpu(d_buf, d_family_rows, result, global_rows):
    """Launch NVRTC kernel for multipolygon coordinate counting (part + ring indirection)."""
    runtime = get_cuda_runtime()
    n = int(d_family_rows.size)
    kernel_group = compile_kernel_group("num-coords-fp64", _NUM_COORDS_FP64, _NUM_COORDS_KERNEL_NAMES)
    kernel = kernel_group["num_coords_multipolygon"]
    d_out = runtime.allocate((n,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_buf.ring_offsets), ptr(d_buf.part_offsets),
             ptr(d_buf.geometry_offsets), ptr(d_family_rows), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        result[global_rows] = runtime.copy_device_to_host(d_out)
    finally:
        runtime.free(d_out)


# ---------------------------------------------------------------------------
# num_geometries_owned
# ---------------------------------------------------------------------------


def num_geometries_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Compute per-geometry part count from offset arrays.

    Simple types return 1, multi types return part count.
    Device-native path uses NVRTC offset_diff kernel (Tier 1).
    """
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.int32)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path: NVRTC offset_diff for multi types ---
        if (
            cp is not None
            and owned.device_state is not None
            and family in owned.device_state.families
        ):
            d_buf = owned.device_state.families[family]

            if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.POLYGON):
                result[global_rows] = 1
            else:
                d_family_rows = cp.asarray(family_rows)
                _offset_diff_gpu(
                    d_buf.geometry_offsets, d_family_rows, result, global_rows,
                )

            continue  # skip host fallback

        # --- Host path (existing logic) ---
        if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.POLYGON):
            result[global_rows] = 1
        else:
            offsets = buf.geometry_offsets
            counts = offsets[family_rows + 1] - offsets[family_rows]
            result[global_rows] = counts

    result[~owned.validity] = 0
    return result


# ---------------------------------------------------------------------------
# num_interior_rings_owned
# ---------------------------------------------------------------------------


def num_interior_rings_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Compute per-geometry interior ring count.

    Only meaningful for Polygon family (ring_count - 1).
    Device-native path uses NVRTC offset_diff_interior_rings kernel (Tier 1).
    """
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.int32)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mask = tags == poly_tag
    if np.any(mask) and GeometryFamily.POLYGON in owned.families:
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path: NVRTC offset_diff_interior_rings ---
        if (
            cp is not None
            and owned.device_state is not None
            and GeometryFamily.POLYGON in owned.device_state.families
        ):
            d_buf = owned.device_state.families[GeometryFamily.POLYGON]
            d_family_rows = cp.asarray(family_rows)
            _offset_diff_gpu(
                d_buf.geometry_offsets, d_family_rows, result, global_rows,
                kernel_name="offset_diff_interior_rings",
            )
        else:
            # --- Host path ---
            buf = owned.families[GeometryFamily.POLYGON]
            offsets = buf.geometry_offsets
            ring_counts = offsets[family_rows + 1] - offsets[family_rows]
            result[global_rows] = np.maximum(ring_counts - 1, 0)

    result[~owned.validity] = 0
    return result


# ---------------------------------------------------------------------------
# get_x_owned / get_y_owned (already have device paths)
# ---------------------------------------------------------------------------


def get_x_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Extract x coordinates for Point geometries from coordinate buffers.

    When device_state is populated with Point family, reads directly from
    device buffers via CuPy without calling _ensure_host_state().
    """
    row_count = owned.row_count
    result = np.full(row_count, np.nan, dtype=np.float64)

    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    mask = owned.tags == point_tag
    if not np.any(mask) or GeometryFamily.POINT not in owned.families:
        return result

    global_rows = np.flatnonzero(mask)
    family_rows = owned.family_row_offsets[global_rows]

    # Device-native path: single D->H transfer of just the result values
    if (
        cp is not None
        and owned.device_state is not None
        and GeometryFamily.POINT in owned.device_state.families
    ):
        d_buf = owned.device_state.families[GeometryFamily.POINT]
        d_family_rows = cp.asarray(family_rows)
        d_coord_indices = d_buf.geometry_offsets[d_family_rows]
        result[global_rows] = cp.asnumpy(d_buf.x[d_coord_indices])
        d_empty = cp.asnumpy(d_buf.empty_mask[d_family_rows])
        if np.any(d_empty):
            result[global_rows[d_empty]] = np.nan
        return result

    # Host path: ensure buffers are materialized
    if not owned.families[GeometryFamily.POINT].host_materialized:
        owned._ensure_host_state()

    buf = owned.families[GeometryFamily.POINT]
    offsets = buf.geometry_offsets
    coord_indices = offsets[family_rows]
    result[global_rows] = buf.x[coord_indices]

    # Handle empty geometries
    if buf.empty_mask.size > 0:
        empty_family_rows = family_rows[buf.empty_mask[family_rows]]
        if len(empty_family_rows) > 0:
            empty_global = global_rows[buf.empty_mask[family_rows]]
            result[empty_global] = np.nan

    return result


def get_y_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Extract y coordinates for Point geometries from coordinate buffers.

    When device_state is populated with Point family, reads directly from
    device buffers via CuPy without calling _ensure_host_state().
    """
    row_count = owned.row_count
    result = np.full(row_count, np.nan, dtype=np.float64)

    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    mask = owned.tags == point_tag
    if not np.any(mask) or GeometryFamily.POINT not in owned.families:
        return result

    global_rows = np.flatnonzero(mask)
    family_rows = owned.family_row_offsets[global_rows]

    # Device-native path: single D->H transfer of just the result values
    if (
        cp is not None
        and owned.device_state is not None
        and GeometryFamily.POINT in owned.device_state.families
    ):
        d_buf = owned.device_state.families[GeometryFamily.POINT]
        d_family_rows = cp.asarray(family_rows)
        d_coord_indices = d_buf.geometry_offsets[d_family_rows]
        result[global_rows] = cp.asnumpy(d_buf.y[d_coord_indices])
        d_empty = cp.asnumpy(d_buf.empty_mask[d_family_rows])
        if np.any(d_empty):
            result[global_rows[d_empty]] = np.nan
        return result

    # Host path: ensure buffers are materialized
    if not owned.families[GeometryFamily.POINT].host_materialized:
        owned._ensure_host_state()

    buf = owned.families[GeometryFamily.POINT]
    offsets = buf.geometry_offsets
    coord_indices = offsets[family_rows]
    result[global_rows] = buf.y[coord_indices]

    # Handle empty geometries
    if buf.empty_mask.size > 0:
        empty_family_rows = family_rows[buf.empty_mask[family_rows]]
        if len(empty_family_rows) > 0:
            empty_global = global_rows[buf.empty_mask[family_rows]]
            result[empty_global] = np.nan

    return result


# ---------------------------------------------------------------------------
# is_closed_owned (Tier 1: NVRTC kernel)
# ---------------------------------------------------------------------------


def is_closed_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Check if geometries are closed (first coord == last coord).

    For LineString: compare first and last coordinate.
    For MultiLineString: all parts must be closed.
    For Polygon/MultiPolygon/Point/MultiPoint: always True.

    Device-native path uses NVRTC kernels for LineString and
    MultiLineString families. Other families are handled as constants.
    """
    row_count = owned.row_count
    result = np.ones(row_count, dtype=bool)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        # Points, MultiPoints, Polygons, MultiPolygons are always closed
        if family not in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path: NVRTC kernel ---
        if (
            cp is not None
            and owned.device_state is not None
            and family in owned.device_state.families
        ):
            d_buf = owned.device_state.families[family]
            d_family_rows = cp.asarray(family_rows)
            n = d_family_rows.size

            runtime = get_cuda_runtime()
            kernel_group = compile_kernel_group(
                "is-closed-fp64", _IS_CLOSED_FP64, _IS_CLOSED_KERNEL_NAMES,
            )
            d_out = runtime.allocate((n,), np.int32, zero=True)
            try:
                ptr = runtime.pointer
                if family is GeometryFamily.LINESTRING:
                    kernel = kernel_group["is_closed_linestring"]
                    params = (
                        (ptr(d_buf.x), ptr(d_buf.y),
                         ptr(d_buf.geometry_offsets), ptr(d_family_rows),
                         ptr(d_out), n),
                        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                    )
                else:  # MULTILINESTRING
                    kernel = kernel_group["is_closed_multilinestring"]
                    params = (
                        (ptr(d_buf.x), ptr(d_buf.y),
                         ptr(d_buf.part_offsets), ptr(d_buf.geometry_offsets),
                         ptr(d_family_rows), ptr(d_out), n),
                        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                    )
                grid, block = runtime.launch_config(kernel, n)
                runtime.launch(kernel, grid=grid, block=block, params=params)
                h_out = runtime.copy_device_to_host(d_out)
                result[global_rows] = h_out.astype(bool)
            finally:
                runtime.free(d_out)

            continue  # skip host fallback

        # --- Host path (existing logic) ---
        if family is GeometryFamily.LINESTRING:
            offsets = buf.geometry_offsets
            for gi, fr in zip(global_rows, family_rows):
                start = int(offsets[fr])
                end = int(offsets[fr + 1])
                if end - start < 2:
                    result[gi] = True
                    continue
                result[gi] = (buf.x[start] == buf.x[end - 1] and
                              buf.y[start] == buf.y[end - 1])

        elif family is GeometryFamily.MULTILINESTRING:
            geom_offsets = buf.geometry_offsets
            part_offsets = buf.part_offsets
            for gi, fr in zip(global_rows, family_rows):
                ps = int(geom_offsets[fr])
                pe = int(geom_offsets[fr + 1])
                closed = True
                for p in range(ps, pe):
                    cs = int(part_offsets[p])
                    ce = int(part_offsets[p + 1])
                    if ce - cs < 2:
                        continue
                    if buf.x[cs] != buf.x[ce - 1] or buf.y[cs] != buf.y[ce - 1]:
                        closed = False
                        break
                result[gi] = closed

    result[~owned.validity] = False
    return result


# ---------------------------------------------------------------------------
# is_ring_owned — fused closure + simplicity test (Tier 2: composition)
# ---------------------------------------------------------------------------


def is_ring_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Check if geometries are valid rings (closed AND simple).

    A geometry is a ring if it is a closed, simple LineString.
    Non-LineString types (Point, Polygon, Multi*) always return False,
    matching Shapely semantics.

    Algorithm:
    1. Check closure via ``is_closed_owned`` (first == last coordinate).
    2. For closed LineStrings only, check simplicity with ring-aware
       adjacency (``is_ring=1``): the first and last segments share an
       endpoint by definition, so they are treated as adjacent and not
       flagged as a self-intersection.
    3. ``is_ring = is_closed AND is_simple_ring_aware`` for LineStrings;
       False for all other families.

    Parameters
    ----------
    owned : OwnedGeometryArray
        The geometry array to check.
    dispatch_mode : ExecutionMode or str
        GPU/CPU/AUTO execution mode.
    precision : PrecisionMode or str
        Precision dispatch mode (ADR-0002). PREDICATE class defaults to fp64.

    Returns
    -------
    np.ndarray of bool
        Per-geometry ring flags. True only for closed, simple LineStrings.
    """
    row_count = owned.row_count
    if row_count == 0:
        return np.array([], dtype=bool)

    selection = plan_dispatch_selection(
        kernel_name="is_ring",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    # Start with all False -- only closed, simple LineStrings become True.
    result = np.zeros(row_count, dtype=bool)

    tags = owned.tags
    linestring_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    linestring_mask = tags == linestring_tag

    # Fast exit: no LineStrings in the array -> all False.
    if not np.any(linestring_mask):
        record_dispatch_event(
            surface="geopandas.array.is_ring",
            operation="is_ring",
            requested=dispatch_mode,
            selected=selection.selected,
            implementation="is_ring_composed",
            reason=selection.reason,
            detail="no linestrings -- all False",
        )
        return result

    # Step 1: closure check (reuse existing GPU/CPU kernel).
    closed = is_closed_owned(owned)

    # Candidate rows: LineStrings that are closed AND valid.
    candidates = linestring_mask & closed & owned.validity
    if not np.any(candidates):
        record_dispatch_event(
            surface="geopandas.array.is_ring",
            operation="is_ring",
            requested=dispatch_mode,
            selected=selection.selected,
            implementation="is_ring_composed",
            reason=selection.reason,
            detail="no closed linestrings -- all False",
        )
        return result

    # Step 2: ring-aware simplicity check for closed LineStrings.
    # Uses is_ring=1 so first-last segment adjacency is skipped.
    family = GeometryFamily.LINESTRING
    buf = owned.families[family]
    family_row_offsets_arr = owned.family_row_offsets

    global_rows = np.flatnonzero(candidates)
    family_rows = family_row_offsets_arr[global_rows]

    actually_used_gpu = False
    use_gpu = (
        selection.selected is ExecutionMode.GPU
        and cp is not None
        and owned.device_state is not None
        and family in owned.device_state.families
    )

    if use_gpu:
        try:
            from vibespatial.constructive.validity import (
                _launch_is_simple_kernel,
            )

            d_buf = owned.device_state.families[family]
            runtime = get_cuda_runtime()
            total_spans = int(d_buf.geometry_offsets.shape[0]) - 1

            if total_spans > 0:
                # Launch simplicity kernel with is_ring=1 for ring-aware
                # adjacency (first-last segment pair treated as adjacent).
                d_span_result = _launch_is_simple_kernel(
                    runtime,
                    d_buf.x,
                    d_buf.y,
                    d_buf.geometry_offsets,
                    total_spans,
                    is_ring=1,
                )
                try:
                    d_family_rows = cp.asarray(family_rows)
                    d_span_result_cp = cp.asarray(d_span_result)
                    h_result = cp.asnumpy(
                        d_span_result_cp[d_family_rows]
                    ).astype(bool)
                    result[global_rows] = h_result
                finally:
                    runtime.free(d_span_result)
            else:
                result[global_rows] = True

            actually_used_gpu = True
        except Exception:
            actually_used_gpu = False  # fall through to CPU

    if not actually_used_gpu:
        # CPU fallback: ring-aware simplicity check.
        from vibespatial.constructive.validity import (
            _linestring_self_intersects,
        )

        x = buf.x
        y = buf.y
        offsets = buf.geometry_offsets
        for gi, fr in zip(global_rows, family_rows):
            start = int(offsets[fr])
            end = int(offsets[fr + 1])
            if start == end:
                result[gi] = True
                continue
            # is_ring=True: treat first-last as adjacent (closure endpoint).
            if not _linestring_self_intersects(
                x, y, start, end, is_ring=True
            ):
                result[gi] = True

        if selection.selected is ExecutionMode.GPU:
            record_fallback_event(
                surface="geopandas.array.is_ring",
                reason="GPU simplicity kernel failed, fell back to CPU",
                d2h_transfer=True,
            )

    impl = (
        "is_ring_gpu_composed"
        if actually_used_gpu
        else "is_ring_cpu_composed"
    )
    record_dispatch_event(
        surface="geopandas.array.is_ring",
        operation="is_ring",
        requested=dispatch_mode,
        selected=(
            ExecutionMode.GPU if actually_used_gpu else ExecutionMode.CPU
        ),
        implementation=impl,
        reason=selection.reason,
        detail="fused is_closed + ring-aware is_simple composition",
    )

    return result


# ---------------------------------------------------------------------------
# is_ccw_owned (Tier 1: NVRTC kernel)
# ---------------------------------------------------------------------------


def is_ccw_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Check if polygon exterior rings are counter-clockwise.

    Uses shoelace signed area: positive = CCW.
    Only meaningful for Polygon and MultiPolygon families. Other types
    return False.

    Device-native path uses NVRTC kernel for shoelace computation.
    """
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=bool)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        if family not in owned.families:
            continue
        buf = owned.families[family]
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path: NVRTC kernel ---
        if (
            cp is not None
            and owned.device_state is not None
            and family in owned.device_state.families
        ):
            d_buf = owned.device_state.families[family]
            d_family_rows = cp.asarray(family_rows)
            n = d_family_rows.size

            runtime = get_cuda_runtime()
            kernel_group = compile_kernel_group(
                "is-ccw-fp64", _IS_CCW_FP64, _IS_CCW_KERNEL_NAMES,
            )
            d_out = runtime.allocate((n,), np.int32, zero=True)
            try:
                ptr = runtime.pointer
                if family is GeometryFamily.POLYGON:
                    kernel = kernel_group["is_ccw_polygon"]
                    params = (
                        (ptr(d_buf.x), ptr(d_buf.y),
                         ptr(d_buf.ring_offsets), ptr(d_buf.geometry_offsets),
                         ptr(d_family_rows), ptr(d_out), n),
                        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                    )
                else:  # MULTIPOLYGON
                    kernel = kernel_group["is_ccw_multipolygon"]
                    params = (
                        (ptr(d_buf.x), ptr(d_buf.y),
                         ptr(d_buf.ring_offsets), ptr(d_buf.part_offsets),
                         ptr(d_buf.geometry_offsets), ptr(d_family_rows),
                         ptr(d_out), n),
                        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                    )
                grid, block = runtime.launch_config(kernel, n)
                runtime.launch(kernel, grid=grid, block=block, params=params)
                h_out = runtime.copy_device_to_host(d_out)
                result[global_rows] = h_out.astype(bool)
            finally:
                runtime.free(d_out)

            continue  # skip host fallback

        # --- Host path (existing logic) ---
        for gi, fr in zip(global_rows, family_rows):
            # Get first ring (exterior) coordinate range
            if family is GeometryFamily.MULTIPOLYGON:
                poly_start = int(buf.geometry_offsets[fr])
                ring_idx = int(buf.part_offsets[poly_start])
            else:
                ring_idx = int(buf.geometry_offsets[fr])

            coord_start = int(buf.ring_offsets[ring_idx])
            coord_end = int(buf.ring_offsets[ring_idx + 1])

            n_coords = coord_end - coord_start
            if n_coords < 3:
                continue
            x = buf.x[coord_start:coord_end]
            y = buf.y[coord_start:coord_end]
            area2 = float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            result[gi] = area2 > 0  # positive = CCW

    result[~owned.validity] = False
    return result


# ---------------------------------------------------------------------------
# get_geometry_owned — extract i-th sub-geometry from Multi* types
# ---------------------------------------------------------------------------

# Mapping from Multi* families to their simple counterparts.
_MULTI_TO_SIMPLE: dict[GeometryFamily, GeometryFamily] = {
    GeometryFamily.MULTIPOINT: GeometryFamily.POINT,
    GeometryFamily.MULTILINESTRING: GeometryFamily.LINESTRING,
    GeometryFamily.MULTIPOLYGON: GeometryFamily.POLYGON,
}

# Simple families (identity at index 0).
_SIMPLE_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.POINT,
    GeometryFamily.LINESTRING,
    GeometryFamily.POLYGON,
})


# ---------------------------------------------------------------------------
# GPU helpers: per-family sub-geometry extraction
# ---------------------------------------------------------------------------

def _get_geometry_multipoint_gpu(
    device_buf: DeviceFamilyGeometryBuffer,
    index: int,
    family_rows: np.ndarray,
) -> tuple[DeviceFamilyGeometryBuffer, Any]:
    """Extract i-th Point from each MultiPoint row on device.

    Returns the device buffer for Point family and a boolean mask of which
    family rows produced a valid result (in-bounds index).

    MultiPoint layout:
      geometry_offsets[row] .. geometry_offsets[row+1] -> coordinate indices
      The i-th point is coordinate at geometry_offsets[row] + i.
    """
    n = len(family_rows)
    d_family_rows = cp.asarray(family_rows)
    d_geom_offsets = device_buf.geometry_offsets

    # Part counts for each row
    d_starts = d_geom_offsets[d_family_rows]
    d_ends = d_geom_offsets[d_family_rows + 1]
    d_part_counts = d_ends - d_starts

    # Resolve negative index and determine validity
    eff_index = index
    if eff_index < 0:
        d_eff_index = d_part_counts + eff_index
    else:
        d_eff_index = cp.full(n, eff_index, dtype=cp.int32)

    d_valid = (d_eff_index >= 0) & (d_eff_index < d_part_counts)

    # Device-side early-exit check — avoid D2H transfer when all OOB
    if int(d_valid.sum().item()) == 0:
        d_empty_x = cp.empty(0, dtype=cp.float64)
        d_empty_y = cp.empty(0, dtype=cp.float64)
        d_empty_geom_offsets = cp.zeros(1, dtype=cp.int32)
        d_empty_mask = cp.empty(0, dtype=cp.bool_)
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=d_empty_x,
            y=d_empty_y,
            geometry_offsets=d_empty_geom_offsets,
            empty_mask=d_empty_mask,
        ), d_valid

    d_valid_local = cp.flatnonzero(d_valid)
    n_valid = int(d_valid_local.size)

    # Coordinate indices for valid rows
    d_coord_idx = d_starts[d_valid_local] + d_eff_index[d_valid_local]

    # Extract coordinates
    d_out_x = device_buf.x[d_coord_idx]
    d_out_y = device_buf.y[d_coord_idx]

    # Build Point offsets: each point has exactly 1 coordinate
    d_out_geom_offsets = cp.arange(n_valid + 1, dtype=cp.int32)
    d_out_empty = cp.zeros(n_valid, dtype=cp.bool_)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_out_empty,
    ), d_valid


def _get_geometry_multilinestring_gpu(
    device_buf: DeviceFamilyGeometryBuffer,
    index: int,
    family_rows: np.ndarray,
) -> tuple[DeviceFamilyGeometryBuffer, Any]:
    """Extract i-th LineString from each MultiLineString row on device.

    MultiLineString layout:
      geometry_offsets[row] .. geometry_offsets[row+1] -> part indices
      part_offsets[part] .. part_offsets[part+1] -> coordinate indices
      The i-th LineString is part geometry_offsets[row] + i.
    """
    n = len(family_rows)
    d_family_rows = cp.asarray(family_rows)
    d_geom_offsets = device_buf.geometry_offsets
    d_part_offsets = device_buf.part_offsets

    d_starts = d_geom_offsets[d_family_rows]
    d_ends = d_geom_offsets[d_family_rows + 1]
    d_part_counts = d_ends - d_starts

    eff_index = index
    if eff_index < 0:
        d_eff_index = d_part_counts + eff_index
    else:
        d_eff_index = cp.full(n, eff_index, dtype=cp.int32)

    d_valid = (d_eff_index >= 0) & (d_eff_index < d_part_counts)

    # Device-side early-exit check — avoid D2H transfer when all OOB
    if int(d_valid.sum().item()) == 0:
        d_empty_x = cp.empty(0, dtype=cp.float64)
        d_empty_y = cp.empty(0, dtype=cp.float64)
        d_empty_geom_offsets = cp.zeros(1, dtype=cp.int32)
        d_empty_mask = cp.empty(0, dtype=cp.bool_)
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.LINESTRING,
            x=d_empty_x,
            y=d_empty_y,
            geometry_offsets=d_empty_geom_offsets,
            empty_mask=d_empty_mask,
        ), d_valid

    d_valid_local = cp.flatnonzero(d_valid)
    n_valid = int(d_valid_local.size)

    # Part index for each valid row
    d_part_idx = d_starts[d_valid_local] + d_eff_index[d_valid_local]

    # Coordinate range for each part
    d_coord_starts = d_part_offsets[d_part_idx]
    d_coord_ends = d_part_offsets[d_part_idx + 1]
    d_coord_lengths = d_coord_ends - d_coord_starts

    # Build output geometry offsets (cumsum of lengths)
    d_out_geom_offsets = cp.zeros(n_valid + 1, dtype=cp.int32)
    cp.cumsum(d_coord_lengths, out=d_out_geom_offsets[1:])

    # Total coordinates
    total_coords = int(d_out_geom_offsets[-1])

    if total_coords == 0:
        d_out_x = cp.empty(0, dtype=cp.float64)
        d_out_y = cp.empty(0, dtype=cp.float64)
    else:
        # Build flat index array to gather coordinates
        d_out_x = cp.empty(total_coords, dtype=cp.float64)
        d_out_y = cp.empty(total_coords, dtype=cp.float64)
        # For each valid row, copy coordinate range
        # Vectorized approach: create an index array mapping output position
        # to source position
        # CuPy repeat requires a Python list; transfer small offset array.
        h_coord_lengths = cp.asnumpy(d_coord_lengths).tolist()
        d_row_ids = cp.repeat(cp.arange(n_valid, dtype=cp.int32), h_coord_lengths)
        d_local_offsets = cp.arange(total_coords, dtype=cp.int32) - d_out_geom_offsets[:-1][d_row_ids]
        d_src_indices = d_coord_starts[d_row_ids] + d_local_offsets
        d_out_x = device_buf.x[d_src_indices]
        d_out_y = device_buf.y[d_src_indices]

    d_out_empty = d_coord_lengths == 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_out_empty,
    ), d_valid


def _get_geometry_multipolygon_gpu(
    device_buf: DeviceFamilyGeometryBuffer,
    index: int,
    family_rows: np.ndarray,
) -> tuple[DeviceFamilyGeometryBuffer, Any]:
    """Extract i-th Polygon from each MultiPolygon row on device.

    MultiPolygon layout:
      geometry_offsets[row] .. geometry_offsets[row+1] -> polygon indices
      part_offsets[polygon] .. part_offsets[polygon+1] -> ring indices
      ring_offsets[ring] .. ring_offsets[ring+1] -> coordinate indices
      The i-th Polygon is polygon geometry_offsets[row] + i.
    """
    n = len(family_rows)
    d_family_rows = cp.asarray(family_rows)
    d_geom_offsets = device_buf.geometry_offsets
    d_part_offsets = device_buf.part_offsets
    d_ring_offsets = device_buf.ring_offsets

    d_starts = d_geom_offsets[d_family_rows]
    d_ends = d_geom_offsets[d_family_rows + 1]
    d_part_counts = d_ends - d_starts

    eff_index = index
    if eff_index < 0:
        d_eff_index = d_part_counts + eff_index
    else:
        d_eff_index = cp.full(n, eff_index, dtype=cp.int32)

    d_valid = (d_eff_index >= 0) & (d_eff_index < d_part_counts)

    # Device-side early-exit check — avoid D2H transfer when all OOB
    if int(d_valid.sum().item()) == 0:
        d_empty_x = cp.empty(0, dtype=cp.float64)
        d_empty_y = cp.empty(0, dtype=cp.float64)
        d_empty_geom_offsets = cp.zeros(1, dtype=cp.int32)
        d_empty_ring_offsets = cp.zeros(1, dtype=cp.int32)
        d_empty_mask = cp.empty(0, dtype=cp.bool_)
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=d_empty_x,
            y=d_empty_y,
            geometry_offsets=d_empty_geom_offsets,
            empty_mask=d_empty_mask,
            ring_offsets=d_empty_ring_offsets,
        ), d_valid

    d_valid_local = cp.flatnonzero(d_valid)
    n_valid = int(d_valid_local.size)

    # Polygon index for each valid row
    d_poly_idx = d_starts[d_valid_local] + d_eff_index[d_valid_local]

    # Ring range for each polygon
    d_ring_starts = d_part_offsets[d_poly_idx]
    d_ring_ends = d_part_offsets[d_poly_idx + 1]
    d_ring_counts = d_ring_ends - d_ring_starts

    # Build output geometry_offsets (maps row -> ring index in output)
    d_out_geom_offsets = cp.zeros(n_valid + 1, dtype=cp.int32)
    cp.cumsum(d_ring_counts, out=d_out_geom_offsets[1:])
    total_rings = int(d_out_geom_offsets[-1])

    if total_rings == 0:
        d_empty_x = cp.empty(0, dtype=cp.float64)
        d_empty_y = cp.empty(0, dtype=cp.float64)
        d_out_ring_offsets = cp.zeros(1, dtype=cp.int32)
        d_out_empty = cp.ones(n_valid, dtype=cp.bool_)
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=d_empty_x,
            y=d_empty_y,
            geometry_offsets=d_out_geom_offsets,
            empty_mask=d_out_empty,
            ring_offsets=d_out_ring_offsets,
        ), d_valid

    # Build flat ring index array to gather source ring indices
    # CuPy repeat requires a Python list; transfer small offset arrays.
    h_ring_counts = cp.asnumpy(d_ring_counts).tolist()
    d_row_ids_ring = cp.repeat(cp.arange(n_valid, dtype=cp.int32), h_ring_counts)
    d_local_ring_offsets = cp.arange(total_rings, dtype=cp.int32) - d_out_geom_offsets[:-1][d_row_ids_ring]
    d_src_ring_indices = d_ring_starts[d_row_ids_ring] + d_local_ring_offsets

    # Coordinate range for each ring
    d_ring_coord_starts = d_ring_offsets[d_src_ring_indices]
    d_ring_coord_ends = d_ring_offsets[d_src_ring_indices + 1]
    d_ring_coord_lengths = d_ring_coord_ends - d_ring_coord_starts

    # Build output ring_offsets
    d_out_ring_offsets = cp.zeros(total_rings + 1, dtype=cp.int32)
    cp.cumsum(d_ring_coord_lengths, out=d_out_ring_offsets[1:])
    total_coords = int(d_out_ring_offsets[-1])

    if total_coords == 0:
        d_out_x = cp.empty(0, dtype=cp.float64)
        d_out_y = cp.empty(0, dtype=cp.float64)
    else:
        # Gather coordinates
        h_ring_coord_lengths = cp.asnumpy(d_ring_coord_lengths).tolist()
        d_ring_ids = cp.repeat(cp.arange(total_rings, dtype=cp.int32), h_ring_coord_lengths)
        d_local_coord_offsets = cp.arange(total_coords, dtype=cp.int32) - d_out_ring_offsets[:-1][d_ring_ids]
        d_src_coord_indices = d_ring_coord_starts[d_ring_ids] + d_local_coord_offsets
        d_out_x = device_buf.x[d_src_coord_indices]
        d_out_y = device_buf.y[d_src_coord_indices]

    d_out_empty = d_ring_counts == 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_out_empty,
        ring_offsets=d_out_ring_offsets,
    ), d_valid


def _pass_through_simple_gpu(
    device_buf: DeviceFamilyGeometryBuffer,
    family: GeometryFamily,
    family_rows: np.ndarray,
    index: int,
) -> tuple[DeviceFamilyGeometryBuffer, Any]:
    """Pass through simple geometries at index 0; mark invalid otherwise.

    For simple types (Point, LineString, Polygon), index=0 returns the
    geometry itself.  Negative indices are resolved: -1 also maps to 0
    since the collection length is 1.  All other indices are out-of-bounds.
    """
    d_family_rows = cp.asarray(family_rows)
    n = int(d_family_rows.size)
    # Simple types act as collections of length 1.
    eff_index = index
    if eff_index < 0:
        eff_index = 1 + eff_index  # length=1, so -1 -> 0, -2 -> -1 (invalid)
    d_valid = cp.full(n, eff_index == 0, dtype=cp.bool_)

    if int(d_valid.sum().item()) == 0:
        if family is GeometryFamily.POLYGON:
            return DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.empty(0, dtype=cp.float64),
                y=cp.empty(0, dtype=cp.float64),
                geometry_offsets=cp.zeros(1, dtype=cp.int32),
                empty_mask=cp.empty(0, dtype=cp.bool_),
                ring_offsets=cp.zeros(1, dtype=cp.int32),
            ), d_valid
        else:
            return DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.empty(0, dtype=cp.float64),
                y=cp.empty(0, dtype=cp.float64),
                geometry_offsets=cp.zeros(1, dtype=cp.int32),
                empty_mask=cp.empty(0, dtype=cp.bool_),
            ), d_valid

    # All rows are valid (eff_index == 0): pass through entire buffer
    # for the given family rows.
    d_geom_offsets = device_buf.geometry_offsets

    d_starts = d_geom_offsets[d_family_rows]
    d_ends = d_geom_offsets[d_family_rows + 1]
    d_lengths = d_ends - d_starts

    # Rebuild geometry offsets
    d_out_geom_offsets = cp.zeros(n + 1, dtype=cp.int32)
    cp.cumsum(d_lengths, out=d_out_geom_offsets[1:])

    if family is GeometryFamily.POLYGON:
        # For Polygon: geometry_offsets -> ring indices, ring_offsets -> coord indices
        # We need to re-gather both ring_offsets and coordinates.
        d_ring_offsets = device_buf.ring_offsets

        total_rings = int(d_out_geom_offsets[-1])
        if total_rings == 0:
            d_out_x = cp.empty(0, dtype=cp.float64)
            d_out_y = cp.empty(0, dtype=cp.float64)
            d_out_ring_offsets = cp.zeros(1, dtype=cp.int32)
        else:
            # CuPy repeat requires a Python list; transfer small offset array.
            h_lengths = cp.asnumpy(d_lengths).tolist()
            # Gather source ring indices
            d_row_ids = cp.repeat(cp.arange(n, dtype=cp.int32), h_lengths)
            d_local_ring = cp.arange(total_rings, dtype=cp.int32) - d_out_geom_offsets[:-1][d_row_ids]
            d_src_ring_idx = d_starts[d_row_ids] + d_local_ring

            d_ring_coord_starts = d_ring_offsets[d_src_ring_idx]
            d_ring_coord_ends = d_ring_offsets[d_src_ring_idx + 1]
            d_ring_coord_lengths = d_ring_coord_ends - d_ring_coord_starts

            d_out_ring_offsets = cp.zeros(total_rings + 1, dtype=cp.int32)
            cp.cumsum(d_ring_coord_lengths, out=d_out_ring_offsets[1:])
            total_coords = int(d_out_ring_offsets[-1])

            if total_coords == 0:
                d_out_x = cp.empty(0, dtype=cp.float64)
                d_out_y = cp.empty(0, dtype=cp.float64)
            else:
                h_ring_coord_lengths = cp.asnumpy(d_ring_coord_lengths).tolist()
                d_ring_ids = cp.repeat(cp.arange(total_rings, dtype=cp.int32), h_ring_coord_lengths)
                d_local_coord = cp.arange(total_coords, dtype=cp.int32) - d_out_ring_offsets[:-1][d_ring_ids]
                d_src_coord_idx = d_ring_coord_starts[d_ring_ids] + d_local_coord
                d_out_x = device_buf.x[d_src_coord_idx]
                d_out_y = device_buf.y[d_src_coord_idx]

        d_out_empty = device_buf.empty_mask[d_family_rows]

        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_out_x,
            y=d_out_y,
            geometry_offsets=d_out_geom_offsets,
            empty_mask=d_out_empty,
            ring_offsets=d_out_ring_offsets,
        ), d_valid

    # Point or LineString: geometry_offsets -> coordinate indices directly.
    total_coords = int(d_out_geom_offsets[-1])
    if total_coords == 0:
        d_out_x = cp.empty(0, dtype=cp.float64)
        d_out_y = cp.empty(0, dtype=cp.float64)
    else:
        h_lengths = cp.asnumpy(d_lengths).tolist()
        d_row_ids = cp.repeat(cp.arange(n, dtype=cp.int32), h_lengths)
        d_local_offsets = cp.arange(total_coords, dtype=cp.int32) - d_out_geom_offsets[:-1][d_row_ids]
        d_src_indices = d_starts[d_row_ids] + d_local_offsets
        d_out_x = device_buf.x[d_src_indices]
        d_out_y = device_buf.y[d_src_indices]

    d_out_empty = device_buf.empty_mask[d_family_rows]

    return DeviceFamilyGeometryBuffer(
        family=family,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=d_out_geom_offsets,
        empty_mask=d_out_empty,
    ), d_valid


# ---------------------------------------------------------------------------
# GPU kernel: get_geometry
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "get_geometry",
    "gpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "polygon", "linestring",
        "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cupy", "constructive", "get_geometry"),
)
def _get_geometry_gpu(
    owned: OwnedGeometryArray,
    index: int,
) -> OwnedGeometryArray:
    """Extract i-th sub-geometry from each row using CuPy offset arithmetic.

    Pure CuPy implementation (Tier 2, ADR-0033).  No custom NVRTC kernels.
    For Multi* types, extracts the i-th part as its simple counterpart.
    For simple types, index=0 returns the geometry itself; other indices
    produce None.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    index : int
        Scalar index of the sub-geometry to extract.  Negative indices
        wrap around (e.g., -1 is the last sub-geometry).

    Returns
    -------
    OwnedGeometryArray
        Extracted sub-geometries.
    """
    d_state = owned._ensure_device_state()

    row_count = owned.row_count
    out_tags = cp.asarray(d_state.tags).copy()
    out_validity = cp.asarray(d_state.validity).copy()
    src_tags = cp.asarray(d_state.tags)
    src_family_row_offsets = cp.asarray(d_state.family_row_offsets)

    new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    # Track the ordered global rows for each output family so that
    # family_row_offsets match the merge order of the device buffers.
    family_global_rows_ordered: dict[GeometryFamily, list[Any]] = {}

    for family, device_buf in d_state.families.items():
        geom_count = device_buf.geometry_offsets.shape[0] - 1
        if geom_count == 0:
            continue

        tag = FAMILY_TAGS[family]
        global_rows = cp.flatnonzero(src_tags == tag)
        if global_rows.size == 0:
            continue

        family_rows = src_family_row_offsets[global_rows]

        if family in _MULTI_TO_SIMPLE:
            simple_family = _MULTI_TO_SIMPLE[family]
            if family is GeometryFamily.MULTIPOINT:
                new_buf, valid_mask = _get_geometry_multipoint_gpu(
                    device_buf, index, family_rows,
                )
            elif family is GeometryFamily.MULTILINESTRING:
                new_buf, valid_mask = _get_geometry_multilinestring_gpu(
                    device_buf, index, family_rows,
                )
            else:  # MULTIPOLYGON
                new_buf, valid_mask = _get_geometry_multipolygon_gpu(
                    device_buf, index, family_rows,
                )

            # Remap tags: Multi* -> simple counterpart for valid rows
            simple_tag = FAMILY_TAGS[simple_family]
            out_tags[global_rows[valid_mask]] = simple_tag
            # Invalidate out-of-bounds rows
            out_validity[global_rows[~valid_mask]] = False

            # Track which global rows (valid only) went into this buffer.
            valid_global = global_rows[valid_mask]

            # Merge into output families (in case both Multi* and simple
            # produce the same output family)
            if simple_family in new_device_families:
                new_device_families[simple_family] = _merge_device_family_buffers(
                    new_device_families[simple_family], new_buf, simple_family,
                )
                family_global_rows_ordered[simple_family].append(valid_global)
            else:
                new_device_families[simple_family] = new_buf
                family_global_rows_ordered[simple_family] = [valid_global]

        elif family in _SIMPLE_FAMILIES:
            new_buf, valid_mask = _pass_through_simple_gpu(
                device_buf, family, family_rows, index,
            )
            # Invalidate out-of-bounds rows
            out_validity[global_rows[~valid_mask]] = False

            # Track which global rows (valid only) went into this buffer.
            valid_global = global_rows[valid_mask]

            if family in new_device_families:
                new_device_families[family] = _merge_device_family_buffers(
                    new_device_families[family], new_buf, family,
                )
                family_global_rows_ordered[family].append(valid_global)
            else:
                new_device_families[family] = new_buf
                family_global_rows_ordered[family] = [valid_global]

    # Recompute family_row_offsets.  The merge order determines which
    # family-local index each global row maps to -- we must follow the
    # same concatenation order used in _merge_device_family_buffers.
    new_family_row_offsets = cp.full(row_count, -1, dtype=cp.int32)
    for row_chunks in family_global_rows_ordered.values():
        ordered_rows = cp.concatenate(row_chunks) if len(row_chunks) > 1 else row_chunks[0]
        new_family_row_offsets[ordered_rows] = cp.arange(
            int(ordered_rows.size), dtype=cp.int32,
        )

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=new_family_row_offsets,
        execution_mode="gpu",
    )


def _merge_device_family_buffers(
    existing: DeviceFamilyGeometryBuffer,
    new: DeviceFamilyGeometryBuffer,
    family: GeometryFamily,
) -> DeviceFamilyGeometryBuffer:
    """Merge two device family buffers by concatenating coordinates and
    adjusting offsets.
    """
    # Coordinate offset for the new buffer
    existing_coord_count = int(existing.x.shape[0])

    d_x = cp.concatenate([existing.x, new.x])
    d_y = cp.concatenate([existing.y, new.y])
    d_empty = cp.concatenate([existing.empty_mask, new.empty_mask])

    if family is GeometryFamily.POLYGON:
        # Polygon has ring_offsets -> coordinates, geometry_offsets -> rings
        existing_ring_count = int(existing.ring_offsets.shape[0]) - 1

        # Shift new ring_offsets by existing coordinate count
        shifted_ring_offsets = new.ring_offsets + existing_coord_count
        d_ring_offsets = cp.concatenate([
            existing.ring_offsets, shifted_ring_offsets[1:],
        ])

        # Shift new geometry_offsets by existing ring count
        shifted_geom_offsets = new.geometry_offsets + existing_ring_count
        d_geom_offsets = cp.concatenate([
            existing.geometry_offsets, shifted_geom_offsets[1:],
        ])

        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x,
            y=d_y,
            geometry_offsets=d_geom_offsets,
            empty_mask=d_empty,
            ring_offsets=d_ring_offsets,
        )

    # Point / LineString: geometry_offsets -> coordinates
    shifted_geom_offsets = new.geometry_offsets + existing_coord_count
    d_geom_offsets = cp.concatenate([
        existing.geometry_offsets, shifted_geom_offsets[1:],
    ])

    return DeviceFamilyGeometryBuffer(
        family=family,
        x=d_x,
        y=d_y,
        geometry_offsets=d_geom_offsets,
        empty_mask=d_empty,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def get_geometry_owned(
    owned: OwnedGeometryArray,
    index: int | np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Extract the i-th sub-geometry from each geometry in the array.

    For Multi* types (MultiPoint, MultiLineString, MultiPolygon), extracts
    the i-th part as its simple counterpart (Point, LineString, Polygon).
    For simple types, index=0 returns the geometry itself; other indices
    produce None.

    Negative indices wrap around: -1 is the last sub-geometry.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    index : int or array-like
        Sub-geometry index.  Scalar index is applied to all rows.
        Array-like index provides per-row indices (falls back to Shapely).
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  CONSTRUCTIVE class stays fp64 by design per
        ADR-0002; wired here for observability.

    Returns
    -------
    OwnedGeometryArray
        Extracted sub-geometries.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    # Array-like index: fall back to CPU (element-wise different indices
    # per row is not easily vectorisable with pure offset arithmetic).
    if not isinstance(index, (int, np.integer)):
        result = get_geometry_cpu(owned, index)
        record_fallback_event(
            surface="geopandas.array.get_geometry",
            reason="array-like index requires element-wise Shapely dispatch",
            detail=f"rows={row_count}, index_type={type(index).__name__}",
            requested=dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode),
            d2h_transfer=True,
        )
        record_dispatch_event(
            surface="geopandas.array.get_geometry",
            operation="get_geometry",
            implementation="get_geometry_cpu_shapely",
            reason="array-like index fallback",
            detail=f"rows={row_count}",
            requested=dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode),
            selected=ExecutionMode.CPU,
        )
        return result

    selection = plan_dispatch_selection(
        kernel_name="get_geometry",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _get_geometry_gpu(owned, int(index))
        if result is not None:
            record_dispatch_event(
                surface="geopandas.array.get_geometry",
                operation="get_geometry",
                implementation="get_geometry_gpu_cupy",
                reason=selection.reason,
                detail=(
                    f"rows={row_count}, index={index}, "
                    f"precision={precision_plan.compute_precision.value} "
                    f"(offset-only, not parameterized)"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    # CPU fallback
    result = get_geometry_cpu(owned, int(index))
    record_fallback_event(
        surface="geopandas.array.get_geometry",
        reason="CPU fallback for get_geometry",
        detail=f"rows={row_count}, index={index}",
        requested=selection.requested,
        d2h_transfer=True,
    )
    record_dispatch_event(
        surface="geopandas.array.get_geometry",
        operation="get_geometry",
        implementation="get_geometry_cpu_shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
