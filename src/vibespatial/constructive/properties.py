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

ADR-0033:
- num_coordinates, num_geometries, num_interior_rings: Tier 1 (NVRTC offset_diff)
  for simple/multi families; Tier 1 (NVRTC nested-loop) for polygon families
- is_closed, is_ccw: Tier 1 (NVRTC)
"""

from __future__ import annotations

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
    FAMILY_TAGS,
    OwnedGeometryArray,
)

# ---------------------------------------------------------------------------
# NVRTC kernels for is_closed and is_ccw (Tier 1, ADR-0033)
# ---------------------------------------------------------------------------

# is_closed: compares first and last coordinate per span
_IS_CLOSED_KERNEL_SOURCE = r"""
extern "C" __global__ void is_closed_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int start = geometry_offsets[row];
    const int end = geometry_offsets[row + 1];

    if (end - start < 2) {{
        out[i] = 1;
        return;
    }}

    out[i] = (x[start] == x[end - 1] && y[start] == y[end - 1]) ? 1 : 0;
}}

extern "C" __global__ void is_closed_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ps = geometry_offsets[row];
    const int pe = geometry_offsets[row + 1];

    int closed = 1;
    for (int p = ps; p < pe; p++) {{
        const int cs = part_offsets[p];
        const int ce = part_offsets[p + 1];
        if (ce - cs < 2) continue;
        if (x[cs] != x[ce - 1] || y[cs] != y[ce - 1]) {{
            closed = 0;
            break;
        }}
    }}
    out[i] = closed;
}}
"""

_IS_CLOSED_KERNEL_NAMES = ("is_closed_linestring", "is_closed_multilinestring")
_IS_CLOSED_FP64 = _IS_CLOSED_KERNEL_SOURCE.format()

# is_ccw: shoelace signed area on exterior ring
_IS_CCW_KERNEL_SOURCE = r"""
extern "C" __global__ void is_ccw_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ring_idx = geometry_offsets[row];
    const int coord_start = ring_offsets[ring_idx];
    const int coord_end = ring_offsets[ring_idx + 1];

    const int ncoords = coord_end - coord_start;
    if (ncoords < 3) {{
        out[i] = 0;
        return;
    }}

    double area2 = 0.0;
    for (int j = coord_start; j < coord_end - 1; j++) {{
        area2 += x[j] * y[j + 1] - x[j + 1] * y[j];
    }}
    out[i] = (area2 > 0.0) ? 1 : 0;
}}

extern "C" __global__ void is_ccw_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int poly_start = geometry_offsets[row];
    const int ring_idx = part_offsets[poly_start];
    const int coord_start = ring_offsets[ring_idx];
    const int coord_end = ring_offsets[ring_idx + 1];

    const int ncoords = coord_end - coord_start;
    if (ncoords < 3) {{
        out[i] = 0;
        return;
    }}

    double area2 = 0.0;
    for (int j = coord_start; j < coord_end - 1; j++) {{
        area2 += x[j] * y[j + 1] - x[j + 1] * y[j];
    }}
    out[i] = (area2 > 0.0) ? 1 : 0;
}}
"""

_IS_CCW_KERNEL_NAMES = ("is_ccw_polygon", "is_ccw_multipolygon")
_IS_CCW_FP64 = _IS_CCW_KERNEL_SOURCE.format()

# ---------------------------------------------------------------------------
# NVRTC offset_diff kernel (Tier 1, ADR-0033)
#
# Replaces CuPy fancy indexing for simple-family offset arithmetic:
#   out[i] = offsets[rows[i] + 1] - offsets[rows[i]]
# Also provides an interior-ring variant that subtracts 1 and clamps to 0.
# ---------------------------------------------------------------------------

_OFFSET_DIFF_KERNEL_SOURCE = r"""
extern "C" __global__ void offset_diff(
    const int* __restrict__ offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = rows[i];
    out[i] = offsets[row + 1] - offsets[row];
}}

extern "C" __global__ void offset_diff_interior_rings(
    const int* __restrict__ offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = rows[i];
    int diff = offsets[row + 1] - offsets[row] - 1;
    out[i] = diff > 0 ? diff : 0;
}}
"""

_OFFSET_DIFF_KERNEL_NAMES = ("offset_diff", "offset_diff_interior_rings")
_OFFSET_DIFF_FP64 = _OFFSET_DIFF_KERNEL_SOURCE.format()

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

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


# ---------------------------------------------------------------------------
# NVRTC kernel for num_coordinates on POLYGON family (ring indirection)
# Needed because CuPy cannot express the double-indirect offset walk
# without Python loops.
# ---------------------------------------------------------------------------

_NUM_COORDS_KERNEL_SOURCE = r"""
extern "C" __global__ void num_coords_polygon(
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ring_start = geometry_offsets[row];
    const int ring_end = geometry_offsets[row + 1];

    int total = 0;
    for (int ri = ring_start; ri < ring_end; ri++) {{
        total += ring_offsets[ri + 1] - ring_offsets[ri];
    }}
    out[i] = total;
}}

extern "C" __global__ void num_coords_multilinestring(
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int part_start = geometry_offsets[row];
    const int part_end = geometry_offsets[row + 1];

    int total = 0;
    for (int pi = part_start; pi < part_end; pi++) {{
        total += part_offsets[pi + 1] - part_offsets[pi];
    }}
    out[i] = total;
}}

extern "C" __global__ void num_coords_multipolygon(
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int poly_start = geometry_offsets[row];
    const int poly_end = geometry_offsets[row + 1];

    int total = 0;
    for (int pi = poly_start; pi < poly_end; pi++) {{
        const int ring_start = part_offsets[pi];
        const int ring_end = part_offsets[pi + 1];
        for (int ri = ring_start; ri < ring_end; ri++) {{
            total += ring_offsets[ri + 1] - ring_offsets[ri];
        }}
    }}
    out[i] = total;
}}
"""

_NUM_COORDS_KERNEL_NAMES = (
    "num_coords_polygon",
    "num_coords_multilinestring",
    "num_coords_multipolygon",
)
_NUM_COORDS_FP64 = _NUM_COORDS_KERNEL_SOURCE.format()

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
            n = int(d_family_rows.size)

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
            n = int(d_family_rows.size)

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
