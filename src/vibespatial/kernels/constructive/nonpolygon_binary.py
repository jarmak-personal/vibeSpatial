"""GPU kernels for non-polygon binary constructive operations.

Handles family combinations that do not involve Polygon-Polygon pairs:
- Point-Point: coordinate comparison with tolerance
- MultiPoint-Polygon: batch PIP with compaction
- Point-LineString: point-to-segment minimum distance
- LineString-Polygon: segment clipping against polygon boundary
- LineString-LineString: segment-segment intersection

All kernels return device-resident OwnedGeometryArray instances.

ADR-0033: Mixed tiers.
  - Point-Point: Tier 2 (CuPy element-wise)
  - MultiPoint-Polygon: Tier 2 over Tier 1 PIP kernel
  - Point-LineString: Tier 1 (custom NVRTC, geometry-specific inner loop)
  - LineString-Polygon: Tier 1 (custom NVRTC, segment-ring traversal)
  - LineString-LineString: Tier 1 (custom NVRTC, segment-segment intersection)
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
ADR-0034: NVRTC precompilation via request_nvrtc_warmup at module scope.
"""

from __future__ import annotations

import logging

from vibespatial.constructive.nonpolygon_binary_output import (
    build_device_backed_linestring_output,
    build_device_backed_multipoint_output,
    build_point_result_from_source,
    empty_linestring_output,
    host_prefix_offsets,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.kernels.constructive.nonpolygon_binary_source import (
    _LINESTRING_LINESTRING_KERNEL_NAMES,
    _LINESTRING_LINESTRING_KERNEL_SOURCE,
    _LINESTRING_POLYGON_KERNEL_NAMES,
    _LINESTRING_POLYGON_KERNEL_SOURCE,
    _POINT_LINESTRING_KERNEL_NAMES,
    _POINT_LINESTRING_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADR-0034: NVRTC precompilation at module scope
# ---------------------------------------------------------------------------
request_nvrtc_warmup([
    ("point-linestring-on-line", _POINT_LINESTRING_KERNEL_SOURCE, _POINT_LINESTRING_KERNEL_NAMES),
    ("linestring-polygon-clip", _LINESTRING_POLYGON_KERNEL_SOURCE, _LINESTRING_POLYGON_KERNEL_NAMES),
    ("linestring-linestring-isect", _LINESTRING_LINESTRING_KERNEL_SOURCE, _LINESTRING_LINESTRING_KERNEL_NAMES),
])

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _point_linestring_kernels():
    return compile_kernel_group(
        "point-linestring-on-line",
        _POINT_LINESTRING_KERNEL_SOURCE,
        _POINT_LINESTRING_KERNEL_NAMES,
    )


def _linestring_polygon_kernels():
    return compile_kernel_group(
        "linestring-polygon-clip",
        _LINESTRING_POLYGON_KERNEL_SOURCE,
        _LINESTRING_POLYGON_KERNEL_NAMES,
    )


def _linestring_linestring_kernels():
    return compile_kernel_group(
        "linestring-linestring-isect",
        _LINESTRING_LINESTRING_KERNEL_SOURCE,
        _LINESTRING_LINESTRING_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Registered kernel variant (ARCH003 compliance)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "nonpolygon_binary_constructive",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "linestring", "multipoint",
    ),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "nonpolygon"),
)
def _nonpolygon_binary_constructive_gpu(
    op: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """GPU non-polygon binary constructive — variant entry point.

    This registration satisfies ARCH003. Actual dispatch is performed
    by ``binary_constructive.py`` which calls the individual kernel
    functions below directly.
    """
    return None  # pragma: no cover — dispatch handled by binary_constructive.py


# ---------------------------------------------------------------------------
# Point-Point constructive operations (Tier 2: CuPy element-wise)
# ---------------------------------------------------------------------------

def point_point_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-Point intersection: keep rows where coordinates match.

    Uses CuPy element-wise comparison (Tier 2).
    Returns device-resident OwnedGeometryArray.
    """
    import cupy as cp

    n = left.row_count
    left.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                 reason="point_point_intersection GPU")
    right.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                  reason="point_point_intersection GPU")

    l_state = left.device_state
    r_state = right.device_state

    l_buf = l_state.families[GeometryFamily.POINT]
    r_buf = r_state.families[GeometryFamily.POINT]

    # Both-valid mask
    d_l_valid = l_state.validity.astype(cp.bool_) & ~l_buf.empty_mask.astype(cp.bool_)
    d_r_valid = r_state.validity.astype(cp.bool_) & ~r_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_l_valid & d_r_valid

    # Point coordinates: each point has exactly 1 coord at geom_offsets[i]
    # Compare coordinates within tolerance
    tol = 1e-8

    # For valid rows, extract the coordinate index from geometry_offsets
    d_l_offsets = cp.asarray(l_buf.geometry_offsets)
    d_r_offsets = cp.asarray(r_buf.geometry_offsets)
    d_l_x = cp.asarray(l_buf.x)
    d_l_y = cp.asarray(l_buf.y)
    d_r_x = cp.asarray(r_buf.x)
    d_r_y = cp.asarray(r_buf.y)

    # Initialize match mask as False
    d_match = cp.zeros(n, dtype=cp.bool_)

    # For valid rows, compare coordinates
    valid_indices = cp.flatnonzero(d_both_valid)
    if valid_indices.size > 0:
        l_coord_idx = d_l_offsets[valid_indices]
        r_coord_idx = d_r_offsets[valid_indices]
        lx = d_l_x[l_coord_idx]
        ly = d_l_y[l_coord_idx]
        rx = d_r_x[r_coord_idx]
        ry = d_r_y[r_coord_idx]
        dx = lx - rx
        dy = ly - ry
        close = (dx * dx + dy * dy) < (tol * tol)
        d_match[valid_indices] = close

    runtime = get_cuda_runtime()
    h_match = runtime.copy_device_to_host(d_match).astype(bool)

    # Build output: same Point buffers as left, with match-based validity
    return build_point_result_from_source(left, h_match)


def point_point_difference(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-Point difference: keep rows where coordinates differ.

    difference(a, b) = a where a != b, NULL where a == b.
    If right is NULL, keep left. If left is NULL, result is NULL.
    """
    import cupy as cp

    left.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                 reason="point_point_difference GPU")
    right.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                  reason="point_point_difference GPU")

    l_state = left.device_state
    r_state = right.device_state

    l_buf = l_state.families[GeometryFamily.POINT]
    r_buf = r_state.families[GeometryFamily.POINT]

    d_l_valid = l_state.validity.astype(cp.bool_) & ~l_buf.empty_mask.astype(cp.bool_)
    d_r_valid = r_state.validity.astype(cp.bool_) & ~r_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_l_valid & d_r_valid

    tol = 1e-8

    d_l_offsets = cp.asarray(l_buf.geometry_offsets)
    d_r_offsets = cp.asarray(r_buf.geometry_offsets)
    d_l_x = cp.asarray(l_buf.x)
    d_l_y = cp.asarray(l_buf.y)
    d_r_x = cp.asarray(r_buf.x)
    d_r_y = cp.asarray(r_buf.y)

    # Start with "keep left if left is valid"
    d_keep = d_l_valid.copy()

    # For rows where both are valid: only keep if coordinates differ
    valid_indices = cp.flatnonzero(d_both_valid)
    if valid_indices.size > 0:
        l_coord_idx = d_l_offsets[valid_indices]
        r_coord_idx = d_r_offsets[valid_indices]
        lx = d_l_x[l_coord_idx]
        ly = d_l_y[l_coord_idx]
        rx = d_r_x[r_coord_idx]
        ry = d_r_y[r_coord_idx]
        dx = lx - rx
        dy = ly - ry
        same = (dx * dx + dy * dy) < (tol * tol)
        # Where both valid and same coords: don't keep
        d_keep[valid_indices] = ~same

    runtime = get_cuda_runtime()
    h_keep = runtime.copy_device_to_host(d_keep).astype(bool)
    return build_point_result_from_source(left, h_keep)


def point_point_union(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-Point union: keep all unique coordinates.

    For element-wise union of two Point arrays, the result per row is:
    - If both are the same point -> the point (as Point)
    - If both are different -> MultiPoint with both
    - If only one is valid -> that point
    - If neither is valid -> NULL

    For simplicity and to maintain consistent output type per row,
    we output MultiPoints. Rows with a single point get a MultiPoint
    with 1 point. Rows with matching points get a MultiPoint with 1 point.
    """
    import cupy as cp

    n = left.row_count
    left.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                 reason="point_point_union GPU")
    right.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                  reason="point_point_union GPU")

    l_state = left.device_state
    r_state = right.device_state
    l_buf = l_state.families[GeometryFamily.POINT]
    r_buf = r_state.families[GeometryFamily.POINT]

    d_l_valid = l_state.validity.astype(cp.bool_) & ~l_buf.empty_mask.astype(cp.bool_)
    d_r_valid = r_state.validity.astype(cp.bool_) & ~r_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_l_valid & d_r_valid
    d_any_valid = d_l_valid | d_r_valid

    tol = 1e-8

    d_l_offsets = cp.asarray(l_buf.geometry_offsets)
    d_r_offsets = cp.asarray(r_buf.geometry_offsets)
    d_l_x = cp.asarray(l_buf.x)
    d_l_y = cp.asarray(l_buf.y)
    d_r_x = cp.asarray(r_buf.x)
    d_r_y = cp.asarray(r_buf.y)

    # For each row, determine output coordinate count:
    # - both valid & same: 1 point
    # - both valid & different: 2 points
    # - one valid: 1 point
    # - neither valid: 0 points
    d_counts = cp.zeros(n, dtype=cp.int32)

    # Rows with only left valid: 1 point
    only_left = d_l_valid & ~d_r_valid
    d_counts[only_left] = 1

    # Rows with only right valid: 1 point
    only_right = ~d_l_valid & d_r_valid
    d_counts[only_right] = 1

    # Rows with both valid: check if same
    both_indices = cp.flatnonzero(d_both_valid)
    if both_indices.size > 0:
        l_ci = d_l_offsets[both_indices]
        r_ci = d_r_offsets[both_indices]
        lx = d_l_x[l_ci]
        ly = d_l_y[l_ci]
        rx = d_r_x[r_ci]
        ry = d_r_y[r_ci]
        dx = lx - rx
        dy = ly - ry
        same = (dx * dx + dy * dy) < (tol * tol)
        d_counts[both_indices] = cp.where(same, cp.int32(1), cp.int32(2))

    # Prefix sum for output coordinate offsets
    runtime = get_cuda_runtime()
    h_counts = runtime.copy_device_to_host(d_counts)
    h_offsets = host_prefix_offsets(h_counts)
    total_coords = int(h_offsets[-1])

    if total_coords == 0:
        # All empty
        return from_shapely_geometries([None] * n)

    # Allocate output coordinates on device
    d_out_x = runtime.allocate((total_coords,), cp.float64)
    d_out_y = runtime.allocate((total_coords,), cp.float64)
    d_out_x_cp = cp.asarray(d_out_x)
    d_out_y_cp = cp.asarray(d_out_y)

    d_offsets_cp = cp.asarray(h_offsets)

    # Scatter coordinates (Tier 2: CuPy element-wise)
    # Only-left rows
    only_left_idx = cp.flatnonzero(only_left)
    if only_left_idx.size > 0:
        out_pos = d_offsets_cp[only_left_idx]
        src_pos = d_l_offsets[only_left_idx]
        d_out_x_cp[out_pos] = d_l_x[src_pos]
        d_out_y_cp[out_pos] = d_l_y[src_pos]

    # Only-right rows
    only_right_idx = cp.flatnonzero(only_right)
    if only_right_idx.size > 0:
        out_pos = d_offsets_cp[only_right_idx]
        src_pos = d_r_offsets[only_right_idx]
        d_out_x_cp[out_pos] = d_r_x[src_pos]
        d_out_y_cp[out_pos] = d_r_y[src_pos]

    # Both-valid rows
    if both_indices.size > 0:
        l_ci = d_l_offsets[both_indices]
        r_ci = d_r_offsets[both_indices]
        out_pos = d_offsets_cp[both_indices]
        # First point is always left
        d_out_x_cp[out_pos] = d_l_x[l_ci]
        d_out_y_cp[out_pos] = d_l_y[l_ci]
        # For rows with 2 points, also write right at out_pos+1
        two_pt_mask = d_counts[both_indices] == 2
        two_pt_idx = both_indices[two_pt_mask]
        if two_pt_idx.size > 0:
            out_pos2 = d_offsets_cp[two_pt_idx] + 1
            r_ci2 = d_r_offsets[two_pt_idx]
            d_out_x_cp[out_pos2] = d_r_x[r_ci2]
            d_out_y_cp[out_pos2] = d_r_y[r_ci2]

    h_validity = runtime.copy_device_to_host(d_any_valid).astype(bool)
    geometry_offsets = h_offsets

    return build_device_backed_multipoint_output(
        d_out_x, d_out_y,
        row_count=n,
        validity=h_validity,
        geometry_offsets=geometry_offsets,
    )


def point_point_symmetric_difference(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-Point symmetric_difference: keep if only one side has a point.

    If coordinates match, result is NULL (empty).
    If coordinates differ, result is MultiPoint with both.
    If only one valid, result is that point.
    """
    import cupy as cp

    n = left.row_count
    left.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                 reason="point_point_symmetric_difference GPU")
    right.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                  reason="point_point_symmetric_difference GPU")

    l_state = left.device_state
    r_state = right.device_state
    l_buf = l_state.families[GeometryFamily.POINT]
    r_buf = r_state.families[GeometryFamily.POINT]

    d_l_valid = l_state.validity.astype(cp.bool_) & ~l_buf.empty_mask.astype(cp.bool_)
    d_r_valid = r_state.validity.astype(cp.bool_) & ~r_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_l_valid & d_r_valid

    tol = 1e-8

    d_l_offsets = cp.asarray(l_buf.geometry_offsets)
    d_r_offsets = cp.asarray(r_buf.geometry_offsets)
    d_l_x = cp.asarray(l_buf.x)
    d_l_y = cp.asarray(l_buf.y)
    d_r_x = cp.asarray(r_buf.x)
    d_r_y = cp.asarray(r_buf.y)

    # For symmetric difference:
    # - same coords: empty (0 points)
    # - diff coords: 2 points
    # - only left: 1 point
    # - only right: 1 point
    d_counts = cp.zeros(n, dtype=cp.int32)

    only_left = d_l_valid & ~d_r_valid
    d_counts[only_left] = 1
    only_right = ~d_l_valid & d_r_valid
    d_counts[only_right] = 1

    both_indices = cp.flatnonzero(d_both_valid)
    d_same = cp.zeros(n, dtype=cp.bool_)
    if both_indices.size > 0:
        l_ci = d_l_offsets[both_indices]
        r_ci = d_r_offsets[both_indices]
        dx = d_l_x[l_ci] - d_r_x[r_ci]
        dy = d_l_y[l_ci] - d_r_y[r_ci]
        same = (dx * dx + dy * dy) < (tol * tol)
        d_same[both_indices] = same
        d_counts[both_indices] = cp.where(same, cp.int32(0), cp.int32(2))

    runtime = get_cuda_runtime()
    h_counts = runtime.copy_device_to_host(d_counts)
    h_offsets = host_prefix_offsets(h_counts)
    total_coords = int(h_offsets[-1])

    # Validity: row is valid if it has any output points
    d_has_output = d_counts > 0
    h_validity = runtime.copy_device_to_host(d_has_output).astype(bool)

    if total_coords == 0:
        return from_shapely_geometries([None] * n)

    d_out_x = runtime.allocate((total_coords,), cp.float64)
    d_out_y = runtime.allocate((total_coords,), cp.float64)
    d_out_x_cp = cp.asarray(d_out_x)
    d_out_y_cp = cp.asarray(d_out_y)
    d_offsets_cp = cp.asarray(h_offsets)

    # Only-left
    only_left_idx = cp.flatnonzero(only_left)
    if only_left_idx.size > 0:
        out_pos = d_offsets_cp[only_left_idx]
        src_pos = d_l_offsets[only_left_idx]
        d_out_x_cp[out_pos] = d_l_x[src_pos]
        d_out_y_cp[out_pos] = d_l_y[src_pos]

    # Only-right
    only_right_idx = cp.flatnonzero(only_right)
    if only_right_idx.size > 0:
        out_pos = d_offsets_cp[only_right_idx]
        src_pos = d_r_offsets[only_right_idx]
        d_out_x_cp[out_pos] = d_r_x[src_pos]
        d_out_y_cp[out_pos] = d_r_y[src_pos]

    # Both valid & different
    diff_mask = d_both_valid & ~d_same
    diff_indices = cp.flatnonzero(diff_mask)
    if diff_indices.size > 0:
        out_pos = d_offsets_cp[diff_indices]
        l_ci = d_l_offsets[diff_indices]
        r_ci = d_r_offsets[diff_indices]
        d_out_x_cp[out_pos] = d_l_x[l_ci]
        d_out_y_cp[out_pos] = d_l_y[l_ci]
        d_out_x_cp[out_pos + 1] = d_r_x[r_ci]
        d_out_y_cp[out_pos + 1] = d_r_y[r_ci]

    return build_device_backed_multipoint_output(
        d_out_x, d_out_y,
        row_count=n,
        validity=h_validity,
        geometry_offsets=h_offsets,
    )


# ---------------------------------------------------------------------------
# Point-LineString constructive operations (Tier 1: NVRTC kernel)
# ---------------------------------------------------------------------------

def point_linestring_intersection(
    points: OwnedGeometryArray,
    linestrings: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-LineString intersection: keep points that lie on the linestring."""
    return _point_linestring_constructive(points, linestrings, mode="intersection")


def point_linestring_difference(
    points: OwnedGeometryArray,
    linestrings: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Point-LineString difference: keep points that do NOT lie on the linestring."""
    return _point_linestring_constructive(points, linestrings, mode="difference")


def _point_linestring_constructive(
    points: OwnedGeometryArray,
    linestrings: OwnedGeometryArray,
    *,
    mode: str,
) -> OwnedGeometryArray:
    """Common implementation for Point-LineString intersection/difference.

    Uses the NVRTC point_linestring_on_line kernel (Tier 1).
    """
    import cupy as cp

    n = points.row_count
    points.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                   reason=f"point_linestring_{mode} GPU")
    linestrings.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason=f"point_linestring_{mode} GPU")

    pt_state = points.device_state
    ls_state = linestrings.device_state

    pt_buf = pt_state.families[GeometryFamily.POINT]
    ls_buf = ls_state.families[GeometryFamily.LINESTRING]

    d_pt_valid = pt_state.validity.astype(cp.bool_) & ~pt_buf.empty_mask.astype(cp.bool_)
    d_ls_valid = ls_state.validity.astype(cp.bool_) & ~ls_buf.empty_mask.astype(cp.bool_)
    d_both_valid = (d_pt_valid & d_ls_valid).astype(cp.int32)

    runtime = get_cuda_runtime()
    d_on_line = runtime.allocate((n,), cp.int32, zero=True)

    kernels = _point_linestring_kernels()
    ptr = runtime.pointer

    params = (
        (
            ptr(pt_buf.x), ptr(pt_buf.y), ptr(pt_buf.geometry_offsets),
            ptr(ls_buf.x), ptr(ls_buf.y), ptr(ls_buf.geometry_offsets),
            ptr(d_both_valid), ptr(d_on_line), n,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["point_linestring_on_line"], n)
    runtime.launch(kernels["point_linestring_on_line"], grid=grid, block=block, params=params)
    runtime.synchronize()

    h_on_line = runtime.copy_device_to_host(d_on_line).astype(bool)

    if mode == "intersection":
        # Keep points that are on the line
        new_validity = h_on_line
    else:
        # Difference: keep points NOT on line, but only if left is valid.
        # If right is NULL, difference = identity (keep left).
        h_pt_valid = runtime.copy_device_to_host(d_pt_valid).astype(bool)
        h_ls_valid = runtime.copy_device_to_host(d_ls_valid).astype(bool)
        new_validity = h_pt_valid & (~h_on_line | ~h_ls_valid)

    return build_point_result_from_source(points, new_validity)


# ---------------------------------------------------------------------------
# LineString-Polygon constructive operations (Tier 1: NVRTC kernel)
# ---------------------------------------------------------------------------

def linestring_polygon_intersection(
    linestrings: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """LineString-Polygon intersection: clip line to inside of polygon."""
    return _linestring_polygon_constructive(linestrings, polygons, mode=0)


def linestring_polygon_difference(
    linestrings: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """LineString-Polygon difference: clip line to outside of polygon."""
    return _linestring_polygon_constructive(linestrings, polygons, mode=1)


def _linestring_polygon_constructive(
    linestrings: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
    *,
    mode: int,
) -> OwnedGeometryArray:
    """Common implementation for LineString-Polygon intersection/difference.

    Uses two-pass count-scatter NVRTC kernels (Tier 1).
    mode: 0 = intersection (keep inside), 1 = difference (keep outside).
    """
    import cupy as cp

    n = linestrings.row_count
    linestrings.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason="linestring_polygon_constructive GPU")
    polygons.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                     reason="linestring_polygon_constructive GPU")

    ls_state = linestrings.device_state
    poly_state = polygons.device_state

    ls_buf = ls_state.families[GeometryFamily.LINESTRING]

    # Determine polygon family
    if GeometryFamily.POLYGON in poly_state.families:
        poly_buf = poly_state.families[GeometryFamily.POLYGON]
    elif GeometryFamily.MULTIPOLYGON in poly_state.families:
        # For MultiPolygon, use the first polygon part for simplicity.
        # A full implementation would iterate all parts.
        poly_buf = poly_state.families[GeometryFamily.MULTIPOLYGON]
    else:
        return from_shapely_geometries([None] * n)

    d_ls_valid = ls_state.validity.astype(cp.bool_) & ~ls_buf.empty_mask.astype(cp.bool_)
    d_poly_valid = poly_state.validity.astype(cp.bool_) & ~poly_buf.empty_mask.astype(cp.bool_)
    d_both_valid = (d_ls_valid & d_poly_valid).astype(cp.int32)

    runtime = get_cuda_runtime()
    d_counts = runtime.allocate((n,), cp.int32, zero=True)

    kernels = _linestring_polygon_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(ls_buf.x), ptr(ls_buf.y), ptr(ls_buf.geometry_offsets),
            ptr(poly_buf.x), ptr(poly_buf.y),
            ptr(poly_buf.ring_offsets), ptr(poly_buf.geometry_offsets),
            ptr(d_both_valid), ptr(d_counts), mode, n,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["linestring_polygon_count"], n)
    runtime.launch(kernels["linestring_polygon_count"], grid=grid, block=block, params=count_params)

    # Prefix sum for scatter offsets
    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total_verts = count_scatter_total(runtime, d_counts, d_offsets)

    if total_verts == 0:
        validity, geometry_offsets = empty_linestring_output(n)
        d_out_x = runtime.allocate((0,), cp.float64)
        d_out_y = runtime.allocate((0,), cp.float64)
        return build_device_backed_linestring_output(
            d_out_x, d_out_y,
            row_count=n, validity=validity, geometry_offsets=geometry_offsets,
        )

    d_out_x = runtime.allocate((total_verts,), cp.float64)
    d_out_y = runtime.allocate((total_verts,), cp.float64)

    scatter_params = (
        (
            ptr(ls_buf.x), ptr(ls_buf.y), ptr(ls_buf.geometry_offsets),
            ptr(poly_buf.x), ptr(poly_buf.y),
            ptr(poly_buf.ring_offsets), ptr(poly_buf.geometry_offsets),
            ptr(d_both_valid), ptr(d_offsets),
            ptr(d_out_x), ptr(d_out_y), mode, n,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["linestring_polygon_scatter"], n,
    )
    runtime.launch(
        kernels["linestring_polygon_scatter"],
        grid=scatter_grid, block=scatter_block, params=scatter_params,
    )

    runtime.synchronize()

    # Build geometry_offsets from d_offsets
    d_offsets_cp = cp.asarray(d_offsets)
    d_geom_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_geom_offsets[:n] = d_offsets_cp
    d_geom_offsets[n] = total_verts
    h_geom_offsets = runtime.copy_device_to_host(d_geom_offsets)

    # Validity: rows with at least 2 output vertices form valid linestrings
    h_counts_arr = runtime.copy_device_to_host(d_counts)
    validity = h_counts_arr >= 2

    return build_device_backed_linestring_output(
        d_out_x, d_out_y,
        row_count=n, validity=validity, geometry_offsets=h_geom_offsets,
    )


# ---------------------------------------------------------------------------
# LineString-LineString intersection (Tier 1: NVRTC kernel)
# ---------------------------------------------------------------------------

def linestring_linestring_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """LineString-LineString intersection: find intersection points.

    Returns a Point or MultiPoint per row depending on intersection count.
    Uses two-pass count-scatter NVRTC kernels (Tier 1).
    """
    import cupy as cp

    n = left.row_count
    left.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                 reason="linestring_linestring_intersection GPU")
    right.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                  reason="linestring_linestring_intersection GPU")

    l_state = left.device_state
    r_state = right.device_state

    l_buf = l_state.families[GeometryFamily.LINESTRING]
    r_buf = r_state.families[GeometryFamily.LINESTRING]

    d_l_valid = l_state.validity.astype(cp.bool_) & ~l_buf.empty_mask.astype(cp.bool_)
    d_r_valid = r_state.validity.astype(cp.bool_) & ~r_buf.empty_mask.astype(cp.bool_)
    d_both_valid = (d_l_valid & d_r_valid).astype(cp.int32)

    runtime = get_cuda_runtime()
    d_counts = runtime.allocate((n,), cp.int32, zero=True)

    kernels = _linestring_linestring_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(l_buf.x), ptr(l_buf.y), ptr(l_buf.geometry_offsets),
            ptr(r_buf.x), ptr(r_buf.y), ptr(r_buf.geometry_offsets),
            ptr(d_both_valid), ptr(d_counts), n,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["linestring_linestring_count"], n)
    runtime.launch(
        kernels["linestring_linestring_count"],
        grid=grid, block=block, params=count_params,
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total_points = count_scatter_total(runtime, d_counts, d_offsets)

    if total_points == 0:
        return from_shapely_geometries([None] * n)

    d_out_x = runtime.allocate((total_points,), cp.float64)
    d_out_y = runtime.allocate((total_points,), cp.float64)

    scatter_params = (
        (
            ptr(l_buf.x), ptr(l_buf.y), ptr(l_buf.geometry_offsets),
            ptr(r_buf.x), ptr(r_buf.y), ptr(r_buf.geometry_offsets),
            ptr(d_both_valid), ptr(d_offsets),
            ptr(d_out_x), ptr(d_out_y), n,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["linestring_linestring_scatter"], n,
    )
    runtime.launch(
        kernels["linestring_linestring_scatter"],
        grid=scatter_grid, block=scatter_block, params=scatter_params,
    )

    runtime.synchronize()

    # Build output: MultiPoint per row (0 or more intersection points)
    d_offsets_cp = cp.asarray(d_offsets)
    d_geom_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_geom_offsets[:n] = d_offsets_cp
    d_geom_offsets[n] = total_points
    h_geom_offsets = runtime.copy_device_to_host(d_geom_offsets)

    h_counts_arr = runtime.copy_device_to_host(d_counts)
    validity = h_counts_arr > 0

    return build_device_backed_multipoint_output(
        d_out_x, d_out_y,
        row_count=n, validity=validity, geometry_offsets=h_geom_offsets,
    )


# ---------------------------------------------------------------------------
# Helper: build result from source Point array with new validity mask
# ---------------------------------------------------------------------------
