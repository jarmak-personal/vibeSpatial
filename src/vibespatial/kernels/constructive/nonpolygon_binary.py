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

import numpy as np

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
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
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
# Device-backed OwnedGeometryArray builders
# ---------------------------------------------------------------------------

def _build_device_backed_point_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident Point OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POINT: point_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POINT,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
    )


def _build_device_backed_linestring_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident LineString OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    ls_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: ls_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.LINESTRING,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
    )


def _build_device_backed_multipoint_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident MultiPoint OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.MULTIPOINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    mp_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOINT),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.MULTIPOINT: mp_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.MULTIPOINT,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
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
    return _build_point_polygon_result_from_source(left, h_match)


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
    return _build_point_polygon_result_from_source(left, h_keep)


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
    h_offsets = np.empty(n + 1, dtype=np.int32)
    h_offsets[0] = 0
    np.cumsum(h_counts, out=h_offsets[1:])
    total_coords = int(h_offsets[-1])

    if total_coords == 0:
        # All empty
        return from_shapely_geometries([None] * n)

    # Allocate output coordinates on device
    d_out_x = runtime.allocate((total_coords,), np.float64)
    d_out_y = runtime.allocate((total_coords,), np.float64)
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

    return _build_device_backed_multipoint_output(
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
    h_offsets = np.empty(n + 1, dtype=np.int32)
    h_offsets[0] = 0
    np.cumsum(h_counts, out=h_offsets[1:])
    total_coords = int(h_offsets[-1])

    # Validity: row is valid if it has any output points
    d_has_output = d_counts > 0
    h_validity = runtime.copy_device_to_host(d_has_output).astype(bool)

    if total_coords == 0:
        return from_shapely_geometries([None] * n)

    d_out_x = runtime.allocate((total_coords,), np.float64)
    d_out_y = runtime.allocate((total_coords,), np.float64)
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

    return _build_device_backed_multipoint_output(
        d_out_x, d_out_y,
        row_count=n,
        validity=h_validity,
        geometry_offsets=h_offsets,
    )


# ---------------------------------------------------------------------------
# MultiPoint-Polygon constructive operations (Tier 2 over Tier 1 PIP)
# ---------------------------------------------------------------------------

def multipoint_polygon_intersection(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon intersection: keep points inside polygon.

    Uses the PIP kernel on exploded points, then compacts and rebuilds
    MultiPoint offsets.  All validity/offset reads are bulk D2H transfers
    (no per-element .get() in loops).
    """
    import cupy as cp

    n = multipoints.row_count
    multipoints.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason="multipoint_polygon_intersection GPU")
    polygons.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                     reason="multipoint_polygon_intersection GPU")

    mp_state = multipoints.device_state
    poly_state = polygons.device_state

    mp_buf = mp_state.families[GeometryFamily.MULTIPOINT]
    # Determine polygon family
    if GeometryFamily.POLYGON in poly_state.families:
        poly_buf = poly_state.families[GeometryFamily.POLYGON]
    elif GeometryFamily.MULTIPOLYGON in poly_state.families:
        poly_buf = poly_state.families[GeometryFamily.MULTIPOLYGON]
    else:
        return from_shapely_geometries([None] * n)

    d_mp_valid = mp_state.validity.astype(cp.bool_) & ~mp_buf.empty_mask.astype(cp.bool_)
    d_poly_valid = poly_state.validity.astype(cp.bool_) & ~poly_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_mp_valid & d_poly_valid

    # Single bulk D2H transfers for offsets and validity
    runtime = get_cuda_runtime()
    h_mp_offsets = runtime.copy_device_to_host(cp.asarray(mp_buf.geometry_offsets))
    h_both_valid = runtime.copy_device_to_host(d_both_valid)
    total_points = int(h_mp_offsets[-1])

    if total_points == 0:
        return from_shapely_geometries([None] * n)

    # Build row_ids on host (no per-element D2H)
    row_ids_h = np.empty(total_points, dtype=np.int32)
    for i in range(n):
        row_ids_h[h_mp_offsets[i]:h_mp_offsets[i + 1]] = i

    # Get coordinates on host for Shapely geometry construction
    h_mp_x = mp_buf.x if mp_buf.host_materialized else get_cuda_runtime().copy_device_to_host(mp_buf.x)
    h_mp_y = mp_buf.y if mp_buf.host_materialized else get_cuda_runtime().copy_device_to_host(mp_buf.y)

    from shapely.geometry import Point as ShapelyPoint

    # Build exploded point geometries and replicated polygon geometries
    poly_shapely = polygons.to_shapely()

    point_geoms = []
    poly_geoms = []
    for i in range(n):
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        for j in range(start, end):
            point_geoms.append(ShapelyPoint(float(h_mp_x[j]), float(h_mp_y[j])))
            poly_geoms.append(poly_shapely[i])

    if len(point_geoms) == 0:
        return from_shapely_geometries([None] * n)

    # Use PIP on exploded arrays
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pt_oga = from_shapely_geometries(point_geoms)
    poly_oga = from_shapely_geometries(poly_geoms)
    pip_mask = point_in_polygon(pt_oga, poly_oga, _return_device=True)
    if hasattr(pip_mask, "__cuda_array_interface__"):
        h_pip = runtime.copy_device_to_host(pip_mask)
    else:
        h_pip = np.asarray(pip_mask, dtype=bool)

    # Compact: for each row, keep only points that are inside
    # Uses bulk h_both_valid array instead of per-element .get()
    result_geoms = []
    for i in range(n):
        if not h_both_valid[i]:
            result_geoms.append(None)
            continue
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        kept = [point_geoms[j] for j in range(start, end) if h_pip[j]]
        if len(kept) == 0:
            result_geoms.append(None)
        elif len(kept) == 1:
            result_geoms.append(kept[0])
        else:
            from shapely.geometry import MultiPoint as ShapelyMultiPoint
            result_geoms.append(ShapelyMultiPoint(kept))

    return from_shapely_geometries(result_geoms)


def multipoint_polygon_difference(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon difference: keep points outside polygon.

    All validity/offset reads are bulk D2H transfers
    (no per-element .get() in loops).
    """
    import cupy as cp

    n = multipoints.row_count
    multipoints.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason="multipoint_polygon_difference GPU")
    polygons.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                     reason="multipoint_polygon_difference GPU")

    mp_state = multipoints.device_state
    poly_state = polygons.device_state

    mp_buf = mp_state.families[GeometryFamily.MULTIPOINT]

    d_mp_valid = mp_state.validity.astype(cp.bool_) & ~mp_buf.empty_mask.astype(cp.bool_)
    d_poly_valid = poly_state.validity.astype(cp.bool_)
    d_both_valid = d_mp_valid & d_poly_valid

    # Single bulk D2H transfers
    runtime = get_cuda_runtime()
    h_mp_offsets = runtime.copy_device_to_host(cp.asarray(mp_buf.geometry_offsets))
    h_mp_valid = runtime.copy_device_to_host(d_mp_valid)
    h_both_valid = runtime.copy_device_to_host(d_both_valid)
    total_points = int(h_mp_offsets[-1])

    if total_points == 0:
        return from_shapely_geometries([None] * n)

    h_mp_x = mp_buf.x if mp_buf.host_materialized else get_cuda_runtime().copy_device_to_host(mp_buf.x)
    h_mp_y = mp_buf.y if mp_buf.host_materialized else get_cuda_runtime().copy_device_to_host(mp_buf.y)

    from shapely.geometry import Point as ShapelyPoint

    poly_shapely = polygons.to_shapely()

    point_geoms = []
    poly_geoms = []
    for i in range(n):
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        for j in range(start, end):
            point_geoms.append(ShapelyPoint(float(h_mp_x[j]), float(h_mp_y[j])))
            poly_geoms.append(poly_shapely[i])

    if len(point_geoms) == 0:
        return from_shapely_geometries([None] * n)

    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pt_oga = from_shapely_geometries(point_geoms)
    poly_oga = from_shapely_geometries(poly_geoms)
    pip_mask = point_in_polygon(pt_oga, poly_oga, _return_device=True)
    if hasattr(pip_mask, "__cuda_array_interface__"):
        h_pip = runtime.copy_device_to_host(pip_mask)
    else:
        h_pip = np.asarray(pip_mask, dtype=bool)

    # Difference: keep points NOT inside.
    # Uses bulk h_mp_valid/h_both_valid arrays instead of per-element .get()
    result_geoms = []
    for i in range(n):
        if not h_mp_valid[i]:
            result_geoms.append(None)
            continue
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        if h_both_valid[i]:
            kept = [point_geoms[j] for j in range(start, end) if not h_pip[j]]
        else:
            # Right is NULL: difference with NULL = identity
            kept = [point_geoms[j] for j in range(start, end)]
        if len(kept) == 0:
            result_geoms.append(None)
        elif len(kept) == 1:
            result_geoms.append(kept[0])
        else:
            from shapely.geometry import MultiPoint as ShapelyMultiPoint
            result_geoms.append(ShapelyMultiPoint(kept))

    return from_shapely_geometries(result_geoms)


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
    d_on_line = runtime.allocate((n,), np.int32, zero=True)

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

    return _build_point_polygon_result_from_source(points, new_validity)


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
    d_counts = runtime.allocate((n,), np.int32, zero=True)

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
        validity = np.zeros(n, dtype=bool)
        geometry_offsets = np.zeros(n + 1, dtype=np.int32)
        d_out_x = runtime.allocate((0,), np.float64)
        d_out_y = runtime.allocate((0,), np.float64)
        return _build_device_backed_linestring_output(
            d_out_x, d_out_y,
            row_count=n, validity=validity, geometry_offsets=geometry_offsets,
        )

    d_out_x = runtime.allocate((total_verts,), np.float64)
    d_out_y = runtime.allocate((total_verts,), np.float64)

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

    return _build_device_backed_linestring_output(
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
    d_counts = runtime.allocate((n,), np.int32, zero=True)

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

    d_out_x = runtime.allocate((total_points,), np.float64)
    d_out_y = runtime.allocate((total_points,), np.float64)

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

    return _build_device_backed_multipoint_output(
        d_out_x, d_out_y,
        row_count=n, validity=validity, geometry_offsets=h_geom_offsets,
    )


# ---------------------------------------------------------------------------
# Helper: build result from source Point array with new validity mask
# ---------------------------------------------------------------------------

def _build_point_polygon_result_from_source(
    points: OwnedGeometryArray,
    new_validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray sharing Point buffers with new validity.

    Mirrors _build_point_polygon_result from binary_constructive.py.
    """
    from vibespatial.geometry.owned import OwnedGeometryDeviceState

    new_device_state = None
    if points.device_state is not None:
        runtime = get_cuda_runtime()
        new_device_state = OwnedGeometryDeviceState(
            validity=runtime.from_host(new_validity),
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
            families=dict(points.device_state.families),
        )

    result = OwnedGeometryArray(
        validity=new_validity,
        tags=points.tags.copy(),
        family_row_offsets=points.family_row_offsets.copy(),
        families=dict(points.families),
        residency=points.residency,
        device_state=new_device_state,
    )
    return result
