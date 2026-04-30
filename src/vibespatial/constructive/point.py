from __future__ import annotations

from functools import lru_cache

import numpy as np

from vibespatial.constructive.point_kernels import (
    _POINT_CONSTRUCTIVE_KERNEL_NAMES,
    _POINT_CONSTRUCTIVE_KERNEL_SOURCE,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import compact_indices

request_warmup(["select_i32", "select_i64"])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import (  # noqa: E402
    Residency,
    TransferTrigger,
    combined_residency,
)

request_nvrtc_warmup([
    ("point-constructive", _POINT_CONSTRUCTIVE_KERNEL_SOURCE, _POINT_CONSTRUCTIVE_KERNEL_NAMES),
])


def _point_constructive_kernels():
    return compile_kernel_group("point-constructive", _POINT_CONSTRUCTIVE_KERNEL_SOURCE, _POINT_CONSTRUCTIVE_KERNEL_NAMES)


def _device_bool_scalar(value, *, reason: str) -> bool:
    import cupy as cp

    scalar = cp.asarray(value).reshape(1)
    host = get_cuda_runtime().copy_device_to_host(scalar, reason=reason)
    return bool(np.asarray(host).reshape(-1)[0])


def _point_rows_and_xy(points: OwnedGeometryArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points._ensure_host_state()
    buffer = points.families[GeometryFamily.POINT]
    point_rows = np.flatnonzero(
        points.validity
        & (points.tags == FAMILY_TAGS[GeometryFamily.POINT])
        & ~buffer.empty_mask[points.family_row_offsets]
    ).astype(np.int32, copy=False)
    family_rows = points.family_row_offsets[point_rows]
    coord_rows = buffer.geometry_offsets[family_rows]
    return point_rows, buffer.x[coord_rows], buffer.y[coord_rows]


def point_owned_from_xy(x: np.ndarray, y: np.ndarray) -> OwnedGeometryArray:
    """Create a point-only OwnedGeometryArray directly from x/y coordinate arrays.

    Avoids Shapely round-trip.  Both arrays must be 1-D float64 of equal length.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    row_count = x.size
    if y.size != row_count:
        raise ValueError("x and y must have the same length")

    geom_offsets = np.arange(row_count + 1, dtype=np.int32)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    bounds = np.column_stack((x, y, x, y))

    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=row_count,
        x=x,
        y=y,
        geometry_offsets=geom_offsets,
        empty_mask=np.zeros(row_count, dtype=bool),
        bounds=bounds,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POINT: point_buffer},
        residency=Residency.HOST,
    )


def point_owned_from_xy_device(x: np.ndarray, y: np.ndarray) -> OwnedGeometryArray:
    """Create a device-resident point OwnedGeometryArray from host x/y arrays.

    Uploads coordinate data directly to the GPU and builds a device-resident
    ``OwnedGeometryArray``.  Host-side x/y arrays are kept in the family
    buffer for backward compatibility with code that reads ``buffer.x``
    directly (e.g. ``predicates/support.py:extract_point_coordinates``).

    Both arrays must be 1-D float64 of equal length.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    row_count = x.size
    if y.size != row_count:
        raise ValueError("x and y must have the same length")
    if row_count == 0:
        return _empty_point_output()

    import cupy as cp

    geom_offsets = np.arange(row_count + 1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    bounds = np.column_stack((x, y, x, y))

    runtime = get_cuda_runtime()
    device_x = runtime.from_host(x)
    device_y = runtime.from_host(y)
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_empty_mask = cp.zeros(row_count, dtype=cp.bool_)
    d_bounds = cp.column_stack((device_x, device_y, device_x, device_y))
    d_validity = cp.ones(row_count, dtype=cp.bool_)
    d_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=cp.int8)
    d_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=device_x,
                y=device_y,
                geometry_offsets=d_geom_offsets,
                empty_mask=d_empty_mask,
                bounds=d_bounds,
            )
        },
        row_count=row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    result.families[GeometryFamily.POINT] = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=row_count,
        x=x,
        y=y,
        geometry_offsets=geom_offsets,
        empty_mask=empty_mask,
        bounds=bounds,
    )
    return result


def _empty_point_output() -> OwnedGeometryArray:
    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=0,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.asarray([0], dtype=np.int32),
        empty_mask=np.empty(0, dtype=bool),
        bounds=np.empty((0, 4), dtype=np.float64),
    )
    return OwnedGeometryArray(
        validity=np.empty(0, dtype=bool),
        tags=np.empty(0, dtype=np.int8),
        family_row_offsets=np.empty(0, dtype=np.int32),
        families={GeometryFamily.POINT: point_buffer},
        residency=Residency.HOST,
    )


@lru_cache(maxsize=32)
def _round_point_unit_circle(quad_segs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return GEOS-compatible clockwise point-buffer unit-circle coordinates."""
    n_arc = 4 * int(quad_segs)
    angles = np.linspace(0.0, -2.0 * np.pi, num=n_arc, endpoint=False, dtype=np.float64)
    unit_x = np.cos(angles)
    unit_y = np.sin(angles)
    q = int(quad_segs)
    unit_x[0] = 1.0
    unit_y[0] = 0.0
    unit_x[q] = 0.0
    unit_y[q] = -1.0
    unit_x[2 * q] = -1.0
    unit_y[2 * q] = 0.0
    unit_x[3 * q] = 0.0
    unit_y[3 * q] = 1.0
    return (
        np.ascontiguousarray(unit_x, dtype=np.float64),
        np.ascontiguousarray(unit_y, dtype=np.float64),
    )


def _point_buffer_host_admission_checked(points: OwnedGeometryArray) -> bool:
    if points._validity is None or points._tags is None or points._family_row_offsets is None:
        return False
    point_buffer = points.families.get(GeometryFamily.POINT)
    if point_buffer is None or point_buffer.host_materialized is False:
        return False
    if not np.all(points._validity):
        raise ValueError("point_buffer_owned_array requires non-null point rows only")
    if np.any(points._tags != FAMILY_TAGS[GeometryFamily.POINT]):
        raise ValueError("point_buffer_owned_array requires point-only rows")
    if np.any(point_buffer.empty_mask):
        raise ValueError("point_buffer_owned_array requires non-null, non-empty point rows only")
    return True


def _build_point_buffers_cpu(
    points: OwnedGeometryArray, radii: np.ndarray, *, quad_segs: int = 1,
) -> OwnedGeometryArray:
    _, x, y = _point_rows_and_xy(points)
    row_count = x.size
    if quad_segs == 1:
        verts_per_ring = 5
        out_x = np.empty(row_count * verts_per_ring, dtype=np.float64)
        out_y = np.empty(row_count * verts_per_ring, dtype=np.float64)
        out_x[0::5] = x + radii
        out_y[0::5] = y
        out_x[1::5] = x
        out_y[1::5] = y - radii
        out_x[2::5] = x - radii
        out_y[2::5] = y
        out_x[3::5] = x
        out_y[3::5] = y + radii
        out_x[4::5] = x + radii
        out_y[4::5] = y
    else:
        n_arc = 4 * quad_segs
        verts_per_ring = n_arc + 1
        cos_a, sin_a = _round_point_unit_circle(quad_segs)
        out_x = np.empty(row_count * verts_per_ring, dtype=np.float64)
        out_y = np.empty(row_count * verts_per_ring, dtype=np.float64)
        for i in range(n_arc):
            out_x[i::verts_per_ring] = x + radii * cos_a[i]
            out_y[i::verts_per_ring] = y + radii * sin_a[i]
        out_x[n_arc::verts_per_ring] = out_x[0::verts_per_ring]
        out_y[n_arc::verts_per_ring] = out_y[0::verts_per_ring]
    return _build_polygon_output(out_x, out_y, x, y, radii, verts_per_ring=verts_per_ring)


def _build_polygon_output(
    out_x: np.ndarray,
    out_y: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    radii: np.ndarray,
    *,
    verts_per_ring: int = 5,
) -> OwnedGeometryArray:
    row_count = point_x.size
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    ring_offsets = np.arange(0, (row_count + 1) * verts_per_ring, verts_per_ring, dtype=np.int32)
    bounds = np.column_stack((point_x - radii, point_y - radii, point_x + radii, point_y + radii))
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=int(row_count),
        x=np.ascontiguousarray(out_x, dtype=np.float64),
        y=np.ascontiguousarray(out_y, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=np.zeros(row_count, dtype=bool),
        ring_offsets=ring_offsets,
        bounds=bounds,
    )
    return OwnedGeometryArray(
        validity=np.ones(row_count, dtype=bool),
        tags=np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8),
        family_row_offsets=np.arange(row_count, dtype=np.int32),
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.HOST,
    )


def _build_device_backed_point_output(
    device_x,
    device_y,
    *,
    row_count: int,
) -> OwnedGeometryArray:
    import cupy as cp

    validity = cp.ones(row_count, dtype=cp.bool_)
    tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=cp.int8)
    family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    device_families = {
        GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=device_x,
            y=device_y,
            geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
            empty_mask=cp.zeros(row_count, dtype=cp.uint8),
            bounds=None,
        )
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
        execution_mode="gpu",
    )


def _build_device_backed_polygon_output(
    device_x,
    device_y,
    *,
    row_count: int,
    bounds,
    verts_per_ring: int = 5,
) -> OwnedGeometryArray:
    import cupy as cp

    runtime = get_cuda_runtime()
    bounds_is_device = hasattr(bounds, "__cuda_array_interface__")
    device_bounds = (
        bounds
        if bounds_is_device
        else None
        if bounds is None
        else runtime.from_host(bounds)
    )
    host_bounds = None if bounds_is_device else bounds
    d_geometry_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_ring_offsets = cp.arange(
        0,
        (row_count + 1) * verts_per_ring,
        verts_per_ring,
        dtype=cp.int32,
    )
    d_empty_mask = cp.zeros(row_count, dtype=cp.bool_)
    d_validity = cp.ones(row_count, dtype=cp.bool_)
    d_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8)
    d_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=np.bool_),
        ring_offsets=None,
        bounds=host_bounds,
        host_materialized=False,
    )
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=device_x,
                y=device_y,
                geometry_offsets=d_geometry_offsets,
                empty_mask=d_empty_mask,
                ring_offsets=d_ring_offsets,
                bounds=device_bounds,
                dense_single_ring_width=verts_per_ring,
            )
        },
        row_count=row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.row_bounds = device_bounds
    result.families[GeometryFamily.POLYGON] = polygon_buffer
    return result


def clip_points_rect_owned(
    points: OwnedGeometryArray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
    boundary_inclusive: bool = True,
) -> OwnedGeometryArray:
    if GeometryFamily.POINT not in points.families or len(points.families) != 1:
        raise ValueError("clip_points_rect_owned requires a point-only OwnedGeometryArray")
    selected_mode = plan_dispatch_selection(
        kernel_name="point_clip",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=points.row_count,
        requested_mode=dispatch_mode,
        current_residency=combined_residency(points),
    ).selected
    point_buffer = points.families[GeometryFamily.POINT]
    if point_buffer.row_count == 0:
        return _empty_point_output()
    if selected_mode is not ExecutionMode.GPU:
        point_rows, x, y = _point_rows_and_xy(points)
        keep = point_rows[(x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)]
        if not boundary_inclusive:
            keep = point_rows[(x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)]
        return points.take(keep)

    keep_rows, output_x, output_y = _clip_points_rect_gpu_arrays(
        points,
        xmin,
        ymin,
        xmax,
        ymax,
        boundary_inclusive=boundary_inclusive,
    )
    runtime = get_cuda_runtime()
    if int(keep_rows.size) == 0:
        runtime.free(keep_rows)
        runtime.free(output_x)
        runtime.free(output_y)
        return _empty_point_output()
    try:
        return _build_device_backed_point_output(output_x, output_y, row_count=int(keep_rows.size))
    finally:
        runtime.free(keep_rows)


def _clip_points_rect_gpu_arrays(
    points: OwnedGeometryArray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    boundary_inclusive: bool,
) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    if GeometryFamily.POINT not in points.families or len(points.families) != 1:
        raise ValueError("_clip_points_rect_gpu_arrays requires a point-only OwnedGeometryArray")

    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="clip_points_rect_owned selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = points._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    device_mask = runtime.allocate((points.row_count,), np.uint8)
    keep_rows = None
    output_x = None
    output_y = None
    success = False
    try:
        kernel = _point_constructive_kernels()["point_rect_mask"]
        ptr = runtime.pointer
        params = (
            (
                ptr(state.family_row_offsets),
                ptr(point_buffer.geometry_offsets),
                ptr(point_buffer.empty_mask),
                ptr(point_buffer.x),
                ptr(point_buffer.y),
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax),
                1 if boundary_inclusive else 0,
                ptr(device_mask),
                points.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernel, points.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        keep_rows = compact_indices(device_mask).values
        if int(keep_rows.size) == 0:
            output_x = runtime.allocate((0,), np.float64)
            output_y = runtime.allocate((0,), np.float64)
            success = True
            return keep_rows, output_x, output_y
        output_x = runtime.allocate((int(keep_rows.size),), np.float64)
        output_y = runtime.allocate((int(keep_rows.size),), np.float64)
        gather_kernel = _point_constructive_kernels()["point_subset_gather"]
        gather_params = (
            (
                ptr(state.family_row_offsets),
                ptr(point_buffer.geometry_offsets),
                ptr(point_buffer.x),
                ptr(point_buffer.y),
                ptr(keep_rows),
                ptr(output_x),
                ptr(output_y),
                int(keep_rows.size),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        gather_grid, gather_block = runtime.launch_config(gather_kernel, int(keep_rows.size))
        runtime.launch(
            gather_kernel,
            grid=gather_grid,
            block=gather_block,
            params=gather_params,
        )
        success = True
        return keep_rows, output_x, output_y
    finally:
        runtime.free(device_mask)
        if not success:
            runtime.free(keep_rows)
            runtime.free(output_x)
            runtime.free(output_y)


def point_buffer_owned_array(
    points: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 1,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    if GeometryFamily.POINT not in points.families or len(points.families) != 1:
        raise ValueError("point_buffer_owned_array requires a point-only OwnedGeometryArray")

    radii = (
        np.full(points.row_count, float(distance), dtype=np.float64)
        if np.isscalar(distance)
        else np.asarray(distance, dtype=np.float64)
    )
    if radii.shape != (points.row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    selected_mode = plan_dispatch_selection(
        kernel_name="point_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=points.row_count,
        requested_mode=dispatch_mode,
        current_residency=combined_residency(points),
    ).selected
    if selected_mode is not ExecutionMode.GPU:
        point_buffer = points.families[GeometryFamily.POINT]
        if not np.all(points.validity):
            raise ValueError("point_buffer_owned_array requires non-null point rows only")
        if np.any(points.tags != FAMILY_TAGS[GeometryFamily.POINT]):
            raise ValueError("point_buffer_owned_array requires point-only rows")
        if np.any(point_buffer.empty_mask):
            raise ValueError("point_buffer_owned_array requires non-null, non-empty point rows only")
        return _build_point_buffers_cpu(points, radii, quad_segs=quad_segs)

    host_admission_checked = _point_buffer_host_admission_checked(points)
    verts_per_ring = 4 * quad_segs + 1
    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="point_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    import cupy as cp

    state = points._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    if not host_admission_checked:
        if not _device_bool_scalar(
            cp.all(state.validity),
            reason="point buffer validity admission scalar fence",
        ):
            raise ValueError("point_buffer_owned_array requires non-null point rows only")
        if not _device_bool_scalar(
            cp.all(state.tags == FAMILY_TAGS[GeometryFamily.POINT]),
            reason="point buffer family-tag admission scalar fence",
        ):
            raise ValueError("point_buffer_owned_array requires point-only rows")
        if _device_bool_scalar(
            cp.any(point_buffer.empty_mask),
            reason="point buffer empty-point admission scalar fence",
        ):
            raise ValueError("point_buffer_owned_array requires non-null, non-empty point rows only")
    device_radii = runtime.from_host(radii)
    device_unit_x = None
    device_unit_y = None
    if quad_segs != 1:
        unit_x, unit_y = _round_point_unit_circle(quad_segs)
        device_unit_x = runtime.from_host(unit_x)
        device_unit_y = runtime.from_host(unit_y)
    device_x = runtime.allocate((points.row_count * verts_per_ring,), np.float64)
    device_y = runtime.allocate((points.row_count * verts_per_ring,), np.float64)
    success = False
    try:
        ptr = runtime.pointer
        if quad_segs == 1:
            kernel = _point_constructive_kernels()["point_buffer_quad1"]
            params = (
                (
                    ptr(state.family_row_offsets),
                    ptr(point_buffer.geometry_offsets),
                    ptr(point_buffer.empty_mask),
                    ptr(point_buffer.x),
                    ptr(point_buffer.y),
                    ptr(device_radii),
                    ptr(device_x),
                    ptr(device_y),
                    points.row_count,
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
                    KERNEL_PARAM_I32,
                ),
            )
        else:
            kernel = _point_constructive_kernels()["point_buffer_round"]
            params = (
                (
                    ptr(state.family_row_offsets),
                    ptr(point_buffer.geometry_offsets),
                    ptr(point_buffer.empty_mask),
                    ptr(point_buffer.x),
                    ptr(point_buffer.y),
                    ptr(device_radii),
                    ptr(device_unit_x),
                    ptr(device_unit_y),
                    ptr(device_x),
                    ptr(device_y),
                    verts_per_ring,
                    points.row_count,
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
        grid, block = runtime.launch_config(kernel, points.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        coord_rows = cp.asarray(point_buffer.geometry_offsets)[
            cp.asarray(state.family_row_offsets, dtype=cp.int64)
        ].astype(cp.int64, copy=False)
        point_x = cp.asarray(point_buffer.x)[coord_rows]
        point_y = cp.asarray(point_buffer.y)[coord_rows]
        d_radii = cp.asarray(device_radii)
        bounds = cp.column_stack(
            (
                point_x - d_radii,
                point_y - d_radii,
                point_x + d_radii,
                point_y + d_radii,
            )
        )
        success = True
        return _build_device_backed_polygon_output(
            device_x,
            device_y,
            row_count=points.row_count,
            bounds=bounds,
            verts_per_ring=verts_per_ring,
        )
    finally:
        runtime.free(device_radii)
        runtime.free(device_unit_x)
        runtime.free(device_unit_y)
        if not success:
            runtime.free(device_x)
            runtime.free(device_y)


def point_buffer_native_tabular_result(
    points: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 1,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
):
    """Return point-buffer output as a private native constructive carrier."""
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    buffered = point_buffer_owned_array(
        points,
        distance,
        quad_segs=quad_segs,
        dispatch_mode=dispatch_mode,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        buffered,
        operation="buffer",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
    )
