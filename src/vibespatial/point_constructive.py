from __future__ import annotations

import numpy as np

from vibespatial.cccl_primitives import compact_indices
from vibespatial.cccl_precompile import request_warmup

request_warmup(["select_i32", "select_i64"])
from vibespatial.cuda_runtime import (  # noqa: E402
    DeviceArray,
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily, get_geometry_buffer_schema  # noqa: E402
from vibespatial.owned_geometry import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.residency import Residency, TransferTrigger  # noqa: E402
from vibespatial.adaptive_runtime import plan_dispatch_selection  # noqa: E402
from vibespatial.precision import KernelClass  # noqa: E402
from vibespatial.runtime import ExecutionMode, has_gpu_runtime  # noqa: E402


_POINT_CONSTRUCTIVE_KERNEL_SOURCE = """
extern "C" __global__ void point_rect_mask(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int inclusive,
    unsigned char* out,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    out[row] = 0;
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  out[row] = inclusive
      ? ((px >= xmin && px <= xmax && py >= ymin && py <= ymax) ? 1 : 0)
      : ((px > xmin && px < xmax && py > ymin && py < ymax) ? 1 : 0);
}

extern "C" __global__ void point_buffer_quad1(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const double* radii,
    double* out_x,
    double* out_y,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  const double radius = radii[row];
  const int base = row * 5;
  out_x[base + 0] = px + radius; out_y[base + 0] = py;
  out_x[base + 1] = px;          out_y[base + 1] = py - radius;
  out_x[base + 2] = px - radius; out_y[base + 2] = py;
  out_x[base + 3] = px;          out_y[base + 3] = py + radius;
  out_x[base + 4] = px + radius; out_y[base + 4] = py;
}

extern "C" __global__ void point_buffer_round(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const double* radii,
    double* out_x,
    double* out_y,
    int quad_segs,
    int verts_per_ring,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  const double radius = radii[row];
  const int base = row * verts_per_ring;
  const int n_arc = 4 * quad_segs;
  const double step = -2.0 * 3.14159265358979323846 / (double)n_arc;
  for (int i = 0; i < n_arc; i++) {
    double angle = (double)i * step;
    out_x[base + i] = px + radius * cos(angle);
    out_y[base + i] = py + radius * sin(angle);
  }
  out_x[base + n_arc] = out_x[base];
  out_y[base + n_arc] = out_y[base];
}

extern "C" __global__ void point_subset_gather(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const int* keep_rows,
    double* out_x,
    double* out_y,
    int out_row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= out_row_count) {
    return;
  }
  const int source_row = keep_rows[row];
  const int point_row = point_row_offsets[source_row];
  const int coord = point_geometry_offsets[point_row];
  out_x[row] = point_x[coord];
  out_y[row] = point_y[coord];
}
"""

POINT_CLIP_GPU_THRESHOLD = 10_000
POINT_BUFFER_GPU_THRESHOLD = 10_000


_POINT_CONSTRUCTIVE_KERNEL_NAMES = ("point_rect_mask", "point_buffer_quad1", "point_buffer_round", "point_subset_gather")

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
request_nvrtc_warmup([
    ("point-constructive", _POINT_CONSTRUCTIVE_KERNEL_SOURCE, _POINT_CONSTRUCTIVE_KERNEL_NAMES),
])


def _point_constructive_kernels():
    return compile_kernel_group("point-constructive", _POINT_CONSTRUCTIVE_KERNEL_SOURCE, _POINT_CONSTRUCTIVE_KERNEL_NAMES)


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


def _select_constructive_mode(
    dispatch_mode: ExecutionMode,
    *,
    row_count: int,
    gpu_threshold: int,
) -> ExecutionMode:
    if dispatch_mode is ExecutionMode.GPU:
        return ExecutionMode.GPU
    if dispatch_mode is ExecutionMode.CPU:
        return ExecutionMode.CPU
    if has_gpu_runtime() and row_count >= gpu_threshold:
        return ExecutionMode.GPU
    return ExecutionMode.CPU


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
        angles = np.linspace(0.0, -2.0 * np.pi, num=n_arc, endpoint=False, dtype=np.float64)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
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
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        bounds=None,
        host_materialized=False,
    )
    runtime = get_cuda_runtime()
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
                    bounds=None,
                )
            },
        ),
    )


def _build_device_backed_polygon_output(
    device_x,
    device_y,
    *,
    row_count: int,
    bounds: np.ndarray | None,
    verts_per_ring: int = 5,
) -> OwnedGeometryArray:
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    ring_offsets = np.arange(0, (row_count + 1) * verts_per_ring, verts_per_ring, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=bounds,
        host_materialized=False,
    )
    runtime = get_cuda_runtime()
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None if bounds is None else runtime.from_host(bounds),
                )
            },
        ),
    )


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

    point_buffer = points.families[GeometryFamily.POINT]
    if not np.all(points.validity):
        raise ValueError("point_buffer_owned_array requires non-null point rows only")
    if np.any(points.tags != FAMILY_TAGS[GeometryFamily.POINT]):
        raise ValueError("point_buffer_owned_array requires point-only rows")
    if np.any(point_buffer.empty_mask):
        raise ValueError("point_buffer_owned_array requires non-null, non-empty point rows only")
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
    ).selected
    if selected_mode is not ExecutionMode.GPU:
        return _build_point_buffers_cpu(points, radii, quad_segs=quad_segs)

    verts_per_ring = 4 * quad_segs + 1
    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="point_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = points._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    device_radii = runtime.from_host(radii)
    device_x = runtime.allocate((points.row_count * verts_per_ring,), np.float64)
    device_y = runtime.allocate((points.row_count * verts_per_ring,), np.float64)
    bounds = None
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
                    ptr(device_x),
                    ptr(device_y),
                    quad_segs,
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
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            )
        grid, block = runtime.launch_config(kernel, points.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        host_point_buffer = points.families[GeometryFamily.POINT]
        if host_point_buffer.host_materialized:
            coord_rows = host_point_buffer.geometry_offsets[points.family_row_offsets]
            point_x = host_point_buffer.x[coord_rows]
            point_y = host_point_buffer.y[coord_rows]
            bounds = np.column_stack((point_x - radii, point_y - radii, point_x + radii, point_y + radii))
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
        if not success:
            runtime.free(device_x)
            runtime.free(device_y)
