"""GPU-native element-wise polygon-vs-rectangle intersection kernel.

Clips each polygon row in ``left`` against an axis-aligned rectangle row in
``right``. The rectangle comes from the right polygon's exact 5-vertex box
coordinates, keeping the work on the GPU and avoiding the generic overlay
pipeline for parcel-grid workloads.

ADR-0033: Tier 1 (custom NVRTC kernel) -- geometry-specific ring traversal and
  rectangle clipping.
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
ADR-0034: NVRTC precompilation via request_nvrtc_warmup at module scope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.constructive.polygon_intersection_cpu import (
    polygon_intersection_cpu as _polygon_intersection_cpu,
)
from vibespatial.constructive.polygon_intersection_output import (
    build_device_backed_polygon_intersection_output,
    build_empty_device_backed_polygon_intersection_output,
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
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import PrecisionPlan

logger = logging.getLogger(__name__)

_MAX_INPUT_VERTS = 256
_KERNEL_NAMES = (
    "polygon_rect_intersection_count",
    "polygon_rect_intersection_scatter",
)
_KERNEL_SOURCE = r"""
#define EPSILON 1e-12

__device__ int clip_edge(
    const double* in_x,
    const double* in_y,
    int in_count,
    double* out_x,
    double* out_y,
    int max_out,
    int edge_type,
    double edge_val
) {
    if (in_count == 0) return 0;
    int out_count = 0;

    double prev_x = in_x[in_count - 1];
    double prev_y = in_y[in_count - 1];

    int prev_inside;
    if (edge_type == 0) prev_inside = (prev_x >= edge_val) ? 1 : 0;
    else if (edge_type == 1) prev_inside = (prev_x <= edge_val) ? 1 : 0;
    else if (edge_type == 2) prev_inside = (prev_y >= edge_val) ? 1 : 0;
    else prev_inside = (prev_y <= edge_val) ? 1 : 0;

    for (int i = 0; i < in_count; ++i) {
        double cur_x = in_x[i];
        double cur_y = in_y[i];

        int cur_inside;
        if (edge_type == 0) cur_inside = (cur_x >= edge_val) ? 1 : 0;
        else if (edge_type == 1) cur_inside = (cur_x <= edge_val) ? 1 : 0;
        else if (edge_type == 2) cur_inside = (cur_y >= edge_val) ? 1 : 0;
        else cur_inside = (cur_y <= edge_val) ? 1 : 0;

        if (cur_inside) {
            if (!prev_inside) {
                double ix, iy;
                if (edge_type <= 1) {
                    double dx = cur_x - prev_x;
                    if (fabs(dx) <= EPSILON) {
                        ix = edge_val;
                        iy = prev_y;
                    } else {
                        double t = (edge_val - prev_x) / dx;
                        ix = edge_val;
                        iy = prev_y + t * (cur_y - prev_y);
                    }
                } else {
                    double dy = cur_y - prev_y;
                    if (fabs(dy) <= EPSILON) {
                        ix = prev_x;
                        iy = edge_val;
                    } else {
                        double t = (edge_val - prev_y) / dy;
                        ix = prev_x + t * (cur_x - prev_x);
                        iy = edge_val;
                    }
                }
                if (out_count < max_out) {
                    out_x[out_count] = ix;
                    out_y[out_count] = iy;
                    ++out_count;
                }
            }
            if (out_count < max_out) {
                out_x[out_count] = cur_x;
                out_y[out_count] = cur_y;
                ++out_count;
            }
        } else if (prev_inside) {
            double ix, iy;
            if (edge_type <= 1) {
                double dx = cur_x - prev_x;
                if (fabs(dx) <= EPSILON) {
                    ix = edge_val;
                    iy = prev_y;
                } else {
                    double t = (edge_val - prev_x) / dx;
                    ix = edge_val;
                    iy = prev_y + t * (cur_y - prev_y);
                }
            } else {
                double dy = cur_y - prev_y;
                if (fabs(dy) <= EPSILON) {
                    ix = prev_x;
                    iy = edge_val;
                } else {
                    double t = (edge_val - prev_y) / dy;
                    ix = prev_x + t * (cur_x - prev_x);
                    iy = edge_val;
                }
            }
            if (out_count < max_out) {
                out_x[out_count] = ix;
                out_y[out_count] = iy;
                ++out_count;
            }
        }

        prev_x = cur_x;
        prev_y = cur_y;
        prev_inside = cur_inside;
    }

    return out_count;
}

__device__ double polygon_area2(
    const double* x,
    const double* y,
    int count
) {
    if (count < 3) return 0.0;
    double area2 = 0.0;
    double prev_x = x[count - 1];
    double prev_y = y[count - 1];
    for (int i = 0; i < count; ++i) {
        const double cur_x = x[i];
        const double cur_y = y[i];
        area2 += prev_x * cur_y - cur_x * prev_y;
        prev_x = cur_x;
        prev_y = cur_y;
    }
    return area2;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_intersection_count(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    int* __restrict__ out_counts,
    int* __restrict__ out_valid,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    if (!left_valid[row] || !right_valid[row]) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        return;
    }

    const int ring_start_idx = left_geom_offsets[row];
    const int ring_end_idx = left_geom_offsets[row + 1];
    if (ring_end_idx - ring_start_idx != 1) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        return;
    }

    const int start = left_ring_offsets[ring_start_idx];
    const int end = left_ring_offsets[ring_start_idx + 1];
    int n = end - start;
    if (n > 1) {
        const double dx = left_x[start] - left_x[end - 1];
        const double dy = left_y[start] - left_y[end - 1];
        if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) {
            --n;
        }
    }

    if (n < 3 || n > 256) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        return;
    }

    const double xmin = rect_xmin[row];
    const double ymin = rect_ymin[row];
    const double xmax = rect_xmax[row];
    const double ymax = rect_ymax[row];
    if (!(xmin < xmax && ymin < ymax)) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        return;
    }

    double buf_a_x[256], buf_a_y[256];
    double buf_b_x[256], buf_b_y[256];
    for (int i = 0; i < n; ++i) {
        buf_a_x[i] = left_x[start + i];
        buf_a_y[i] = left_y[start + i];
    }

    double edges[4] = {xmin, xmax, ymin, ymax};
    int count = n;
    double* src_x = buf_a_x;
    double* src_y = buf_a_y;
    double* dst_x = buf_b_x;
    double* dst_y = buf_b_y;

    for (int edge = 0; edge < 4; ++edge) {
        count = clip_edge(src_x, src_y, count, dst_x, dst_y, 256, edge, edges[edge]);
        if (count == 0) break;
        double* tmp;
        tmp = src_x; src_x = dst_x; dst_x = tmp;
        tmp = src_y; src_y = dst_y; dst_y = tmp;
    }

    if (count < 3 || fabs(polygon_area2(src_x, src_y, count)) <= EPSILON) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        return;
    }

    out_counts[row] = count + 1;
    out_valid[row] = 1;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_intersection_scatter(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    const int* __restrict__ out_offsets,
    const int* __restrict__ out_counts,
    const int* __restrict__ out_valid,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !out_valid[row]) return;
    if (!left_valid[row] || !right_valid[row]) return;

    const int ring_start_idx = left_geom_offsets[row];
    const int ring_end_idx = left_geom_offsets[row + 1];
    if (ring_end_idx - ring_start_idx != 1) return;

    const int out_start = out_offsets[row];
    const int expected = out_counts[row];
    if (expected <= 0) return;

    const int start = left_ring_offsets[ring_start_idx];
    const int end = left_ring_offsets[ring_start_idx + 1];
    int n = end - start;
    if (n > 1) {
        const double dx = left_x[start] - left_x[end - 1];
        const double dy = left_y[start] - left_y[end - 1];
        if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) {
            --n;
        }
    }
    if (n < 3 || n > 256) return;

    const double xmin = rect_xmin[row];
    const double ymin = rect_ymin[row];
    const double xmax = rect_xmax[row];
    const double ymax = rect_ymax[row];

    double buf_a_x[256], buf_a_y[256];
    double buf_b_x[256], buf_b_y[256];
    for (int i = 0; i < n; ++i) {
        buf_a_x[i] = left_x[start + i];
        buf_a_y[i] = left_y[start + i];
    }

    double edges[4] = {xmin, xmax, ymin, ymax};
    int count = n;
    double* src_x = buf_a_x;
    double* src_y = buf_a_y;
    double* dst_x = buf_b_x;
    double* dst_y = buf_b_y;

    for (int edge = 0; edge < 4; ++edge) {
        count = clip_edge(src_x, src_y, count, dst_x, dst_y, 256, edge, edges[edge]);
        if (count == 0) return;
        double* tmp;
        tmp = src_x; src_x = dst_x; dst_x = tmp;
        tmp = src_y; src_y = dst_y; dst_y = tmp;
    }

    if (count < 3) return;

    const int max_copy = expected - 1;
    for (int i = 0; i < count && i < max_copy; ++i) {
        out_x[out_start + i] = src_x[i];
        out_y[out_start + i] = src_y[i];
    }
    out_x[out_start + count] = src_x[0];
    out_y[out_start + count] = src_y[0];
}
"""

request_nvrtc_warmup([
    ("polygon-rect-intersection", _KERNEL_SOURCE, _KERNEL_NAMES),
])

request_warmup(["exclusive_scan_i32"])


def _polygon_rect_intersection_kernels():
    return compile_kernel_group(
        "polygon-rect-intersection",
        _KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


def _extract_polygon_family_device_buffer(owned: OwnedGeometryArray):
    if GeometryFamily.POLYGON not in owned.families:
        return None, None
    host_buf = owned.families[GeometryFamily.POLYGON]
    if host_buf.row_count == 0:
        return None, None
    state = owned._ensure_device_state()
    device_buf = (
        state.families[GeometryFamily.POLYGON]
        if GeometryFamily.POLYGON in state.families
        else None
    )
    return device_buf, host_buf


def _device_is_dense_single_ring_polygons(polygon_buf, row_count: int) -> bool:
    if polygon_buf is None or row_count <= 0:
        return False
    if int(polygon_buf.geometry_offsets.size) != row_count + 1:
        return False
    geom_counts = polygon_buf.geometry_offsets[1:] - polygon_buf.geometry_offsets[:-1]
    if not bool(cp.all(geom_counts == 1).item()):
        return False
    if bool(cp.any(polygon_buf.empty_mask).item()):
        return False
    return True


def _device_rectangle_bounds(polygon_buf, row_count: int):
    if not _device_is_dense_single_ring_polygons(polygon_buf, row_count):
        return None
    if int(polygon_buf.ring_offsets.size) != row_count + 1:
        return None
    expected_offsets = cp.arange(0, (row_count + 1) * 5, 5, dtype=cp.int32)
    if not bool(cp.all(polygon_buf.ring_offsets == expected_offsets).item()):
        return None
    if int(polygon_buf.x.size) != row_count * 5 or int(polygon_buf.y.size) != row_count * 5:
        return None

    x = polygon_buf.x.reshape(row_count, 5)
    y = polygon_buf.y.reshape(row_count, 5)
    if not bool(cp.all(cp.isclose(x[:, 0], x[:, 4])).item()):
        return None
    if not bool(cp.all(cp.isclose(y[:, 0], y[:, 4])).item()):
        return None

    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    axis_aligned = ((cp.abs(dx) < 1e-12) ^ (cp.abs(dy) < 1e-12))
    if not bool(cp.all(axis_aligned).item()):
        return None

    return (
        cp.min(x[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.min(y[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.max(x[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.max(y[:, :4], axis=1).astype(cp.float64, copy=False),
    )


def _host_is_dense_single_ring_polygons(polygon_buf, row_count: int) -> bool:
    if polygon_buf is None or row_count <= 0:
        return False
    if int(polygon_buf.geometry_offsets.size) != row_count + 1:
        return False
    for row in range(row_count):
        if int(polygon_buf.geometry_offsets[row + 1]) - int(polygon_buf.geometry_offsets[row]) != 1:
            return False
        if bool(polygon_buf.empty_mask[row]):
            return False
    return True


def _host_rectangle_bounds(polygon_buf, row_count: int):
    if not _host_is_dense_single_ring_polygons(polygon_buf, row_count):
        return None
    if polygon_buf.ring_offsets is None or int(polygon_buf.ring_offsets.size) != row_count + 1:
        return None
    for row in range(row_count + 1):
        if int(polygon_buf.ring_offsets[row]) != row * 5:
            return None
    if int(polygon_buf.x.size) != row_count * 5 or int(polygon_buf.y.size) != row_count * 5:
        return None
    epsilon = 1e-12
    xmin: list[float] = []
    ymin: list[float] = []
    xmax: list[float] = []
    ymax: list[float] = []
    for row in range(row_count):
        base = row * 5
        if abs(float(polygon_buf.x[base]) - float(polygon_buf.x[base + 4])) > epsilon:
            return None
        if abs(float(polygon_buf.y[base]) - float(polygon_buf.y[base + 4])) > epsilon:
            return None
        row_x = [float(polygon_buf.x[base + index]) for index in range(4)]
        row_y = [float(polygon_buf.y[base + index]) for index in range(4)]
        for edge in range(4):
            dx = float(polygon_buf.x[base + edge + 1]) - float(polygon_buf.x[base + edge])
            dy = float(polygon_buf.y[base + edge + 1]) - float(polygon_buf.y[base + edge])
            x_axis = abs(dx) < epsilon
            y_axis = abs(dy) < epsilon
            if x_axis == y_axis:
                return None
        xmin.append(min(row_x))
        ymin.append(min(row_y))
        xmax.append(max(row_x))
        ymax.append(max(row_y))
    return xmin, ymin, xmax, ymax


def polygon_rect_intersection_can_handle(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    if cp is None or left.row_count != right.row_count or left.row_count == 0:
        return False
    if set(left.families) != {GeometryFamily.POLYGON}:
        return False
    if set(right.families) != {GeometryFamily.POLYGON}:
        return False
    if left.families[GeometryFamily.POLYGON].row_count != left.row_count:
        return False
    if right.families[GeometryFamily.POLYGON].row_count != right.row_count:
        return False

    left_host = left.families[GeometryFamily.POLYGON]
    right_host = right.families[GeometryFamily.POLYGON]
    if not _host_is_dense_single_ring_polygons(left_host, left.row_count):
        return False
    if left_host.ring_offsets is None:
        return False
    max_input_verts = 0
    for row in range(left.row_count):
        ring_span = int(left_host.ring_offsets[row + 1]) - int(left_host.ring_offsets[row])
        if ring_span > max_input_verts:
            max_input_verts = ring_span
    if max_input_verts > (_MAX_INPUT_VERTS + 1):
        return False
    return _host_rectangle_bounds(right_host, right.row_count) is not None


def _polygon_rect_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    runtime = get_cuda_runtime()
    n = left.row_count

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_rect_intersection selected GPU execution",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_rect_intersection selected GPU execution",
    )

    left_dev, left_host = _extract_polygon_family_device_buffer(left)
    right_dev, right_host = _extract_polygon_family_device_buffer(right)
    if left_dev is None or right_dev is None:
        return _build_empty_result(n, runtime_selection)
    if left_host.row_count != n or right_host.row_count != n:
        raise ValueError(
            "polygon_rect_intersection GPU path requires polygon-only inputs "
            f"(left family rows={left_host.row_count}, "
            f"right family rows={right_host.row_count}, expected={n})"
        )
    if not _device_is_dense_single_ring_polygons(left_dev, n):
        raise ValueError("left operand is not a dense single-ring polygon batch")

    rect_bounds = _device_rectangle_bounds(right_dev, n)
    if rect_bounds is None:
        raise ValueError("right operand is not an axis-aligned rectangle batch")
    d_xmin, d_ymin, d_xmax, d_ymax = rect_bounds

    left_state = left.device_state
    right_state = right.device_state
    d_left_valid = (
        left_state.validity.astype(cp.bool_) & ~left_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)
    d_right_valid = (
        right_state.validity.astype(cp.bool_) & ~right_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)

    d_counts = runtime.allocate((n,), cp.int32, zero=True)
    d_valid = runtime.allocate((n,), cp.int32, zero=True)

    kernels = _polygon_rect_intersection_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            n,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_intersection_count"], n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total_verts = count_scatter_total(runtime, d_counts, d_offsets)
    if total_verts == 0:
        return _build_empty_result(n, runtime_selection)

    d_out_x = runtime.allocate((total_verts,), cp.float64)
    d_out_y = runtime.allocate((total_verts,), cp.float64)

    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_offsets),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_out_x),
            ptr(d_out_y),
            n,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_intersection_scatter"], n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_ring_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_ring_offsets[:n] = cp.asarray(d_offsets)
    d_ring_offsets[n] = total_verts

    return build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid.astype(cp.bool_),
        ring_offsets=d_ring_offsets,
        runtime_selection=runtime_selection,
    )


def _build_empty_result(n: int, runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    return build_empty_device_backed_polygon_intersection_output(
        row_count=n,
        runtime_selection=runtime_selection,
    )


@register_kernel_variant(
    "polygon_rect_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=False,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "intersection", "rectangle", "clip"),
)
def _polygon_rect_intersection_gpu_variant(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    return _polygon_rect_intersection_gpu(
        left,
        right,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
    )


def polygon_rect_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    if left.row_count != right.row_count:
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )
    n = left.row_count
    if n == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="polygon_rect_intersection",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )
    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _polygon_rect_intersection_gpu(
            left,
            right,
            runtime_selection=selection,
            precision_plan=precision_plan,
        )
        record_dispatch_event(
            surface="vibespatial.kernels.constructive.polygon_rect_intersection",
            operation="polygon_rect_intersection",
            implementation="polygon_rect_intersection_gpu",
            reason=selection.reason,
            detail=(
                f"rows={n}, "
                f"precision={precision_plan.compute_precision.value}"
            ),
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    result = _polygon_intersection_cpu(left, right, precision=precision)
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_rect_intersection",
        operation="polygon_rect_intersection",
        implementation="polygon_rect_intersection_cpu",
        reason=selection.reason,
        detail=f"rows={n}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
