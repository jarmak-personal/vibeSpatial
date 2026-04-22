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
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, combined_residency
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
_MAX_BOUNDARY_SPLIT_COMPONENTS = 16
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

__device__ int compact_vertices(
    double* x,
    double* y,
    int count
) {
    if (count <= 1) return count;

    int out_count = 0;
    for (int i = 0; i < count; ++i) {
        const double cur_x = x[i];
        const double cur_y = y[i];
        if (out_count > 0) {
            const double dx = cur_x - x[out_count - 1];
            const double dy = cur_y - y[out_count - 1];
            if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) {
                continue;
            }
        }
        x[out_count] = cur_x;
        y[out_count] = cur_y;
        ++out_count;
    }

    if (out_count > 1) {
        const double dx = x[out_count - 1] - x[0];
        const double dy = y[out_count - 1] - y[0];
        if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) {
            --out_count;
        }
    }
    return out_count;
}

__device__ int remove_colinear_vertices(
    const double* in_x,
    const double* in_y,
    int count,
    double* out_x,
    double* out_y,
    int max_out
) {
    if (count <= 2) return count;

    int out_count = 0;
    for (int i = 0; i < count; ++i) {
        const int prev = (i + count - 1) % count;
        const int next = (i + 1) % count;
        const double ax = in_x[i] - in_x[prev];
        const double ay = in_y[i] - in_y[prev];
        const double bx = in_x[next] - in_x[i];
        const double by = in_y[next] - in_y[i];
        const double cross = ax * by - ay * bx;
        const double scale = fabs(ax) + fabs(ay) + fabs(bx) + fabs(by) + 1.0;
        if (fabs(cross) <= EPSILON * scale) {
            continue;
        }
        if (out_count < max_out) {
            out_x[out_count] = in_x[i];
            out_y[out_count] = in_y[i];
            ++out_count;
        }
    }
    return out_count;
}

__device__ int finalize_clipped_vertices(
    double* src_x,
    double* src_y,
    double* tmp_x,
    double* tmp_y,
    int count,
    int max_out
) {
    count = compact_vertices(src_x, src_y, count);
    count = remove_colinear_vertices(src_x, src_y, count, tmp_x, tmp_y, max_out);
    for (int i = 0; i < count && i < max_out; ++i) {
        src_x[i] = tmp_x[i];
        src_y[i] = tmp_y[i];
    }
    count = compact_vertices(src_x, src_y, count);
    count = remove_colinear_vertices(src_x, src_y, count, tmp_x, tmp_y, max_out);
    for (int i = 0; i < count && i < max_out; ++i) {
        src_x[i] = tmp_x[i];
        src_y[i] = tmp_y[i];
    }
    return compact_vertices(src_x, src_y, count);
}

__device__ int has_repeated_rect_boundary_segments(
    const double* x,
    const double* y,
    int count,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    int xmin_segments = 0;
    int xmax_segments = 0;
    int ymin_segments = 0;
    int ymax_segments = 0;

    for (int i = 0; i < count; ++i) {
        const int next = (i + 1) % count;
        const double x0 = x[i];
        const double y0 = y[i];
        const double x1 = x[next];
        const double y1 = y[next];
        if (fabs(x0 - x1) <= EPSILON && fabs(y0 - y1) <= EPSILON) {
            continue;
        }
        if (fabs(x0 - x1) <= EPSILON) {
            const double span = fabs(y1 - y0);
            if (span <= EPSILON) continue;
            if (fabs(x0 - xmin) <= EPSILON) ++xmin_segments;
            else if (fabs(x0 - xmax) <= EPSILON) ++xmax_segments;
        } else if (fabs(y0 - y1) <= EPSILON) {
            const double span = fabs(x1 - x0);
            if (span <= EPSILON) continue;
            if (fabs(y0 - ymin) <= EPSILON) ++ymin_segments;
            else if (fabs(y0 - ymax) <= EPSILON) ++ymax_segments;
        }
    }

    return (
        xmin_segments > 1
        || xmax_segments > 1
        || ymin_segments > 1
        || ymax_segments > 1
    ) ? 1 : 0;
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
    int* __restrict__ out_boundary_overlap,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    if (!left_valid[row] || !right_valid[row]) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        return;
    }

    const int ring_start_idx = left_geom_offsets[row];
    const int ring_end_idx = left_geom_offsets[row + 1];
    if (ring_end_idx - ring_start_idx != 1) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
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
        out_boundary_overlap[row] = 0;
        return;
    }

    const double xmin = rect_xmin[row];
    const double ymin = rect_ymin[row];
    const double xmax = rect_xmax[row];
    const double ymax = rect_ymax[row];
    if (!(xmin < xmax && ymin < ymax)) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        return;
    }

    int boundary_overlap = 0;
    double prev_seg_x = left_x[start + n - 1];
    double prev_seg_y = left_y[start + n - 1];
    for (int i = 0; i < n; ++i) {
        const double cur_seg_x = left_x[start + i];
        const double cur_seg_y = left_y[start + i];

        if (fabs(prev_seg_x - cur_seg_x) <= EPSILON) {
            if (fabs(prev_seg_x - xmin) <= EPSILON || fabs(prev_seg_x - xmax) <= EPSILON) {
                const double seg_min = fmin(prev_seg_y, cur_seg_y);
                const double seg_max = fmax(prev_seg_y, cur_seg_y);
                const double overlap_min = fmax(seg_min, ymin);
                const double overlap_max = fmin(seg_max, ymax);
                if ((overlap_max - overlap_min) > EPSILON) {
                    boundary_overlap = 1;
                    break;
                }
            }
        } else if (fabs(prev_seg_y - cur_seg_y) <= EPSILON) {
            if (fabs(prev_seg_y - ymin) <= EPSILON || fabs(prev_seg_y - ymax) <= EPSILON) {
                const double seg_min = fmin(prev_seg_x, cur_seg_x);
                const double seg_max = fmax(prev_seg_x, cur_seg_x);
                const double overlap_min = fmax(seg_min, xmin);
                const double overlap_max = fmin(seg_max, xmax);
                if ((overlap_max - overlap_min) > EPSILON) {
                    boundary_overlap = 1;
                    break;
                }
            }
        }

        prev_seg_x = cur_seg_x;
        prev_seg_y = cur_seg_y;
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
        count = compact_vertices(dst_x, dst_y, count);
        count = remove_colinear_vertices(dst_x, dst_y, count, src_x, src_y, 256);
        if (count == 0) break;
    }
    count = finalize_clipped_vertices(src_x, src_y, dst_x, dst_y, count, 256);

    if (count < 3 || fabs(polygon_area2(src_x, src_y, count)) <= EPSILON) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        return;
    }

    if (has_repeated_rect_boundary_segments(src_x, src_y, count, xmin, ymin, xmax, ymax)) {
        boundary_overlap = 1;
    }
    out_boundary_overlap[row] = boundary_overlap;
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
        count = compact_vertices(dst_x, dst_y, count);
        count = remove_colinear_vertices(dst_x, dst_y, count, src_x, src_y, 256);
        if (count == 0) return;
    }
    count = finalize_clipped_vertices(src_x, src_y, dst_x, dst_y, count, 256);

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

_BOUNDARY_SPLIT_KERNEL_NAMES = (
    "polygon_rect_boundary_split_count",
    "polygon_rect_boundary_split_scatter",
)
_BOUNDARY_SPLIT_KERNEL_SOURCE = r"""
#define EPSILON 1e-12
#define MAX_INPUT_VERTS 256
#define MAX_COMPONENTS 16

__device__ int point_side(
    double x,
    double y,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    if (fabs(y - ymin) <= EPSILON && x >= xmin - EPSILON && x <= xmax + EPSILON) return 0;
    if (fabs(x - xmax) <= EPSILON && y >= ymin - EPSILON && y <= ymax + EPSILON) return 1;
    if (fabs(y - ymax) <= EPSILON && x >= xmin - EPSILON && x <= xmax + EPSILON) return 2;
    if (fabs(x - xmin) <= EPSILON && y >= ymin - EPSILON && y <= ymax + EPSILON) return 3;
    return -1;
}

__device__ int is_rect_boundary_edge(
    double x0,
    double y0,
    double x1,
    double y1,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) return 0;
    if (fabs(y0 - y1) <= EPSILON) {
        if (
            (fabs(y0 - ymin) <= EPSILON || fabs(y0 - ymax) <= EPSILON)
            && x0 >= xmin - EPSILON && x0 <= xmax + EPSILON
            && x1 >= xmin - EPSILON && x1 <= xmax + EPSILON
        ) return 1;
    }
    if (fabs(x0 - x1) <= EPSILON) {
        if (
            (fabs(x0 - xmin) <= EPSILON || fabs(x0 - xmax) <= EPSILON)
            && y0 >= ymin - EPSILON && y0 <= ymax + EPSILON
            && y1 >= ymin - EPSILON && y1 <= ymax + EPSILON
        ) return 1;
    }
    return 0;
}

__device__ double boundary_t(
    double x,
    double y,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    const double w = xmax - xmin;
    const double h = ymax - ymin;
    const int side = point_side(x, y, xmin, ymin, xmax, ymax);
    if (side == 0) return x - xmin;
    if (side == 1) return w + (y - ymin);
    if (side == 2) return w + h + (xmax - x);
    if (side == 3) return w + h + w + (ymax - y);
    return -1.0;
}

__device__ void point_at_corner_t(
    int corner,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double* out_x,
    double* out_y
) {
    if (corner == 0) {
        *out_x = xmin; *out_y = ymin;
    } else if (corner == 1) {
        *out_x = xmax; *out_y = ymin;
    } else if (corner == 2) {
        *out_x = xmax; *out_y = ymax;
    } else {
        *out_x = xmin; *out_y = ymax;
    }
}

__device__ int boundary_corner_count(
    double from_x,
    double from_y,
    double to_x,
    double to_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    const double w = xmax - xmin;
    const double h = ymax - ymin;
    const double perimeter = 2.0 * (w + h);
    if (!(perimeter > EPSILON)) return -1;
    const double from_t = boundary_t(from_x, from_y, xmin, ymin, xmax, ymax);
    const double to_t = boundary_t(to_x, to_y, xmin, ymin, xmax, ymax);
    if (from_t < 0.0 || to_t < 0.0) return -1;
    double cw = to_t - from_t;
    if (cw <= EPSILON) cw += perimeter;
    const double ccw = perimeter - cw;
    const int clockwise = (cw <= ccw) ? 1 : 0;
    const double distance = clockwise ? cw : ccw;
    const double corners[4] = {0.0, w, w + h, w + h + w};
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        double delta = clockwise ? (corners[i] - from_t) : (from_t - corners[i]);
        if (delta <= EPSILON) delta += perimeter;
        if (delta < distance - EPSILON) ++count;
    }
    return count;
}

__device__ void append_boundary_path_area(
    double from_x,
    double from_y,
    double to_x,
    double to_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double* prev_x,
    double* prev_y,
    double* area2
) {
    const double w = xmax - xmin;
    const double h = ymax - ymin;
    const double perimeter = 2.0 * (w + h);
    double current_t = boundary_t(from_x, from_y, xmin, ymin, xmax, ymax);
    const double to_t = boundary_t(to_x, to_y, xmin, ymin, xmax, ymax);
    if (current_t < 0.0 || to_t < 0.0 || !(perimeter > EPSILON)) return;

    double cw = to_t - current_t;
    if (cw <= EPSILON) cw += perimeter;
    const double ccw = perimeter - cw;
    const int clockwise = (cw <= ccw) ? 1 : 0;
    const double corners[4] = {0.0, w, w + h, w + h + w};
    double remaining = clockwise ? cw : ccw;

    while (remaining > EPSILON) {
        int best = -1;
        double best_delta = perimeter + 1.0;
        for (int i = 0; i < 4; ++i) {
            double delta = clockwise ? (corners[i] - current_t) : (current_t - corners[i]);
            if (delta <= EPSILON) delta += perimeter;
            if (delta < remaining - EPSILON && delta < best_delta) {
                best_delta = delta;
                best = i;
            }
        }
        if (best < 0) break;
        double cx, cy;
        point_at_corner_t(best, xmin, ymin, xmax, ymax, &cx, &cy);
        *area2 += (*prev_x) * cy - cx * (*prev_y);
        *prev_x = cx;
        *prev_y = cy;
        current_t = corners[best];
        remaining -= best_delta;
    }
}

__device__ int append_boundary_path_points(
    double from_x,
    double from_y,
    double to_x,
    double to_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double* out_x,
    double* out_y,
    int cursor,
    int limit
) {
    const double w = xmax - xmin;
    const double h = ymax - ymin;
    const double perimeter = 2.0 * (w + h);
    double current_t = boundary_t(from_x, from_y, xmin, ymin, xmax, ymax);
    const double to_t = boundary_t(to_x, to_y, xmin, ymin, xmax, ymax);
    if (current_t < 0.0 || to_t < 0.0 || !(perimeter > EPSILON)) return cursor;

    double cw = to_t - current_t;
    if (cw <= EPSILON) cw += perimeter;
    const double ccw = perimeter - cw;
    const int clockwise = (cw <= ccw) ? 1 : 0;
    const double corners[4] = {0.0, w, w + h, w + h + w};
    double remaining = clockwise ? cw : ccw;

    while (remaining > EPSILON && cursor < limit) {
        int best = -1;
        double best_delta = perimeter + 1.0;
        for (int i = 0; i < 4; ++i) {
            double delta = clockwise ? (corners[i] - current_t) : (current_t - corners[i]);
            if (delta <= EPSILON) delta += perimeter;
            if (delta < remaining - EPSILON && delta < best_delta) {
                best_delta = delta;
                best = i;
            }
        }
        if (best < 0) break;
        point_at_corner_t(best, xmin, ymin, xmax, ymax, &out_x[cursor], &out_y[cursor]);
        ++cursor;
        current_t = corners[best];
        remaining -= best_delta;
    }
    return cursor;
}

__device__ double component_area2(
    const double* x,
    const double* y,
    int n,
    int chain_start,
    int chain_count,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    const int first_idx = chain_start;
    double first_x = x[first_idx];
    double first_y = y[first_idx];
    double prev_x = first_x;
    double prev_y = first_y;
    double area2 = 0.0;
    for (int k = 1; k < chain_count; ++k) {
        const int idx = (chain_start + k) % n;
        const double cur_x = x[idx];
        const double cur_y = y[idx];
        area2 += prev_x * cur_y - cur_x * prev_y;
        prev_x = cur_x;
        prev_y = cur_y;
    }
    append_boundary_path_area(
        prev_x, prev_y, first_x, first_y,
        xmin, ymin, xmax, ymax,
        &prev_x, &prev_y, &area2
    );
    area2 += prev_x * first_y - first_x * prev_y;
    return area2;
}

__device__ int analyze_components(
    const double* x,
    const double* y,
    int n,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int* component_vertex_counts
) {
    if (n < 4 || n > MAX_INPUT_VERTS) return 0;
    int boundary_edges[MAX_INPUT_VERTS];
    int first_boundary = -1;
    int boundary_count = 0;
    for (int i = 0; i < n; ++i) {
        const int next = (i + 1) % n;
        const int is_boundary = is_rect_boundary_edge(
            x[i], y[i], x[next], y[next], xmin, ymin, xmax, ymax
        );
        boundary_edges[i] = is_boundary;
        if (is_boundary) {
            if (first_boundary < 0) first_boundary = i;
            ++boundary_count;
        }
    }
    if (first_boundary < 0 || boundary_count < 2) return 0;

    const int start_vertex = (first_boundary + 1) % n;
    int chain_start = -1;
    int chain_count = 0;
    int comp_count = 0;
    for (int step = 0; step < n; ++step) {
        const int i = (start_vertex + step) % n;
        const int next = (i + 1) % n;
        if (chain_count == 0) {
            chain_start = i;
            chain_count = 1;
        }
        if (boundary_edges[i]) {
            if (chain_count >= 2) {
                const int end_idx = i;
                const int start_idx = chain_start;
                const int corner_count = boundary_corner_count(
                    x[end_idx], y[end_idx], x[start_idx], y[start_idx],
                    xmin, ymin, xmax, ymax
                );
                if (corner_count < 0) return 0;
                const int vertex_count = chain_count + corner_count + 1;
                const double area2 = component_area2(
                    x, y, n, chain_start, chain_count,
                    xmin, ymin, xmax, ymax
                );
                if (vertex_count >= 4 && fabs(area2) > EPSILON) {
                    if (comp_count >= MAX_COMPONENTS) return 0;
                    component_vertex_counts[comp_count] = vertex_count;
                    ++comp_count;
                }
            }
            chain_count = 0;
            chain_start = -1;
        } else {
            (void)next;
            ++chain_count;
        }
    }
    return comp_count;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_count(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    int* __restrict__ component_counts,
    int* __restrict__ component_vertex_counts,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    component_counts[row] = 0;
    for (int c = 0; c < MAX_COMPONENTS; ++c) {
        component_vertex_counts[row * MAX_COMPONENTS + c] = 0;
    }
    if (!clipped_valid[row]) return;

    const int geom_start = clipped_geom_offsets[row];
    const int geom_end = clipped_geom_offsets[row + 1];
    if (geom_end - geom_start != 1) return;
    const int start = clipped_ring_offsets[geom_start];
    const int end = clipped_ring_offsets[geom_start + 1];
    int n = end - start;
    if (n > 1) {
        const double dx = clipped_x[start] - clipped_x[end - 1];
        const double dy = clipped_y[start] - clipped_y[end - 1];
        if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) --n;
    }
    if (n < 4 || n > MAX_INPUT_VERTS) return;

    double x[MAX_INPUT_VERTS], y[MAX_INPUT_VERTS];
    for (int i = 0; i < n; ++i) {
        x[i] = clipped_x[start + i];
        y[i] = clipped_y[start + i];
    }

    int local_counts[MAX_COMPONENTS];
    for (int c = 0; c < MAX_COMPONENTS; ++c) local_counts[c] = 0;
    const int count = analyze_components(
        x, y, n,
        rect_xmin[row], rect_ymin[row], rect_xmax[row], rect_ymax[row],
        local_counts
    );
    component_counts[row] = count;
    for (int c = 0; c < count; ++c) {
        component_vertex_counts[row * MAX_COMPONENTS + c] = local_counts[c];
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_scatter(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    const int* __restrict__ component_offsets,
    const int* __restrict__ ring_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !clipped_valid[row]) return;

    const int geom_start = clipped_geom_offsets[row];
    const int geom_end = clipped_geom_offsets[row + 1];
    if (geom_end - geom_start != 1) return;
    const int start = clipped_ring_offsets[geom_start];
    const int end = clipped_ring_offsets[geom_start + 1];
    int n = end - start;
    if (n > 1) {
        const double dx = clipped_x[start] - clipped_x[end - 1];
        const double dy = clipped_y[start] - clipped_y[end - 1];
        if ((dx * dx + dy * dy) <= (EPSILON * EPSILON)) --n;
    }
    if (n < 4 || n > MAX_INPUT_VERTS) return;

    double x[MAX_INPUT_VERTS], y[MAX_INPUT_VERTS];
    for (int i = 0; i < n; ++i) {
        x[i] = clipped_x[start + i];
        y[i] = clipped_y[start + i];
    }

    int boundary_edges[MAX_INPUT_VERTS];
    int first_boundary = -1;
    int boundary_count = 0;
    const double xmin = rect_xmin[row];
    const double ymin = rect_ymin[row];
    const double xmax = rect_xmax[row];
    const double ymax = rect_ymax[row];
    for (int i = 0; i < n; ++i) {
        const int next = (i + 1) % n;
        const int is_boundary = is_rect_boundary_edge(
            x[i], y[i], x[next], y[next], xmin, ymin, xmax, ymax
        );
        boundary_edges[i] = is_boundary;
        if (is_boundary) {
            if (first_boundary < 0) first_boundary = i;
            ++boundary_count;
        }
    }
    if (first_boundary < 0 || boundary_count < 2) return;

    const int start_vertex = (first_boundary + 1) % n;
    int chain_start = -1;
    int chain_count = 0;
    int comp_count = 0;
    const int comp_base = component_offsets[row];
    for (int step = 0; step < n; ++step) {
        const int i = (start_vertex + step) % n;
        if (chain_count == 0) {
            chain_start = i;
            chain_count = 1;
        }
        if (boundary_edges[i]) {
            if (chain_count >= 2) {
                const int end_idx = i;
                const int start_idx = chain_start;
                const int corner_count = boundary_corner_count(
                    x[end_idx], y[end_idx], x[start_idx], y[start_idx],
                    xmin, ymin, xmax, ymax
                );
                const int vertex_count = chain_count + corner_count + 1;
                const double area2 = component_area2(
                    x, y, n, chain_start, chain_count,
                    xmin, ymin, xmax, ymax
                );
                if (
                    corner_count >= 0
                    && vertex_count >= 4
                    && fabs(area2) > EPSILON
                    && comp_count < MAX_COMPONENTS
                ) {
                    const int component = comp_base + comp_count;
                    int cursor = ring_offsets[component];
                    const int limit = ring_offsets[component + 1];
                    for (int k = 0; k < chain_count && cursor < limit; ++k) {
                        const int idx = (chain_start + k) % n;
                        out_x[cursor] = x[idx];
                        out_y[cursor] = y[idx];
                        ++cursor;
                    }
                    const int last_idx = (chain_start + chain_count - 1) % n;
                    cursor = append_boundary_path_points(
                        x[last_idx], y[last_idx], x[start_idx], y[start_idx],
                        xmin, ymin, xmax, ymax,
                        out_x, out_y, cursor, limit
                    );
                    if (cursor < limit) {
                        out_x[cursor] = x[start_idx];
                        out_y[cursor] = y[start_idx];
                    }
                    ++comp_count;
                }
            }
            chain_count = 0;
            chain_start = -1;
        } else {
            ++chain_count;
        }
    }
}
"""

request_nvrtc_warmup([
    ("polygon-rect-boundary-split", _BOUNDARY_SPLIT_KERNEL_SOURCE, _BOUNDARY_SPLIT_KERNEL_NAMES),
])


def _polygon_rect_intersection_kernels():
    return compile_kernel_group(
        "polygon-rect-intersection",
        _KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


def _polygon_rect_boundary_split_kernels():
    return compile_kernel_group(
        "polygon-rect-boundary-split",
        _BOUNDARY_SPLIT_KERNEL_SOURCE,
        _BOUNDARY_SPLIT_KERNEL_NAMES,
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


def polygon_rect_split_boundary_components(
    clipped: OwnedGeometryArray,
    rectangles: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Split repeated-boundary rectangle clip rings into MultiPolygon parts.

    ``polygon_rect_intersection`` emits one polygon ring per row. Concave mask
    clips can produce disconnected intersections; in that case the single ring
    contains repeated rectangle-boundary connector segments. This helper removes
    those connector edges and closes each component along the rectangle boundary
    on the GPU. Unsupported shapes return ``None`` so callers keep the generic
    GPU make-valid fallback.
    """
    if cp is None or clipped.row_count == 0 or clipped.row_count != rectangles.row_count:
        return None
    if set(clipped.families) != {GeometryFamily.POLYGON}:
        return None
    if set(rectangles.families) != {GeometryFamily.POLYGON}:
        return None

    clipped.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_rect boundary split selected GPU execution",
    )
    rectangles.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_rect boundary split selected GPU execution",
    )

    clipped_dev, _clipped_host = _extract_polygon_family_device_buffer(clipped)
    rect_dev, _rect_host = _extract_polygon_family_device_buffer(rectangles)
    row_count = clipped.row_count
    if clipped_dev is None or rect_dev is None:
        return None
    if not _device_is_dense_single_ring_polygons(clipped_dev, row_count):
        return None
    rect_bounds = _device_rectangle_bounds(rect_dev, row_count)
    if rect_bounds is None:
        return None
    d_xmin, d_ymin, d_xmax, d_ymax = rect_bounds

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _polygon_rect_boundary_split_kernels()
    state = clipped._ensure_device_state()
    d_valid = cp.asarray(state.validity).astype(cp.int32, copy=False)
    d_component_counts = runtime.allocate((row_count,), cp.int32, zero=True)
    d_component_vertex_counts_matrix = runtime.allocate(
        (row_count * _MAX_BOUNDARY_SPLIT_COMPONENTS,),
        cp.int32,
        zero=True,
    )

    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_boundary_split_count"],
        row_count,
    )
    runtime.launch(
        kernels["polygon_rect_boundary_split_count"],
        grid=count_grid,
        block=count_block,
        params=(
            (
                ptr(clipped_dev.x),
                ptr(clipped_dev.y),
                ptr(clipped_dev.ring_offsets),
                ptr(clipped_dev.geometry_offsets),
                ptr(d_xmin),
                ptr(d_ymin),
                ptr(d_xmax),
                ptr(d_ymax),
                ptr(d_valid),
                ptr(d_component_counts),
                ptr(d_component_vertex_counts_matrix),
                row_count,
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
            ),
        ),
    )

    # This is a narrow exact replacement for disconnected rect-clip rows. If a
    # row does not split into multiple components, leave it to generic repair.
    if bool(cp.any(d_component_counts < 2).item()):
        return None

    d_geometry_offsets = exclusive_sum(d_component_counts, synchronize=False)
    total_parts = count_scatter_total(runtime, d_component_counts, d_geometry_offsets)
    if total_parts <= 0:
        return None

    slots = cp.arange(
        row_count * _MAX_BOUNDARY_SPLIT_COMPONENTS,
        dtype=cp.int32,
    )
    slot_rows = slots // _MAX_BOUNDARY_SPLIT_COMPONENTS
    slot_components = slots - slot_rows * _MAX_BOUNDARY_SPLIT_COMPONENTS
    valid_slots = slot_components < d_component_counts[slot_rows]
    d_component_vertex_counts = d_component_vertex_counts_matrix[slots[valid_slots]]
    if int(d_component_vertex_counts.size) != total_parts:
        return None

    d_ring_offsets = cp.empty(total_parts + 1, dtype=cp.int32)
    d_ring_offsets[0] = 0
    cp.cumsum(
        d_component_vertex_counts.astype(cp.int32, copy=False),
        out=d_ring_offsets[1:],
    )
    total_vertices = int(d_ring_offsets[-1])
    if total_vertices <= 0:
        return None

    d_out_x = runtime.allocate((total_vertices,), cp.float64)
    d_out_y = runtime.allocate((total_vertices,), cp.float64)
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_boundary_split_scatter"],
        row_count,
    )
    runtime.launch(
        kernels["polygon_rect_boundary_split_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=(
            (
                ptr(clipped_dev.x),
                ptr(clipped_dev.y),
                ptr(clipped_dev.ring_offsets),
                ptr(clipped_dev.geometry_offsets),
                ptr(d_xmin),
                ptr(d_ymin),
                ptr(d_xmax),
                ptr(d_ymax),
                ptr(d_valid),
                ptr(d_geometry_offsets),
                ptr(d_ring_offsets),
                ptr(d_out_x),
                ptr(d_out_y),
                row_count,
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
                KERNEL_PARAM_I32,
            ),
        ),
    )

    multipolygon_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=cp.concatenate([
            d_geometry_offsets,
            cp.asarray([total_parts], dtype=cp.int32),
        ]),
        empty_mask=cp.zeros(row_count, dtype=cp.bool_),
        part_offsets=cp.arange(total_parts + 1, dtype=cp.int32),
        ring_offsets=d_ring_offsets,
        bounds=None,
    )
    return build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: multipolygon_buffer},
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            dtype=cp.int8,
        ),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
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


def _host_max_input_vertices(polygon_buf, row_count: int) -> int | None:
    if polygon_buf is None or polygon_buf.ring_offsets is None:
        return None
    if int(polygon_buf.ring_offsets.size) != row_count + 1:
        return None
    max_input_verts = 0
    for row in range(row_count):
        ring_span = int(polygon_buf.ring_offsets[row + 1]) - int(polygon_buf.ring_offsets[row])
        if ring_span > max_input_verts:
            max_input_verts = ring_span
    return max_input_verts


def _device_max_input_vertices(polygon_buf, row_count: int) -> int | None:
    if polygon_buf is None or polygon_buf.ring_offsets is None:
        return None
    if int(polygon_buf.ring_offsets.size) != row_count + 1:
        return None
    ring_spans = polygon_buf.ring_offsets[1:] - polygon_buf.ring_offsets[:-1]
    if int(ring_spans.size) != row_count:
        return None
    if int(ring_spans.size) == 0:
        return 0
    return int(cp.max(ring_spans).item())


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
    left_device = (
        None
        if left.device_state is None or GeometryFamily.POLYGON not in left.device_state.families
        else left.device_state.families[GeometryFamily.POLYGON]
    )
    right_device = (
        None
        if right.device_state is None or GeometryFamily.POLYGON not in right.device_state.families
        else right.device_state.families[GeometryFamily.POLYGON]
    )

    if left_host.host_materialized:
        if not _host_is_dense_single_ring_polygons(left_host, left.row_count):
            return False
        max_input_verts = _host_max_input_vertices(left_host, left.row_count)
    else:
        if not _device_is_dense_single_ring_polygons(left_device, left.row_count):
            return False
        max_input_verts = _device_max_input_vertices(left_device, left.row_count)

    if max_input_verts is None or max_input_verts > (_MAX_INPUT_VERTS + 1):
        return False

    if right_host.host_materialized:
        return _host_rectangle_bounds(right_host, right.row_count) is not None
    return _device_rectangle_bounds(right_device, right.row_count) is not None


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
    d_boundary_overlap = runtime.allocate((n,), cp.int32, zero=True)

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
            ptr(d_boundary_overlap),
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

    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid.astype(cp.bool_),
        ring_offsets=d_ring_offsets,
        runtime_selection=runtime_selection,
    )
    result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(cp.bool_)
    return result


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
        current_residency=combined_residency(left, right),
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
