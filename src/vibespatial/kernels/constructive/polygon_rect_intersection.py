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
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.device_functions.intersection_point import (
    INTERSECTION_POINT_DEVICE,
)
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    DeviceFixedGeometrySizeMetadata,
    OwnedGeometryArray,
    build_device_resident_owned,
    device_select_owned_capacity_partitions,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, combined_residency
from vibespatial.runtime._runtime import has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_pairwise_product_work_from_owned,
)
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
_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + INTERSECTION_POINT_DEVICE
    + r"""
#define EPSILON 1e-12

__device__ __forceinline__ int rect_clip_intersection(
    double ax,
    double ay,
    double bx,
    double by,
    int edge_type,
    double edge_val,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double* out_x,
    double* out_y
) {
    if (edge_type <= 1) {
        if (ax == edge_val && bx == edge_val) {
            return 1;
        }
        if (ax == edge_val) {
            *out_x = ax;
            *out_y = ay;
            return 1;
        }
        if (bx == edge_val) {
            *out_x = bx;
            *out_y = by;
            return 1;
        }
        return vs_proper_intersection_point_dd(
            ax, ay, bx, by,
            edge_val, ymin, edge_val, ymax,
            out_x, out_y
        );
    }
    if (ay == edge_val && by == edge_val) {
        return 1;
    }
    if (ay == edge_val) {
        *out_x = ax;
        *out_y = ay;
        return 1;
    }
    if (by == edge_val) {
        *out_x = bx;
        *out_y = by;
        return 1;
    }
    return vs_proper_intersection_point_dd(
        ax, ay, bx, by,
        xmin, edge_val, xmax, edge_val,
        out_x, out_y
    );
}

__device__ int clip_edge(
    const double* in_x,
    const double* in_y,
    const int* in_support,
    int in_count,
    double* out_x,
    double* out_y,
    int* out_support,
    int max_out,
    int edge_type,
    double edge_val,
    double xmin,
    double ymin,
    double xmax,
    double ymax
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
                if (!rect_clip_intersection(
                        prev_x, prev_y, cur_x, cur_y,
                        edge_type, edge_val,
                        xmin, ymin, xmax, ymax,
                        &ix, &iy)) continue;
                if (out_count < max_out) {
                    out_x[out_count] = ix;
                    out_y[out_count] = iy;
                    out_support[out_count] = -(edge_type + 1);
                    ++out_count;
                }
            }
            if (out_count < max_out) {
                out_x[out_count] = cur_x;
                out_y[out_count] = cur_y;
                out_support[out_count] = in_support[i];
                ++out_count;
            }
        } else if (prev_inside) {
            double ix, iy;
            if (!rect_clip_intersection(
                    prev_x, prev_y, cur_x, cur_y,
                    edge_type, edge_val,
                    xmin, ymin, xmax, ymax,
                    &ix, &iy)) continue;
            if (out_count < max_out) {
                out_x[out_count] = ix;
                out_y[out_count] = iy;
                out_support[out_count] = in_support[i];
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
    int* support,
    int count
) {
    if (count <= 1) return count;

    int out_count = 0;
    for (int i = 0; i < count; ++i) {
        const double cur_x = x[i];
        const double cur_y = y[i];
        if (out_count > 0) {
            if (cur_x == x[out_count - 1] && cur_y == y[out_count - 1]) {
                continue;
            }
        }
        x[out_count] = cur_x;
        y[out_count] = cur_y;
        support[out_count] = support[i];
        ++out_count;
    }

    if (out_count > 1) {
        if (x[out_count - 1] == x[0] && y[out_count - 1] == y[0]) {
            support[0] = support[out_count - 1];
            --out_count;
        }
    }
    return out_count;
}

__device__ int remove_colinear_vertices(
    const double* in_x,
    const double* in_y,
    const int* in_support,
    int count,
    double* out_x,
    double* out_y,
    int* out_support,
    int max_out
) {
    if (count <= 2) return count;

    int out_count = 0;
    for (int i = 0; i < count; ++i) {
        const int prev = (i + count - 1) % count;
        const int next = (i + 1) % count;
        if (vs_orient2d(
                in_x[prev], in_y[prev],
                in_x[i], in_y[i],
                in_x[next], in_y[next]) == 0) {
            continue;
        }
        if (out_count < max_out) {
            out_x[out_count] = in_x[i];
            out_y[out_count] = in_y[i];
            out_support[out_count] = in_support[i];
            ++out_count;
        }
    }
    return out_count;
}

__device__ int finalize_clipped_vertices(
    double* src_x,
    double* src_y,
    int* src_support,
    double* tmp_x,
    double* tmp_y,
    int* tmp_support,
    int count,
    int max_out
) {
    count = compact_vertices(src_x, src_y, src_support, count);
    count = remove_colinear_vertices(
        src_x, src_y, src_support, count,
        tmp_x, tmp_y, tmp_support, max_out
    );
    for (int i = 0; i < count && i < max_out; ++i) {
        src_x[i] = tmp_x[i];
        src_y[i] = tmp_y[i];
        src_support[i] = tmp_support[i];
    }
    count = compact_vertices(src_x, src_y, src_support, count);
    count = remove_colinear_vertices(
        src_x, src_y, src_support, count,
        tmp_x, tmp_y, tmp_support, max_out
    );
    for (int i = 0; i < count && i < max_out; ++i) {
        src_x[i] = tmp_x[i];
        src_y[i] = tmp_y[i];
        src_support[i] = tmp_support[i];
    }
    return compact_vertices(src_x, src_y, src_support, count);
}

__device__ void canonicalize_rect_boundary_vertices(
    double* x,
    double* y,
    const int* support,
    int count,
    const double* source_x,
    const double* source_y,
    int source_count,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    for (int i = 0; i < count; ++i) {
        int edge_type = -1;
        double edge_val = 0.0;
        if (x[i] == xmin) {
            edge_type = 0;
            edge_val = xmin;
        } else if (x[i] == xmax) {
            edge_type = 1;
            edge_val = xmax;
        } else if (y[i] == ymin) {
            edge_type = 2;
            edge_val = ymin;
        } else if (y[i] == ymax) {
            edge_type = 3;
            edge_val = ymax;
        }
        if (edge_type < 0) continue;

        int source_edge = support[i];
        if (source_edge < 0) {
            source_edge = support[(i + 1) % count];
        }
        if (source_edge < 0 || source_edge >= source_count) continue;
        const int source_prev = source_edge == 0 ? source_count - 1 : source_edge - 1;
        double canonical_x = x[i];
        double canonical_y = y[i];
        if (rect_clip_intersection(
                source_x[source_prev], source_y[source_prev],
                source_x[source_edge], source_y[source_edge],
                edge_type, edge_val,
                xmin, ymin, xmax, ymax,
                &canonical_x, &canonical_y)) {
            x[i] = canonical_x;
            y[i] = canonical_y;
        }
    }
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

__device__ int polygon_has_area_dimension(
    const double* x,
    const double* y,
    int count
) {
    if (count < 3) return 0;
    for (int i = 1; i + 1 < count; ++i) {
        if (vs_orient2d(x[0], y[0], x[i], y[i], x[i + 1], y[i + 1]) != 0) {
            return 1;
        }
    }
    return 0;
}

__device__ int point_strictly_inside_rect(
    double x,
    double y,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    return (
        x > xmin + EPSILON
        && x < xmax - EPSILON
        && y > ymin + EPSILON
        && y < ymax - EPSILON
    ) ? 1 : 0;
}

__device__ int point_on_segment_eps(
    double px,
    double py,
    double ax,
    double ay,
    double bx,
    double by
) {
    const double cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    const double scale = fabs(bx - ax) + fabs(by - ay) + 1.0;
    if (fabs(cross) > EPSILON * scale) return 0;
    return (
        px >= fmin(ax, bx) - EPSILON
        && px <= fmax(ax, bx) + EPSILON
        && py >= fmin(ay, by) - EPSILON
        && py <= fmax(ay, by) + EPSILON
    ) ? 1 : 0;
}

__device__ int orientation_sign_eps(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy
) {
    const double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const double scale = (
        fabs(bx - ax) + fabs(by - ay) + fabs(cx - ax) + fabs(cy - ay) + 1.0
    );
    if (fabs(cross) <= EPSILON * scale) return 0;
    return cross > 0.0 ? 1 : -1;
}

__device__ int segments_intersect_or_touch(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy
) {
    const int o1 = orientation_sign_eps(ax, ay, bx, by, cx, cy);
    const int o2 = orientation_sign_eps(ax, ay, bx, by, dx, dy);
    const int o3 = orientation_sign_eps(cx, cy, dx, dy, ax, ay);
    const int o4 = orientation_sign_eps(cx, cy, dx, dy, bx, by);
    if (o1 == 0 && point_on_segment_eps(cx, cy, ax, ay, bx, by)) return 1;
    if (o2 == 0 && point_on_segment_eps(dx, dy, ax, ay, bx, by)) return 1;
    if (o3 == 0 && point_on_segment_eps(ax, ay, cx, cy, dx, dy)) return 1;
    if (o4 == 0 && point_on_segment_eps(bx, by, cx, cy, dx, dy)) return 1;
    return (o1 != o2 && o3 != o4) ? 1 : 0;
}

__device__ int segment_has_strict_rect_interior(
    double x0,
    double y0,
    double x1,
    double y1,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    if (
        point_strictly_inside_rect(x0, y0, xmin, ymin, xmax, ymax)
        || point_strictly_inside_rect(x1, y1, xmin, ymin, xmax, ymax)
    ) {
        return 1;
    }

    double t0 = 0.0;
    double t1 = 1.0;
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double p[4] = {-dx, dx, -dy, dy};
    const double q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};
    for (int edge = 0; edge < 4; ++edge) {
        if (fabs(p[edge]) <= EPSILON) {
            if (q[edge] < -EPSILON) return 0;
            continue;
        }
        const double r = q[edge] / p[edge];
        if (p[edge] < 0.0) {
            if (r > t1) return 0;
            if (r > t0) t0 = r;
        } else {
            if (r < t0) return 0;
            if (r < t1) t1 = r;
        }
    }
    if ((t1 - t0) <= EPSILON) return 0;
    const double tm = 0.5 * (t0 + t1);
    return point_strictly_inside_rect(
        x0 + tm * dx,
        y0 + tm * dy,
        xmin,
        ymin,
        xmax,
        ymax
    );
}

__device__ int rect_boundary_overlap_class(
    double x0,
    double y0,
    double x1,
    double y1,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    double signed_area2
) {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    int has_overlap = 0;
    double mid_x = 0.0;
    double mid_y = 0.0;

    if (fabs(dx) <= EPSILON) {
        if (fabs(x0 - xmin) <= EPSILON || fabs(x0 - xmax) <= EPSILON) {
            const double overlap_min = fmax(fmin(y0, y1), ymin);
            const double overlap_max = fmin(fmax(y0, y1), ymax);
            if ((overlap_max - overlap_min) > EPSILON) {
                has_overlap = 1;
                mid_x = x0;
                mid_y = 0.5 * (overlap_min + overlap_max);
            }
        }
    } else if (fabs(dy) <= EPSILON) {
        if (fabs(y0 - ymin) <= EPSILON || fabs(y0 - ymax) <= EPSILON) {
            const double overlap_min = fmax(fmin(x0, x1), xmin);
            const double overlap_max = fmin(fmax(x0, x1), xmax);
            if ((overlap_max - overlap_min) > EPSILON) {
                has_overlap = 1;
                mid_x = 0.5 * (overlap_min + overlap_max);
                mid_y = y0;
            }
        }
    }
    if (!has_overlap) return 0;

    const double scale = fmax(fmax(xmax - xmin, ymax - ymin), 1.0);
    const double offset = 1e-9 * scale;
    double nx;
    double ny;
    if (signed_area2 >= 0.0) {
        nx = -dy;
        ny = dx;
    } else {
        nx = dy;
        ny = -dx;
    }
    const double norm = sqrt(nx * nx + ny * ny);
    if (norm <= EPSILON) return 1;
    return point_strictly_inside_rect(
        mid_x + offset * nx / norm,
        mid_y + offset * ny / norm,
        xmin,
        ymin,
        xmax,
        ymax
    ) ? 2 : 1;
}

__device__ int segment_intersects_rect_boundary(
    double x0,
    double y0,
    double x1,
    double y1,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    if (segments_intersect_or_touch(x0, y0, x1, y1, xmin, ymin, xmax, ymin)) return 1;
    if (segments_intersect_or_touch(x0, y0, x1, y1, xmax, ymin, xmax, ymax)) return 1;
    if (segments_intersect_or_touch(x0, y0, x1, y1, xmax, ymax, xmin, ymax)) return 1;
    if (segments_intersect_or_touch(x0, y0, x1, y1, xmin, ymax, xmin, ymin)) return 1;
    return 0;
}

__device__ int polygon_rect_lower_dim_candidate(
    const double* x,
    const double* y,
    int count,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    const double signed_area2 = polygon_area2(x, y, count);
    for (int i = 0; i < count; ++i) {
        const int next = (i + 1) % count;
        const double x0 = x[i];
        const double y0 = y[i];
        const double x1 = x[next];
        const double y1 = y[next];
        if (fabs(x0 - x1) <= EPSILON && fabs(y0 - y1) <= EPSILON) {
            continue;
        }
        const int overlap_class = rect_boundary_overlap_class(
            x0,
            y0,
            x1,
            y1,
            xmin,
            ymin,
            xmax,
            ymax,
            signed_area2
        );
        if (overlap_class == 1) return 1;
        if (overlap_class == 2) continue;

        if (
            !segment_has_strict_rect_interior(x0, y0, x1, y1, xmin, ymin, xmax, ymax)
            && segment_intersects_rect_boundary(x0, y0, x1, y1, xmin, ymin, xmax, ymax)
        ) {
            return 1;
        }
    }
    return 0;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_intersection_count(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_family_rows,
    const long long* __restrict__ selected_rows,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    int* __restrict__ out_counts,
    int* __restrict__ out_valid,
    int* __restrict__ out_boundary_overlap,
    int* __restrict__ out_exact_polygon_only,
    int* __restrict__ out_lower_dimensional_remnant,
    const int source_row_count,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    out_lower_dimensional_remnant[row] = 0;
    const long long source_row_ll = selected_rows ? selected_rows[row] : (long long)row;
    if (source_row_ll < 0 || source_row_ll >= (long long)source_row_count) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
        return;
    }
    const int source_row = (int)source_row_ll;

    if (!left_valid[source_row] || !right_valid[row]) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
        return;
    }

    const int family_row = left_family_rows[source_row];
    if (family_row < 0) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
        return;
    }

    const int ring_start_idx = left_geom_offsets[family_row];
    const int ring_end_idx = left_geom_offsets[family_row + 1];
    if (ring_end_idx - ring_start_idx != 1) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
        return;
    }

    const int start = left_ring_offsets[ring_start_idx];
    const int end = left_ring_offsets[ring_start_idx + 1];
    int n = end - start;
    if (n > 1) {
        if (left_x[start] == left_x[end - 1] && left_y[start] == left_y[end - 1]) {
            --n;
        }
    }

    if (n < 3 || n > 256) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
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
        out_exact_polygon_only[row] = 0;
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
    int buf_a_support[256], buf_b_support[256];
    for (int i = 0; i < n; ++i) {
        buf_a_x[i] = left_x[start + i];
        buf_a_y[i] = left_y[start + i];
        buf_a_support[i] = i;
    }
    const int lower_dim_candidate = polygon_rect_lower_dim_candidate(
        buf_a_x,
        buf_a_y,
        n,
        xmin,
        ymin,
        xmax,
        ymax
    );
    out_lower_dimensional_remnant[row] = lower_dim_candidate;

    double edges[4] = {xmin, xmax, ymin, ymax};
    int count = n;
    double* src_x = buf_a_x;
    double* src_y = buf_a_y;
    double* dst_x = buf_b_x;
    double* dst_y = buf_b_y;
    int* src_support = buf_a_support;
    int* dst_support = buf_b_support;

    for (int edge = 0; edge < 4; ++edge) {
        count = clip_edge(
            src_x, src_y, src_support, count,
            dst_x, dst_y, dst_support, 256,
            edge, edges[edge], xmin, ymin, xmax, ymax
        );
        count = compact_vertices(dst_x, dst_y, dst_support, count);
        count = remove_colinear_vertices(
            dst_x, dst_y, dst_support, count,
            src_x, src_y, src_support, 256
        );
        if (count == 0) break;
    }
    count = finalize_clipped_vertices(
        src_x, src_y, src_support,
        dst_x, dst_y, dst_support,
        count, 256
    );
    canonicalize_rect_boundary_vertices(
        src_x, src_y, src_support, count,
        left_x + start, left_y + start, n,
        xmin, ymin, xmax, ymax
    );

    if (count < 3 || !polygon_has_area_dimension(src_x, src_y, count)) {
        out_counts[row] = 0;
        out_valid[row] = 0;
        out_boundary_overlap[row] = 0;
        out_exact_polygon_only[row] = 0;
        return;
    }

    if (has_repeated_rect_boundary_segments(src_x, src_y, count, xmin, ymin, xmax, ymax)) {
        boundary_overlap = 1;
    }
    out_boundary_overlap[row] = boundary_overlap;
    out_exact_polygon_only[row] = lower_dim_candidate ? 0 : 1;
    out_counts[row] = count + 1;
    out_valid[row] = 1;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_intersection_scatter(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_family_rows,
    const long long* __restrict__ selected_rows,
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
    const int source_row_count,
    const int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !out_valid[row]) return;
    const long long source_row_ll = selected_rows ? selected_rows[row] : (long long)row;
    if (source_row_ll < 0 || source_row_ll >= (long long)source_row_count) return;
    const int source_row = (int)source_row_ll;
    if (!left_valid[source_row] || !right_valid[row]) return;

    const int family_row = left_family_rows[source_row];
    if (family_row < 0) return;

    const int ring_start_idx = left_geom_offsets[family_row];
    const int ring_end_idx = left_geom_offsets[family_row + 1];
    if (ring_end_idx - ring_start_idx != 1) return;

    const int out_start = out_offsets[row];
    const int expected = out_counts[row];
    if (expected <= 0) return;

    const int start = left_ring_offsets[ring_start_idx];
    const int end = left_ring_offsets[ring_start_idx + 1];
    int n = end - start;
    if (n > 1) {
        if (left_x[start] == left_x[end - 1] && left_y[start] == left_y[end - 1]) {
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
    int buf_a_support[256], buf_b_support[256];
    for (int i = 0; i < n; ++i) {
        buf_a_x[i] = left_x[start + i];
        buf_a_y[i] = left_y[start + i];
        buf_a_support[i] = i;
    }

    double edges[4] = {xmin, xmax, ymin, ymax};
    int count = n;
    double* src_x = buf_a_x;
    double* src_y = buf_a_y;
    double* dst_x = buf_b_x;
    double* dst_y = buf_b_y;
    int* src_support = buf_a_support;
    int* dst_support = buf_b_support;

    for (int edge = 0; edge < 4; ++edge) {
        count = clip_edge(
            src_x, src_y, src_support, count,
            dst_x, dst_y, dst_support, 256,
            edge, edges[edge], xmin, ymin, xmax, ymax
        );
        count = compact_vertices(dst_x, dst_y, dst_support, count);
        count = remove_colinear_vertices(
            dst_x, dst_y, dst_support, count,
            src_x, src_y, src_support, 256
        );
        if (count == 0) return;
    }
    count = finalize_clipped_vertices(
        src_x, src_y, src_support,
        dst_x, dst_y, dst_support,
        count, 256
    );
    canonicalize_rect_boundary_vertices(
        src_x, src_y, src_support, count,
        left_x + start, left_y + start, n,
        xmin, ymin, xmax, ymax
    );

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
)

request_nvrtc_warmup(
    [
        ("polygon-rect-intersection", _KERNEL_SOURCE, _KERNEL_NAMES),
    ]
)

request_warmup(["exclusive_scan_i32"])

_BOUNDARY_SPLIT_KERNEL_NAMES = (
    "polygon_rect_boundary_split_count",
    "polygon_rect_boundary_split_count_selected",
    "polygon_rect_boundary_split_scatter",
    "polygon_rect_boundary_split_scatter_selected",
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

__device__ void count_boundary_split_row(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    int* __restrict__ component_counts,
    int* __restrict__ component_vertex_counts,
    const int row,
    const int output_row
) {
    component_counts[output_row] = 0;
    for (int c = 0; c < MAX_COMPONENTS; ++c) {
        component_vertex_counts[output_row * MAX_COMPONENTS + c] = 0;
    }
    if (!clipped_valid[row]) return;

    const int family_row = clipped_family_rows[row];
    if (family_row < 0) return;

    const int geom_start = clipped_geom_offsets[family_row];
    const int geom_end = clipped_geom_offsets[family_row + 1];
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
    component_counts[output_row] = count;
    for (int c = 0; c < count; ++c) {
        component_vertex_counts[output_row * MAX_COMPONENTS + c] = local_counts[c];
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_count(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
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
    count_boundary_split_row(
        clipped_x, clipped_y, clipped_ring_offsets, clipped_geom_offsets,
        clipped_family_rows, rect_xmin, rect_ymin, rect_xmax, rect_ymax,
        clipped_valid, component_counts, component_vertex_counts, row, row
    );
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_count_selected(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    const long long* __restrict__ selected_rows,
    int* __restrict__ component_counts,
    int* __restrict__ component_vertex_counts,
    const int selected_count
) {
    const int output_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_row >= selected_count) return;
    const long long source_row_ll = selected_rows[output_row];
    if (source_row_ll < 0LL || source_row_ll > 2147483647LL) return;
    count_boundary_split_row(
        clipped_x, clipped_y, clipped_ring_offsets, clipped_geom_offsets,
        clipped_family_rows, rect_xmin, rect_ymin, rect_xmax, rect_ymax,
        clipped_valid, component_counts, component_vertex_counts,
        (int)source_row_ll, output_row
    );
}

__device__ void scatter_boundary_split_row(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    const int* __restrict__ component_offsets,
    const int* __restrict__ ring_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int row,
    const int output_row
) {
    if (!clipped_valid[row]) return;

    const int family_row = clipped_family_rows[row];
    if (family_row < 0) return;

    const int geom_start = clipped_geom_offsets[family_row];
    const int geom_end = clipped_geom_offsets[family_row + 1];
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
    const int comp_base = component_offsets[output_row];
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

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_scatter(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
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
    if (row >= row_count) return;
    scatter_boundary_split_row(
        clipped_x, clipped_y, clipped_ring_offsets, clipped_geom_offsets,
        clipped_family_rows, rect_xmin, rect_ymin, rect_xmax, rect_ymax,
        clipped_valid, component_offsets, ring_offsets, out_x, out_y,
        row, row
    );
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_split_scatter_selected(
    const double* __restrict__ clipped_x,
    const double* __restrict__ clipped_y,
    const int* __restrict__ clipped_ring_offsets,
    const int* __restrict__ clipped_geom_offsets,
    const int* __restrict__ clipped_family_rows,
    const double* __restrict__ rect_xmin,
    const double* __restrict__ rect_ymin,
    const double* __restrict__ rect_xmax,
    const double* __restrict__ rect_ymax,
    const int* __restrict__ clipped_valid,
    const long long* __restrict__ selected_rows,
    const int* __restrict__ component_offsets,
    const int* __restrict__ ring_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int selected_count
) {
    const int output_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_row >= selected_count) return;
    const long long source_row_ll = selected_rows[output_row];
    if (source_row_ll < 0LL || source_row_ll > 2147483647LL) return;
    scatter_boundary_split_row(
        clipped_x, clipped_y, clipped_ring_offsets, clipped_geom_offsets,
        clipped_family_rows, rect_xmin, rect_ymin, rect_xmax, rect_ymax,
        clipped_valid, component_offsets, ring_offsets, out_x, out_y,
        (int)source_row_ll, output_row
    );
}
"""

request_nvrtc_warmup(
    [
        (
            "polygon-rect-boundary-split",
            _BOUNDARY_SPLIT_KERNEL_SOURCE,
            _BOUNDARY_SPLIT_KERNEL_NAMES,
        ),
    ]
)

_BOUNDARY_POINT_CONTACT_KERNEL_NAMES = (
    "polygon_rect_boundary_point_contact_count",
    "polygon_rect_boundary_point_contact_scatter",
)
_BOUNDARY_CONTACT_MAX_SEGMENTS = 32
_BOUNDARY_POINT_CONTACT_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + INTERSECTION_POINT_DEVICE
    + r"""
__device__ int point_on_segment(
    const double px,
    const double py,
    const double ax,
    const double ay,
    const double bx,
    const double by
) {
    if (vs_orient2d(ax, ay, bx, by, px, py) != 0) return 0;
    return (
        px >= fmin(ax, bx) && px <= fmax(ax, bx) &&
        py >= fmin(ay, by) && py <= fmax(ay, by)
    );
}

__device__ int overlap_interval_point_or_line(
    const double a0,
    const double a1,
    const double b0,
    const double b1,
    double* out_value,
    int* is_line
) {
    const double amin = fmin(a0, a1);
    const double amax = fmax(a0, a1);
    const double bmin = fmin(b0, b1);
    const double bmax = fmax(b0, b1);
    const double lo = fmax(amin, bmin);
    const double hi = fmin(amax, bmax);
    if (hi < lo) return 0;
    if (hi > lo) {
        *is_line = 1;
        return 1;
    }
    *out_value = lo;
    *is_line = 0;
    return 1;
}

__device__ int segment_contact_point(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double cx,
    const double cy,
    const double dx,
    const double dy,
    double* out_x,
    double* out_y,
    int* line_overlap
) {
    const int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
    const int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
    const int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
    const int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);

    if (o1 == 0 && o2 == 0 && o3 == 0 && o4 == 0) {
        double interval_value = 0.0;
        int is_line = 0;
        const double rx = bx - ax;
        const double ry = by - ay;
        if (fabs(rx) >= fabs(ry)) {
            if (!overlap_interval_point_or_line(ax, bx, cx, dx, &interval_value, &is_line)) {
                return 0;
            }
            if (is_line) {
                *line_overlap = 1;
                return 1;
            }
        } else {
            if (!overlap_interval_point_or_line(ay, by, cy, dy, &interval_value, &is_line)) {
                return 0;
            }
            if (is_line) {
                *line_overlap = 1;
                return 1;
            }
        }
        if (point_on_segment(ax, ay, cx, cy, dx, dy)) {
            *out_x = ax; *out_y = ay; return 1;
        }
        if (point_on_segment(bx, by, cx, cy, dx, dy)) {
            *out_x = bx; *out_y = by; return 1;
        }
        if (point_on_segment(cx, cy, ax, ay, bx, by)) {
            *out_x = cx; *out_y = cy; return 1;
        }
        if (point_on_segment(dx, dy, ax, ay, bx, by)) {
            *out_x = dx; *out_y = dy; return 1;
        }
        return 0;
    }

    if (o1 == 0 && point_on_segment(cx, cy, ax, ay, bx, by)) {
        *out_x = cx; *out_y = cy; return 1;
    }
    if (o2 == 0 && point_on_segment(dx, dy, ax, ay, bx, by)) {
        *out_x = dx; *out_y = dy; return 1;
    }
    if (o3 == 0 && point_on_segment(ax, ay, cx, cy, dx, dy)) {
        *out_x = ax; *out_y = ay; return 1;
    }
    if (o4 == 0 && point_on_segment(bx, by, cx, cy, dx, dy)) {
        *out_x = bx; *out_y = by; return 1;
    }
    if (
        ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
        ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))
    ) {
        return vs_proper_intersection_point_dd(
            ax, ay, bx, by, cx, cy, dx, dy, out_x, out_y
        );
    }
    return 0;
}

__device__ int collinear_rect_overlap_segment(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double cx,
    const double cy,
    const double dx,
    const double dy,
    double* out_x0,
    double* out_y0,
    double* out_x1,
    double* out_y1
) {
    if (ax != bx) {
        const double lo = fmax(fmin(ax, bx), fmin(cx, dx));
        const double hi = fmin(fmax(ax, bx), fmax(cx, dx));
        if (!(lo < hi)) return 0;
        *out_x0 = lo; *out_y0 = ay;
        *out_x1 = hi; *out_y1 = ay;
        return 1;
    }
    const double lo = fmax(fmin(ay, by), fmin(cy, dy));
    const double hi = fmin(fmax(ay, by), fmax(cy, dy));
    if (!(lo < hi)) return 0;
    *out_x0 = ax; *out_y0 = lo;
    *out_x1 = ax; *out_y1 = hi;
    return 1;
}

__device__ int add_unique_contact_point(
    double* points_x,
    double* points_y,
    int* point_count,
    const double x,
    const double y
) {
    for (int point = 0; point < *point_count; ++point) {
        if (points_x[point] == x && points_y[point] == y) return 1;
    }
    if (*point_count >= 64) return 0;
    points_x[*point_count] = x;
    points_y[*point_count] = y;
    *point_count += 1;
    return 1;
}

__device__ int collect_rect_polygon_boundary_contacts(
    const double* __restrict__ mask_x,
    const double* __restrict__ mask_y,
    const int segment_count,
    const double xmin,
    const double ymin,
    const double xmax,
    const double ymax,
    double* points_x,
    double* points_y,
    double* lines_x0,
    double* lines_y0,
    double* lines_x1,
    double* lines_y1,
    int* line_count
) {
    if (!(isfinite(xmin) && isfinite(ymin) && isfinite(xmax) && isfinite(ymax))) return 0;
    if (!(xmin < xmax && ymin < ymax)) return 0;
    const double edge_x0[4] = {xmin, xmax, xmax, xmin};
    const double edge_y0[4] = {ymin, ymin, ymax, ymax};
    const double edge_x1[4] = {xmax, xmax, xmin, xmin};
    const double edge_y1[4] = {ymin, ymax, ymax, ymin};
    int point_count = 0;
    for (int edge = 0; edge < 4; ++edge) {
        const double ax = edge_x0[edge];
        const double ay = edge_y0[edge];
        const double bx = edge_x1[edge];
        const double by = edge_y1[edge];
        const int edge_line_start = *line_count;
        for (int segment = 0; segment < segment_count; ++segment) {
            const double cx = mask_x[segment];
            const double cy = mask_y[segment];
            const double dx = mask_x[segment + 1];
            const double dy = mask_y[segment + 1];
            double px = 0.0;
            double py = 0.0;
            int line_overlap = 0;
            if (!segment_contact_point(
                    ax, ay, bx, by, cx, cy, dx, dy,
                    &px, &py, &line_overlap
                )) {
                continue;
            }
            if (!line_overlap) {
                add_unique_contact_point(
                    points_x, points_y, &point_count, px, py
                );
                continue;
            }
            double overlap_x0;
            double overlap_y0;
            double overlap_x1;
            double overlap_y1;
            if (!collinear_rect_overlap_segment(
                    ax, ay, bx, by, cx, cy, dx, dy,
                    &overlap_x0, &overlap_y0, &overlap_x1, &overlap_y1
                )) {
                continue;
            }
            int insert = *line_count;
            const double overlap_lo = ax != bx ? overlap_x0 : overlap_y0;
            while (insert > edge_line_start) {
                const int prior = insert - 1;
                const double prior_lo = ax != bx ? lines_x0[prior] : lines_y0[prior];
                if (prior_lo <= overlap_lo) break;
                lines_x0[insert] = lines_x0[prior];
                lines_y0[insert] = lines_y0[prior];
                lines_x1[insert] = lines_x1[prior];
                lines_y1[insert] = lines_y1[prior];
                insert = prior;
            }
            lines_x0[insert] = overlap_x0;
            lines_y0[insert] = overlap_y0;
            lines_x1[insert] = overlap_x1;
            lines_y1[insert] = overlap_y1;
            *line_count += 1;
        }
        int write = edge_line_start;
        for (int read = edge_line_start; read < *line_count; ++read) {
            if (write == edge_line_start) {
                lines_x0[write] = lines_x0[read];
                lines_y0[write] = lines_y0[read];
                lines_x1[write] = lines_x1[read];
                lines_y1[write] = lines_y1[read];
                write += 1;
                continue;
            }
            const int prior = write - 1;
            const double prior_hi = ax != bx ? lines_x1[prior] : lines_y1[prior];
            const double current_lo = ax != bx ? lines_x0[read] : lines_y0[read];
            if (current_lo <= prior_hi) {
                if (ax != bx) {
                    lines_x1[prior] = fmax(lines_x1[prior], lines_x1[read]);
                } else {
                    lines_y1[prior] = fmax(lines_y1[prior], lines_y1[read]);
                }
            } else {
                lines_x0[write] = lines_x0[read];
                lines_y0[write] = lines_y0[read];
                lines_x1[write] = lines_x1[read];
                lines_y1[write] = lines_y1[read];
                write += 1;
            }
        }
        *line_count = write;
    }
    int write = 0;
    for (int point = 0; point < point_count; ++point) {
        int covered = 0;
        for (int line = 0; line < *line_count; ++line) {
            if (point_on_segment(
                    points_x[point], points_y[point],
                    lines_x0[line], lines_y0[line],
                    lines_x1[line], lines_y1[line]
                )) {
                covered = 1;
                break;
            }
        }
        if (!covered) {
            points_x[write] = points_x[point];
            points_y[write] = points_y[point];
            write += 1;
        }
    }
    return write;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_point_contact_count(
    const double* __restrict__ mask_x,
    const double* __restrict__ mask_y,
    const double* __restrict__ xmin,
    const double* __restrict__ ymin,
    const double* __restrict__ xmax,
    const double* __restrict__ ymax,
    int* __restrict__ out_counts,
    int* __restrict__ out_line_counts,
    const long long n,
    const int segment_count
) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double px[64];
    double py[64];
    double lx0[32];
    double ly0[32];
    double lx1[32];
    double ly1[32];
    int line_count = 0;
    out_counts[row] = collect_rect_polygon_boundary_contacts(
        mask_x, mask_y, segment_count,
        xmin[row], ymin[row], xmax[row], ymax[row],
        px, py, lx0, ly0, lx1, ly1, &line_count
    );
    out_line_counts[row] = line_count;
}

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_rect_boundary_point_contact_scatter(
    const double* __restrict__ mask_x,
    const double* __restrict__ mask_y,
    const double* __restrict__ xmin,
    const double* __restrict__ ymin,
    const double* __restrict__ xmax,
    const double* __restrict__ ymax,
    const int* __restrict__ counts,
    const int* __restrict__ line_counts,
    const int* __restrict__ multipoint_offsets,
    const int* __restrict__ multiline_offsets,
    double* __restrict__ point_x,
    double* __restrict__ point_y,
    double* __restrict__ multipoint_x,
    double* __restrict__ multipoint_y,
    double* __restrict__ line_x,
    double* __restrict__ line_y,
    double* __restrict__ multiline_x,
    double* __restrict__ multiline_y,
    const long long n,
    const int segment_count
) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (
        row >= n || (counts[row] <= 0 && line_counts[row] <= 0)
    ) return;
    double px[64];
    double py[64];
    double lx0[32];
    double ly0[32];
    double lx1[32];
    double ly1[32];
    int line_count = 0;
    const int count = collect_rect_polygon_boundary_contacts(
        mask_x, mask_y, segment_count,
        xmin[row], ymin[row], xmax[row], ymax[row],
        px, py, lx0, ly0, lx1, ly1, &line_count
    );
    if (count != counts[row] || line_count != line_counts[row]) return;
    if (count == 1) {
        point_x[row] = px[0];
        point_y[row] = py[0];
    } else if (count > 1) {
        const int start = multipoint_offsets[row];
        for (int point = 0; point < count; ++point) {
            multipoint_x[start + point] = px[point];
            multipoint_y[start + point] = py[point];
        }
    }
    if (line_count == 1) {
        const int start = ((int)row) * 2;
        line_x[start] = lx0[0];
        line_y[start] = ly0[0];
        line_x[start + 1] = lx1[0];
        line_y[start + 1] = ly1[0];
    } else if (line_count > 1) {
        const int part_start = multiline_offsets[row];
        for (int line = 0; line < line_count; ++line) {
            const int coord = (part_start + line) * 2;
            multiline_x[coord] = lx0[line];
            multiline_y[coord] = ly0[line];
            multiline_x[coord + 1] = lx1[line];
            multiline_y[coord + 1] = ly1[line];
        }
    }
}
"""
)

request_nvrtc_warmup(
    [
        (
            "polygon-rect-boundary-point-contact",
            _BOUNDARY_POINT_CONTACT_KERNEL_SOURCE,
            _BOUNDARY_POINT_CONTACT_KERNEL_NAMES,
        ),
    ]
)

_POLYGON_SHAPE_MASK_BOUNDS_KERNEL_NAMES = ("polygon_shape_mask_bounds",)
_POLYGON_SHAPE_MASK_BOUNDS_KERNEL_SOURCE = r"""
#define EPSILON 1e-12

extern "C" __global__ void __launch_bounds__(256, 4)
polygon_shape_mask_bounds(
    const signed char* __restrict__ tags,
    const unsigned char* __restrict__ validity,
    const int* __restrict__ family_rows,
    const int* __restrict__ geom_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ x,
    const double* __restrict__ y,
    unsigned char* __restrict__ out_simple,
    unsigned char* __restrict__ out_rect,
    double* __restrict__ out_bounds,
    const int row_count,
    const int polygon_rows,
    const int polygon_tag,
    const int max_input_vertices
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const double nan_value = 0.0 / 0.0;
    out_simple[row] = 0;
    out_rect[row] = 0;
    out_bounds[row * 4 + 0] = nan_value;
    out_bounds[row * 4 + 1] = nan_value;
    out_bounds[row * 4 + 2] = nan_value;
    out_bounds[row * 4 + 3] = nan_value;

    const int family_row = family_rows[row];
    if (
        !validity[row]
        || (int)tags[row] != polygon_tag
        || family_row < 0
        || family_row >= polygon_rows
        || empty_mask[family_row]
    ) {
        return;
    }

    const int ring_start = geom_offsets[family_row];
    const int ring_stop = geom_offsets[family_row + 1];
    if (ring_stop <= ring_start) return;

    const int coord_start = ring_offsets[ring_start];
    const int coord_stop = ring_offsets[ring_stop];
    if (coord_stop <= coord_start) return;

    double xmin = x[coord_start];
    double xmax = xmin;
    double ymin = y[coord_start];
    double ymax = ymin;
    for (int coord = coord_start + 1; coord < coord_stop; ++coord) {
        const double px = x[coord];
        const double py = y[coord];
        xmin = fmin(xmin, px);
        xmax = fmax(xmax, px);
        ymin = fmin(ymin, py);
        ymax = fmax(ymax, py);
    }
    out_bounds[row * 4 + 0] = xmin;
    out_bounds[row * 4 + 1] = ymin;
    out_bounds[row * 4 + 2] = xmax;
    out_bounds[row * 4 + 3] = ymax;

    const int single_ring = (ring_stop - ring_start) == 1;
    const int ring_coord_start = ring_offsets[ring_start];
    const int ring_coord_stop = ring_offsets[ring_start + 1];
    const int vertex_count = ring_coord_stop - ring_coord_start;
    if (
        single_ring
        && vertex_count >= 4
        && vertex_count <= max_input_vertices + 1
        && xmin < xmax
        && ymin < ymax
    ) {
        out_simple[row] = 1;
    }
    if (!single_ring || vertex_count != 5 || !(xmin < xmax && ymin < ymax)) {
        return;
    }

    const double scale = fmax(fmax(fabs(xmax - xmin), fabs(ymax - ymin)), 1.0);
    const double tol = 1e-9 * scale;
    if (
        fabs(x[ring_coord_start] - x[ring_coord_start + 4]) > tol
        || fabs(y[ring_coord_start] - y[ring_coord_start + 4]) > tol
    ) {
        return;
    }

    for (int offset = 0; offset < 5; ++offset) {
        const double px = x[ring_coord_start + offset];
        const double py = y[ring_coord_start + offset];
        const int x_at_side = (
            fabs(px - xmin) <= tol || fabs(px - xmax) <= tol
        );
        const int y_at_side = (
            fabs(py - ymin) <= tol || fabs(py - ymax) <= tol
        );
        if (!x_at_side || !y_at_side) return;
    }
    for (int offset = 0; offset < 4; ++offset) {
        const double x0 = x[ring_coord_start + offset];
        const double y0 = y[ring_coord_start + offset];
        const double x1 = x[ring_coord_start + offset + 1];
        const double y1 = y[ring_coord_start + offset + 1];
        const int same_x = fabs(x1 - x0) <= tol;
        const int same_y = fabs(y1 - y0) <= tol;
        if (same_x == same_y) return;
    }
    out_rect[row] = 1;
}
"""

request_nvrtc_warmup(
    [
        (
            "polygon-shape-mask-bounds",
            _POLYGON_SHAPE_MASK_BOUNDS_KERNEL_SOURCE,
            _POLYGON_SHAPE_MASK_BOUNDS_KERNEL_NAMES,
        ),
    ]
)


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


def _polygon_rect_boundary_point_contact_kernels():
    return compile_kernel_group(
        "polygon-rect-boundary-point-contact",
        _BOUNDARY_POINT_CONTACT_KERNEL_SOURCE,
        _BOUNDARY_POINT_CONTACT_KERNEL_NAMES,
    )


def _polygon_shape_mask_bounds_kernels():
    return compile_kernel_group(
        "polygon-shape-mask-bounds",
        _POLYGON_SHAPE_MASK_BOUNDS_KERNEL_SOURCE,
        _POLYGON_SHAPE_MASK_BOUNDS_KERNEL_NAMES,
    )


def _polygon_rect_device_to_host(device_array: object, *, reason: str):
    return get_cuda_runtime().copy_device_to_host(device_array, reason=reason)


def _polygon_rect_bool_scalar(value: object, *, reason: str) -> bool:
    return bool(_polygon_rect_device_to_host(cp.asarray(value).reshape(1), reason=reason)[0])


def _polygon_rect_int_scalar(value: object, *, reason: str) -> int:
    return int(_polygon_rect_device_to_host(cp.asarray(value).reshape(1), reason=reason)[0])


def _extract_polygon_family_device_buffer(owned: OwnedGeometryArray):
    if GeometryFamily.POLYGON not in owned.families:
        return None, None
    host_buf = owned.families[GeometryFamily.POLYGON]
    if host_buf.row_count == 0:
        return None, None
    state = owned._ensure_device_state()
    device_buf = (
        state.families[GeometryFamily.POLYGON] if GeometryFamily.POLYGON in state.families else None
    )
    return device_buf, host_buf


def _device_dense_single_ring_width(polygon_buf, row_count: int) -> int | None:
    """Return cached fixed-width one-ring proof for a device polygon buffer."""
    if polygon_buf is None or row_count <= 0:
        return None
    width = getattr(polygon_buf, "dense_single_ring_width", None)
    if width is None:
        fixed_size = getattr(polygon_buf, "fixed_size", None)
        if (
            fixed_size is not None
            and fixed_size.first_level_count_per_row == 1
            and fixed_size.coord_count_per_row is not None
        ):
            width = int(fixed_size.coord_count_per_row)
        elif (
            polygon_buf.ring_offsets is not None
            and int(polygon_buf.geometry_offsets.size) == row_count + 1
            and int(polygon_buf.ring_offsets.size) == row_count + 1
            and int(polygon_buf.empty_mask.size) == row_count
            and int(polygon_buf.x.size) == int(polygon_buf.y.size)
            and int(polygon_buf.x.size) > 0
            and int(polygon_buf.x.size) % row_count == 0
        ):
            width = int(polygon_buf.x.size) // row_count
        else:
            return None
    width = int(width)
    if width <= 0:
        return None
    if polygon_buf.ring_offsets is None:
        return None
    if (
        int(polygon_buf.geometry_offsets.size) != row_count + 1
        or int(polygon_buf.ring_offsets.size) != row_count + 1
        or int(polygon_buf.empty_mask.size) != row_count
        or int(polygon_buf.x.size) != row_count * width
        or int(polygon_buf.y.size) != row_count * width
    ):
        return None
    return width


def _bounded_polygon_rect_vertex_capacity(
    polygon_buf,
    physical_rows: int,
    logical_rows: int,
) -> int | None:
    """Return a safe small-shape output capacity for rect-clipped polygons."""
    if logical_rows <= 0 or physical_rows <= 0:
        return 0
    width = _device_dense_single_ring_width(polygon_buf, physical_rows)
    if width is None:
        # The NVRTC clipper bounds its local queues and emitted row width at
        # MAX_INPUT_VERTS. Unknown-width row-indirected sources therefore use
        # that kernel contract directly instead of reading an exact total.
        per_row_capacity = _MAX_INPUT_VERTS
    elif int(width) > _MAX_INPUT_VERTS + 1:
        # A row-indirected partition can share a wide physical source while all
        # of its active logical lanes satisfy the per-row clip contract. Finite
        # rectangle bounds are the activity carrier; inactive lanes are counted
        # as zero by the kernel before it reads source structure. Size from the
        # logical clip capacity instead of rejecting the unrelated physical row.
        per_row_capacity = _MAX_INPUT_VERTS
    else:
        # Fixed-width metadata admits a tighter Sutherland-Hodgman capacity.
        per_row_capacity = min(int(width) * 16 + 16, _MAX_INPUT_VERTS)
    capacity = int(logical_rows) * per_row_capacity
    if capacity <= 0:
        return 0
    if capacity >= 2_147_483_647:
        return None
    return capacity


def device_trusted_single_ring_polygon_batch(
    owned: OwnedGeometryArray,
    *,
    max_input_vertices: int = _MAX_INPUT_VERTS,
) -> bool:
    """Return True when device metadata proves logical rows fit this kernel."""
    if cp is None or owned.row_count <= 0:
        return False
    if set(owned.families) != {GeometryFamily.POLYGON}:
        return False
    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None:
        return False
    physical_rows = max(int(polygon.geometry_offsets.size) - 1, 0)
    width = _device_dense_single_ring_width(polygon, physical_rows)
    return width is not None and 4 <= int(width) <= int(max_input_vertices) + 1


def device_trusted_rectangle_bounds_matrix(owned: OwnedGeometryArray):
    """Return row-aligned device bounds when metadata proves rectangle rows."""
    if cp is None or owned.row_count <= 0:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None:
        return None
    if bool(getattr(polygon, "axis_aligned_rectangles", False)):
        row_bounds = getattr(state, "row_bounds", None)
        if row_bounds is not None:
            bounds = cp.asarray(row_bounds, dtype=cp.float64)
            if tuple(bounds.shape) == (int(owned.row_count), 4):
                return bounds
    physical_rows = max(int(polygon.geometry_offsets.size) - 1, 0)
    if physical_rows <= 0:
        return None
    if _device_dense_single_ring_width(polygon, physical_rows) != 5:
        return None
    if not bool(getattr(polygon, "axis_aligned_rectangles", False)):
        return None
    row_count = int(owned.row_count)
    if not getattr(owned, "is_indexed_view", False) and physical_rows == row_count:
        bounds = _device_rectangle_bounds(polygon, row_count)
    else:
        bounds = _device_logical_rectangle_bounds(polygon, state, row_count)
    if bounds is None:
        return None
    return cp.column_stack(bounds).astype(cp.float64, copy=False)


def _device_is_dense_single_ring_polygons(polygon_buf, row_count: int) -> bool:
    if polygon_buf is None or row_count <= 0:
        return False
    if _device_dense_single_ring_width(polygon_buf, row_count) is not None:
        return True
    if int(polygon_buf.geometry_offsets.size) != row_count + 1:
        return False
    geom_counts = polygon_buf.geometry_offsets[1:] - polygon_buf.geometry_offsets[:-1]
    if not _polygon_rect_bool_scalar(
        cp.all(geom_counts == 1),
        reason="polygon-rectangle dense single-ring scalar fence",
    ):
        return False
    if _polygon_rect_bool_scalar(
        cp.any(polygon_buf.empty_mask),
        reason="polygon-rectangle empty-mask scalar fence",
    ):
        return False
    return True


def _device_rectangle_bounds(polygon_buf, row_count: int):
    dense_width = _device_dense_single_ring_width(polygon_buf, row_count)
    if dense_width is not None:
        if dense_width != 5:
            return None
        if bool(getattr(polygon_buf, "axis_aligned_rectangles", False)):
            if polygon_buf.bounds is not None:
                bounds = cp.asarray(polygon_buf.bounds)
                if tuple(int(dim) for dim in bounds.shape) == (row_count, 4):
                    # Raw kernels read these as pointer-linear vectors; CuPy
                    # columns from an AoS bounds matrix are strided views.
                    bounds_soa = cp.ascontiguousarray(
                        bounds.astype(cp.float64, copy=False).T,
                    )
                    return (
                        bounds_soa[0],
                        bounds_soa[1],
                        bounds_soa[2],
                        bounds_soa[3],
                    )
            x = polygon_buf.x.reshape(row_count, 5)
            y = polygon_buf.y.reshape(row_count, 5)
            return (
                cp.min(x[:, :4], axis=1).astype(cp.float64, copy=False),
                cp.min(y[:, :4], axis=1).astype(cp.float64, copy=False),
                cp.max(x[:, :4], axis=1).astype(cp.float64, copy=False),
                cp.max(y[:, :4], axis=1).astype(cp.float64, copy=False),
            )
    else:
        if not _device_is_dense_single_ring_polygons(polygon_buf, row_count):
            return None
        if polygon_buf.ring_offsets is None or int(polygon_buf.ring_offsets.size) != row_count + 1:
            return None
        expected_offsets = cp.arange(0, (row_count + 1) * 5, 5, dtype=cp.int32)
        if not _polygon_rect_bool_scalar(
            cp.all(polygon_buf.ring_offsets == expected_offsets),
            reason="polygon-rectangle ring-offset scalar fence",
        ):
            return None
    if int(polygon_buf.x.size) != row_count * 5 or int(polygon_buf.y.size) != row_count * 5:
        return None

    x = polygon_buf.x.reshape(row_count, 5)
    y = polygon_buf.y.reshape(row_count, 5)
    if not _polygon_rect_bool_scalar(
        cp.all(cp.isclose(x[:, 0], x[:, 4])),
        reason="polygon-rectangle x-closure scalar fence",
    ):
        return None
    if not _polygon_rect_bool_scalar(
        cp.all(cp.isclose(y[:, 0], y[:, 4])),
        reason="polygon-rectangle y-closure scalar fence",
    ):
        return None

    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    axis_aligned = (cp.abs(dx) < 1e-12) ^ (cp.abs(dy) < 1e-12)
    if not _polygon_rect_bool_scalar(
        cp.all(axis_aligned),
        reason="polygon-rectangle axis-aligned scalar fence",
    ):
        return None

    return (
        cp.min(x[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.min(y[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.max(x[:, :4], axis=1).astype(cp.float64, copy=False),
        cp.max(y[:, :4], axis=1).astype(cp.float64, copy=False),
    )


def _device_logical_rectangle_bounds(polygon_buf, state, row_count: int):
    """Return rectangle bounds for logical rows backed by device family offsets."""
    if polygon_buf is None or polygon_buf.ring_offsets is None:
        return None
    physical_rows = max(int(polygon_buf.geometry_offsets.size) - 1, 0)
    if physical_rows <= 0:
        return None
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_family_valid = (
        d_validity
        & (d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON]))
        & (d_family_rows >= 0)
        & (d_family_rows < physical_rows)
    )
    d_safe_rows = cp.clip(
        d_family_rows,
        cp.int32(0),
        cp.int32(physical_rows - 1),
    ).astype(cp.int64, copy=False)
    d_empty = cp.asarray(polygon_buf.empty_mask, dtype=cp.bool_)[d_safe_rows]
    d_family_valid &= ~d_empty

    dense_width = _device_dense_single_ring_width(polygon_buf, physical_rows)
    if dense_width != 5:
        return None
    if (
        bool(getattr(polygon_buf, "axis_aligned_rectangles", False))
        and polygon_buf.bounds is not None
    ):
        bounds = cp.asarray(polygon_buf.bounds, dtype=cp.float64)
        if tuple(int(dim) for dim in bounds.shape) == (physical_rows, 4):
            gathered = bounds[d_safe_rows].reshape(row_count, 4)
            gathered = cp.where(d_family_valid[:, None], gathered, cp.nan)
            bounds_soa = cp.ascontiguousarray(gathered.T)
            return bounds_soa[0], bounds_soa[1], bounds_soa[2], bounds_soa[3]

    if int(polygon_buf.x.size) != physical_rows * 5 or int(polygon_buf.y.size) != physical_rows * 5:
        return None
    x = cp.asarray(polygon_buf.x, dtype=cp.float64).reshape(physical_rows, 5)[d_safe_rows]
    y = cp.asarray(polygon_buf.y, dtype=cp.float64).reshape(physical_rows, 5)[d_safe_rows]
    xmin = cp.min(x[:, :4], axis=1).astype(cp.float64, copy=False)
    ymin = cp.min(y[:, :4], axis=1).astype(cp.float64, copy=False)
    xmax = cp.max(x[:, :4], axis=1).astype(cp.float64, copy=False)
    ymax = cp.max(y[:, :4], axis=1).astype(cp.float64, copy=False)
    xmin = cp.where(d_family_valid, xmin, cp.nan)
    ymin = cp.where(d_family_valid, ymin, cp.nan)
    xmax = cp.where(d_family_valid, xmax, cp.nan)
    ymax = cp.where(d_family_valid, ymax, cp.nan)
    return xmin, ymin, xmax, ymax


def device_single_ring_polygon_mask(
    owned: OwnedGeometryArray,
    *,
    max_input_vertices: int = _MAX_INPUT_VERTS,
):
    """Return a device mask for logical rows that are simple single-ring polygons."""
    if cp is None:
        return None
    if owned.row_count == 0:
        return cp.zeros(0, dtype=cp.bool_)
    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None or polygon.ring_offsets is None:
        return cp.zeros(owned.row_count, dtype=cp.bool_)
    polygon_rows = int(polygon.geometry_offsets.size) - 1
    if polygon_rows <= 0:
        return cp.zeros(owned.row_count, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_family_valid = (d_tags == polygon_tag) & (d_family_rows >= 0) & (d_family_rows < polygon_rows)
    d_safe_rows = cp.clip(d_family_rows, cp.int32(0), cp.int32(polygon_rows - 1)).astype(
        cp.int64,
        copy=False,
    )
    d_geom_offsets = cp.asarray(polygon.geometry_offsets, dtype=cp.int32)
    d_ring_offsets = cp.asarray(polygon.ring_offsets, dtype=cp.int32)
    d_geom_starts = d_geom_offsets[d_safe_rows]
    d_geom_ends = d_geom_offsets[d_safe_rows + 1]
    d_single_ring = (d_geom_ends - d_geom_starts) == 1
    d_coord_starts = d_ring_offsets[d_geom_starts]
    d_coord_ends = d_ring_offsets[d_geom_ends]
    d_vertex_counts = d_coord_ends - d_coord_starts
    d_empty = cp.asarray(polygon.empty_mask, dtype=cp.bool_)[d_safe_rows]
    return (
        d_family_valid
        & cp.asarray(state.validity, dtype=cp.bool_)
        & ~d_empty
        & d_single_ring
        & (d_vertex_counts >= 4)
        & (d_vertex_counts <= (int(max_input_vertices) + 1))
    )


def device_polygon_shape_mask_bounds(
    owned: OwnedGeometryArray,
    *,
    max_input_vertices: int = _MAX_INPUT_VERTS,
):
    """Return simple mask, rectangle mask, and polygon bounds as one rowset carrier."""
    if cp is None:
        return None
    row_count = int(owned.row_count)
    if row_count == 0:
        return (
            cp.zeros(0, dtype=cp.bool_),
            cp.zeros(0, dtype=cp.bool_),
            cp.empty((0, 4), dtype=cp.float64),
        )

    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None or polygon.ring_offsets is None:
        return (
            cp.zeros(row_count, dtype=cp.bool_),
            cp.zeros(row_count, dtype=cp.bool_),
            cp.full((row_count, 4), cp.nan, dtype=cp.float64),
        )
    polygon_rows = int(polygon.geometry_offsets.size) - 1
    if polygon_rows <= 0 or int(polygon.x.size) <= 0:
        return (
            cp.zeros(row_count, dtype=cp.bool_),
            cp.zeros(row_count, dtype=cp.bool_),
            cp.full((row_count, 4), cp.nan, dtype=cp.float64),
        )

    d_simple = cp.empty(row_count, dtype=cp.bool_)
    d_rect = cp.empty(row_count, dtype=cp.bool_)
    d_bounds = cp.empty(row_count * 4, dtype=cp.float64)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    d_geom_offsets = cp.asarray(polygon.geometry_offsets, dtype=cp.int32)
    d_ring_offsets = cp.asarray(polygon.ring_offsets, dtype=cp.int32)
    d_empty = cp.asarray(polygon.empty_mask, dtype=cp.bool_)
    d_x = cp.asarray(polygon.x, dtype=cp.float64)
    d_y = cp.asarray(polygon.y, dtype=cp.float64)

    runtime = get_cuda_runtime()
    kernels = _polygon_shape_mask_bounds_kernels()
    ptr = runtime.pointer
    params = (
        (
            ptr(d_tags),
            ptr(d_validity),
            ptr(d_family_rows),
            ptr(d_geom_offsets),
            ptr(d_ring_offsets),
            ptr(d_empty),
            ptr(d_x),
            ptr(d_y),
            ptr(d_simple),
            ptr(d_rect),
            ptr(d_bounds),
            row_count,
            polygon_rows,
            int(FAMILY_TAGS[GeometryFamily.POLYGON]),
            int(max_input_vertices),
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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(
        kernels["polygon_shape_mask_bounds"],
        row_count,
    )
    runtime.launch(
        kernels["polygon_shape_mask_bounds"],
        grid=grid,
        block=block,
        params=params,
    )
    return d_simple, d_rect, d_bounds.reshape(row_count, 4)


def device_rectangle_polygon_mask_and_bounds(owned: OwnedGeometryArray):
    """Return ``(device_mask, device_bounds)`` for logical rectangle polygon rows."""
    if cp is None:
        return None
    if owned.row_count == 0:
        return cp.zeros(0, dtype=cp.bool_), cp.empty((0, 4), dtype=cp.float64)
    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None or polygon.ring_offsets is None:
        return cp.zeros(owned.row_count, dtype=cp.bool_), cp.full(
            (owned.row_count, 4),
            cp.nan,
            dtype=cp.float64,
        )
    polygon_rows = int(polygon.geometry_offsets.size) - 1
    coord_count = int(polygon.x.size)
    if polygon_rows <= 0 or coord_count <= 0:
        return cp.zeros(owned.row_count, dtype=cp.bool_), cp.full(
            (owned.row_count, 4),
            cp.nan,
            dtype=cp.float64,
        )

    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_family_valid = (d_tags == polygon_tag) & (d_family_rows >= 0) & (d_family_rows < polygon_rows)
    d_safe_rows = cp.clip(d_family_rows, cp.int32(0), cp.int32(polygon_rows - 1)).astype(
        cp.int64,
        copy=False,
    )
    d_geom_offsets = cp.asarray(polygon.geometry_offsets, dtype=cp.int32)
    d_ring_offsets = cp.asarray(polygon.ring_offsets, dtype=cp.int32)
    d_geom_starts = d_geom_offsets[d_safe_rows]
    d_geom_ends = d_geom_offsets[d_safe_rows + 1]
    d_single_ring = (d_geom_ends - d_geom_starts) == 1
    d_coord_starts = d_ring_offsets[d_geom_starts]
    d_coord_ends = d_ring_offsets[d_geom_ends]
    d_five_coords = (d_coord_ends - d_coord_starts) == 5

    d_offsets = (
        d_coord_starts[:, None].astype(cp.int64, copy=False)
        + cp.arange(
            5,
            dtype=cp.int64,
        )[None, :]
    )
    d_offsets = cp.clip(d_offsets, cp.int64(0), cp.int64(coord_count - 1))
    d_x = cp.asarray(polygon.x, dtype=cp.float64)[d_offsets]
    d_y = cp.asarray(polygon.y, dtype=cp.float64)[d_offsets]
    d_minx = cp.min(d_x[:, :4], axis=1)
    d_maxx = cp.max(d_x[:, :4], axis=1)
    d_miny = cp.min(d_y[:, :4], axis=1)
    d_maxy = cp.max(d_y[:, :4], axis=1)
    d_scale = cp.maximum(cp.maximum(cp.abs(d_maxx - d_minx), cp.abs(d_maxy - d_miny)), 1.0)
    d_tol = 1e-9 * d_scale
    d_tol_2d = d_tol[:, None]
    d_closed = (cp.abs(d_x[:, 0] - d_x[:, 4]) <= d_tol) & (cp.abs(d_y[:, 0] - d_y[:, 4]) <= d_tol)
    d_x_at_side = (cp.abs(d_x - d_minx[:, None]) <= d_tol_2d) | (
        cp.abs(d_x - d_maxx[:, None]) <= d_tol_2d
    )
    d_y_at_side = (cp.abs(d_y - d_miny[:, None]) <= d_tol_2d) | (
        cp.abs(d_y - d_maxy[:, None]) <= d_tol_2d
    )
    d_edge_same_x = cp.abs(d_x[:, 1:] - d_x[:, :-1]) <= d_tol_2d
    d_edge_same_y = cp.abs(d_y[:, 1:] - d_y[:, :-1]) <= d_tol_2d
    d_empty = cp.asarray(polygon.empty_mask, dtype=cp.bool_)[d_safe_rows]
    d_rect = (
        d_family_valid
        & cp.asarray(state.validity, dtype=cp.bool_)
        & ~d_empty
        & d_single_ring
        & d_five_coords
        & d_closed
        & cp.all(d_x_at_side, axis=1)
        & cp.all(d_y_at_side, axis=1)
        & cp.all(cp.logical_xor(d_edge_same_x, d_edge_same_y), axis=1)
        & (d_minx < d_maxx)
        & (d_miny < d_maxy)
    )
    d_bounds = cp.column_stack((d_minx, d_miny, d_maxx, d_maxy)).astype(
        cp.float64,
        copy=False,
    )
    d_bounds = cp.where(d_rect[:, None], d_bounds, cp.nan)
    return d_rect, d_bounds


def _bounds_columns_from_device_bounds(device_bounds, row_count: int):
    d_bounds = cp.asarray(device_bounds, dtype=cp.float64)
    if d_bounds.ndim != 2 or int(d_bounds.shape[0]) != row_count or int(d_bounds.shape[1]) != 4:
        raise ValueError("polygon-rectangle bounds carrier must have shape (rows, 4)")
    d_bounds_soa = cp.ascontiguousarray(d_bounds.T)
    return d_bounds_soa[0], d_bounds_soa[1], d_bounds_soa[2], d_bounds_soa[3]


def polygon_rect_boundary_contacts_from_bounds(
    mask: OwnedGeometryArray,
    rect_bounds,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> (
    tuple[
        OwnedGeometryArray,
        object,
        OwnedGeometryArray,
        object,
        object,
    ]
    | None
):
    """Build point and line rectangle/polygon boundary intersections on device.

    Physical shape: a device rowset of rectangle bounds against one single-ring
    polygon mask. Point and collinear-line contacts are emitted as separate
    row-aligned capacities so callers can compose both without geometry-object
    reconstruction. The admitted carrier is one physically singular ring with
    at most ``_BOUNDARY_CONTACT_MAX_SEGMENTS`` segments; line components are
    ordered and merged at segment capacity instead of using a fixed part queue.
    """
    if cp is None or not has_gpu_runtime():
        return None
    n = int(getattr(rect_bounds, "shape", (0,))[0])
    if n == 0:
        return None
    if mask.row_count != 1:
        return None
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    if mask.residency is not Residency.DEVICE:
        mask.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_rect boundary point contacts selected GPU mask",
        )
    state = mask._ensure_device_state(preserve_indexed_view=True)
    mask_dev = state.families.get(GeometryFamily.POLYGON)
    if (
        mask_dev is None
        or mask_dev.ring_offsets is None
        or int(mask_dev.geometry_offsets.size) != 2
        or int(mask_dev.ring_offsets.size) != 2
        or int(mask_dev.x.size) != int(mask_dev.y.size)
    ):
        return None
    if set(mask.families) != {GeometryFamily.POLYGON}:
        return None
    segment_capacity = int(mask_dev.x.size) - 1
    if segment_capacity < 3 or segment_capacity > _BOUNDARY_CONTACT_MAX_SEGMENTS:
        return None

    d_xmin, d_ymin, d_xmax, d_ymax = _bounds_columns_from_device_bounds(
        rect_bounds,
        n,
    )
    runtime = get_cuda_runtime()
    kernels = _polygon_rect_boundary_point_contact_kernels()
    ptr = runtime.pointer
    d_counts = runtime.allocate((n,), cp.int32, zero=True)
    d_line_counts = runtime.allocate((n,), cp.int32, zero=True)
    count_params = (
        (
            ptr(mask_dev.x),
            ptr(mask_dev.y),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_counts),
            ptr(d_line_counts),
            n,
            segment_capacity,
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
            KERNEL_PARAM_I64,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_boundary_point_contact_count"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_boundary_point_contact_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )
    d_output_mask = d_counts > 0
    d_single_mask = d_output_mask & (d_counts == 1)
    d_multi_mask = d_output_mask & (d_counts > 1)
    d_multi_counts = cp.where(d_multi_mask, d_counts, cp.int32(0)).astype(
        cp.int32,
        copy=False,
    )
    d_multi_offsets = exclusive_sum(d_multi_counts, synchronize=False)
    d_line_output_mask = d_line_counts > 0
    d_single_line_mask = d_line_output_mask & (d_line_counts == 1)
    d_multiline_mask = d_line_output_mask & (d_line_counts > 1)
    d_multiline_counts = cp.where(
        d_multiline_mask,
        d_line_counts,
        cp.int32(0),
    ).astype(cp.int32, copy=False)
    d_multiline_offsets = exclusive_sum(
        d_multiline_counts,
        synchronize=False,
    )
    multipoint_capacity = n * (segment_capacity * 2)
    multiline_part_capacity = n * segment_capacity
    d_point_x = runtime.allocate((n,), cp.float64)
    d_point_y = runtime.allocate((n,), cp.float64)
    d_multipoint_x = runtime.allocate((multipoint_capacity,), cp.float64)
    d_multipoint_y = runtime.allocate((multipoint_capacity,), cp.float64)
    d_line_x = runtime.allocate((n * 2,), cp.float64)
    d_line_y = runtime.allocate((n * 2,), cp.float64)
    d_multiline_x = runtime.allocate((multiline_part_capacity * 2,), cp.float64)
    d_multiline_y = runtime.allocate((multiline_part_capacity * 2,), cp.float64)
    scatter_params = (
        (
            ptr(mask_dev.x),
            ptr(mask_dev.y),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_counts),
            ptr(d_line_counts),
            ptr(d_multi_offsets),
            ptr(d_multiline_offsets),
            ptr(d_point_x),
            ptr(d_point_y),
            ptr(d_multipoint_x),
            ptr(d_multipoint_y),
            ptr(d_line_x),
            ptr(d_line_y),
            ptr(d_multiline_x),
            ptr(d_multiline_y),
            n,
            segment_capacity,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I64,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_boundary_point_contact_scatter"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_boundary_point_contact_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_multipoint_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_multipoint_offsets[:n] = d_multi_offsets
    d_multipoint_offsets[n] = cp.sum(d_multi_counts, dtype=cp.int32)
    d_row_ids = cp.arange(n, dtype=cp.int32)
    d_tags_out = cp.where(
        d_single_mask,
        cp.int8(FAMILY_TAGS[GeometryFamily.POINT]),
        cp.where(
            d_multi_mask,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT]),
            cp.int8(-1),
        ),
    )
    d_family_row_offsets = cp.where(
        d_output_mask,
        d_row_ids,
        cp.int32(-1),
    )
    device_families = {
        GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=d_point_x,
            y=d_point_y,
            geometry_offsets=cp.arange(n + 1, dtype=cp.int32),
            empty_mask=~d_single_mask,
            bounds=None,
            fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=1),
        ),
        GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=d_multipoint_x,
            y=d_multipoint_y,
            geometry_offsets=d_multipoint_offsets,
            empty_mask=~d_multi_mask,
            bounds=None,
        ),
    }

    point_result = build_device_resident_owned(
        device_families=device_families,
        row_count=n,
        tags=d_tags_out,
        validity=d_output_mask,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    if point_result.device_state is not None:
        point_result.device_state.trusted_all_valid = False
        point_result.device_state.trusted_all_non_empty = False

    d_multiline_geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_multiline_geometry_offsets[:n] = d_multiline_offsets
    d_multiline_geometry_offsets[n] = cp.sum(
        d_multiline_counts,
        dtype=cp.int32,
    )
    d_line_tags = cp.where(
        d_single_line_mask,
        cp.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]),
        cp.where(
            d_multiline_mask,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING]),
            cp.int8(-1),
        ),
    )
    d_line_family_rows = cp.where(
        d_line_output_mask,
        d_row_ids,
        cp.int32(-1),
    )
    line_result = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_line_x,
                y=d_line_y,
                geometry_offsets=cp.arange(n + 1, dtype=cp.int32) * cp.int32(2),
                empty_mask=~d_single_line_mask,
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=2),
            ),
            GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTILINESTRING,
                x=d_multiline_x,
                y=d_multiline_y,
                geometry_offsets=d_multiline_geometry_offsets,
                part_offsets=(cp.arange(multiline_part_capacity + 1, dtype=cp.int32) * cp.int32(2)),
                empty_mask=~d_multiline_mask,
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(
                    max_first_level_count_per_row=segment_capacity,
                    max_coord_count_per_row=segment_capacity * 2,
                ),
            ),
        },
        row_count=n,
        tags=d_line_tags,
        validity=d_line_output_mask,
        family_row_offsets=d_line_family_rows,
        execution_mode="gpu",
    )
    if line_result.device_state is not None:
        line_result.device_state.trusted_all_valid = False
        line_result.device_state.trusted_all_non_empty = False
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_rect_intersection",
        operation="polygon_rect_boundary_contacts",
        implementation="polygon_rect_boundary_contacts_gpu",
        reason=(
            "rectangle-cell polygon-mask boundary point and line contacts were "
            "assembled directly from device bounds"
        ),
        detail=(
            f"rows={n}; point_capacity={n}; "
            f"multipoint_capacity={multipoint_capacity}; "
            f"line_capacity={n}; multiline_part_capacity={multiline_part_capacity}; "
            f"mask_segments={segment_capacity}; "
            "workload_shape=rowset_segment_capacity_rectangle_boundary_contacts"
        ),
        requested=requested,
        selected=ExecutionMode.GPU,
    )
    return (
        point_result,
        d_output_mask,
        line_result,
        d_line_output_mask,
        cp.zeros(n, dtype=cp.bool_),
    )


def rectangle_rectangle_boundary_intersections_from_bounds(
    left_bounds,
    right_bounds,
    *,
    active_mask=None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Build exact lower-dimensional rectangle intersections at row capacity.

    Positive-area overlap belongs to the polygon area carrier.  This carrier
    emits only zero-width edge contacts and zero-width/zero-height corner
    contacts, with null rows for area overlap and disjoint pairs.
    """
    if cp is None or not has_gpu_runtime():
        return None

    d_left = cp.asarray(left_bounds, dtype=cp.float64)
    if d_left.ndim != 2 or int(d_left.shape[1]) != 4:
        raise ValueError("left rectangle bounds must have shape (rows, 4)")
    row_count = int(d_left.shape[0])
    d_right = cp.asarray(right_bounds, dtype=cp.float64)
    if d_right.ndim == 1:
        if int(d_right.size) != 4:
            raise ValueError("right rectangle bounds must contain four values")
        d_right = cp.broadcast_to(d_right.reshape(1, 4), (row_count, 4))
    elif d_right.ndim != 2 or tuple(d_right.shape) != (row_count, 4):
        raise ValueError("right rectangle bounds must be scalar or row-aligned")

    if active_mask is None:
        d_active = cp.ones(row_count, dtype=cp.bool_)
    else:
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != row_count:
            raise ValueError("rectangle boundary activity must match row capacity")

    d_xmin = cp.maximum(d_left[:, 0], d_right[:, 0])
    d_ymin = cp.maximum(d_left[:, 1], d_right[:, 1])
    d_xmax = cp.minimum(d_left[:, 2], d_right[:, 2])
    d_ymax = cp.minimum(d_left[:, 3], d_right[:, 3])
    d_finite = cp.all(cp.isfinite(d_left), axis=1) & cp.all(
        cp.isfinite(d_right),
        axis=1,
    )
    d_x_positive = d_xmax > d_xmin
    d_y_positive = d_ymax > d_ymin
    d_x_zero = d_xmax == d_xmin
    d_y_zero = d_ymax == d_ymin
    d_line = d_active & d_finite & ((d_x_zero & d_y_positive) | (d_y_zero & d_x_positive))
    d_point = d_active & d_finite & d_x_zero & d_y_zero
    d_valid = d_line | d_point

    d_vertical = d_x_zero & d_y_positive
    d_line_x = cp.column_stack(
        (d_xmin, cp.where(d_vertical, d_xmin, d_xmax)),
    ).reshape(-1)
    d_line_y = cp.column_stack(
        (d_ymin, cp.where(d_vertical, d_ymax, d_ymin)),
    ).reshape(-1)
    d_rows = cp.arange(row_count, dtype=cp.int32)
    d_tags = cp.where(
        d_line,
        cp.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]),
        cp.where(
            d_point,
            cp.int8(FAMILY_TAGS[GeometryFamily.POINT]),
            cp.int8(-1),
        ),
    )
    d_family_rows = cp.where(d_valid, d_rows, cp.int32(-1))
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_line_x,
                y=d_line_y,
                geometry_offsets=cp.arange(
                    0,
                    2 * row_count + 1,
                    2,
                    dtype=cp.int32,
                ),
                empty_mask=~d_line,
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=2),
            ),
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=d_xmin.copy(),
                y=d_ymin.copy(),
                geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
                empty_mask=~d_point,
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=1),
            ),
        },
        row_count=row_count,
        tags=d_tags,
        validity=d_valid,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )
    state = result._ensure_device_state(preserve_indexed_view=True)
    d_nan = cp.asarray(cp.nan, dtype=cp.float64)
    state.row_bounds = cp.where(
        d_valid[:, None],
        cp.column_stack((d_xmin, d_ymin, d_xmax, d_ymax)),
        d_nan,
    )
    state.trusted_all_valid = True if row_count == 0 else False
    state.trusted_all_non_empty = True if row_count == 0 else False
    state.trusted_family_domain = (
        GeometryFamily.LINESTRING,
        GeometryFamily.POINT,
    )

    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_rect_intersection",
        operation="rectangle_rectangle_boundary_intersection",
        implementation="rectangle_rectangle_boundary_bounds_gpu",
        reason=(
            "axis-aligned rectangle boundary contacts were assembled from resident fp64 bounds"
        ),
        detail=(
            f"rows={row_count}; line_capacity={row_count}; "
            f"point_capacity={row_count}; workload_shape=rectangle_pair_bounds"
        ),
        requested=requested,
        selected=ExecutionMode.GPU,
    )
    return result


def polygon_rect_intersection_from_bounds(
    left: OwnedGeometryArray,
    rect_bounds,
    *,
    source_rows=None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OwnedGeometryArray:
    """Clip logical polygon rows by row-aligned device rectangle bounds.

    This is the row-indirected carrier for mixed/few-right overlay batches: the
    subject polygons can remain indexed or repeated, while rectangle geometry is
    represented by a device ``(rows, 4)`` bounds table.
    """
    if cp is None:
        raise RuntimeError("CuPy is required for polygon-rectangle bounds clipping")
    source_row_count = int(left.row_count)
    d_source_rows = None
    if source_rows is None:
        n = source_row_count
    else:
        d_source_rows = cp.asarray(source_rows, dtype=cp.int64)
        n = int(d_source_rows.size)
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    selection = RuntimeSelection(
        requested=requested,
        selected=ExecutionMode.GPU,
        reason="GPU row-indirected polygon-rectangle bounds intersection selected",
    )
    if n == 0:
        return _build_empty_result(n, selection)

    if left.residency is not Residency.DEVICE:
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_rect_intersection_from_bounds selected GPU execution",
        )
    state = left._ensure_device_state(preserve_indexed_view=True)
    left_dev = state.families.get(GeometryFamily.POLYGON)
    if left_dev is None or left_dev.ring_offsets is None:
        return _build_empty_result(n, selection)
    polygon_rows = int(left_dev.geometry_offsets.size) - 1
    if polygon_rows <= 0:
        return _build_empty_result(n, selection)

    d_xmin, d_ymin, d_xmax, d_ymax = _bounds_columns_from_device_bounds(
        rect_bounds,
        n,
    )
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_valid = (
        (d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON]))
        & (d_family_rows >= 0)
        & (d_family_rows < polygon_rows)
    )
    d_safe_rows = cp.clip(d_family_rows, cp.int32(0), cp.int32(polygon_rows - 1)).astype(
        cp.int64,
        copy=False,
    )
    d_left_valid = (
        d_family_valid
        & cp.asarray(state.validity, dtype=cp.bool_)
        & ~cp.asarray(left_dev.empty_mask, dtype=cp.bool_)[d_safe_rows]
    ).astype(cp.int32, copy=False)
    d_right_valid = (
        cp.isfinite(d_xmin)
        & cp.isfinite(d_ymin)
        & cp.isfinite(d_xmax)
        & cp.isfinite(d_ymax)
        & (d_xmin < d_xmax)
        & (d_ymin < d_ymax)
    ).astype(cp.int32, copy=False)

    runtime = get_cuda_runtime()
    d_counts = runtime.allocate((n,), cp.int32, zero=True)
    d_valid = runtime.allocate((n,), cp.int32, zero=True)
    d_boundary_overlap = runtime.allocate((n,), cp.int32, zero=True)
    d_exact_polygon_only = runtime.allocate((n,), cp.int32, zero=True)
    d_lower_dimensional_remnant = runtime.allocate((n,), cp.int32, zero=True)
    kernels = _polygon_rect_intersection_kernels()
    ptr = runtime.pointer
    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_family_rows),
            ptr(d_source_rows),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_boundary_overlap),
            ptr(d_exact_polygon_only),
            ptr(d_lower_dimensional_remnant),
            source_row_count,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_intersection_count"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    bounded_capacity = _bounded_polygon_rect_vertex_capacity(
        left_dev,
        polygon_rows,
        n,
    )
    if bounded_capacity is None or bounded_capacity <= 0:
        raise RuntimeError("polygon-rectangle row-indirected output exceeds int32 vertex capacity")
    output_capacity = int(bounded_capacity)
    d_logical_total = d_offsets[-1] + d_counts[-1]

    d_out_x = runtime.allocate((output_capacity,), cp.float64)
    d_out_y = runtime.allocate((output_capacity,), cp.float64)
    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(d_family_rows),
            ptr(d_source_rows),
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
            source_row_count,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_intersection_scatter"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_ring_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_ring_offsets[:n] = cp.asarray(d_offsets)
    d_ring_offsets[n] = d_logical_total
    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid.astype(cp.bool_),
        ring_offsets=d_ring_offsets,
        runtime_selection=selection,
    )
    result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(cp.bool_)
    result._polygon_rect_exact_polygon_only = d_exact_polygon_only.astype(cp.bool_)
    result._polygon_intersection_lower_dimensional_remnant = (
        d_lower_dimensional_remnant.astype(cp.bool_)
    )
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_rect_intersection",
        operation="polygon_rect_intersection",
        implementation="polygon_rect_intersection_row_indirected_bounds_gpu",
        reason=selection.reason,
        detail=f"rows={n}",
        requested=selection.requested,
        selected=ExecutionMode.GPU,
    )
    return result


def polygon_rect_split_boundary_component_replacements(
    clipped: OwnedGeometryArray,
    rectangles: OwnedGeometryArray | None = None,
    *,
    rect_bounds=None,
    eligible_mask=None,
) -> tuple[OwnedGeometryArray, object] | None:
    """Return replacement rows for repeated-boundary rectangle clip rings.

    ``polygon_rect_intersection`` emits one polygon ring per row. Concave mask
    clips can produce disconnected intersections; in that case the single ring
    contains repeated rectangle-boundary connector segments. This helper removes
    those connector edges and closes each component along the rectangle boundary.

    The output retains ``clipped`` row capacity and carries a device split mask.
    Callers select the replacement capacity through row indirection, so dynamic
    split cardinality and nested component counts remain device-resident.
    """
    if cp is None or clipped.row_count == 0:
        return None
    if set(clipped.families) != {GeometryFamily.POLYGON}:
        return None

    row_count = clipped.row_count
    if clipped.device_state is None or clipped.residency is not Residency.DEVICE:
        clipped.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_rect boundary split selected GPU execution",
        )
    clipped_state = clipped._ensure_device_state(preserve_indexed_view=True)
    clipped_dev = clipped_state.families.get(GeometryFamily.POLYGON)
    if clipped_dev is None:
        return None
    if rect_bounds is None:
        if (
            rectangles is None
            or clipped.row_count != rectangles.row_count
            or set(rectangles.families) != {GeometryFamily.POLYGON}
        ):
            return None
        if rectangles.device_state is None or rectangles.residency is not Residency.DEVICE:
            rectangles.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="polygon_rect boundary split selected GPU execution",
            )
        rect_state = rectangles._ensure_device_state(preserve_indexed_view=True)
        rect_dev = rect_state.families.get(GeometryFamily.POLYGON)
        if rect_dev is None:
            return None
        device_bounds = (
            _device_rectangle_bounds(rect_dev, row_count)
            if not getattr(rectangles, "is_indexed_view", False)
            else _device_logical_rectangle_bounds(rect_dev, rect_state, row_count)
        )
        if device_bounds is None:
            return None
        d_xmin, d_ymin, d_xmax, d_ymax = device_bounds
    else:
        d_xmin, d_ymin, d_xmax, d_ymax = _bounds_columns_from_device_bounds(
            rect_bounds,
            row_count,
        )

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _polygon_rect_boundary_split_kernels()
    polygon_rows = max(int(clipped_dev.geometry_offsets.size) - 1, 0)
    if polygon_rows <= 0:
        return None
    d_family_rows = cp.asarray(clipped_state.family_row_offsets, dtype=cp.int32)
    d_safe_rows = cp.clip(
        d_family_rows,
        cp.int32(0),
        cp.int32(polygon_rows - 1),
    ).astype(cp.int64, copy=False)
    d_valid = (
        cp.asarray(clipped_state.validity, dtype=cp.bool_)
        & (
            cp.asarray(clipped_state.tags, dtype=cp.int8)
            == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
        )
        & (d_family_rows >= 0)
        & (d_family_rows < polygon_rows)
        & ~cp.asarray(clipped_dev.empty_mask, dtype=cp.bool_)[d_safe_rows]
    ).astype(cp.int32, copy=False)
    if eligible_mask is not None:
        d_eligible = cp.asarray(eligible_mask, dtype=cp.bool_)
        if d_eligible.ndim != 1 or int(d_eligible.size) != row_count:
            raise ValueError("boundary split eligibility must align with clipped rows")
        d_valid &= d_eligible.astype(cp.int32, copy=False)
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
                ptr(d_family_rows),
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    d_split_mask = d_component_counts >= 2
    d_capacity_component_counts = cp.where(
        d_split_mask,
        d_component_counts,
        cp.int32(0),
    ).astype(cp.int32, copy=False)
    d_geometry_offsets = exclusive_sum(
        d_capacity_component_counts,
        synchronize=False,
    )

    component_capacity = row_count * _MAX_BOUNDARY_SPLIT_COMPONENTS
    slots = cp.arange(component_capacity, dtype=cp.int32)
    slot_rows = slots // _MAX_BOUNDARY_SPLIT_COMPONENTS
    slot_components = slots - slot_rows * _MAX_BOUNDARY_SPLIT_COMPONENTS
    valid_slots = slot_components < d_capacity_component_counts[slot_rows]
    d_component_positions = (d_geometry_offsets[slot_rows] + slot_components).astype(
        cp.int32, copy=False
    )
    d_component_vertex_counts = cp.zeros(component_capacity, dtype=cp.int32)
    cp.maximum.at(
        d_component_vertex_counts,
        cp.where(valid_slots, d_component_positions, cp.int32(0)),
        cp.where(
            valid_slots,
            d_component_vertex_counts_matrix[slots],
            cp.int32(0),
        ),
    )
    d_ring_offsets = cp.empty(component_capacity + 1, dtype=cp.int32)
    d_ring_offsets[0] = 0
    cp.cumsum(d_component_vertex_counts, out=d_ring_offsets[1:])

    # Every emitted component partitions the source ring chain and adds at
    # most three rectangle corners plus closure.  Size from physical input
    # coordinates and component capacity; logical use remains in offsets.
    output_vertices = int(clipped_dev.x.size) + 4 * component_capacity
    if output_vertices <= 0 or output_vertices >= 2_147_483_647:
        return None

    d_out_x = runtime.allocate((output_vertices,), cp.float64)
    d_out_y = runtime.allocate((output_vertices,), cp.float64)
    scatter_kernel_name = "polygon_rect_boundary_split_scatter"
    scatter_grid, scatter_block = runtime.launch_config(
        kernels[scatter_kernel_name],
        row_count,
    )
    d_split_valid = d_valid & d_split_mask.astype(cp.int32, copy=False)
    scatter_params = (
        (
            ptr(clipped_dev.x),
            ptr(clipped_dev.y),
            ptr(clipped_dev.ring_offsets),
            ptr(clipped_dev.geometry_offsets),
            ptr(d_family_rows),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_split_valid),
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(
        kernels[scatter_kernel_name],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_multipolygon_geometry_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    d_multipolygon_geometry_offsets[:row_count] = d_geometry_offsets
    d_multipolygon_geometry_offsets[row_count] = cp.sum(
        d_capacity_component_counts,
        dtype=cp.int32,
    )
    multipolygon_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=d_out_x,
        y=d_out_y,
        geometry_offsets=d_multipolygon_geometry_offsets,
        empty_mask=~d_split_mask,
        part_offsets=cp.arange(component_capacity + 1, dtype=cp.int32),
        ring_offsets=d_ring_offsets,
        bounds=None,
    )
    split_owned = build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: multipolygon_buffer},
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            dtype=cp.int8,
        ),
        validity=d_split_mask,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    return split_owned, d_split_mask


def polygon_rect_split_boundary_component_replacements_from_bounds(
    clipped: OwnedGeometryArray,
    rect_bounds,
    eligible_mask,
) -> tuple[OwnedGeometryArray, object] | None:
    """Return row-capacity boundary-split replacements from rectangle bounds."""
    return polygon_rect_split_boundary_component_replacements(
        clipped,
        rect_bounds=rect_bounds,
        eligible_mask=eligible_mask,
    )


def polygon_rect_split_boundary_components(
    clipped: OwnedGeometryArray,
    rectangles: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Split repeated-boundary rectangle clip rings into row-aligned output."""
    replacements = polygon_rect_split_boundary_component_replacements(
        clipped,
        rectangles,
    )
    if replacements is None:
        return None
    split_owned, d_split_mask = replacements
    if split_owned.row_count != clipped.row_count:
        return None
    return device_select_owned_capacity_partitions(
        clipped,
        [(split_owned, d_split_mask)],
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
    dense_width = _device_dense_single_ring_width(polygon_buf, row_count)
    if dense_width is not None:
        return dense_width
    if int(polygon_buf.ring_offsets.size) != row_count + 1:
        return None
    ring_spans = polygon_buf.ring_offsets[1:] - polygon_buf.ring_offsets[:-1]
    if int(ring_spans.size) != row_count:
        return None
    if int(ring_spans.size) == 0:
        return 0
    return _polygon_rect_int_scalar(
        cp.max(ring_spans),
        reason="polygon-rectangle max-input-vertices scalar fence",
    )


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
    left_indexed = bool(getattr(left, "is_indexed_view", False))
    right_indexed = bool(getattr(right, "is_indexed_view", False))
    if not left_indexed and left.families[GeometryFamily.POLYGON].row_count != left.row_count:
        return False
    if not right_indexed and right.families[GeometryFamily.POLYGON].row_count != right.row_count:
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

    if left_indexed:
        if not device_trusted_single_ring_polygon_batch(left):
            return False
        max_input_verts = _MAX_INPUT_VERTS + 1
    elif left_host.host_materialized:
        if not _host_is_dense_single_ring_polygons(left_host, left.row_count):
            return False
        max_input_verts = _host_max_input_vertices(left_host, left.row_count)
    else:
        if not _device_is_dense_single_ring_polygons(left_device, left.row_count):
            return False
        max_input_verts = _device_max_input_vertices(left_device, left.row_count)

    if max_input_verts is None or max_input_verts > (_MAX_INPUT_VERTS + 1):
        return False

    if right_indexed:
        return device_trusted_rectangle_bounds_matrix(right) is not None
    if right_host.host_materialized:
        return _host_rectangle_bounds(right_host, right.row_count) is not None
    return _device_rectangle_bounds(right_device, right.row_count) is not None


def rectangle_intersection_can_handle(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    """Return True when both resident operands are proven rectangle batches."""
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
    if left.device_state is None or right.device_state is None:
        return False
    if GeometryFamily.POLYGON not in left.device_state.families:
        return False
    if GeometryFamily.POLYGON not in right.device_state.families:
        return False
    left_device = left.device_state.families[GeometryFamily.POLYGON]
    right_device = right.device_state.families[GeometryFamily.POLYGON]
    left_bounds = _device_rectangle_bounds(left_device, left.row_count)
    if left_bounds is None:
        return False
    return _device_rectangle_bounds(right_device, right.row_count) is not None


def rectangle_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute row-aligned rectangle intersections with fixed-width output."""
    if cp is None:
        raise RuntimeError("CuPy is required for rectangle intersection")
    if left.row_count != right.row_count:
        raise ValueError("rectangle intersection requires row-aligned inputs")
    n = left.row_count
    runtime_selection = RuntimeSelection(
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
        reason="GPU rectangle-rectangle intersection selected fixed-width output",
    )
    if n == 0:
        return _build_empty_result(n, runtime_selection)

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="rectangle_intersection selected GPU execution",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="rectangle_intersection selected GPU execution",
    )
    left_dev, _left_host = _extract_polygon_family_device_buffer(left)
    right_dev, _right_host = _extract_polygon_family_device_buffer(right)
    if left_dev is None or right_dev is None:
        return _build_empty_result(n, runtime_selection)
    left_bounds = _device_rectangle_bounds(left_dev, n)
    right_bounds = _device_rectangle_bounds(right_dev, n)
    if left_bounds is None or right_bounds is None:
        raise ValueError("rectangle intersection requires rectangle metadata or certification")

    l_xmin, l_ymin, l_xmax, l_ymax = left_bounds
    r_xmin, r_ymin, r_xmax, r_ymax = right_bounds
    xmin = cp.maximum(l_xmin, r_xmin)
    ymin = cp.maximum(l_ymin, r_ymin)
    xmax = cp.minimum(l_xmax, r_xmax)
    ymax = cp.minimum(l_ymax, r_ymax)
    d_left_valid = left.device_state.validity.astype(cp.bool_) & ~left_dev.empty_mask.astype(
        cp.bool_
    )
    d_right_valid = right.device_state.validity.astype(cp.bool_) & ~right_dev.empty_mask.astype(
        cp.bool_
    )
    d_valid = d_left_valid & d_right_valid & (xmin < xmax) & (ymin < ymax)
    d_empty = ~d_valid

    out_x = cp.empty(n * 5, dtype=cp.float64)
    out_y = cp.empty(n * 5, dtype=cp.float64)
    out_x[0::5] = xmin
    out_y[0::5] = ymin
    out_x[1::5] = xmax
    out_y[1::5] = ymin
    out_x[2::5] = xmax
    out_y[2::5] = ymax
    out_x[3::5] = xmin
    out_y[3::5] = ymax
    out_x[4::5] = xmin
    out_y[4::5] = ymin
    out_bounds = cp.column_stack((xmin, ymin, xmax, ymax))
    if bool(int(d_empty.size)):
        out_bounds[d_empty] = cp.nan

    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=out_x,
                y=out_y,
                geometry_offsets=cp.arange(n + 1, dtype=cp.int32),
                empty_mask=d_empty,
                ring_offsets=cp.arange(0, (n + 1) * 5, 5, dtype=cp.int32),
                bounds=out_bounds,
                dense_single_ring_width=5,
                axis_aligned_rectangles=True,
            )
        },
        row_count=n,
        tags=cp.full(n, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=d_valid,
        family_row_offsets=cp.arange(n, dtype=cp.int32),
        execution_mode="gpu",
    )
    eps = 1e-12
    d_boundary_overlap = d_valid & (
        (cp.abs(xmin - r_xmin) <= eps)
        | (cp.abs(xmax - r_xmax) <= eps)
        | (cp.abs(ymin - r_ymin) <= eps)
        | (cp.abs(ymax - r_ymax) <= eps)
    )
    result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(cp.bool_, copy=False)
    result._polygon_rect_exact_polygon_only = d_valid.astype(cp.bool_, copy=False)
    result.runtime_history.append(runtime_selection)
    return result


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
    d_exact_polygon_only = runtime.allocate((n,), cp.int32, zero=True)
    d_lower_dimensional_remnant = runtime.allocate((n,), cp.int32, zero=True)

    kernels = _polygon_rect_intersection_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(left_state.family_row_offsets),
            ptr(None),
            ptr(d_xmin),
            ptr(d_ymin),
            ptr(d_xmax),
            ptr(d_ymax),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            ptr(d_boundary_overlap),
            ptr(d_exact_polygon_only),
            ptr(d_lower_dimensional_remnant),
            n,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    count_grid, count_block = runtime.launch_config(
        kernels["polygon_rect_intersection_count"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_count"],
        grid=count_grid,
        block=count_block,
        params=count_params,
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    output_capacity = _bounded_polygon_rect_vertex_capacity(
        left_dev,
        n,
        n,
    )
    if output_capacity is None or output_capacity <= 0:
        raise RuntimeError("dense polygon-rectangle intersection lacks a bounded vertex capacity")
    d_logical_total = d_offsets[-1] + d_counts[-1]

    d_out_x = runtime.allocate((output_capacity,), cp.float64)
    d_out_y = runtime.allocate((output_capacity,), cp.float64)

    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(left_state.family_row_offsets),
            ptr(None),
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_rect_intersection_scatter"],
        n,
    )
    runtime.launch(
        kernels["polygon_rect_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=scatter_params,
    )

    d_ring_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_ring_offsets[:n] = cp.asarray(d_offsets)
    d_ring_offsets[n] = d_logical_total

    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid.astype(cp.bool_),
        ring_offsets=d_ring_offsets,
        runtime_selection=runtime_selection,
    )
    result._polygon_rect_boundary_overlap = d_boundary_overlap.astype(cp.bool_)
    result._polygon_rect_exact_polygon_only = d_exact_polygon_only.astype(cp.bool_)
    result._polygon_intersection_lower_dimensional_remnant = (
        d_lower_dimensional_remnant.astype(cp.bool_)
    )
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
        raise ValueError(f"row count mismatch: left={left.row_count}, right={right.row_count}")
    n = left.row_count
    if n == 0:
        return from_shapely_geometries([])

    pair_work = estimate_pairwise_product_work_from_owned(
        left,
        right,
        pair_unit="segment",
        output_row_count=n,
        primary_unit_name="polygon-rect-intersection-segment-pair",
    )
    output_coordinate_capacity = n * _MAX_INPUT_VERTS
    selection = plan_dispatch_selection(
        kernel_name="polygon_rect_intersection",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        work_estimate=PhysicalWorkEstimate(
            row_count=n,
            coordinate_count=pair_work.coordinate_count,
            segment_count=pair_work.segment_count,
            segment_pair_count=pair_work.segment_pair_count,
            part_count=pair_work.part_count,
            ring_count=pair_work.ring_count,
            output_row_count=n,
            output_byte_count=output_coordinate_capacity * 16,
            temporary_byte_count=output_coordinate_capacity * 32,
            primary_unit_count=max(
                pair_work.dispatch_unit_count(),
                output_coordinate_capacity,
            ),
            primary_unit_name="polygon-rect-intersection-segment-pair",
        ),
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
            detail=(f"rows={n}, precision={precision_plan.compute_precision.value}"),
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
