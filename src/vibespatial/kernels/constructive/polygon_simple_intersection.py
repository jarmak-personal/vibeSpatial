"""Validated bounded simple-polygon intersection carrier.

This module is intentionally narrower than the full overlay graph.  It handles
aligned single-ring polygon pairs by collecting boundary/inside vertices,
building one candidate ring, validating that every candidate edge stays inside
both source polygons, and returning a device row-aligned polygon carrier plus a
device support mask.  Rows that fail the validation stay unsupported so callers
can route them to the exact topology carrier without changing semantics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vibespatial.constructive.polygon_intersection_output import (
    build_device_backed_polygon_intersection_output,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.device_functions.constructed_orientation import (
    CONSTRUCTED_ORIENTATION_DEVICE,
)
from vibespatial.cuda.device_functions.intersection_point import (
    INTERSECTION_POINT_DEVICE,
)
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.segment_crossing import SEGMENT_CROSSING_DEVICE
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionPlan

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

logger = logging.getLogger(__name__)

_MAX_INPUT_VERTS = 64
_MAX_CANDIDATE_VERTS = 192
_KERNEL_NAMES = (
    "polygon_simple_intersection_count",
    "polygon_simple_intersection_scatter",
)

_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + SEGMENT_CROSSING_DEVICE
    + INTERSECTION_POINT_DEVICE
    + CONSTRUCTED_ORIENTATION_DEVICE
    + (
        "#define MAX_INPUT_VERTS "
        + str(_MAX_INPUT_VERTS)
        + "\n#define MAX_CANDIDATE_VERTS "
        + str(_MAX_CANDIDATE_VERTS)
        + r"""

__device__ __forceinline__ double ps_abs(double value) {
    return value < 0.0 ? -value : value;
}

__device__ __forceinline__ double ps_max(double a, double b) {
    return a > b ? a : b;
}

__device__ __forceinline__ int ps_same_point(
    double ax, double ay, double bx, double by
) {
    return ax == bx && ay == by;
}

__device__ __forceinline__ int ps_constructed_orient(
    double ax, double ay,
    double bx, double by,
    double px, double py
) {
    return vs_constructed_orient(ax, ay, bx, by, px, py);
}

__device__ __forceinline__ int ps_on_segment(
    double px, double py,
    double ax, double ay,
    double bx, double by
) {
    if (ps_constructed_orient(ax, ay, bx, by, px, py) != 0) {
        return 0;
    }
    const double coordinate_scale = ps_max(
        1.0,
        ps_max(
            ps_max(ps_abs(ax), ps_abs(ay)),
            ps_max(ps_abs(bx), ps_abs(by))
        )
    );
    const double coordinate_error =
        8.0 * 2.2204460492503131e-16 * coordinate_scale;
    return (
        px >= fmin(ax, bx) - coordinate_error &&
        px <= fmax(ax, bx) + coordinate_error &&
        py >= fmin(ay, by) - coordinate_error &&
        py <= fmax(ay, by) + coordinate_error
    );
}

__device__ int ps_strip_closed_ring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int end
) {
    int n = end - start;
    if (n > 1 && ps_same_point(x[start], y[start], x[end - 1], y[end - 1])) {
        n -= 1;
    }
    return n;
}

__device__ int ps_point_in_ring_inclusive(
    double px,
    double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int n
) {
    int inside = 0;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const double xi = x[start + i];
        const double yi = y[start + i];
        const double xj = x[start + j];
        const double yj = y[start + j];
        if (ps_on_segment(px, py, xj, yj, xi, yi)) {
            return 1;
        }
        const int crosses = ((yi > py) != (yj > py));
        if (crosses) {
            const double x_at_y = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if (px < x_at_y) {
                inside = !inside;
            }
        }
    }
    return inside;
}

__device__ int ps_point_on_ring_exact(
    double px,
    double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int n
) {
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const double ax = x[start + j];
        const double ay = y[start + j];
        const double bx = x[start + i];
        const double by = y[start + i];
        if (
            vs_orient2d(ax, ay, bx, by, px, py) == 0 &&
            vs_point_on_segment_collinear(px, py, ax, ay, bx, by)
        ) {
            return 1;
        }
    }
    return 0;
}

__device__ int ps_ring_strictly_inside_ring(
    const double* __restrict__ subject_x,
    const double* __restrict__ subject_y,
    int subject_start,
    int subject_n,
    const double* __restrict__ container_x,
    const double* __restrict__ container_y,
    int container_start,
    int container_n
) {
    for (int i = 0; i < subject_n; ++i) {
        const double px = subject_x[subject_start + i];
        const double py = subject_y[subject_start + i];
        if (
            ps_point_on_ring_exact(
                px, py,
                container_x, container_y, container_start, container_n
            ) ||
            !ps_point_in_ring_inclusive(
                px, py,
                container_x, container_y, container_start, container_n
            )
        ) {
            return 0;
        }
    }
    for (int si = 0, sj = subject_n - 1; si < subject_n; sj = si++) {
        const double sax = subject_x[subject_start + sj];
        const double say = subject_y[subject_start + sj];
        const double sbx = subject_x[subject_start + si];
        const double sby = subject_y[subject_start + si];
        for (int ci = 0, cj = container_n - 1; ci < container_n; cj = ci++) {
            const double cax = container_x[container_start + cj];
            const double cay = container_y[container_start + cj];
            const double cbx = container_x[container_start + ci];
            const double cby = container_y[container_start + ci];
            if (
                vs_segments_properly_cross(
                    sax, say, sbx, sby,
                    cax, cay, cbx, cby
                ) ||
                (
                    vs_orient2d(sax, say, sbx, sby, cax, cay) == 0 &&
                    vs_point_on_segment_collinear(cax, cay, sax, say, sbx, sby)
                ) ||
                (
                    vs_orient2d(sax, say, sbx, sby, cbx, cby) == 0 &&
                    vs_point_on_segment_collinear(cbx, cby, sax, say, sbx, sby)
                )
            ) {
                return 0;
            }
        }
    }
    return 1;
}

__device__ int ps_append_unique(
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int count,
    double x,
    double y
) {
    for (int i = 0; i < count; ++i) {
        if (ps_same_point(out_x[i], out_y[i], x, y)) {
            return count;
        }
    }
    if (count >= MAX_CANDIDATE_VERTS) {
        return -1;
    }
    out_x[count] = x;
    out_y[count] = y;
    return count + 1;
}

__device__ int ps_segment_intersections(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    double dx, double dy,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int count
) {
    const int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
    const int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
    const int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
    const int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);

    if (o1 == 0 && ps_on_segment(cx, cy, ax, ay, bx, by)) {
        count = ps_append_unique(out_x, out_y, count, cx, cy);
        if (count < 0) return -1;
    }
    if (o2 == 0 && ps_on_segment(dx, dy, ax, ay, bx, by)) {
        count = ps_append_unique(out_x, out_y, count, dx, dy);
        if (count < 0) return -1;
    }
    if (o3 == 0 && ps_on_segment(ax, ay, cx, cy, dx, dy)) {
        count = ps_append_unique(out_x, out_y, count, ax, ay);
        if (count < 0) return -1;
    }
    if (o4 == 0 && ps_on_segment(bx, by, cx, cy, dx, dy)) {
        count = ps_append_unique(out_x, out_y, count, bx, by);
        if (count < 0) return -1;
    }

    if (
        ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
        ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))
    ) {
        double ix;
        double iy;
        if (!vs_proper_intersection_point_dd(
            ax, ay, bx, by, cx, cy, dx, dy, &ix, &iy
        )) {
            return count;
        }
        count = ps_append_unique(out_x, out_y, count, ix, iy);
        if (count < 0) return -1;
    }
    return count;
}

__device__ int ps_source_pair_has_uncertain_incidence(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    int left_start,
    int left_n,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    int right_start,
    int right_n
) {
    for (int li = 0; li < left_n; ++li) {
        const double px = left_x[left_start + li];
        const double py = left_y[left_start + li];
        for (int ri = 0; ri < right_n; ++ri) {
            const int rj = ri == 0 ? right_n - 1 : ri - 1;
            if (vs_source_incidence_is_uncertain(
                    right_x[right_start + rj], right_y[right_start + rj],
                    right_x[right_start + ri], right_y[right_start + ri],
                    px, py)) {
                return 1;
            }
        }
    }
    for (int ri = 0; ri < right_n; ++ri) {
        const double px = right_x[right_start + ri];
        const double py = right_y[right_start + ri];
        for (int li = 0; li < left_n; ++li) {
            const int lj = li == 0 ? left_n - 1 : li - 1;
            if (vs_source_incidence_is_uncertain(
                    left_x[left_start + lj], left_y[left_start + lj],
                    left_x[left_start + li], left_y[left_start + li],
                    px, py)) {
                return 1;
            }
        }
    }
    return 0;
}

__device__ double ps_ring_area(
    const double* __restrict__ xs,
    const double* __restrict__ ys,
    int n
) {
    const double origin_x = xs[0];
    const double origin_y = ys[0];
    double area = 0.0;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        area +=
            (xs[j] - origin_x) * (ys[i] - origin_y) -
            (xs[i] - origin_x) * (ys[j] - origin_y);
    }
    return 0.5 * area;
}

__device__ int ps_edge_crosses_ring_interior(
    double ax, double ay,
    double bx, double by,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int n
) {
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const double cx = x[start + j];
        const double cy = y[start + j];
        const double dx = x[start + i];
        const double dy = y[start + i];
        const int o1 = ps_constructed_orient(ax, ay, bx, by, cx, cy);
        const int o2 = ps_constructed_orient(ax, ay, bx, by, dx, dy);
        const int o3 = ps_constructed_orient(cx, cy, dx, dy, ax, ay);
        const int o4 = ps_constructed_orient(cx, cy, dx, dy, bx, by);
        if (o1 == 0 && o2 == 0) {
            continue;
        }
        if (
            ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
            ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))
        ) {
            return 1;
        }
    }
    return 0;
}

__device__ int ps_build_validated_candidate(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    int left_start,
    int left_n,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    int right_start,
    int right_n,
    double* __restrict__ cand_x,
    double* __restrict__ cand_y
) {
    if (ps_ring_strictly_inside_ring(
            left_x, left_y, left_start, left_n,
            right_x, right_y, right_start, right_n)) {
        for (int i = 0; i < left_n; ++i) {
            cand_x[i] = left_x[left_start + i];
            cand_y[i] = left_y[left_start + i];
        }
        return left_n;
    }
    if (ps_ring_strictly_inside_ring(
            right_x, right_y, right_start, right_n,
            left_x, left_y, left_start, left_n)) {
        for (int i = 0; i < right_n; ++i) {
            cand_x[i] = right_x[right_start + i];
            cand_y[i] = right_y[right_start + i];
        }
        return right_n;
    }
    if (ps_source_pair_has_uncertain_incidence(
            left_x, left_y, left_start, left_n,
            right_x, right_y, right_start, right_n)) {
        return -1;
    }
    int count = 0;
    for (int i = 0; i < left_n; ++i) {
        const double x = left_x[left_start + i];
        const double y = left_y[left_start + i];
        if (ps_point_in_ring_inclusive(x, y, right_x, right_y, right_start, right_n)) {
            count = ps_append_unique(cand_x, cand_y, count, x, y);
            if (count < 0) return -1;
        }
    }
    for (int i = 0; i < right_n; ++i) {
        const double x = right_x[right_start + i];
        const double y = right_y[right_start + i];
        if (ps_point_in_ring_inclusive(x, y, left_x, left_y, left_start, left_n)) {
            count = ps_append_unique(cand_x, cand_y, count, x, y);
            if (count < 0) return -1;
        }
    }
    for (int li = 0, lj = left_n - 1; li < left_n; lj = li++) {
        const double ax = left_x[left_start + lj];
        const double ay = left_y[left_start + lj];
        const double bx = left_x[left_start + li];
        const double by = left_y[left_start + li];
        for (int ri = 0, rj = right_n - 1; ri < right_n; rj = ri++) {
            count = ps_segment_intersections(
                ax, ay, bx, by,
                right_x[right_start + rj], right_y[right_start + rj],
                right_x[right_start + ri], right_y[right_start + ri],
                cand_x, cand_y, count
            );
            if (count < 0) return -1;
        }
    }
    if (count == 0) {
        return 0;
    }
    if (count < 3) {
        // A point or line contact is a valid intersection, but this carrier
        // can only encode polygonal output.  Route it to the exact mixed-
        // dimensional topology carrier instead of claiming an empty result.
        return -1;
    }

    const double origin_x = cand_x[0];
    const double origin_y = cand_y[0];
    double cx = 0.0;
    double cy = 0.0;
    for (int i = 0; i < count; ++i) {
        cx += cand_x[i] - origin_x;
        cy += cand_y[i] - origin_y;
    }
    cx = origin_x + cx / (double)count;
    cy = origin_y + cy / (double)count;

    for (int i = 1; i < count; ++i) {
        double tx = cand_x[i];
        double ty = cand_y[i];
        double ta = atan2(ty - cy, tx - cx);
        double td = (tx - cx) * (tx - cx) + (ty - cy) * (ty - cy);
        int j = i - 1;
        while (j >= 0) {
            const double ja = atan2(cand_y[j] - cy, cand_x[j] - cx);
            const double jd = (
                (cand_x[j] - cx) * (cand_x[j] - cx) +
                (cand_y[j] - cy) * (cand_y[j] - cy)
            );
            if (ja < ta || (ja == ta && jd <= td)) {
                break;
            }
            cand_x[j + 1] = cand_x[j];
            cand_y[j + 1] = cand_y[j];
            --j;
        }
        cand_x[j + 1] = tx;
        cand_y[j + 1] = ty;
    }

    int topology_orientation = 0;
    for (int i = 1; i + 1 < count; ++i) {
        topology_orientation = ps_constructed_orient(
            cand_x[0], cand_y[0],
            cand_x[i], cand_y[i],
            cand_x[i + 1], cand_y[i + 1]
        );
        if (topology_orientation != 0) {
            break;
        }
    }
    if (topology_orientation == 0) {
        // Intersection coordinates are constructed values.  Distinct fp64
        // payloads can still denote one collinear topological event within
        // their propagated arithmetic error.  The exact graph must decide
        // whether that event is a line, point, or empty result.
        return -1;
    }
    const double area = ps_ring_area(cand_x, cand_y, count);

    for (int i = 0, j = count - 1; i < count; j = i++) {
        const double ax = cand_x[j];
        const double ay = cand_y[j];
        const double bx = cand_x[i];
        const double by = cand_y[i];
        const double mx = 0.5 * (ax + bx);
        const double my = 0.5 * (ay + by);
        if (
            !ps_point_in_ring_inclusive(mx, my, left_x, left_y, left_start, left_n) ||
            !ps_point_in_ring_inclusive(mx, my, right_x, right_y, right_start, right_n)
        ) {
            return -1;
        }
        if (
            ps_edge_crosses_ring_interior(ax, ay, bx, by, left_x, left_y, left_start, left_n) ||
            ps_edge_crosses_ring_interior(ax, ay, bx, by, right_x, right_y, right_start, right_n)
        ) {
            return -1;
        }
    }
    if (area < 0.0 || (area == 0.0 && topology_orientation < 0)) {
        for (int i = 0; i < count / 2; ++i) {
            const int k = count - 1 - i;
            const double tx = cand_x[i];
            const double ty = cand_y[i];
            cand_x[i] = cand_x[k];
            cand_y[i] = cand_y[k];
            cand_x[k] = tx;
            cand_y[k] = ty;
        }
    }
    return count;
}

__device__ int ps_prepare_row(
    int idx,
    const unsigned char* __restrict__ left_valid,
    const signed char* __restrict__ left_tags,
    const int* __restrict__ left_family_rows,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const unsigned char* __restrict__ left_empty,
    int left_polygon_rows,
    const unsigned char* __restrict__ right_valid,
    const signed char* __restrict__ right_tags,
    const int* __restrict__ right_family_rows,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const unsigned char* __restrict__ right_empty,
    int right_polygon_rows,
    int polygon_tag,
    int* __restrict__ left_start,
    int* __restrict__ left_n,
    int* __restrict__ right_start,
    int* __restrict__ right_n
) {
    const int lrow = left_family_rows[idx];
    const int rrow = right_family_rows[idx];
    if (
        !left_valid[idx] || !right_valid[idx] ||
        ((int)left_tags[idx]) != polygon_tag ||
        ((int)right_tags[idx]) != polygon_tag ||
        lrow < 0 || rrow < 0 ||
        lrow >= left_polygon_rows ||
        rrow >= right_polygon_rows ||
        left_empty[lrow] || right_empty[rrow]
    ) {
        return 0;
    }
    const int l_geom_start = left_geom_offsets[lrow];
    const int l_geom_end = left_geom_offsets[lrow + 1];
    const int r_geom_start = right_geom_offsets[rrow];
    const int r_geom_end = right_geom_offsets[rrow + 1];
    if ((l_geom_end - l_geom_start) != 1 || (r_geom_end - r_geom_start) != 1) {
        return 0;
    }
    *left_start = left_ring_offsets[l_geom_start];
    const int left_end = left_ring_offsets[l_geom_start + 1];
    *right_start = right_ring_offsets[r_geom_start];
    const int right_end = right_ring_offsets[r_geom_start + 1];
    *left_n = ps_strip_closed_ring(left_x, left_y, *left_start, left_end);
    *right_n = ps_strip_closed_ring(right_x, right_y, *right_start, right_end);
    if (
        *left_n < 3 || *right_n < 3 ||
        *left_n > MAX_INPUT_VERTS ||
        *right_n > MAX_INPUT_VERTS
    ) {
        return 0;
    }
    return 1;
}

extern "C" __global__ __launch_bounds__(128, 2) void polygon_simple_intersection_count(
    const unsigned char* __restrict__ left_valid,
    const signed char* __restrict__ left_tags,
    const int* __restrict__ left_family_rows,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const unsigned char* __restrict__ left_empty,
    int left_polygon_rows,
    const unsigned char* __restrict__ right_valid,
    const signed char* __restrict__ right_tags,
    const int* __restrict__ right_family_rows,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const unsigned char* __restrict__ right_empty,
    int right_polygon_rows,
    int polygon_tag,
    int n,
    int* __restrict__ counts,
    unsigned char* __restrict__ valid,
    unsigned char* __restrict__ supported
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    counts[idx] = 0;
    valid[idx] = 0;
    supported[idx] = 0;

    int left_start = 0, left_n = 0, right_start = 0, right_n = 0;
    if (!ps_prepare_row(
        idx,
        left_valid, left_tags, left_family_rows,
        left_x, left_y, left_ring_offsets, left_geom_offsets, left_empty, left_polygon_rows,
        right_valid, right_tags, right_family_rows,
        right_x, right_y, right_ring_offsets, right_geom_offsets, right_empty, right_polygon_rows,
        polygon_tag,
        &left_start, &left_n, &right_start, &right_n
    )) {
        return;
    }
    double cand_x[MAX_CANDIDATE_VERTS];
    double cand_y[MAX_CANDIDATE_VERTS];
    const int out_n = ps_build_validated_candidate(
        left_x, left_y, left_start, left_n,
        right_x, right_y, right_start, right_n,
        cand_x, cand_y
    );
    if (out_n < 0) {
        return;
    }
    supported[idx] = 1;
    if (out_n >= 3) {
        valid[idx] = 1;
        counts[idx] = out_n + 1;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void polygon_simple_intersection_scatter(
    const unsigned char* __restrict__ left_valid,
    const signed char* __restrict__ left_tags,
    const int* __restrict__ left_family_rows,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const unsigned char* __restrict__ left_empty,
    int left_polygon_rows,
    const unsigned char* __restrict__ right_valid,
    const signed char* __restrict__ right_tags,
    const int* __restrict__ right_family_rows,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const unsigned char* __restrict__ right_empty,
    int right_polygon_rows,
    int polygon_tag,
    int n,
    const int* __restrict__ offsets,
    const unsigned char* __restrict__ valid,
    double* __restrict__ out_x,
    double* __restrict__ out_y
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || !valid[idx]) return;

    int left_start = 0, left_n = 0, right_start = 0, right_n = 0;
    if (!ps_prepare_row(
        idx,
        left_valid, left_tags, left_family_rows,
        left_x, left_y, left_ring_offsets, left_geom_offsets, left_empty, left_polygon_rows,
        right_valid, right_tags, right_family_rows,
        right_x, right_y, right_ring_offsets, right_geom_offsets, right_empty, right_polygon_rows,
        polygon_tag,
        &left_start, &left_n, &right_start, &right_n
    )) {
        return;
    }
    double cand_x[MAX_CANDIDATE_VERTS];
    double cand_y[MAX_CANDIDATE_VERTS];
    const int out_n = ps_build_validated_candidate(
        left_x, left_y, left_start, left_n,
        right_x, right_y, right_start, right_n,
        cand_x, cand_y
    );
    if (out_n < 3) {
        return;
    }
    const int offset = offsets[idx];
    for (int i = 0; i < out_n; ++i) {
        out_x[offset + i] = cand_x[i];
        out_y[offset + i] = cand_y[i];
    }
    out_x[offset + out_n] = cand_x[0];
    out_y[offset + out_n] = cand_y[0];
}
"""
    )
)

request_nvrtc_warmup(
    [
        ("polygon-simple-intersection", _KERNEL_SOURCE, _KERNEL_NAMES),
    ]
)


def _polygon_simple_intersection_kernels():
    return compile_kernel_group(
        "polygon-simple-intersection",
        _KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


def _polygon_family_buffers(owned: OwnedGeometryArray):
    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None or polygon.ring_offsets is None:
        return None, state
    if int(polygon.geometry_offsets.size) <= 1:
        return None, state
    return polygon, state


def polygon_simple_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> tuple[OwnedGeometryArray, object] | None:
    """Return row-aligned simple-polygon intersections and support mask.

    Physical shape: aligned pairwise single-ring polygon rows with bounded
    vertex counts.  Work units are source vertices, source segment pairs,
    candidate output vertices, and output bytes.  The native output is a
    row-aligned ``OwnedGeometryArray`` plus a device boolean support mask; rows
    with unsupported topology are invalid in the result and ``False`` in the
    mask so callers can route exactly those rows to the full overlay graph.
    """
    if cp is None:
        return None
    if left.row_count != right.row_count:
        return None
    n = int(left.row_count)
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    selection = RuntimeSelection(
        requested=requested,
        selected=ExecutionMode.GPU,
        reason="GPU bounded simple-polygon intersection selected",
    )
    if n == 0:
        return None

    if left.residency is not Residency.DEVICE:
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_simple_intersection selected GPU left geometry",
        )
    if right.residency is not Residency.DEVICE:
        right.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon_simple_intersection selected GPU right geometry",
        )

    left_poly, left_state = _polygon_family_buffers(left)
    right_poly, right_state = _polygon_family_buffers(right)
    if left_poly is None or right_poly is None:
        return None
    left_polygon_rows = int(left_poly.geometry_offsets.size) - 1
    right_polygon_rows = int(right_poly.geometry_offsets.size) - 1
    if left_polygon_rows <= 0 or right_polygon_rows <= 0:
        return None

    runtime = get_cuda_runtime()
    kernels = _polygon_simple_intersection_kernels()
    ptr = runtime.pointer

    d_counts = runtime.allocate((n,), cp.int32, zero=True)
    d_valid = runtime.allocate((n,), cp.bool_, zero=True)
    d_supported = runtime.allocate((n,), cp.bool_, zero=True)

    base_params = (
        ptr(cp.asarray(left_state.validity, dtype=cp.bool_)),
        ptr(cp.asarray(left_state.tags, dtype=cp.int8)),
        ptr(cp.asarray(left_state.family_row_offsets, dtype=cp.int32)),
        ptr(left_poly.x),
        ptr(left_poly.y),
        ptr(left_poly.ring_offsets),
        ptr(left_poly.geometry_offsets),
        ptr(left_poly.empty_mask),
        left_polygon_rows,
        ptr(cp.asarray(right_state.validity, dtype=cp.bool_)),
        ptr(cp.asarray(right_state.tags, dtype=cp.int8)),
        ptr(cp.asarray(right_state.family_row_offsets, dtype=cp.int32)),
        ptr(right_poly.x),
        ptr(right_poly.y),
        ptr(right_poly.ring_offsets),
        ptr(right_poly.geometry_offsets),
        ptr(right_poly.empty_mask),
        right_polygon_rows,
        int(FAMILY_TAGS[GeometryFamily.POLYGON]),
        n,
    )
    base_types = (
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
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
    )
    grid, block = runtime.launch_config(
        kernels["polygon_simple_intersection_count"],
        n,
    )
    runtime.launch(
        kernels["polygon_simple_intersection_count"],
        grid=grid,
        block=block,
        params=(
            base_params + (ptr(d_counts), ptr(d_valid), ptr(d_supported)),
            base_types + (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR),
        ),
    )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    output_capacity = n * (_MAX_CANDIDATE_VERTS + 1)
    d_out_x = runtime.allocate((output_capacity,), cp.float64)
    d_out_y = runtime.allocate((output_capacity,), cp.float64)

    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_simple_intersection_scatter"],
        n,
    )
    runtime.launch(
        kernels["polygon_simple_intersection_scatter"],
        grid=scatter_grid,
        block=scatter_block,
        params=(
            base_params + (ptr(d_offsets), ptr(d_valid), ptr(d_out_x), ptr(d_out_y)),
            base_types + (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR),
        ),
    )

    d_ring_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_ring_offsets[:n] = d_offsets
    d_ring_offsets[n] = d_offsets[-1] + d_counts[-1]
    result = build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=d_valid,
        ring_offsets=d_ring_offsets,
        runtime_selection=selection,
    )
    result._simple_polygon_intersection_supported = d_supported
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_simple_intersection",
        operation="polygon_intersection",
        implementation="validated_simple_polygon_intersection_gpu",
        reason=(
            "bounded single-ring polygon pairs were assembled and validated "
            "as a row-aligned native carrier before exact topology fallback"
        ),
        detail=(
            f"rows={n}; max_input_vertices={_MAX_INPUT_VERTS}; "
            "workload_shape=aligned_pairwise_simple_polygon_candidate_ring"
        ),
        requested=requested,
        selected=ExecutionMode.GPU,
    )
    return result, d_supported


@register_kernel_variant(
    "polygon_simple_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=False,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "intersection", "simple-polygon"),
)
def _polygon_simple_intersection_gpu_variant(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    del runtime_selection, precision_plan
    result = polygon_simple_intersection(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result is None:
        raise RuntimeError("polygon_simple_intersection declined")
    return result[0]
