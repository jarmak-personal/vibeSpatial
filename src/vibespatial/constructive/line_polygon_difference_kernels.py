"""CUDA source for exact boundary refinement of line/polygon intervals."""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE

_LINE_POLYGON_INTERVAL_KERNEL_SOURCE = ORIENT2D_DEVICE + r"""
__device__ inline bool vs_interval_on_ring_edge(
    double ax, double ay,
    double bx, double by,
    const double* x,
    const double* y,
    const int* ring_offsets,
    int ring_start,
    int ring_end
) {
    for (int ring = ring_start; ring < ring_end; ++ring) {
        const int cs = ring_offsets[ring];
        const int ce = ring_offsets[ring + 1];
        for (int coord = cs + 1; coord < ce; ++coord) {
            const double ex0 = x[coord - 1];
            const double ey0 = y[coord - 1];
            const double ex1 = x[coord];
            const double ey1 = y[coord];
            if (vs_orient2d(ex0, ey0, ex1, ey1, ax, ay) != 0
                || vs_orient2d(ex0, ey0, ex1, ey1, bx, by) != 0) {
                continue;
            }
            const double minx = fmin(ex0, ex1);
            const double maxx = fmax(ex0, ex1);
            const double miny = fmin(ey0, ey1);
            const double maxy = fmax(ey0, ey1);
            if (ax >= minx && ax <= maxx && ay >= miny && ay <= maxy
                && bx >= minx && bx <= maxx && by >= miny && by <= maxy) {
                return true;
            }
        }
    }
    return false;
}

extern "C" __global__ void __launch_bounds__(256, 4)
refine_line_polygon_boundary_intervals(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const double* __restrict__ dst_x,
    const double* __restrict__ dst_y,
    const unsigned char* __restrict__ active,
    const unsigned char* __restrict__ right_validity,
    const signed char* __restrict__ right_tags,
    const int* __restrict__ right_family_rows,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    unsigned char* __restrict__ out,
    int interval_count,
    int family_tag,
    int is_multipolygon
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int interval = lane; interval < interval_count; interval += stride) {
        if (!active[interval] || !right_validity[interval]
            || right_tags[interval] != (signed char)family_tag) {
            continue;
        }
        const int family_row = right_family_rows[interval];
        if (family_row < 0 || empty_mask[family_row]) continue;

        bool covered = false;
        if (is_multipolygon) {
            const int polygon_start = geometry_offsets[family_row];
            const int polygon_end = geometry_offsets[family_row + 1];
            for (int polygon = polygon_start; polygon < polygon_end && !covered; ++polygon) {
                covered = vs_interval_on_ring_edge(
                    src_x[interval], src_y[interval],
                    dst_x[interval], dst_y[interval],
                    polygon_x, polygon_y, ring_offsets,
                    part_offsets[polygon], part_offsets[polygon + 1]);
            }
        } else {
            covered = vs_interval_on_ring_edge(
                src_x[interval], src_y[interval],
                dst_x[interval], dst_y[interval],
                polygon_x, polygon_y, ring_offsets,
                geometry_offsets[family_row], geometry_offsets[family_row + 1]);
        }
        if (covered) out[interval] = 1u;
    }
}
"""

_LINE_POLYGON_INTERVAL_KERNEL_NAMES = (
    "refine_line_polygon_boundary_intervals",
)

__all__ = [
    "_LINE_POLYGON_INTERVAL_KERNEL_NAMES",
    "_LINE_POLYGON_INTERVAL_KERNEL_SOURCE",
]
