"""NVRTC source for grouped mixed-dimensional constructive reduction."""

from __future__ import annotations

from vibespatial.cuda.device_functions.point_on_segment import POINT_ON_SEGMENT_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

_GROUPED_MIXED_UNION_KERNEL_NAMES = ("grouped_points_on_atomic_edges",)

_GROUPED_MIXED_UNION_KERNEL_SOURCE = (
    SPATIAL_TOLERANCE_PREAMBLE
    + POINT_ON_SEGMENT_DEVICE.format()
    + r"""
extern "C" __global__ void __launch_bounds__(256, 4)
grouped_points_on_atomic_edges(
    const double* __restrict__ point_x,
    const double* __restrict__ point_y,
    const int* __restrict__ point_group_rows,
    const unsigned char* __restrict__ point_active,
    const long long* __restrict__ edge_group_offsets,
    const double* __restrict__ edge_x0,
    const double* __restrict__ edge_y0,
    const double* __restrict__ edge_x1,
    const double* __restrict__ edge_y1,
    unsigned char* __restrict__ out_covered,
    const int point_capacity
) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int point = thread >> 5;
    const int lane = thread & 31;
    if (point >= point_capacity) return;

    if (!point_active[point]) {
        if (lane == 0) out_covered[point] = 0;
        return;
    }

    const int group = point_group_rows[point];
    const long long start = edge_group_offsets[group];
    const long long stop = edge_group_offsets[group + 1];
    const double px = point_x[point];
    const double py = point_y[point];
    unsigned int covered = 0;

    for (long long base = start; base < stop; base += 32) {
        const long long edge = base + lane;
        bool lane_covered = false;
        if (edge < stop) {
            lane_covered = vs_point_on_segment(
                px,
                py,
                edge_x0[edge],
                edge_y0[edge],
                edge_x1[edge],
                edge_y1[edge],
                VS_SPATIAL_EPSILON
            );
        }
        covered |= __ballot_sync(0xFFFFFFFFu, lane_covered);
        if (covered != 0) break;
    }

    if (lane == 0) out_covered[point] = covered != 0;
}
"""
)

__all__ = [
    "_GROUPED_MIXED_UNION_KERNEL_NAMES",
    "_GROUPED_MIXED_UNION_KERNEL_SOURCE",
]
