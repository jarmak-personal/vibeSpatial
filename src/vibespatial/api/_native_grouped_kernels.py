"""CUDA source for bounded native grouped reductions.

These reducers are segment-shaped: their temporary footprint is the output
vector, independent of source-row cardinality.  They exist for NativeGrouped
pipelines where histogram/bincount lowering has the wrong memory shape.  One
warp cooperates on each segment so a skewed group cannot serialize millions
of values through one lane.
"""

from __future__ import annotations

_NATIVE_GROUPED_BOUNDED_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
segmented_u8_min_bounded_warp(
    const unsigned char* __restrict__ values,
    const long long* __restrict__ offsets,
    const long long* __restrict__ sorted_order,
    const int* __restrict__ group_ids,
    const unsigned char* __restrict__ group_validity,
    unsigned char* __restrict__ out,
    unsigned int* __restrict__ error_flag,
    long long ordered_row_count,
    long long value_row_count,
    int observed_group_count,
    int output_group_count,
    int identity_value
) {
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int group = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (group >= observed_group_count) return;
    const int output_group = group_ids[group];
    const long long start = offsets[group];
    const long long end = offsets[group + 1];
    if (start < 0 || end < start || end > ordered_row_count
        || (group == 0 && start != 0)
        || (group == observed_group_count - 1 && end != ordered_row_count)
        || output_group < 0 || output_group >= output_group_count) {
        if (lane == 0) atomicOr(error_flag, 1u);
        return;
    }
    if (group_validity && !group_validity[group]) {
        if (lane == 0) out[output_group] = 0u;
        return;
    }
    const unsigned int active = __activemask();
    for (long long position = start + lane; ; position += 32) {
        const bool present = position < end;
        const long long row = present
            ? (sorted_order ? sorted_order[position] : position)
            : 0;
        const bool invalid_row = present && (row < 0 || row >= value_row_count);
        if (__ballot_sync(active, invalid_row) != 0u) {
            if (lane == 0) {
                out[output_group] = 0u;
                atomicOr(error_flag, 1u);
            }
            return;
        }
        const bool failed = present && !invalid_row && values[row] == 0u;
        if (__ballot_sync(active, failed) != 0u) {
            if (lane == 0) out[output_group] = 0u;
            return;
        }
        if (__ballot_sync(active, position + 32 < end) == 0u) break;
    }
    if (lane == 0) out[output_group] = (unsigned char)identity_value;
}

extern "C" __global__ void __launch_bounds__(256, 4)
map_grouped_bool_rows_bounded(
    const unsigned char* __restrict__ values,
    const unsigned char* __restrict__ left_validity,
    const signed char* __restrict__ left_tags,
    const int* __restrict__ family_rows,
    const unsigned char* __restrict__ right_validity,
    unsigned char* __restrict__ out,
    int row_count,
    int family_tag
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int row = lane; row < row_count; row += stride) {
        const int family_row = family_rows[row];
        out[row] = (
            left_validity[row]
            && right_validity[row]
            && left_tags[row] == (signed char)family_tag
            && family_row >= 0
            && values[family_row]
        ) ? 1u : 0u;
    }
}
"""

_NATIVE_GROUPED_BOUNDED_KERNEL_NAMES = (
    "segmented_u8_min_bounded_warp",
    "map_grouped_bool_rows_bounded",
)


__all__ = [
    "_NATIVE_GROUPED_BOUNDED_KERNEL_NAMES",
    "_NATIVE_GROUPED_BOUNDED_KERNEL_SOURCE",
]
