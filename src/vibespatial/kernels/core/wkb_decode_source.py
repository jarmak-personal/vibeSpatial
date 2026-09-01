"""NVRTC source for byte-authoritative, endian-aware WKB decode."""

from __future__ import annotations

_WKB_DECODE_KERNEL_SOURCE = r"""

__device__ __forceinline__ unsigned int read_u32_ordered(
    const unsigned char* src, const unsigned char byte_order
) {
    if (byte_order == 1) {
        return (unsigned int)src[0]
             | ((unsigned int)src[1] << 8)
             | ((unsigned int)src[2] << 16)
             | ((unsigned int)src[3] << 24);
    }
    return ((unsigned int)src[0] << 24)
         | ((unsigned int)src[1] << 16)
         | ((unsigned int)src[2] << 8)
         | (unsigned int)src[3];
}

__device__ __forceinline__ double read_f64_ordered(
    const unsigned char* src, const unsigned char byte_order
) {
    unsigned long long bits = 0ULL;
    if (byte_order == 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) bits |= ((unsigned long long)src[i]) << (8 * i);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) bits |= ((unsigned long long)src[i]) << (8 * (7 - i));
    }
    return __longlong_as_double((long long)bits);
}

__device__ __forceinline__ bool is_nan_bits(const double value) {
    const unsigned long long bits = (unsigned long long)__double_as_longlong(value);
    return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL)
        && ((bits & 0x000FFFFFFFFFFFFFULL) != 0ULL);
}

__device__ __forceinline__ unsigned char classify_type(
    const unsigned int type_id, signed char* family
) {
    const bool ewkb_z = (type_id & 0x80000000U) != 0U;
    const bool ewkb_m = (type_id & 0x40000000U) != 0U;
    const bool ewkb_srid = (type_id & 0x20000000U) != 0U;
    const unsigned int ewkb_base = type_id & 0x1FFFFFFFU;
    const bool iso_dimensional = type_id >= 1000U && type_id < 4000U;
    const unsigned int iso_base = iso_dimensional ? type_id % 1000U : type_id;
    const unsigned int candidate = ewkb_srid ? ewkb_base : iso_base;
    *family = -1;
    if (ewkb_z || ewkb_m || iso_dimensional) return 14;
    if (candidate == 7U) return 15;
    if (ewkb_srid && candidate >= 1U && candidate <= 7U) return 13;
    if (candidate < 1U || candidate > 6U) return 16;
    *family = (signed char)(candidate - 1U);
    return 0;
}

__device__ __forceinline__ bool take_bytes(
    unsigned long long* cursor,
    const unsigned long long end,
    const unsigned long long byte_count
) {
    if (*cursor > end || byte_count > end - *cursor) return false;
    *cursor += byte_count;
    return true;
}

extern "C" {

__global__ void __launch_bounds__(256, 2) wkb_plan_summary(
    const unsigned char* __restrict__ statuses,
    const signed char* __restrict__ family_tags,
    const unsigned char* __restrict__ native_mask,
    const int* __restrict__ part_counts,
    const int* __restrict__ ring_counts,
    const int* __restrict__ coordinate_counts,
    unsigned long long* __restrict__ aggregate,
    const int count
) {
    __shared__ unsigned long long bins[47];
    for (int index = (int)threadIdx.x; index < 47; index += (int)blockDim.x) {
        bins[index] = 0ULL;
    }
    __syncthreads();

    const int stride = (int)(blockDim.x * gridDim.x);
    for (int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         row < count; row += stride) {
        const unsigned int status = (unsigned int)statuses[row];
        if (status < 20U) atomicAdd(&bins[status], 1ULL);
        if (native_mask[row]) {
            const int family = (int)family_tags[row];
            if (family >= 0 && family < 6) {
                atomicAdd(&bins[20 + family], 1ULL);
                atomicAdd(
                    &bins[29 + family],
                    (unsigned long long)(unsigned int)part_counts[row]
                );
                atomicAdd(
                    &bins[35 + family],
                    (unsigned long long)(unsigned int)ring_counts[row]
                );
                atomicAdd(
                    &bins[41 + family],
                    (unsigned long long)(unsigned int)coordinate_counts[row]
                );
            }
        }
        atomicAdd(&bins[26], (unsigned long long)(unsigned int)part_counts[row]);
        atomicAdd(&bins[27], (unsigned long long)(unsigned int)ring_counts[row]);
        atomicAdd(&bins[28], (unsigned long long)(unsigned int)coordinate_counts[row]);
    }
    __syncthreads();
    for (int index = (int)threadIdx.x; index < 47; index += (int)blockDim.x) {
        atomicAdd(&aggregate[index], bins[index]);
    }
}

__global__ void __launch_bounds__(256, 2) wkb_structural_scan(
    const unsigned char* __restrict__ payload,
    const long long payload_size,
    const long long* __restrict__ record_offsets,
    const unsigned char* __restrict__ validity,
    const int use_validity,
    unsigned char* __restrict__ statuses,
    signed char* __restrict__ family_tags,
    unsigned char* __restrict__ root_byte_orders,
    unsigned char* __restrict__ empty_flags,
    int* __restrict__ primary_counts,
    int* __restrict__ part_counts,
    int* __restrict__ ring_counts,
    int* __restrict__ coordinate_counts,
    const int count
) {
    const int stride = (int)(blockDim.x * gridDim.x);
    for (int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         row < count; row += stride) {
        statuses[row] = 11;
        family_tags[row] = -1;
        root_byte_orders[row] = 255;
        empty_flags[row] = 0;
        primary_counts[row] = 0;
        part_counts[row] = 0;
        ring_counts[row] = 0;
        coordinate_counts[row] = 0;

        if (use_validity && !validity[row]) {
            statuses[row] = 0;
            continue;
        }
        const long long signed_start = record_offsets[row];
        const long long signed_end = record_offsets[row + 1];
        if (
            signed_start < 0 || signed_end < signed_start
            || signed_end > payload_size
        ) continue;
        const unsigned long long start = (unsigned long long)signed_start;
        const unsigned long long end = (unsigned long long)signed_end;
        if (end - start < 5ULL) continue;

        unsigned long long cursor = start;
        const unsigned char root_order = payload[cursor++];
        if (root_order > 1) {
            statuses[row] = 10;
            continue;
        }
        root_byte_orders[row] = root_order;
        const unsigned int type_id = read_u32_ordered(payload + cursor, root_order);
        cursor += 4ULL;
        signed char family = -1;
        const unsigned char type_status = classify_type(type_id, &family);
        if (type_status != 0) {
            statuses[row] = type_status;
            continue;
        }
        family_tags[row] = family;

        unsigned long long parts = 0ULL;
        unsigned long long rings = 0ULL;
        unsigned long long coords = 0ULL;
        unsigned int primary = 1U;
        bool mixed_endian = false;
        bool malformed = false;
        bool overflow = false;
        bool family_mismatch = false;
        bool semantic_invalid = false;

        if (family == 0) {
            if (!take_bytes(&cursor, end, 16ULL)) malformed = true;
            else {
                const double x = read_f64_ordered(payload + start + 5ULL, root_order);
                const double y = read_f64_ordered(payload + start + 13ULL, root_order);
                const bool empty = is_nan_bits(x) && is_nan_bits(y);
                empty_flags[row] = empty ? 1 : 0;
                coords = empty ? 0ULL : 1ULL;
            }
        } else {
            if (!take_bytes(&cursor, end, 4ULL)) malformed = true;
            else {
                primary = read_u32_ordered(payload + start + 5ULL, root_order);
                if (primary > 0x7FFFFFFFU) {
                    statuses[row] = 12;
                    continue;
                }
            }
        }

        if (!malformed && !overflow && family == 1) {
            coords = (unsigned long long)primary;
            if ((unsigned long long)primary > (end - cursor) / 16ULL) malformed = true;
            else cursor += (unsigned long long)primary * 16ULL;
            semantic_invalid = primary == 1U;
            empty_flags[row] = primary == 0U ? 1 : 0;
        } else if (!malformed && !overflow && family == 2) {
            rings = (unsigned long long)primary;
            for (unsigned int ring = 0; ring < primary && !malformed && !overflow; ++ring) {
                if (!take_bytes(&cursor, end, 4ULL)) {
                    malformed = true;
                    break;
                }
                const unsigned int npts = read_u32_ordered(payload + cursor - 4ULL, root_order);
                if (npts > 0x7FFFFFFFU) {
                    overflow = true;
                    break;
                }
                if ((unsigned long long)npts > (end - cursor) / 16ULL) {
                    malformed = true;
                    break;
                }
                cursor += (unsigned long long)npts * 16ULL;
                coords += (unsigned long long)npts;
                if (coords > 0x7FFFFFFFULL) overflow = true;
            }
            empty_flags[row] = primary == 0U ? 1 : 0;
        } else if (!malformed && !overflow && family >= 3) {
            parts = (unsigned long long)primary;
            const signed char expected_child = (signed char)(family - 3);
            for (unsigned int part = 0; part < primary && !malformed && !overflow; ++part) {
                if (end - cursor < 5ULL) {
                    malformed = true;
                    break;
                }
                const unsigned char child_order = payload[cursor];
                if (child_order > 1) {
                    statuses[row] = 10;
                    malformed = true;
                    break;
                }
                if (child_order != root_order) mixed_endian = true;
                const unsigned int child_type = read_u32_ordered(payload + cursor + 1ULL, child_order);
                signed char child_family = -1;
                const unsigned char child_status = classify_type(child_type, &child_family);
                if (child_status != 0) {
                    statuses[row] = child_status;
                    malformed = true;
                    break;
                }
                if (child_family != expected_child) {
                    family_mismatch = true;
                    break;
                }
                cursor += 5ULL;
                if (family == 3) {
                    if (!take_bytes(&cursor, end, 16ULL)) {
                        malformed = true;
                        break;
                    }
                    coords += 1ULL;
                } else {
                    if (!take_bytes(&cursor, end, 4ULL)) {
                        malformed = true;
                        break;
                    }
                    const unsigned int child_count = read_u32_ordered(
                        payload + cursor - 4ULL, child_order
                    );
                    if (child_count > 0x7FFFFFFFU) {
                        overflow = true;
                        break;
                    }
                    if (family == 4) {
                        if ((unsigned long long)child_count > (end - cursor) / 16ULL) {
                            malformed = true;
                            break;
                        }
                        cursor += (unsigned long long)child_count * 16ULL;
                        coords += (unsigned long long)child_count;
                        semantic_invalid = semantic_invalid || child_count == 1U;
                    } else {
                        rings += (unsigned long long)child_count;
                        if (rings > 0x7FFFFFFFULL) {
                            overflow = true;
                            break;
                        }
                        for (unsigned int ring = 0;
                             ring < child_count && !malformed && !overflow; ++ring) {
                            if (!take_bytes(&cursor, end, 4ULL)) {
                                malformed = true;
                                break;
                            }
                            const unsigned int npts = read_u32_ordered(
                                payload + cursor - 4ULL, child_order
                            );
                            if (npts > 0x7FFFFFFFU) {
                                overflow = true;
                                break;
                            }
                            if ((unsigned long long)npts > (end - cursor) / 16ULL) {
                                malformed = true;
                                break;
                            }
                            cursor += (unsigned long long)npts * 16ULL;
                            coords += (unsigned long long)npts;
                            if (coords > 0x7FFFFFFFULL) overflow = true;
                        }
                    }
                }
                if (coords > 0x7FFFFFFFULL) overflow = true;
            }
            empty_flags[row] = primary == 0U ? 1 : 0;
        }

        if (family == 1 || family == 3) parts = 0ULL;
        if (family == 4) rings = 0ULL;
        primary_counts[row] = (int)primary;
        part_counts[row] = parts <= 0x7FFFFFFFULL ? (int)parts : 0;
        ring_counts[row] = rings <= 0x7FFFFFFFULL ? (int)rings : 0;
        coordinate_counts[row] = coords <= 0x7FFFFFFFULL ? (int)coords : 0;

        if (family_mismatch) statuses[row] = 17;
        else if (overflow) statuses[row] = 12;
        else if (semantic_invalid) statuses[row] = 19;
        else if (malformed && statuses[row] == 11) statuses[row] = 11;
        else if (malformed) { }
        else if (cursor != end) statuses[row] = 18;
        else statuses[row] = mixed_endian ? 3 : (root_order == 1 ? 1 : 2);
    }
}

__global__ void __launch_bounds__(256, 2) emit_polygon_ring_tasks(
    const unsigned char* __restrict__ payload,
    const long long* __restrict__ record_offsets,
    const int* __restrict__ row_indexes,
    const int* __restrict__ ring_bases,
    const int* __restrict__ coordinate_bases,
    long long* __restrict__ task_byte_offsets,
    int* __restrict__ task_counts,
    int* __restrict__ task_output_offsets,
    unsigned char* __restrict__ task_byte_orders,
    int* __restrict__ ring_offsets_out,
    const int count
) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    const int row = row_indexes[tid];
    const long long start = record_offsets[row];
    const unsigned char order = payload[start];
    const unsigned int ring_count = read_u32_ordered(payload + start + 5, order);
    unsigned long long cursor = (unsigned long long)start + 9ULL;
    int coordinate = coordinate_bases[tid];
    const int ring_base = ring_bases[tid];
    for (unsigned int ring = 0; ring < ring_count; ++ring) {
        const unsigned int npts = read_u32_ordered(payload + cursor, order);
        const int task = ring_base + (int)ring;
        ring_offsets_out[task] = coordinate;
        task_byte_offsets[task] = (long long)(cursor + 4ULL);
        task_counts[task] = (int)npts;
        task_output_offsets[task] = coordinate;
        task_byte_orders[task] = order;
        cursor += 4ULL + (unsigned long long)npts * 16ULL;
        coordinate += (int)npts;
    }
}

__global__ void __launch_bounds__(256, 2) emit_multipoint_tasks(
    const unsigned char* __restrict__ payload,
    const long long* __restrict__ record_offsets,
    const int* __restrict__ row_indexes,
    const int* __restrict__ coordinate_bases,
    long long* __restrict__ task_byte_offsets,
    int* __restrict__ task_counts,
    int* __restrict__ task_output_offsets,
    unsigned char* __restrict__ task_byte_orders,
    const int count
) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    const int row = row_indexes[tid];
    const long long start = record_offsets[row];
    unsigned long long cursor = (unsigned long long)start + 9ULL;
    const unsigned char root_order = payload[start];
    const unsigned int parts = read_u32_ordered(payload + start + 5, root_order);
    const int base = coordinate_bases[tid];
    for (unsigned int part = 0; part < parts; ++part) {
        const unsigned char order = payload[cursor];
        const int task = base + (int)part;
        task_byte_offsets[task] = (long long)(cursor + 5ULL);
        task_counts[task] = 1;
        task_output_offsets[task] = task;
        task_byte_orders[task] = order;
        cursor += 21ULL;
    }
}

__global__ void __launch_bounds__(256, 2) emit_multilinestring_part_tasks(
    const unsigned char* __restrict__ payload,
    const long long* __restrict__ record_offsets,
    const int* __restrict__ row_indexes,
    const int* __restrict__ part_bases,
    const int* __restrict__ coordinate_bases,
    long long* __restrict__ task_byte_offsets,
    int* __restrict__ task_counts,
    int* __restrict__ task_output_offsets,
    unsigned char* __restrict__ task_byte_orders,
    int* __restrict__ part_offsets_out,
    const int count
) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    const int row = row_indexes[tid];
    const long long start = record_offsets[row];
    unsigned long long cursor = (unsigned long long)start + 9ULL;
    const unsigned char root_order = payload[start];
    const unsigned int parts = read_u32_ordered(payload + start + 5, root_order);
    const int part_base = part_bases[tid];
    int coordinate = coordinate_bases[tid];
    for (unsigned int part = 0; part < parts; ++part) {
        const unsigned char order = payload[cursor];
        const unsigned int npts = read_u32_ordered(payload + cursor + 5ULL, order);
        const int task = part_base + (int)part;
        part_offsets_out[task] = coordinate;
        task_byte_offsets[task] = (long long)(cursor + 9ULL);
        task_counts[task] = (int)npts;
        task_output_offsets[task] = coordinate;
        task_byte_orders[task] = order;
        cursor += 9ULL + (unsigned long long)npts * 16ULL;
        coordinate += (int)npts;
    }
}

__global__ void __launch_bounds__(256, 2) emit_multipolygon_ring_tasks(
    const unsigned char* __restrict__ payload,
    const long long* __restrict__ record_offsets,
    const int* __restrict__ row_indexes,
    const int* __restrict__ polygon_bases,
    const int* __restrict__ ring_bases,
    const int* __restrict__ coordinate_bases,
    long long* __restrict__ task_byte_offsets,
    int* __restrict__ task_counts,
    int* __restrict__ task_output_offsets,
    unsigned char* __restrict__ task_byte_orders,
    int* __restrict__ part_offsets_out,
    int* __restrict__ ring_offsets_out,
    const int count
) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    const int row = row_indexes[tid];
    const long long start = record_offsets[row];
    unsigned long long cursor = (unsigned long long)start + 9ULL;
    const unsigned char root_order = payload[start];
    const unsigned int polygons = read_u32_ordered(payload + start + 5, root_order);
    const int polygon_base = polygon_bases[tid];
    int ring = ring_bases[tid];
    int coordinate = coordinate_bases[tid];
    for (unsigned int polygon = 0; polygon < polygons; ++polygon) {
        const unsigned char order = payload[cursor];
        const unsigned int rings = read_u32_ordered(payload + cursor + 5ULL, order);
        part_offsets_out[polygon_base + (int)polygon] = ring;
        cursor += 9ULL;
        for (unsigned int local_ring = 0; local_ring < rings; ++local_ring) {
            const unsigned int npts = read_u32_ordered(payload + cursor, order);
            ring_offsets_out[ring] = coordinate;
            task_byte_offsets[ring] = (long long)(cursor + 4ULL);
            task_counts[ring] = (int)npts;
            task_output_offsets[ring] = coordinate;
            task_byte_orders[ring] = order;
            cursor += 4ULL + (unsigned long long)npts * 16ULL;
            coordinate += (int)npts;
            ring += 1;
        }
    }
}

#define DEFINE_POINT_DECODER(NAME, ORDER) \
__global__ void __launch_bounds__(256, 2) NAME( \
    const unsigned char* __restrict__ payload, \
    const long long* __restrict__ record_offsets, \
    const int* __restrict__ row_indexes, \
    const int* __restrict__ family_positions, \
    double* __restrict__ x_out, double* __restrict__ y_out, const int count \
) { \
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x); \
    if (tid >= count) return; \
    const int row = row_indexes[tid]; \
    const int out = family_positions[tid]; \
    const long long start = record_offsets[row]; \
    x_out[out] = read_f64_ordered(payload + start + 5, ORDER); \
    y_out[out] = read_f64_ordered(payload + start + 13, ORDER); \
}

DEFINE_POINT_DECODER(decode_point_rows_le, 1)
DEFINE_POINT_DECODER(decode_point_rows_be, 0)

#define DEFINE_LINE_DECODER(NAME, ORDER) \
__global__ void __launch_bounds__(256, 2) NAME( \
    const unsigned char* __restrict__ payload, \
    const long long* __restrict__ record_offsets, \
    const int* __restrict__ row_indexes, \
    const int* __restrict__ family_positions, \
    const int* __restrict__ coordinate_offsets, \
    double* __restrict__ x_out, double* __restrict__ y_out, const int count \
) { \
    const int task = (int)blockIdx.x; \
    if (task >= count) return; \
    const int row = row_indexes[task]; \
    const int family_position = family_positions[task]; \
    const unsigned int npts = read_u32_ordered(payload + record_offsets[row] + 5, ORDER); \
    const long long byte_start = record_offsets[row] + 9; \
    const int output_start = coordinate_offsets[family_position]; \
    for (unsigned int point = threadIdx.x; point < npts; point += blockDim.x) { \
        const long long byte_offset = byte_start + (long long)point * 16LL; \
        x_out[output_start + (int)point] = read_f64_ordered(payload + byte_offset, ORDER); \
        y_out[output_start + (int)point] = read_f64_ordered(payload + byte_offset + 8, ORDER); \
    } \
}

DEFINE_LINE_DECODER(decode_linestring_rows_le, 1)
DEFINE_LINE_DECODER(decode_linestring_rows_be, 0)

#define DEFINE_TASK_DECODER(NAME, ORDER) \
__global__ void __launch_bounds__(256, 2) NAME( \
    const unsigned char* __restrict__ payload, \
    const long long* __restrict__ task_byte_offsets, \
    const int* __restrict__ task_counts, \
    const int* __restrict__ task_output_offsets, \
    const int* __restrict__ task_indexes, \
    double* __restrict__ x_out, double* __restrict__ y_out, const int count \
) { \
    const int selected = (int)blockIdx.x; \
    if (selected >= count) return; \
    const int task = task_indexes[selected]; \
    const long long byte_start = task_byte_offsets[task]; \
    const int npts = task_counts[task]; \
    const int output_start = task_output_offsets[task]; \
    for (int point = threadIdx.x; point < npts; point += blockDim.x) { \
        const long long byte_offset = byte_start + (long long)point * 16LL; \
        x_out[output_start + point] = read_f64_ordered(payload + byte_offset, ORDER); \
        y_out[output_start + point] = read_f64_ordered(payload + byte_offset + 8, ORDER); \
    } \
}

DEFINE_TASK_DECODER(decode_coordinate_tasks_le, 1)
DEFINE_TASK_DECODER(decode_coordinate_tasks_be, 0)

}  // extern "C"
"""


_WKB_DECODE_KERNEL_NAMES = (
    "wkb_plan_summary",
    "wkb_structural_scan",
    "emit_polygon_ring_tasks",
    "emit_multipoint_tasks",
    "emit_multilinestring_part_tasks",
    "emit_multipolygon_ring_tasks",
    "decode_point_rows_le",
    "decode_point_rows_be",
    "decode_linestring_rows_le",
    "decode_linestring_rows_be",
    "decode_coordinate_tasks_le",
    "decode_coordinate_tasks_be",
)
