"""CUDA sources for the exact polygon-part y-edge directory.

The directory changes the physical point-location workload from every edge in
a candidate polygon to the edges whose y interval contains the query point.
All coordinates and bin calculations remain fp64.  Edge codes use bit 31 to
encode a ring-closing edge; coordinate and ring counts must therefore fit in
31 bits before this variant is admitted.
"""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE

PART_Y_BIN_COUNT = 8

_POINT_LOCATION_PART_Y_INDEX_SOURCE = (
    ORIENT2D_DEVICE
    + f"#define VS_PART_Y_BIN_COUNT {PART_Y_BIN_COUNT}\n"
    + r"""
#define VS_RING_CLOSURE_FLAG 0x80000000u
#define VS_EDGE_INDEX_MASK 0x7fffffffu

extern "C" __device__ __forceinline__ int vs_part_y_bin(
    double value,
    double minimum,
    double maximum
) {
    if (!(maximum > minimum)) return 0;
    const double scaled =
        (value - minimum) * ((double)VS_PART_Y_BIN_COUNT / (maximum - minimum));
    int bin = (int)floor(scaled);
    if (bin < 0) return 0;
    if (bin >= VS_PART_Y_BIN_COUNT) return VS_PART_Y_BIN_COUNT - 1;
    return bin;
}

extern "C" __device__ __forceinline__ void vs_count_edge_bins(
    double ay,
    double by,
    double minimum,
    double maximum,
    unsigned int* counts
) {
    int first = vs_part_y_bin(fmin(ay, by), minimum, maximum);
    int last = vs_part_y_bin(fmax(ay, by), minimum, maximum);
    for (int bin = first; bin <= last; ++bin) counts[bin] += 1u;
}

extern "C" __global__ void count_polygon_part_y_bins(
    int part_count,
    const int* part_ring_offsets,
    const int* ring_offsets,
    const double* y,
    double* part_ymin,
    double* part_ymax,
    unsigned int* counts
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int part = lane; part < part_count; part += stride) {
        unsigned int* part_counts = counts + ((long long)part * VS_PART_Y_BIN_COUNT);
        for (int bin = 0; bin < VS_PART_Y_BIN_COUNT; ++bin) part_counts[bin] = 0u;
        const int ring_start = part_ring_offsets[part];
        const int ring_end = part_ring_offsets[part + 1];
        double minimum = 1.7976931348623157e+308;
        double maximum = -1.7976931348623157e+308;
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int coord_start = ring_offsets[ring];
            const int coord_end = ring_offsets[ring + 1];
            for (int coord = coord_start; coord < coord_end; ++coord) {
                minimum = fmin(minimum, y[coord]);
                maximum = fmax(maximum, y[coord]);
            }
        }
        part_ymin[part] = minimum;
        part_ymax[part] = maximum;
        if (!isfinite(minimum) || !isfinite(maximum)) continue;
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int coord_start = ring_offsets[ring];
            const int coord_end = ring_offsets[ring + 1];
            if (coord_end <= coord_start) continue;
            vs_count_edge_bins(
                y[coord_end - 1], y[coord_start], minimum, maximum, part_counts);
            for (int coord = coord_start + 1; coord < coord_end; ++coord) {
                vs_count_edge_bins(
                    y[coord - 1], y[coord], minimum, maximum, part_counts);
            }
        }
    }
}

extern "C" __device__ __forceinline__ void vs_scatter_edge_bins(
    unsigned int edge_code,
    double ay,
    double by,
    double minimum,
    double maximum,
    unsigned long long* cursors,
    unsigned int* entries
) {
    int first = vs_part_y_bin(fmin(ay, by), minimum, maximum);
    int last = vs_part_y_bin(fmax(ay, by), minimum, maximum);
    for (int bin = first; bin <= last; ++bin) {
        entries[cursors[bin]++] = edge_code;
    }
}

extern "C" __global__ void scatter_polygon_part_y_bins(
    int part_count,
    const int* part_ring_offsets,
    const int* ring_offsets,
    const double* y,
    const double* part_ymin,
    const double* part_ymax,
    const long long* offsets,
    unsigned int* entries
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int part = lane; part < part_count; part += stride) {
        const long long base = (long long)part * VS_PART_Y_BIN_COUNT;
        unsigned long long cursors[VS_PART_Y_BIN_COUNT];
        for (int bin = 0; bin < VS_PART_Y_BIN_COUNT; ++bin) {
            cursors[bin] = (unsigned long long)offsets[base + bin];
        }
        const double minimum = part_ymin[part];
        const double maximum = part_ymax[part];
        if (!isfinite(minimum) || !isfinite(maximum)) continue;
        const int ring_start = part_ring_offsets[part];
        const int ring_end = part_ring_offsets[part + 1];
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int coord_start = ring_offsets[ring];
            const int coord_end = ring_offsets[ring + 1];
            if (coord_end <= coord_start) continue;
            vs_scatter_edge_bins(
                VS_RING_CLOSURE_FLAG | (unsigned int)ring,
                y[coord_end - 1],
                y[coord_start],
                minimum,
                maximum,
                cursors,
                entries);
            for (int coord = coord_start + 1; coord < coord_end; ++coord) {
                vs_scatter_edge_bins(
                    (unsigned int)coord,
                    y[coord - 1],
                    y[coord],
                    minimum,
                    maximum,
                    cursors,
                    entries);
            }
        }
    }
}

extern "C" __device__ __forceinline__ unsigned char vs_prepared_part_location(
    double px,
    double py,
    int part,
    const int* ring_offsets,
    const double* x,
    const double* y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries
) {
    const double minimum = part_ymin[part];
    const double maximum = part_ymax[part];
    if (py < minimum || py > maximum) return 0;
    const int bin = vs_part_y_bin(py, minimum, maximum);
    const long long key = (long long)part * VS_PART_Y_BIN_COUNT + bin;
    const long long start = offsets[key];
    const long long end = start + (long long)counts[key];
    bool inside = false;
    for (long long position = start; position < end; ++position) {
        const unsigned int code = entries[position];
        int i;
        int j;
        if ((code & VS_RING_CLOSURE_FLAG) != 0u) {
            const int ring = (int)(code & VS_EDGE_INDEX_MASK);
            i = ring_offsets[ring];
            j = ring_offsets[ring + 1] - 1;
        } else {
            i = (int)code;
            j = i - 1;
        }
        const double ax = x[j];
        const double ay = y[j];
        const double bx = x[i];
        const double by = y[i];
        const bool crosses_ray = (ay > py) != (by > py);
        const double minx = fmin(ax, bx);
        const double maxx = fmax(ax, bx);
        const bool boundary_bbox =
            py >= fmin(ay, by) && py <= fmax(ay, by)
            && px >= minx && px <= maxx;
        if (!crosses_ray && !boundary_bbox) continue;
        if (crosses_ray && px < minx) {
            inside = !inside;
            continue;
        }
        if (crosses_ray && px > maxx) continue;
        const int orientation = vs_orient2d(ax, ay, bx, by, px, py);
        if (boundary_bbox && orientation == 0) return 1;
        if (crosses_ray && ((orientation > 0) == (by > ay))) inside = !inside;
    }
    return inside ? 2 : 0;
}

extern "C" __global__ void point_in_polygon_prepared_part_y_index(
    const int* candidate_rows,
    const int* candidate_rows_right,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int candidate_count
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int count = logical_count != 0 ? logical_count[0] : candidate_count;
    const int offset = source_offset != 0 ? (int)source_offset[0] : 0;
    for (int local = lane; local < count; local += stride) {
        const int index = offset + local;
        const int point_row = point_row_offsets[candidate_rows[index]];
        const int polygon_row = polygon_row_offsets[candidate_rows_right[index]];
        if (point_row < 0 || polygon_row < 0
            || point_empty_mask[point_row] || polygon_empty_mask[polygon_row]) {
            out[index] = 0;
            continue;
        }
        const int point_coord = point_geometry_offsets[point_row];
        out[index] = vs_prepared_part_location(
            point_x[point_coord], point_y[point_coord], polygon_row,
            ring_offsets, polygon_x, polygon_y,
            part_ymin, part_ymax, counts, offsets, entries);
    }
}

extern "C" __global__ void point_in_multipolygon_prepared_part_y_index(
    const int* candidate_rows,
    const int* candidate_rows_right,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int candidate_count
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int count = logical_count != 0 ? logical_count[0] : candidate_count;
    const int offset = source_offset != 0 ? (int)source_offset[0] : 0;
    for (int local = lane; local < count; local += stride) {
        const int index = offset + local;
        const int point_row = point_row_offsets[candidate_rows[index]];
        const int polygon_row = polygon_row_offsets[candidate_rows_right[index]];
        if (point_row < 0 || polygon_row < 0
            || point_empty_mask[point_row] || polygon_empty_mask[polygon_row]) {
            out[index] = 0;
            continue;
        }
        const int point_coord = point_geometry_offsets[point_row];
        const double px = point_x[point_coord];
        const double py = point_y[point_coord];
        unsigned char best = 0;
        const int part_start = polygon_geometry_offsets[polygon_row];
        const int part_end = polygon_geometry_offsets[polygon_row + 1];
        for (int part = part_start; part < part_end; ++part) {
            const unsigned char location = vs_prepared_part_location(
                px, py, part, ring_offsets, polygon_x, polygon_y,
                part_ymin, part_ymax, counts, offsets, entries);
            if (location == 1) {
                best = 1;
                break;
            }
            if (location == 2) {
                best = 2;
                break;
            }
        }
        out[index] = best;
    }
}

#define VS_PROFILE_COUNTER_COUNT 16
#define VS_PROFILE_HISTOGRAM_MAX 4096
#define VS_PROFILE_SAMPLES_PER_LAUNCH 128

extern "C" __device__ __forceinline__ bool vs_profile_should_sample(
    int local,
    int count,
    unsigned long long sample_count
) {
    if (sample_count == 0ull || count <= 0) return false;
    const unsigned long long before =
        ((unsigned long long)local * sample_count) / (unsigned long long)count;
    const unsigned long long after =
        ((unsigned long long)(local + 1) * sample_count)
        / (unsigned long long)count;
    return after != before;
}

extern "C" __global__ void reserve_point_region_profile_samples(
    unsigned long long* summary,
    unsigned long long* sample_plan,
    const int* logical_count,
    int sample_limit,
    int candidate_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int count = logical_count != 0 ? logical_count[0] : candidate_count;
    const unsigned long long used = summary[14];
    const unsigned long long limit =
        sample_limit > 0 ? (unsigned long long)sample_limit : 0ull;
    const unsigned long long remaining = limit > used ? limit - used : 0ull;
    const unsigned long long available =
        count > 0 ? (unsigned long long)count : 0ull;
    const unsigned long long launch_cap =
        min(available, (unsigned long long)VS_PROFILE_SAMPLES_PER_LAUNCH);
    const unsigned long long sample_count = min(launch_cap, remaining);
    sample_plan[0] = sample_count;
    summary[14] = used + sample_count;
}

extern "C" __global__ void point_region_profile_sample_mask(
    int count,
    unsigned long long sample_count,
    unsigned char* out
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int local = lane; local < count; local += stride) {
        out[local] = vs_profile_should_sample(local, count, sample_count) ? 1u : 0u;
    }
}

extern "C" __device__ __forceinline__ unsigned char
vs_prepared_part_location_profiled(
    double px,
    double py,
    int part,
    const int* ring_offsets,
    const double* x,
    const double* y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned long long* active_parts,
    unsigned long long* edges_visited,
    unsigned long long* orient2d_calls
) {
    const double minimum = part_ymin[part];
    const double maximum = part_ymax[part];
    if (py < minimum || py > maximum) return 0;
    *active_parts += 1ull;
    const int bin = vs_part_y_bin(py, minimum, maximum);
    const long long key = (long long)part * VS_PART_Y_BIN_COUNT + bin;
    const long long start = offsets[key];
    const long long end = start + (long long)counts[key];
    *edges_visited += (unsigned long long)(end - start);
    bool inside = false;
    for (long long position = start; position < end; ++position) {
        const unsigned int code = entries[position];
        int i;
        int j;
        if ((code & VS_RING_CLOSURE_FLAG) != 0u) {
            const int ring = (int)(code & VS_EDGE_INDEX_MASK);
            i = ring_offsets[ring];
            j = ring_offsets[ring + 1] - 1;
        } else {
            i = (int)code;
            j = i - 1;
        }
        const double ax = x[j];
        const double ay = y[j];
        const double bx = x[i];
        const double by = y[i];
        const bool crosses_ray = (ay > py) != (by > py);
        const double minx = fmin(ax, bx);
        const double maxx = fmax(ax, bx);
        const bool boundary_bbox =
            py >= fmin(ay, by) && py <= fmax(ay, by)
            && px >= minx && px <= maxx;
        if (!crosses_ray && !boundary_bbox) continue;
        if (crosses_ray && px < minx) {
            inside = !inside;
            continue;
        }
        if (crosses_ray && px > maxx) continue;
        *orient2d_calls += 1ull;
        const int orientation = vs_orient2d(ax, ay, bx, by, px, py);
        if (boundary_bbox && orientation == 0) return 1;
        if (crosses_ray && ((orientation > 0) == (by > ay))) inside = !inside;
    }
    return inside ? 2 : 0;
}

extern "C" __device__ __forceinline__ void vs_profile_commit_block(
    unsigned long long* block_summary,
    unsigned long long* summary,
    const unsigned long long* local
) {
    for (int metric = 0; metric < VS_PROFILE_COUNTER_COUNT; ++metric) {
        if (metric >= 11 && metric <= 13) {
            atomicMax(block_summary + metric, local[metric]);
        } else if (local[metric] != 0ull) {
            atomicAdd(block_summary + metric, local[metric]);
        }
    }
    __syncthreads();
    if (threadIdx.x < VS_PROFILE_COUNTER_COUNT) {
        const int metric = threadIdx.x;
        if (metric >= 11 && metric <= 13) {
            atomicMax(summary + metric, block_summary[metric]);
        } else if (block_summary[metric] != 0ull) {
            atomicAdd(summary + metric, block_summary[metric]);
        }
    }
}

extern "C" __global__ void point_in_polygon_prepared_part_y_index_profiled(
    const int* candidate_rows,
    const int* candidate_rows_right,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned long long* summary,
    unsigned long long* parts_histogram,
    unsigned long long* edges_histogram,
    const unsigned long long* sample_plan,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int candidate_count
) {
    __shared__ unsigned long long block_summary[VS_PROFILE_COUNTER_COUNT];
    if (threadIdx.x < VS_PROFILE_COUNTER_COUNT) block_summary[threadIdx.x] = 0ull;
    __syncthreads();
    unsigned long long local_summary[VS_PROFILE_COUNTER_COUNT] = {0ull};
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int count = logical_count != 0 ? logical_count[0] : candidate_count;
    const int offset = source_offset != 0 ? (int)source_offset[0] : 0;
    const unsigned long long sample_count = sample_plan[0];
    for (int local = lane; local < count; local += stride) {
        const int index = offset + local;
        local_summary[0] += 1ull;
        const int point_row = point_row_offsets[candidate_rows[index]];
        const int polygon_row = polygon_row_offsets[candidate_rows_right[index]];
        unsigned long long parts = 0ull;
        unsigned long long active = 0ull;
        unsigned long long edges = 0ull;
        unsigned long long orient = 0ull;
        unsigned char result = 0;
        if (point_row >= 0 && polygon_row >= 0
            && !point_empty_mask[point_row] && !polygon_empty_mask[polygon_row]) {
            local_summary[1] += 1ull;
            parts = 1ull;
            const int point_coord = point_geometry_offsets[point_row];
            result = vs_prepared_part_location_profiled(
                point_x[point_coord], point_y[point_coord], polygon_row,
                ring_offsets, polygon_x, polygon_y, part_ymin, part_ymax,
                counts, offsets, entries, &active, &edges, &orient);
        }
        out[index] = result;
        local_summary[2] += parts;
        local_summary[3] += active;
        local_summary[4] += edges;
        local_summary[5] += orient;
        local_summary[6] += active == 0ull;
        local_summary[7] += edges == 0ull;
        local_summary[8] += result == 1u;
        local_summary[9] += result == 2u;
        local_summary[10] += result == 0u;
        local_summary[11] = max(local_summary[11], parts);
        local_summary[12] = max(local_summary[12], active);
        local_summary[13] = max(local_summary[13], edges);
        if (vs_profile_should_sample(local, count, sample_count)) {
            const int parts_bin = (int)min(parts, (unsigned long long)VS_PROFILE_HISTOGRAM_MAX);
            const int edges_bin = (int)min(edges, (unsigned long long)VS_PROFILE_HISTOGRAM_MAX);
            atomicAdd(parts_histogram + parts_bin, 1ull);
            atomicAdd(edges_histogram + edges_bin, 1ull);
            local_summary[15] += 1ull;
        }
    }
    vs_profile_commit_block(block_summary, summary, local_summary);
}

extern "C" __global__ void point_in_multipolygon_prepared_part_y_index_profiled(
    const int* candidate_rows,
    const int* candidate_rows_right,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned long long* summary,
    unsigned long long* parts_histogram,
    unsigned long long* edges_histogram,
    const unsigned long long* sample_plan,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int candidate_count
) {
    __shared__ unsigned long long block_summary[VS_PROFILE_COUNTER_COUNT];
    if (threadIdx.x < VS_PROFILE_COUNTER_COUNT) block_summary[threadIdx.x] = 0ull;
    __syncthreads();
    unsigned long long local_summary[VS_PROFILE_COUNTER_COUNT] = {0ull};
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int count = logical_count != 0 ? logical_count[0] : candidate_count;
    const int offset = source_offset != 0 ? (int)source_offset[0] : 0;
    const unsigned long long sample_count = sample_plan[0];
    for (int local = lane; local < count; local += stride) {
        const int index = offset + local;
        local_summary[0] += 1ull;
        const int point_row = point_row_offsets[candidate_rows[index]];
        const int polygon_row = polygon_row_offsets[candidate_rows_right[index]];
        unsigned long long parts = 0ull;
        unsigned long long active = 0ull;
        unsigned long long edges = 0ull;
        unsigned long long orient = 0ull;
        unsigned char best = 0;
        if (point_row >= 0 && polygon_row >= 0
            && !point_empty_mask[point_row] && !polygon_empty_mask[polygon_row]) {
            local_summary[1] += 1ull;
            const int point_coord = point_geometry_offsets[point_row];
            const double px = point_x[point_coord];
            const double py = point_y[point_coord];
            const int part_start = polygon_geometry_offsets[polygon_row];
            const int part_end = polygon_geometry_offsets[polygon_row + 1];
            for (int part = part_start; part < part_end; ++part) {
                parts += 1ull;
                const unsigned char location = vs_prepared_part_location_profiled(
                    px, py, part, ring_offsets, polygon_x, polygon_y,
                    part_ymin, part_ymax, counts, offsets, entries,
                    &active, &edges, &orient);
                if (location == 1) {
                    best = 1;
                    break;
                }
                if (location == 2) {
                    best = 2;
                    break;
                }
            }
        }
        out[index] = best;
        local_summary[2] += parts;
        local_summary[3] += active;
        local_summary[4] += edges;
        local_summary[5] += orient;
        local_summary[6] += active == 0ull;
        local_summary[7] += edges == 0ull;
        local_summary[8] += best == 1u;
        local_summary[9] += best == 2u;
        local_summary[10] += best == 0u;
        local_summary[11] = max(local_summary[11], parts);
        local_summary[12] = max(local_summary[12], active);
        local_summary[13] = max(local_summary[13], edges);
        if (vs_profile_should_sample(local, count, sample_count)) {
            const int parts_bin = (int)min(parts, (unsigned long long)VS_PROFILE_HISTOGRAM_MAX);
            const int edges_bin = (int)min(edges, (unsigned long long)VS_PROFILE_HISTOGRAM_MAX);
            atomicAdd(parts_histogram + parts_bin, 1ull);
            atomicAdd(edges_histogram + edges_bin, 1ull);
            local_summary[15] += 1ull;
        }
    }
    vs_profile_commit_block(block_summary, summary, local_summary);
}
"""
)

_POINT_LOCATION_PROFILE_MARKER = "#define VS_PROFILE_COUNTER_COUNT 16"
(
    _POINT_LOCATION_PART_Y_INDEX_SOURCE,
    _POINT_LOCATION_PART_Y_INDEX_PROFILE_TAIL,
) = _POINT_LOCATION_PART_Y_INDEX_SOURCE.split(
    _POINT_LOCATION_PROFILE_MARKER,
    maxsplit=1,
)
_POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE = (
    _POINT_LOCATION_PART_Y_INDEX_SOURCE
    + _POINT_LOCATION_PROFILE_MARKER
    + _POINT_LOCATION_PART_Y_INDEX_PROFILE_TAIL
)

POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES = (
    "count_polygon_part_y_bins",
    "scatter_polygon_part_y_bins",
    "point_in_polygon_prepared_part_y_index",
    "point_in_multipolygon_prepared_part_y_index",
)

POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES = (
    "reserve_point_region_profile_samples",
    "point_region_profile_sample_mask",
    "point_in_polygon_prepared_part_y_index_profiled",
    "point_in_multipolygon_prepared_part_y_index_profiled",
)
