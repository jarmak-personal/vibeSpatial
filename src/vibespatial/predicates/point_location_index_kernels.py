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
"""
)

POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES = (
    "count_polygon_part_y_bins",
    "scatter_polygon_part_y_bins",
    "point_in_polygon_prepared_part_y_index",
    "point_in_multipolygon_prepared_part_y_index",
)
