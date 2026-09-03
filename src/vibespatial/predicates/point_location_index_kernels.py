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
SUPPORTED_PART_Y_BIN_COUNTS = (8, 16, 32, 64, 128, 256)
COVERAGE_GRID_WIDTH = 4
_COVERAGE_GRID_WIDTH_BY_BIN_COUNT = {
    8: 4,
    16: 4,
    32: 8,
    64: 8,
    128: 16,
    256: 16,
}

_POINT_LOCATION_PART_Y_INDEX_SOURCE = (
    ORIENT2D_DEVICE
    + f"#define VS_PART_Y_BIN_COUNT {PART_Y_BIN_COUNT}\n"
    + f"#define VS_COVERAGE_GRID_WIDTH {COVERAGE_GRID_WIDTH}\n"
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

extern "C" __global__ void compute_polygon_part_y_bounds(
    int part_count,
    const int* __restrict__ part_ring_offsets,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ y,
    double* __restrict__ part_ymin,
    double* __restrict__ part_ymax
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int part = lane; part < part_count; part += stride) {
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
    }
}

extern "C" __global__ void compute_polygon_part_x_bounds(
    int part_count,
    const int* __restrict__ part_ring_offsets,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ x,
    double* __restrict__ part_xmin,
    double* __restrict__ part_xmax
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int part = lane; part < part_count; part += stride) {
        const int ring_start = part_ring_offsets[part];
        const int ring_end = part_ring_offsets[part + 1];
        double minimum = 1.7976931348623157e+308;
        double maximum = -1.7976931348623157e+308;
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int coord_start = ring_offsets[ring];
            const int coord_end = ring_offsets[ring + 1];
            for (int coord = coord_start; coord < coord_end; ++coord) {
                minimum = fmin(minimum, x[coord]);
                maximum = fmax(maximum, x[coord]);
            }
        }
        part_xmin[part] = minimum;
        part_xmax[part] = maximum;
    }
}

extern "C" __device__ __forceinline__ int vs_owner_from_offsets(
    int child,
    int owner_count,
    const int* __restrict__ offsets
) {
    int low = 0;
    int high = owner_count;
    while (low < high) {
        const int middle = low + ((high - low) >> 1);
        if (offsets[middle + 1] <= child) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < owner_count ? low : -1;
}

extern "C" __global__ void map_polygon_rings_to_parts(
    int ring_count,
    int part_count,
    const int* __restrict__ part_ring_offsets,
    int* __restrict__ ring_parts
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int ring = lane; ring < ring_count; ring += stride) {
        ring_parts[ring] = vs_owner_from_offsets(
            ring, part_count, part_ring_offsets);
    }
}

extern "C" __device__ __forceinline__ bool vs_edge_context(
    int coord,
    int ring_count,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ ring_parts,
    const double* __restrict__ y,
    int* ring,
    int* part,
    unsigned int* edge_code,
    double* ay,
    double* by
) {
    const int owner = vs_owner_from_offsets(coord, ring_count, ring_offsets);
    if (owner < 0) return false;
    const int coord_start = ring_offsets[owner];
    const int coord_end = ring_offsets[owner + 1];
    if (coord_end <= coord_start) return false;
    *ring = owner;
    *part = ring_parts[owner];
    if (coord == coord_start) {
        *edge_code = VS_RING_CLOSURE_FLAG | (unsigned int)owner;
        *ay = y[coord_end - 1];
        *by = y[coord_start];
    } else {
        *edge_code = (unsigned int)coord;
        *ay = y[coord - 1];
        *by = y[coord];
    }
    return *part >= 0;
}

extern "C" __global__ void count_polygon_edge_y_bin_memberships(
    int coordinate_count,
    int ring_count,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ ring_parts,
    const double* __restrict__ y,
    const double* __restrict__ part_ymin,
    const double* __restrict__ part_ymax,
    unsigned int* __restrict__ counts
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int coord = lane; coord < coordinate_count; coord += stride) {
        int ring;
        int part;
        unsigned int edge_code;
        double ay;
        double by;
        if (!vs_edge_context(
                coord, ring_count, ring_offsets, ring_parts, y,
                &ring, &part, &edge_code, &ay, &by)) continue;
        const double minimum = part_ymin[part];
        const double maximum = part_ymax[part];
        if (!isfinite(minimum) || !isfinite(maximum)) continue;
        const int first = vs_part_y_bin(fmin(ay, by), minimum, maximum);
        const int last = vs_part_y_bin(fmax(ay, by), minimum, maximum);
        const long long base = (long long)part * VS_PART_Y_BIN_COUNT;
        for (int bin = first; bin <= last; ++bin) {
            atomicAdd(counts + base + bin, 1u);
        }
    }
}

extern "C" __global__ void scatter_polygon_edge_y_bin_memberships(
    int coordinate_count,
    int ring_count,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ ring_parts,
    const double* __restrict__ y,
    const double* __restrict__ part_ymin,
    const double* __restrict__ part_ymax,
    const long long* __restrict__ offsets,
    unsigned int* __restrict__ cursors,
    unsigned int* __restrict__ entries
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int coord = lane; coord < coordinate_count; coord += stride) {
        int ring;
        int part;
        unsigned int edge_code;
        double ay;
        double by;
        if (!vs_edge_context(
                coord, ring_count, ring_offsets, ring_parts, y,
                &ring, &part, &edge_code, &ay, &by)) continue;
        const double minimum = part_ymin[part];
        const double maximum = part_ymax[part];
        if (!isfinite(minimum) || !isfinite(maximum)) continue;
        const int first = vs_part_y_bin(fmin(ay, by), minimum, maximum);
        const int last = vs_part_y_bin(fmax(ay, by), minimum, maximum);
        const long long base = (long long)part * VS_PART_Y_BIN_COUNT;
        for (int bin = first; bin <= last; ++bin) {
            const long long key = base + bin;
            const unsigned int local = atomicAdd(cursors + key, 1u);
            entries[offsets[key] + (long long)local] = edge_code;
        }
    }
}

template <bool CollectMetrics>
__device__ __forceinline__ unsigned char vs_prepared_part_location_core(
    double px,
    double py,
    int part,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ part_xmin,
    const double* __restrict__ part_xmax,
    const double* __restrict__ part_ymin,
    const double* __restrict__ part_ymax,
    const unsigned char* __restrict__ coverage,
    const unsigned int* __restrict__ counts,
    const long long* __restrict__ offsets,
    const unsigned int* __restrict__ entries,
    unsigned long long* active_parts,
    unsigned long long* edges_visited,
    unsigned long long* orient2d_calls
) {
    const double minimum = part_ymin[part];
    const double maximum = part_ymax[part];
    if (py < minimum || py > maximum) return 0;
    if (CollectMetrics) *active_parts += 1ull;
    if (coverage != 0 && part_xmin != 0 && part_xmax != 0) {
        const double xmin = part_xmin[part];
        const double xmax = part_xmax[part];
        if (isfinite(xmin) && isfinite(xmax)
            && isfinite(minimum) && isfinite(maximum)
            && xmax > xmin && maximum > minimum) {
            if (px < xmin || px > xmax) return 0u;
            int cell_x = (int)floor(
                (px - xmin) * ((double)VS_COVERAGE_GRID_WIDTH / (xmax - xmin)));
            int cell_y = (int)floor(
                (py - minimum)
                * ((double)VS_COVERAGE_GRID_WIDTH / (maximum - minimum)));
            cell_x = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, cell_x));
            cell_y = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, cell_y));
            const unsigned char state = coverage[
                (long long)part * VS_COVERAGE_GRID_WIDTH * VS_COVERAGE_GRID_WIDTH
                + cell_y * VS_COVERAGE_GRID_WIDTH + cell_x
            ];
            if (state == 1u) return 0u;
            if (state == 2u) return 2u;
        }
    }
    const int bin = vs_part_y_bin(py, minimum, maximum);
    const long long key = (long long)part * VS_PART_Y_BIN_COUNT + bin;
    const long long start = offsets[key];
    const long long end = start + (long long)counts[key];
    if (CollectMetrics) {
        *edges_visited += (unsigned long long)(end - start);
    }
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
        if (CollectMetrics) *orient2d_calls += 1ull;
        const int orientation = vs_orient2d(ax, ay, bx, by, px, py);
        if (boundary_bbox && orientation == 0) return 1;
        if (crosses_ray && ((orientation > 0) == (by > ay))) inside = !inside;
    }
    return inside ? 2 : 0;
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
    return vs_prepared_part_location_core<false>(
        px, py, part, ring_offsets, x, y,
        0, 0, part_ymin, part_ymax, 0,
        counts, offsets, entries, 0, 0, 0);
}

extern "C" __device__ __forceinline__ bool vs_point_in_closed_rect(
    double px,
    double py,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    return px >= xmin && px <= xmax && py >= ymin && py <= ymax;
}

extern "C" __device__ __forceinline__ bool vs_point_on_closed_segment(
    double ax,
    double ay,
    double bx,
    double by,
    double px,
    double py
) {
    return px >= fmin(ax, bx) && px <= fmax(ax, bx)
        && py >= fmin(ay, by) && py <= fmax(ay, by)
        && vs_orient2d(ax, ay, bx, by, px, py) == 0;
}

extern "C" __device__ __forceinline__ bool vs_closed_segments_intersect(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy
) {
    if (fmax(ax, bx) < fmin(cx, dx) || fmax(cx, dx) < fmin(ax, bx)
        || fmax(ay, by) < fmin(cy, dy) || fmax(cy, dy) < fmin(ay, by)) {
        return false;
    }
    const int abc = vs_orient2d(ax, ay, bx, by, cx, cy);
    const int abd = vs_orient2d(ax, ay, bx, by, dx, dy);
    const int cda = vs_orient2d(cx, cy, dx, dy, ax, ay);
    const int cdb = vs_orient2d(cx, cy, dx, dy, bx, by);
    if (abc == 0 && vs_point_on_closed_segment(ax, ay, bx, by, cx, cy)) return true;
    if (abd == 0 && vs_point_on_closed_segment(ax, ay, bx, by, dx, dy)) return true;
    if (cda == 0 && vs_point_on_closed_segment(cx, cy, dx, dy, ax, ay)) return true;
    if (cdb == 0 && vs_point_on_closed_segment(cx, cy, dx, dy, bx, by)) return true;
    return (abc > 0) != (abd > 0) && (cda > 0) != (cdb > 0);
}

extern "C" __device__ __forceinline__ bool vs_segment_intersects_closed_cell(
    double ax,
    double ay,
    double bx,
    double by,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) {
    if (vs_point_in_closed_rect(ax, ay, xmin, ymin, xmax, ymax)
        || vs_point_in_closed_rect(bx, by, xmin, ymin, xmax, ymax)) {
        return true;
    }
    return vs_closed_segments_intersect(
            ax, ay, bx, by, xmin, ymin, xmax, ymin)
        || vs_closed_segments_intersect(
            ax, ay, bx, by, xmax, ymin, xmax, ymax)
        || vs_closed_segments_intersect(
            ax, ay, bx, by, xmax, ymax, xmin, ymax)
        || vs_closed_segments_intersect(
            ax, ay, bx, by, xmin, ymax, xmin, ymin);
}

extern "C" __global__ void initialize_polygon_part_coverage_cells(
    int part_count,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ part_xmin,
    const double* __restrict__ part_xmax,
    const double* __restrict__ part_ymin,
    const double* __restrict__ part_ymax,
    const unsigned int* __restrict__ counts,
    const long long* __restrict__ offsets,
    const unsigned int* __restrict__ entries,
    unsigned char* __restrict__ coverage
) {
    const int cells_per_part = VS_COVERAGE_GRID_WIDTH * VS_COVERAGE_GRID_WIDTH;
    const long long cell_count = (long long)part_count * cells_per_part;
    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long cell = lane; cell < cell_count; cell += stride) {
        const int part = (int)(cell / cells_per_part);
        const int local = (int)(cell - (long long)part * cells_per_part);
        const int cell_x = local % VS_COVERAGE_GRID_WIDTH;
        const int cell_y = local / VS_COVERAGE_GRID_WIDTH;
        const double xmin = part_xmin[part];
        const double xmax = part_xmax[part];
        const double ymin = part_ymin[part];
        const double ymax = part_ymax[part];
        if (!isfinite(xmin) || !isfinite(xmax)
            || !isfinite(ymin) || !isfinite(ymax)
            || !(xmax > xmin) || !(ymax > ymin)) {
            coverage[cell] = 0u;
            continue;
        }
        const double px = xmin + ((double)cell_x + 0.5)
            * ((xmax - xmin) / (double)VS_COVERAGE_GRID_WIDTH);
        const double py = ymin + ((double)cell_y + 0.5)
            * ((ymax - ymin) / (double)VS_COVERAGE_GRID_WIDTH);
        const unsigned char location = vs_prepared_part_location(
            px, py, part, ring_offsets, x, y,
            part_ymin, part_ymax, counts, offsets, entries);
        coverage[cell] = location == 1u ? 0u : (location == 2u ? 2u : 1u);
    }
}

extern "C" __global__ void mark_polygon_edge_coverage_cells(
    int coordinate_count,
    int ring_count,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ ring_parts,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ part_xmin,
    const double* __restrict__ part_xmax,
    const double* __restrict__ part_ymin,
    const double* __restrict__ part_ymax,
    unsigned char* __restrict__ coverage
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int cells_per_part = VS_COVERAGE_GRID_WIDTH * VS_COVERAGE_GRID_WIDTH;
    for (int coord = lane; coord < coordinate_count; coord += stride) {
        int ring;
        int part;
        unsigned int edge_code;
        double ay;
        double by;
        if (!vs_edge_context(
                coord, ring_count, ring_offsets, ring_parts, y,
                &ring, &part, &edge_code, &ay, &by)) continue;
        int i;
        int j;
        if ((edge_code & VS_RING_CLOSURE_FLAG) != 0u) {
            i = ring_offsets[ring];
            j = ring_offsets[ring + 1] - 1;
        } else {
            i = (int)edge_code;
            j = i - 1;
        }
        const double ax = x[j];
        const double bx = x[i];
        const double xmin = part_xmin[part];
        const double xmax = part_xmax[part];
        const double ymin = part_ymin[part];
        const double ymax = part_ymax[part];
        const long long base = (long long)part * cells_per_part;
        if (!isfinite(ax) || !isfinite(ay) || !isfinite(bx) || !isfinite(by)
            || !isfinite(xmin) || !isfinite(xmax)
            || !isfinite(ymin) || !isfinite(ymax)
            || !(xmax > xmin) || !(ymax > ymin)) {
            for (int cell = 0; cell < cells_per_part; ++cell) {
                coverage[base + cell] = 0u;
            }
            continue;
        }
        const double xscale =
            (double)VS_COVERAGE_GRID_WIDTH / (xmax - xmin);
        const double yscale =
            (double)VS_COVERAGE_GRID_WIDTH / (ymax - ymin);
        int first_x = (int)floor((fmin(ax, bx) - xmin) * xscale) - 1;
        int last_x = (int)floor((fmax(ax, bx) - xmin) * xscale);
        int first_y = (int)floor((fmin(ay, by) - ymin) * yscale) - 1;
        int last_y = (int)floor((fmax(ay, by) - ymin) * yscale);
        first_x = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, first_x));
        last_x = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, last_x));
        first_y = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, first_y));
        last_y = max(0, min(VS_COVERAGE_GRID_WIDTH - 1, last_y));
        const double cell_width =
            (xmax - xmin) / (double)VS_COVERAGE_GRID_WIDTH;
        const double cell_height =
            (ymax - ymin) / (double)VS_COVERAGE_GRID_WIDTH;
        for (int cell_y = first_y; cell_y <= last_y; ++cell_y) {
            const double cell_ymin = ymin + (double)cell_y * cell_height;
            const double cell_ymax = cell_y == VS_COVERAGE_GRID_WIDTH - 1
                ? ymax : cell_ymin + cell_height;
            for (int cell_x = first_x; cell_x <= last_x; ++cell_x) {
                const double cell_xmin = xmin + (double)cell_x * cell_width;
                const double cell_xmax = cell_x == VS_COVERAGE_GRID_WIDTH - 1
                    ? xmax : cell_xmin + cell_width;
                if (vs_segment_intersects_closed_cell(
                        ax, ay, bx, by,
                        cell_xmin, cell_ymin, cell_xmax, cell_ymax)) {
                    coverage[
                        base + cell_y * VS_COVERAGE_GRID_WIDTH + cell_x
                    ] = 0u;
                }
            }
        }
    }
}

extern "C" __device__ __forceinline__ unsigned char
vs_prepared_part_location_with_coverage(
    double px,
    double py,
    int part,
    const int* ring_offsets,
    const double* x,
    const double* y,
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries
) {
    return vs_prepared_part_location_core<false>(
        px, py, part, ring_offsets, x, y,
        part_xmin, part_xmax, part_ymin, part_ymax, coverage,
        counts, offsets, entries, 0, 0, 0);
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
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
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
        out[index] = vs_prepared_part_location_with_coverage(
            point_x[point_coord], point_y[point_coord], polygon_row,
            ring_offsets, polygon_x, polygon_y,
            part_xmin, part_xmax, part_ymin, part_ymax, coverage,
            counts, offsets, entries);
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
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
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
            const unsigned char location = vs_prepared_part_location_with_coverage(
                px, py, part, ring_offsets, polygon_x, polygon_y,
                part_xmin, part_xmax, part_ymin, part_ymax, coverage,
                counts, offsets, entries);
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
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
    const unsigned int* counts,
    const long long* offsets,
    const unsigned int* entries,
    unsigned long long* active_parts,
    unsigned long long* edges_visited,
    unsigned long long* orient2d_calls
) {
    return vs_prepared_part_location_core<true>(
        px, py, part, ring_offsets, x, y,
        part_xmin, part_xmax, part_ymin, part_ymax, coverage,
        counts, offsets, entries,
        active_parts, edges_visited, orient2d_calls);
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
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
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
                ring_offsets, polygon_x, polygon_y,
                part_xmin, part_xmax, part_ymin, part_ymax, coverage,
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
    const double* part_xmin,
    const double* part_xmax,
    const double* part_ymin,
    const double* part_ymax,
    const unsigned char* coverage,
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
                    part_xmin, part_xmax, part_ymin, part_ymax, coverage,
                    counts, offsets, entries,
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
    "compute_polygon_part_y_bounds",
    "compute_polygon_part_x_bounds",
    "map_polygon_rings_to_parts",
    "count_polygon_edge_y_bin_memberships",
    "scatter_polygon_edge_y_bin_memberships",
    "initialize_polygon_part_coverage_cells",
    "mark_polygon_edge_coverage_cells",
    "point_in_polygon_prepared_part_y_index",
    "point_in_multipolygon_prepared_part_y_index",
)

POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES = (
    "reserve_point_region_profile_samples",
    "point_region_profile_sample_mask",
    "point_in_polygon_prepared_part_y_index_profiled",
    "point_in_multipolygon_prepared_part_y_index_profiled",
)


def _source_for_bin_count(source: str, bin_count: int) -> str:
    """Return one compile-time-uniform width variant."""
    width = int(bin_count)
    if width not in SUPPORTED_PART_Y_BIN_COUNTS:
        raise ValueError(
            f"unsupported polygon part-y bin count {width}; "
            f"expected one of {SUPPORTED_PART_Y_BIN_COUNTS}"
        )
    coverage_width = coverage_grid_width_for_bin_count(width)
    source = source.replace(
        f"#define VS_PART_Y_BIN_COUNT {PART_Y_BIN_COUNT}\n",
        f"#define VS_PART_Y_BIN_COUNT {width}\n",
        1,
    )
    return source.replace(
        f"#define VS_COVERAGE_GRID_WIDTH {COVERAGE_GRID_WIDTH}\n",
        f"#define VS_COVERAGE_GRID_WIDTH {coverage_width}\n",
        1,
    )


def coverage_grid_width_for_bin_count(bin_count: int) -> int:
    """Return the bounded conservative grid width paired with one y tier."""
    width = int(bin_count)
    try:
        return _COVERAGE_GRID_WIDTH_BY_BIN_COUNT[width]
    except KeyError as exc:
        raise ValueError(
            f"unsupported polygon part-y bin count {width}; "
            f"expected one of {SUPPORTED_PART_Y_BIN_COUNTS}"
        ) from exc


def point_location_part_y_index_source(bin_count: int) -> str:
    """Return the production kernel source for ``bin_count`` bins per part."""
    return _source_for_bin_count(_POINT_LOCATION_PART_Y_INDEX_SOURCE, bin_count)


def point_location_part_y_index_profile_source(bin_count: int) -> str:
    """Return the profiling kernel source for ``bin_count`` bins per part."""
    return _source_for_bin_count(_POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE, bin_count)
