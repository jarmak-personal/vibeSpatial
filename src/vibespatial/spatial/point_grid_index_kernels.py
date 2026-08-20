"""CUDA source for conservative point-grid spatial-index candidates."""

from __future__ import annotations

POINT_GRID_INDEX_KERNEL_NAMES = (
    "point_grid_query_counts",
    "point_grid_query_scatter",
    "point_grid_candidate_not_in_other_superset",
)

_POINT_GRID_INDEX_SOURCE = r"""
extern "C" __device__ __forceinline__ bool vs_point_grid_window(
    const double* bounds,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int grid_size,
    int* cell_x0,
    int* cell_y0,
    int* cell_x1,
    int* cell_y1
) {
    const double qxmin = bounds[0];
    const double qymin = bounds[1];
    const double qxmax = bounds[2];
    const double qymax = bounds[3];
    if (!isfinite(qxmin) || !isfinite(qymin)
        || !isfinite(qxmax) || !isfinite(qymax)
        || qxmax < xmin || qxmin > xmax
        || qymax < ymin || qymin > ymax) {
        return false;
    }
    const double xscale = (double)grid_size / (xmax - xmin);
    const double yscale = (double)grid_size / (ymax - ymin);
    int ix0 = (int)floor((qxmin - xmin) * xscale);
    int iy0 = (int)floor((qymin - ymin) * yscale);
    int ix1 = (int)floor((qxmax - xmin) * xscale);
    int iy1 = (int)floor((qymax - ymin) * yscale);
    ix0 = max(0, min(grid_size - 1, ix0));
    iy0 = max(0, min(grid_size - 1, iy0));
    ix1 = max(0, min(grid_size - 1, ix1));
    iy1 = max(0, min(grid_size - 1, iy1));
    *cell_x0 = ix0;
    *cell_y0 = iy0;
    *cell_x1 = ix1;
    *cell_y1 = iy1;
    return ix0 <= ix1 && iy0 <= iy1;
}

extern "C" __global__ void point_grid_query_counts(
    const double* query_bounds,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int grid_size,
    const long long* integral_counts,
    long long* query_counts,
    int query_count
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int integral_stride = grid_size + 1;
    for (int query = lane; query < query_count; query += stride) {
        int x0, y0, x1, y1;
        if (!vs_point_grid_window(
                query_bounds + ((long long)query * 4),
                xmin, ymin, xmax, ymax, grid_size,
                &x0, &y0, &x1, &y1)) {
            query_counts[query] = 0;
            continue;
        }
        const long long bottom_right =
            integral_counts[(long long)(y1 + 1) * integral_stride + (x1 + 1)];
        const long long above =
            integral_counts[(long long)y0 * integral_stride + (x1 + 1)];
        const long long left =
            integral_counts[(long long)(y1 + 1) * integral_stride + x0];
        const long long corner =
            integral_counts[(long long)y0 * integral_stride + x0];
        query_counts[query] = bottom_right - above - left + corner;
    }
}

extern "C" __global__ void point_grid_query_scatter(
    const double* query_bounds,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int grid_size,
    const int* cell_counts,
    const long long* cell_offsets,
    const int* sorted_tree_rows,
    const long long* query_offsets,
    unsigned long long* query_cursors,
    int* out_left,
    int* out_right,
    int query_count
) {
    const int query = blockIdx.x;
    if (query >= query_count) return;
    int x0, y0, x1, y1;
    if (!vs_point_grid_window(
            query_bounds + ((long long)query * 4),
            xmin, ymin, xmax, ymax, grid_size,
            &x0, &y0, &x1, &y1)) {
        return;
    }
    const int width = x1 - x0 + 1;
    const int cell_count = width * (y1 - y0 + 1);
    for (int local_cell = threadIdx.x;
         local_cell < cell_count;
         local_cell += blockDim.x) {
        const int cell_x = x0 + (local_cell % width);
        const int cell_y = y0 + (local_cell / width);
        const int cell = cell_y * grid_size + cell_x;
        const int point_count = cell_counts[cell];
        if (point_count == 0) continue;
        const unsigned long long destination = atomicAdd(
            query_cursors + query,
            (unsigned long long)point_count);
        const long long source = cell_offsets[cell];
        for (int point = 0; point < point_count; ++point) {
            out_left[destination + point] = query;
            out_right[destination + point] = sorted_tree_rows[source + point];
        }
    }
}

extern "C" __global__ void point_grid_candidate_not_in_other_superset(
    const int* candidate_query_rows,
    const int* candidate_tree_rows,
    const double* query_bounds,
    const long long* other_query_counts,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int grid_size,
    long long pair_budget,
    unsigned char* out,
    int candidate_count
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const double xscale = (double)grid_size / (xmax - xmin);
    const double yscale = (double)grid_size / (ymax - ymin);
    for (int candidate = lane; candidate < candidate_count; candidate += stride) {
        const int query = candidate_query_rows[candidate];
        if (other_query_counts[query] > pair_budget) {
            out[candidate] = 0u;
            continue;
        }
        const int tree_row = candidate_tree_rows[candidate];
        const int point_row = point_row_offsets[tree_row];
        if (point_row < 0 || point_empty_mask[point_row]) {
            out[candidate] = 1u;
            continue;
        }
        const int point_coord = point_geometry_offsets[point_row];
        const double px = point_x[point_coord];
        const double py = point_y[point_coord];
        if (!isfinite(px) || !isfinite(py)) {
            out[candidate] = 1u;
            continue;
        }
        int x0, y0, x1, y1;
        if (!vs_point_grid_window(
                query_bounds + ((long long)query * 4),
                xmin, ymin, xmax, ymax, grid_size,
                &x0, &y0, &x1, &y1)) {
            out[candidate] = 1u;
            continue;
        }
        int cell_x = (int)floor((px - xmin) * xscale);
        int cell_y = (int)floor((py - ymin) * yscale);
        cell_x = max(0, min(grid_size - 1, cell_x));
        cell_y = max(0, min(grid_size - 1, cell_y));
        const bool seen =
            cell_x >= x0 && cell_x <= x1 && cell_y >= y0 && cell_y <= y1;
        out[candidate] = seen ? 0u : 1u;
    }
}
"""
