from __future__ import annotations

from vibespatial.cccl_precompile import request_warmup
from vibespatial.cuda_runtime import get_cuda_runtime, make_kernel_cache_key

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "merge_sort_u64_i32",
    "lower_bound_i32", "lower_bound_u64",
    "upper_bound_i32", "upper_bound_u64",
    "segmented_reduce_min_f64",
])

_SPATIAL_QUERY_KERNEL_SOURCE = """
#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

extern "C" __global__ void point_regular_grid_candidates(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    double origin_x,
    double origin_y,
    double cell_width,
    double cell_height,
    int cols,
    int rows,
    int polygon_count,
    int* out_right_indices,
    unsigned char* out_counts,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int base = row * 4;
  out_right_indices[base + 0] = -1;
  out_right_indices[base + 1] = -1;
  out_right_indices[base + 2] = -1;
  out_right_indices[base + 3] = -1;
  out_counts[row] = 0;

  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    return;
  }

  const double px = point_x[point_geometry_offsets[point_row]];
  const double py = point_y[point_geometry_offsets[point_row]];
  const double xmax = origin_x + ((double) cols) * cell_width;
  const double ymax = origin_y + ((double) rows) * cell_height;
  const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
  if (px < origin_x - tol || px > xmax + tol || py < origin_y - tol || py > ymax + tol) {
    return;
  }

  int candidate_cols[2];
  int candidate_rows[2];
  int col_count = 0;
  int row_count_local = 0;

  const double fx = (px - origin_x) / cell_width;
  const int edge_col = (int) llround(fx);
  if (fabs(fx - ((double) edge_col)) <= tol / cell_width) {
    if (edge_col <= 0) {
      candidate_cols[col_count++] = 0;
    } else if (edge_col >= cols) {
      candidate_cols[col_count++] = cols - 1;
    } else {
      candidate_cols[col_count++] = edge_col - 1;
      candidate_cols[col_count++] = edge_col;
    }
  } else {
    const int col = (int) floor(fx);
    if (col < 0 || col >= cols) {
      return;
    }
    candidate_cols[col_count++] = col;
  }

  const double fy = (py - origin_y) / cell_height;
  const int edge_row = (int) llround(fy);
  if (fabs(fy - ((double) edge_row)) <= tol / cell_height) {
    if (edge_row <= 0) {
      candidate_rows[row_count_local++] = 0;
    } else if (edge_row >= rows) {
      candidate_rows[row_count_local++] = rows - 1;
    } else {
      candidate_rows[row_count_local++] = edge_row - 1;
      candidate_rows[row_count_local++] = edge_row;
    }
  } else {
    const int row_id = (int) floor(fy);
    if (row_id < 0 || row_id >= rows) {
      return;
    }
    candidate_rows[row_count_local++] = row_id;
  }

  int output_count = 0;
  for (int row_pos = 0; row_pos < row_count_local; ++row_pos) {
    for (int col_pos = 0; col_pos < col_count; ++col_pos) {
      const int polygon_row = candidate_rows[row_pos] * cols + candidate_cols[col_pos];
      if (polygon_row < 0 || polygon_row >= polygon_count) {
        continue;
      }
      bool duplicate = false;
      for (int existing = 0; existing < output_count; ++existing) {
        if (out_right_indices[base + existing] == polygon_row) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        out_right_indices[base + output_count] = polygon_row;
        ++output_count;
      }
    }
  }
  out_counts[row] = (unsigned char) output_count;
}

extern "C" __global__ void point_regular_grid_scatter_pairs(
    const int* candidate_right_indices,
    const int* candidate_offsets,
    const int* candidate_counts,
    int* out_left_indices,
    int* out_right_indices,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int count = candidate_counts[row];
  if (count <= 0) {
    return;
  }
  const int source_base = row * 4;
  const int target_base = candidate_offsets[row];
  for (int pos = 0; pos < count; ++pos) {
    out_left_indices[target_base + pos] = row;
    out_right_indices[target_base + pos] = candidate_right_indices[source_base + pos];
  }
}

extern "C" __global__ void regular_grid_box_overlap_count(
        const double* query_bounds,
        double origin_x,
        double origin_y,
        double cell_width,
        double cell_height,
        int cols,
        int rows,
        int polygon_count,
        int* out_counts,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }
    const int base = q * 4;
    const double minx = query_bounds[base + 0];
    const double miny = query_bounds[base + 1];
    const double maxx = query_bounds[base + 2];
    const double maxy = query_bounds[base + 3];
    if (isnan(minx) || isnan(miny) || isnan(maxx) || isnan(maxy)) {
        out_counts[q] = 0;
        return;
    }

    const double xmax = origin_x + ((double) cols) * cell_width;
    const double ymax = origin_y + ((double) rows) * cell_height;
    const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
    if (maxx < origin_x - tol || minx > xmax + tol || maxy < origin_y - tol || miny > ymax + tol) {
        out_counts[q] = 0;
        return;
    }

    double fx_min = (minx - origin_x) / cell_width;
    double fx_max = (maxx - origin_x) / cell_width;
    double fy_min = (miny - origin_y) / cell_height;
    double fy_max = (maxy - origin_y) / cell_height;

    int start_col = (int) floor(fx_min - (tol / cell_width));
    int end_col = (int) floor(fx_max + (tol / cell_width));
    int start_row = (int) floor(fy_min - (tol / cell_height));
    int end_row = (int) floor(fy_max + (tol / cell_height));

    if (start_col < 0) start_col = 0;
    if (start_row < 0) start_row = 0;
    if (end_col >= cols) end_col = cols - 1;
    if (end_row >= rows) end_row = rows - 1;

    if (start_col > end_col || start_row > end_row) {
        out_counts[q] = 0;
        return;
    }

    const int count = (end_col - start_col + 1) * (end_row - start_row + 1);
    out_counts[q] = count < 0 ? 0 : count;
}

extern "C" __global__ void regular_grid_box_overlap_scatter(
        const double* query_bounds,
        double origin_x,
        double origin_y,
        double cell_width,
        double cell_height,
        int cols,
        int rows,
        int polygon_count,
        const int* offsets,
        int* out_left,
        int* out_right,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }
    const int base = q * 4;
    const double minx = query_bounds[base + 0];
    const double miny = query_bounds[base + 1];
    const double maxx = query_bounds[base + 2];
    const double maxy = query_bounds[base + 3];
    if (isnan(minx) || isnan(miny) || isnan(maxx) || isnan(maxy)) {
        return;
    }

    const double xmax = origin_x + ((double) cols) * cell_width;
    const double ymax = origin_y + ((double) rows) * cell_height;
    const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
    if (maxx < origin_x - tol || minx > xmax + tol || maxy < origin_y - tol || miny > ymax + tol) {
        return;
    }

    double fx_min = (minx - origin_x) / cell_width;
    double fx_max = (maxx - origin_x) / cell_width;
    double fy_min = (miny - origin_y) / cell_height;
    double fy_max = (maxy - origin_y) / cell_height;

    int start_col = (int) floor(fx_min - (tol / cell_width));
    int end_col = (int) floor(fx_max + (tol / cell_width));
    int start_row = (int) floor(fy_min - (tol / cell_height));
    int end_row = (int) floor(fy_max + (tol / cell_height));

    if (start_col < 0) start_col = 0;
    if (start_row < 0) start_row = 0;
    if (end_col >= cols) end_col = cols - 1;
    if (end_row >= rows) end_row = rows - 1;
    if (start_col > end_col || start_row > end_row) {
        return;
    }

    int write_pos = offsets[q];
    for (int row_id = start_row; row_id <= end_row; ++row_id) {
        for (int col_id = start_col; col_id <= end_col; ++col_id) {
            const int polygon_row = row_id * cols + col_id;
            if (polygon_row < 0 || polygon_row >= polygon_count) {
                continue;
            }
            out_left[write_pos] = q;
            out_right[write_pos] = polygon_row;
            ++write_pos;
        }
    }
}

extern "C" __global__ void point_box_query_mask(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    double minx,
    double miny,
    double maxx,
    double maxy,
    int predicate_mode,
    unsigned char* out_mask,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }

  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    out_mask[row] = 0;
    return;
  }

  const int coord_row = point_geometry_offsets[point_row];
  const double px = point_x[coord_row];
  const double py = point_y[coord_row];
  const bool inclusive = (minx <= px && px <= maxx && miny <= py && py <= maxy);
  const bool strict_inside = (minx < px && px < maxx && miny < py && py < maxy);
  bool match = false;
  if (predicate_mode == 1) {
    match = strict_inside;
  } else if (predicate_mode == 2) {
    match = inclusive && !strict_inside;
  } else {
    match = inclusive;
  }
  out_mask[row] = match ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Generic GPU bbox overlap candidate generation
// ---------------------------------------------------------------------------
// Tier 1 NVRTC kernel per ADR-0033: geometry-specific bbox overlap test.
// Scalar path (Q=1): one thread per tree row, checks overlap with single query.
// Multi-query path (Q>1): one thread per query row, loops over all tree rows.

extern "C" __global__ void bbox_overlap_tree_mask(
    const double* tree_bounds,
    double query_minx,
    double query_miny,
    double query_maxx,
    double query_maxy,
    unsigned char* out_mask,
    int tree_count
) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tree_count) {
    return;
  }
  const int base = t * 4;
  const double tminx = tree_bounds[base + 0];
  const double tminy = tree_bounds[base + 1];
  const double tmaxx = tree_bounds[base + 2];
  const double tmaxy = tree_bounds[base + 3];
  if (isnan(tminx) || isnan(tminy) || isnan(tmaxx) || isnan(tmaxy)) {
    out_mask[t] = 0;
    return;
  }
  out_mask[t] = (query_minx <= tmaxx && query_maxx >= tminx &&
                  query_miny <= tmaxy && query_maxy >= tminy) ? 1 : 0;
}

extern "C" __global__ void bbox_overlap_multi_count(
    const double* query_bounds,
    const double* tree_bounds,
    int query_count,
    int tree_count,
    int* out_counts
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_count) {
    return;
  }
  const int qbase = q * 4;
  const double qminx = query_bounds[qbase + 0];
  const double qminy = query_bounds[qbase + 1];
  const double qmaxx = query_bounds[qbase + 2];
  const double qmaxy = query_bounds[qbase + 3];
  if (isnan(qminx) || isnan(qminy) || isnan(qmaxx) || isnan(qmaxy)) {
    out_counts[q] = 0;
    return;
  }
  int count = 0;
  for (int t = 0; t < tree_count; t++) {
    const int tbase = t * 4;
    const double tminx = tree_bounds[tbase + 0];
    const double tminy = tree_bounds[tbase + 1];
    const double tmaxx = tree_bounds[tbase + 2];
    const double tmaxy = tree_bounds[tbase + 3];
    if (isnan(tminx)) {
      continue;
    }
    if (qminx <= tmaxx && qmaxx >= tminx &&
        qminy <= tmaxy && qmaxy >= tminy) {
      count++;
    }
  }
  out_counts[q] = count;
}

extern "C" __global__ void bbox_overlap_multi_scatter(
    const double* query_bounds,
    const double* tree_bounds,
    int query_count,
    int tree_count,
    const int* offsets,
    int* out_left,
    int* out_right
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_count) {
    return;
  }
  const int qbase = q * 4;
  const double qminx = query_bounds[qbase + 0];
  const double qminy = query_bounds[qbase + 1];
  const double qmaxx = query_bounds[qbase + 2];
  const double qmaxy = query_bounds[qbase + 3];
  if (isnan(qminx) || isnan(qminy) || isnan(qmaxx) || isnan(qmaxy)) {
    return;
  }
  int write_pos = offsets[q];
  for (int t = 0; t < tree_count; t++) {
    const int tbase = t * 4;
    const double tminx = tree_bounds[tbase + 0];
    const double tminy = tree_bounds[tbase + 1];
    const double tmaxx = tree_bounds[tbase + 2];
    const double tmaxy = tree_bounds[tbase + 3];
    if (isnan(tminx)) {
      continue;
    }
    if (qminx <= tmaxx && qmaxx >= tminx &&
        qminy <= tmaxy && qmaxy >= tminy) {
      out_left[write_pos] = q;
      out_right[write_pos] = t;
      write_pos++;
    }
  }
}

// ---------------------------------------------------------------------------
// Point-point Euclidean distance for nearest-neighbour candidate pairs.
// Tier 1 NVRTC kernel per ADR-0033: geometry-specific distance computation.
// One thread per candidate pair.  NaN coordinates → INFINITY distance.
// When *exclusive* is set, pairs with identical coordinates also get INFINITY
// so that the downstream segmented-min naturally ignores them.
// ---------------------------------------------------------------------------
extern "C" __global__ void point_point_distance_pairs(
    const double* query_x,
    const double* query_y,
    const double* tree_x,
    const double* tree_y,
    const int* left_idx,
    const int* right_idx,
    double* out_distances,
    int exclusive,
    int pair_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;
  const int li = left_idx[i];
  const int ri = right_idx[i];
  const double qx = query_x[li];
  const double qy = query_y[li];
  const double tx = tree_x[ri];
  const double ty = tree_y[ri];
  if (isnan(qx) || isnan(qy) || isnan(tx) || isnan(ty)) {
    out_distances[i] = INFINITY;
    return;
  }
  if (exclusive && qx == tx && qy == ty) {
    out_distances[i] = INFINITY;
    return;
  }
  const double dx = qx - tx;
  const double dy = qy - ty;
  out_distances[i] = sqrt(dx * dx + dy * dy);
}

extern "C" __global__ void point_point_distance_pairs_from_owned(
        const unsigned char* query_validity,
        const signed char* query_tags,
        const int* query_family_row_offsets,
        const int* query_geometry_offsets,
        const unsigned char* query_empty_mask,
        const double* query_x,
        const double* query_y,
        int query_point_tag,
        const unsigned char* tree_validity,
        const signed char* tree_tags,
        const int* tree_family_row_offsets,
        const int* tree_geometry_offsets,
        const unsigned char* tree_empty_mask,
        const double* tree_x,
        const double* tree_y,
        int tree_point_tag,
        const int* left_idx,
        const int* right_idx,
        double* out_distances,
        int exclusive,
        int pair_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) return;

    const int li = left_idx[i];
    const int ri = right_idx[i];

    if (!query_validity[li] || !tree_validity[ri]) {
        out_distances[i] = INFINITY;
        return;
    }
    if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_point_tag) {
        out_distances[i] = INFINITY;
        return;
    }

    const int qrow = query_family_row_offsets[li];
    const int trow = tree_family_row_offsets[ri];
    if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {
        out_distances[i] = INFINITY;
        return;
    }

    const int qcoord = query_geometry_offsets[qrow];
    const int tcoord = tree_geometry_offsets[trow];
    const double qx = query_x[qcoord];
    const double qy = query_y[qcoord];
    const double tx = tree_x[tcoord];
    const double ty = tree_y[tcoord];
    if (isnan(qx) || isnan(qy) || isnan(tx) || isnan(ty)) {
        out_distances[i] = INFINITY;
        return;
    }
    if (exclusive && qx == tx && qy == ty) {
        out_distances[i] = INFINITY;
        return;
    }
    const double dx = qx - tx;
    const double dy = qy - ty;
    out_distances[i] = sqrt(dx * dx + dy * dy);
}

extern "C" __global__ void point_x_window_count(
        const double* query_x,
        const double* query_y,
        const double* tree_y,
        const double* sorted_tree_x,
        const int* sorted_tree_rows,
        const int* start_idx,
        const int* end_idx,
        double max_distance,
        int exclusive,
        int* out_counts,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    if (isnan(qx) || isnan(qy)) {
        out_counts[q] = 0;
        return;
    }

    const int start = start_idx[q];
    const int stop = end_idx[q];
    int count = 0;
    for (int idx = start; idx < stop; ++idx) {
        const int tree_row = sorted_tree_rows[idx];
        const double tx = sorted_tree_x[idx];
        const double ty = tree_y[tree_row];
        if (isnan(ty)) {
            continue;
        }
        if (exclusive && qx == tx && qy == ty) {
            continue;
        }
        if (fabs(ty - qy) <= max_distance) {
            ++count;
        }
    }
    out_counts[q] = count;
}

extern "C" __global__ void point_x_window_scatter(
        const double* query_x,
        const double* query_y,
        const double* tree_y,
        const double* sorted_tree_x,
        const int* sorted_tree_rows,
        const int* start_idx,
        const int* end_idx,
        const int* offsets,
        double max_distance,
        int exclusive,
        int* out_left,
        int* out_right,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    if (isnan(qx) || isnan(qy)) {
        return;
    }

    const int start = start_idx[q];
    const int stop = end_idx[q];
    int write_pos = offsets[q];
    for (int idx = start; idx < stop; ++idx) {
        const int tree_row = sorted_tree_rows[idx];
        const double tx = sorted_tree_x[idx];
        const double ty = tree_y[tree_row];
        if (isnan(ty)) {
            continue;
        }
        if (exclusive && qx == tx && qy == ty) {
            continue;
        }
        if (fabs(ty - qy) <= max_distance) {
            out_left[write_pos] = q;
            out_right[write_pos] = tree_row;
            ++write_pos;
        }
    }
}

extern "C" __global__ void point_regular_grid_nearest_count(
        const double* query_x,
        const double* query_y,
        double origin_x,
        double origin_y,
        double cell_width,
        double cell_height,
        int cols,
        int rows,
        int point_count,
        double max_distance,
        int exclusive,
        int* out_counts,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    if (isnan(qx) || isnan(qy)) {
        out_counts[q] = 0;
        return;
    }

    const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
    const int start_col = max(0, (int) floor(((qx - max_distance) - origin_x) / cell_width - (tol / cell_width)));
    const int end_col = min(cols - 1, (int) floor(((qx + max_distance) - origin_x) / cell_width + (tol / cell_width)));
    const int start_row = max(0, (int) floor(((qy - max_distance) - origin_y) / cell_height - (tol / cell_height)));
    const int end_row = min(rows - 1, (int) floor(((qy + max_distance) - origin_y) / cell_height + (tol / cell_height)));
    if (start_col > end_col || start_row > end_row) {
        out_counts[q] = 0;
        return;
    }

    int count = 0;
    for (int row_id = start_row; row_id <= end_row; ++row_id) {
        for (int col_id = start_col; col_id <= end_col; ++col_id) {
            const int tree_row = row_id * cols + col_id;
            if (tree_row < 0 || tree_row >= point_count) {
                continue;
            }
            const double tx = origin_x + ((double) col_id) * cell_width;
            const double ty = origin_y + ((double) row_id) * cell_height;
            if (exclusive && qx == tx && qy == ty) {
                continue;
            }
            ++count;
        }
    }
    out_counts[q] = count;
}

extern "C" __global__ void point_regular_grid_nearest_scatter(
        const double* query_x,
        const double* query_y,
        double origin_x,
        double origin_y,
        double cell_width,
        double cell_height,
        int cols,
        int rows,
        int point_count,
        const int* offsets,
        double max_distance,
        int exclusive,
        int* out_left,
        int* out_right,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    if (isnan(qx) || isnan(qy)) {
        return;
    }

    const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
    const int start_col = max(0, (int) floor(((qx - max_distance) - origin_x) / cell_width - (tol / cell_width)));
    const int end_col = min(cols - 1, (int) floor(((qx + max_distance) - origin_x) / cell_width + (tol / cell_width)));
    const int start_row = max(0, (int) floor(((qy - max_distance) - origin_y) / cell_height - (tol / cell_height)));
    const int end_row = min(rows - 1, (int) floor(((qy + max_distance) - origin_y) / cell_height + (tol / cell_height)));
    if (start_col > end_col || start_row > end_row) {
        return;
    }

    int write_pos = offsets[q];
    for (int row_id = start_row; row_id <= end_row; ++row_id) {
        for (int col_id = start_col; col_id <= end_col; ++col_id) {
            const int tree_row = row_id * cols + col_id;
            if (tree_row < 0 || tree_row >= point_count) {
                continue;
            }
            const double tx = origin_x + ((double) col_id) * cell_width;
            const double ty = origin_y + ((double) row_id) * cell_height;
            if (exclusive && qx == tx && qy == ty) {
                continue;
            }
            out_left[write_pos] = q;
            out_right[write_pos] = tree_row;
            ++write_pos;
        }
    }
}

extern "C" __global__ void point_nearest_min_sq_from_sorted_x(
        const double* query_x,
        const double* query_y,
        const double* sorted_tree_x,
        const double* tree_y,
        const int* sorted_tree_rows,
        const int* insert_idx,
        int tree_count,
        int exclusive,
        double* out_min_sq,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    if (isnan(qx) || isnan(qy) || tree_count <= 0) {
        out_min_sq[q] = INFINITY;
        return;
    }

    int left = insert_idx[q] - 1;
    int right = insert_idx[q];
    double best_sq = INFINITY;

    while (left >= 0 || right < tree_count) {
        double left_dx = INFINITY;
        double right_dx = INFINITY;
        if (left >= 0) {
            left_dx = fabs(qx - sorted_tree_x[left]);
        }
        if (right < tree_count) {
            right_dx = fabs(sorted_tree_x[right] - qx);
        }

        const double next_dx = fmin(left_dx, right_dx);
        if (best_sq < INFINITY && next_dx * next_dx > best_sq) {
            break;
        }

        int idx = -1;
        if (left_dx <= right_dx) {
            idx = left;
            --left;
        } else {
            idx = right;
            ++right;
        }

        if (idx < 0 || idx >= tree_count) {
            continue;
        }
        const int tree_row = sorted_tree_rows[idx];
        const double tx = sorted_tree_x[idx];
        const double ty = tree_y[tree_row];
        if (isnan(ty)) {
            continue;
        }
        if (exclusive && tx == qx && ty == qy) {
            continue;
        }
        const double dx = tx - qx;
        const double dy = ty - qy;
        const double sq = dx * dx + dy * dy;
        if (sq < best_sq) {
            best_sq = sq;
        }
    }

    out_min_sq[q] = best_sq;
}

extern "C" __global__ void point_nearest_tie_count_from_sorted_x(
        const double* query_x,
        const double* query_y,
        const double* sorted_tree_x,
        const double* tree_y,
        const int* sorted_tree_rows,
        const int* start_idx,
        const int* end_idx,
        const double* min_sq,
        int exclusive,
        int* out_counts,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    const double best_sq = min_sq[q];
    if (isnan(qx) || isnan(qy) || !isfinite(best_sq)) {
        out_counts[q] = 0;
        return;
    }

    const double best = sqrt(best_sq);
    const double tol = 1e-8 + 1e-5 * fabs(best);
    const int start = start_idx[q];
    const int stop = end_idx[q];
    int count = 0;
    for (int idx = start; idx < stop; ++idx) {
        const int tree_row = sorted_tree_rows[idx];
        const double tx = sorted_tree_x[idx];
        const double ty = tree_y[tree_row];
        if (isnan(ty)) {
            continue;
        }
        if (exclusive && tx == qx && ty == qy) {
            continue;
        }
        const double dx = tx - qx;
        const double dy = ty - qy;
        const double dist = sqrt(dx * dx + dy * dy);
        if (fabs(dist - best) <= tol) {
            ++count;
        }
    }
    out_counts[q] = count;
}

extern "C" __global__ void point_nearest_tie_scatter_from_sorted_x(
        const double* query_x,
        const double* query_y,
        const double* sorted_tree_x,
        const double* tree_y,
        const int* sorted_tree_rows,
        const int* start_idx,
        const int* end_idx,
        const int* offsets,
        const double* min_sq,
        int exclusive,
        int* out_left,
        int* out_right,
        int query_count
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) {
        return;
    }

    const double qx = query_x[q];
    const double qy = query_y[q];
    const double best_sq = min_sq[q];
    if (isnan(qx) || isnan(qy) || !isfinite(best_sq)) {
        return;
    }

    const double best = sqrt(best_sq);
    const double tol = 1e-8 + 1e-5 * fabs(best);
    const int start = start_idx[q];
    const int stop = end_idx[q];
    int write_pos = offsets[q];
    for (int idx = start; idx < stop; ++idx) {
        const int tree_row = sorted_tree_rows[idx];
        const double tx = sorted_tree_x[idx];
        const double ty = tree_y[tree_row];
        if (isnan(ty)) {
            continue;
        }
        if (exclusive && tx == qx && ty == qy) {
            continue;
        }
        const double dx = tx - qx;
        const double dy = ty - qy;
        const double dist = sqrt(dx * dx + dy * dy);
        if (fabs(dist - best) <= tol) {
            out_left[write_pos] = q;
            out_right[write_pos] = tree_row;
            ++write_pos;
        }
    }
}

// ---------------------------------------------------------------------------
// Build keep-mask for nearest pairs (return_all=True variant).
// Tier 1 NVRTC: fuses isclose + max_distance filter in a single pass.
// Uses NumPy isclose defaults: atol=1e-8, rtol=1e-5.
// ---------------------------------------------------------------------------
extern "C" __global__ void nearest_keep_mask(
    const double* distances,
    const double* min_distances,
    const int* left_idx,
    unsigned char* out_keep,
    double max_distance,
    int pair_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;
  const double d = distances[i];
  const double mind = min_distances[left_idx[i]];
  const double atol = 1e-8;
  const double rtol = 1e-5;
  const bool close = fabs(d - mind) <= (atol + rtol * fabs(mind));
  const bool within = (d <= max_distance);
  out_keep[i] = (close && within) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Select only the first kept pair per query segment (return_all=False).
// One thread per segment.  Walks the segment in order and marks only the
// first pair whose keep_mask entry is set.
// ---------------------------------------------------------------------------
extern "C" __global__ void nearest_first_per_segment(
    const unsigned char* keep_mask,
    unsigned char* out_mask,
    const int* seg_starts,
    const int* seg_ends,
    int num_segments
) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= num_segments) return;
  const int start = seg_starts[seg];
  const int end = seg_ends[seg];
  bool found = false;
  for (int i = start; i < end; i++) {
    if (!found && keep_mask[i]) {
      out_mask[i] = 1;
      found = true;
    } else {
      out_mask[i] = 0;
    }
  }
}
"""

_SPATIAL_QUERY_KERNEL_NAMES = (
    "point_regular_grid_candidates",
    "point_regular_grid_scatter_pairs",
    "regular_grid_box_overlap_count",
    "regular_grid_box_overlap_scatter",
    "point_box_query_mask",
    "bbox_overlap_tree_mask",
    "bbox_overlap_multi_count",
    "bbox_overlap_multi_scatter",
    "point_point_distance_pairs",
    "point_point_distance_pairs_from_owned",
    "point_x_window_count",
    "point_x_window_scatter",
    "point_regular_grid_nearest_count",
    "point_regular_grid_nearest_scatter",
    "point_nearest_min_sq_from_sorted_x",
    "point_nearest_tie_count_from_sorted_x",
    "point_nearest_tie_scatter_from_sorted_x",
    "nearest_keep_mask",
    "nearest_first_per_segment",
)


def _spatial_query_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("spatial-query", _SPATIAL_QUERY_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_SPATIAL_QUERY_KERNEL_SOURCE,
        kernel_names=_SPATIAL_QUERY_KERNEL_NAMES,
    )


_MORTON_RANGE_KERNEL_SOURCE = """
extern "C" __device__ unsigned long long _mr_spread_bits_32(unsigned int value) {
  unsigned long long x = (unsigned long long) value;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

extern "C" __global__ void morton_range_from_bounds(
    const double* query_bounds,
    double total_minx, double total_miny,
    double total_maxx, double total_maxy,
    unsigned long long* out_range_low,
    unsigned long long* out_range_high,
    int query_count
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_count) return;
  const int base = q * 4;
  const double qx0 = query_bounds[base];
  const double qy0 = query_bounds[base + 1];
  const double qx1 = query_bounds[base + 2];
  const double qy1 = query_bounds[base + 3];

  if (isnan(qx0) || isnan(qy0) || isnan(qx1) || isnan(qy1)) {
    out_range_low[q] = 1ULL;
    out_range_high[q] = 0ULL;
    return;
  }

  const double span_x = fmax(total_maxx - total_minx, 1e-12);
  const double span_y = fmax(total_maxy - total_miny, 1e-12);
  unsigned int nx0 = (unsigned int) max(min(llround(((qx0 - total_minx) / span_x) * 65535.0), 65535LL), 0LL);
  unsigned int ny0 = (unsigned int) max(min(llround(((qy0 - total_miny) / span_y) * 65535.0), 65535LL), 0LL);
  unsigned int nx1 = (unsigned int) max(min(llround(((qx1 - total_minx) / span_x) * 65535.0), 65535LL), 0LL);
  unsigned int ny1 = (unsigned int) max(min(llround(((qy1 - total_miny) / span_y) * 65535.0), 65535LL), 0LL);

  unsigned long long m_low = _mr_spread_bits_32(nx0) | (_mr_spread_bits_32(ny0) << 1);
  unsigned long long m_high = _mr_spread_bits_32(nx1) | (_mr_spread_bits_32(ny1) << 1);
  if (m_low > m_high) {
    unsigned long long tmp = m_low;
    m_low = m_high;
    m_high = tmp;
  }

  unsigned long long xor_val = m_low ^ m_high;
  if (xor_val == 0ULL) {
    out_range_low[q] = m_low;
    out_range_high[q] = m_high;
  } else {
    int leading = __clzll(xor_val);
    unsigned long long mask = ~((1ULL << (64 - leading)) - 1ULL);
    out_range_low[q] = m_low & mask;
    out_range_high[q] = (m_low & mask) | (~mask);
  }
}

extern "C" __global__ void morton_range_count(
    const unsigned long long* starts,
    const unsigned long long* ends,
    const double* sorted_tree_bounds,
    const double* query_bounds,
    int* out_counts,
    int query_count
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_count) return;

  const unsigned long long start = starts[q];
  const unsigned long long end = ends[q];
  const int qbase = q * 4;
  const double qx0 = query_bounds[qbase];
  const double qy0 = query_bounds[qbase + 1];
  const double qx1 = query_bounds[qbase + 2];
  const double qy1 = query_bounds[qbase + 3];

  int count = 0;
  if (!isnan(qx0)) {
    for (unsigned long long i = start; i < end; i++) {
      const int tbase = (int)i * 4;
      const double tx0 = sorted_tree_bounds[tbase];
      const double ty0 = sorted_tree_bounds[tbase + 1];
      const double tx1 = sorted_tree_bounds[tbase + 2];
      const double ty1 = sorted_tree_bounds[tbase + 3];
      if (!isnan(tx0) && qx0 <= tx1 && qx1 >= tx0 && qy0 <= ty1 && qy1 >= ty0) {
        count++;
      }
    }
  }
  out_counts[q] = count;
}

extern "C" __global__ void morton_range_scatter(
    const unsigned long long* starts,
    const unsigned long long* ends,
    const int* order,
    const double* sorted_tree_bounds,
    const double* query_bounds,
    const int* offsets,
    int* out_left,
    int* out_right,
    int query_count
) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_count) return;

  const unsigned long long start = starts[q];
  const unsigned long long end = ends[q];
  const int qbase = q * 4;
  const double qx0 = query_bounds[qbase];
  const double qy0 = query_bounds[qbase + 1];
  const double qx1 = query_bounds[qbase + 2];
  const double qy1 = query_bounds[qbase + 3];

  int write_pos = offsets[q];
  if (!isnan(qx0)) {
    for (unsigned long long i = start; i < end; i++) {
      const int tbase = (int)i * 4;
      const double tx0 = sorted_tree_bounds[tbase];
      const double ty0 = sorted_tree_bounds[tbase + 1];
      const double tx1 = sorted_tree_bounds[tbase + 2];
      const double ty1 = sorted_tree_bounds[tbase + 3];
      if (!isnan(tx0) && qx0 <= tx1 && qx1 >= tx0 && qy0 <= ty1 && qy1 >= ty0) {
        out_left[write_pos] = q;
        out_right[write_pos] = order[(int)i];
        write_pos++;
      }
    }
  }
}
"""

_MORTON_RANGE_KERNEL_NAMES = (
    "morton_range_from_bounds",
    "morton_range_count",
    "morton_range_scatter",
)

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("spatial-query", _SPATIAL_QUERY_KERNEL_SOURCE, _SPATIAL_QUERY_KERNEL_NAMES),
    ("morton-range", _MORTON_RANGE_KERNEL_SOURCE, _MORTON_RANGE_KERNEL_NAMES),
])


def _morton_range_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("morton-range", _MORTON_RANGE_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_MORTON_RANGE_KERNEL_SOURCE,
        kernel_names=_MORTON_RANGE_KERNEL_NAMES,
    )
