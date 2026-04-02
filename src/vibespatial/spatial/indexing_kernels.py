"""NVRTC kernel sources for spatial indexing (Morton keys, MBR sweep, segment MBR)."""

from __future__ import annotations

from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

_INDEXING_KERNEL_SOURCE = SPATIAL_TOLERANCE_PREAMBLE + """
extern "C" __device__ unsigned long long spread_bits_32(unsigned int value) {
  unsigned long long x = (unsigned long long) value;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

extern "C" __global__ void morton_keys_from_bounds(
    const double* bounds,
    double minx,
    double miny,
    double maxx,
    double maxy,
    unsigned long long* out_keys,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int base = row * 4;
  const double bx0 = bounds[base + 0];
  const double by0 = bounds[base + 1];
  const double bx1 = bounds[base + 2];
  const double by1 = bounds[base + 3];
  if (isnan(bx0) || isnan(by0) || isnan(bx1) || isnan(by1)) {
    out_keys[row] = 0xFFFFFFFFFFFFFFFFULL;
    return;
  }
  const double span_x = fmax(maxx - minx, VS_SPATIAL_EPSILON);
  const double span_y = fmax(maxy - miny, VS_SPATIAL_EPSILON);
  const double center_x = (bx0 + bx1) * 0.5;
  const double center_y = (by0 + by1) * 0.5;
  const unsigned int norm_x = (unsigned int) llround(((center_x - minx) / span_x) * 65535.0);
  const unsigned int norm_y = (unsigned int) llround(((center_y - miny) / span_y) * 65535.0);
  out_keys[row] = spread_bits_32(norm_x) | (spread_bits_32(norm_y) << 1);
}

/* Sort-and-sweep MBR overlap test.
 *
 * Operates on a concatenated array of geometries, sorted by minx.
 * Each thread handles one element and sweeps forward through the sorted
 * array.  The sweep terminates when sorted_minx[j] > current_maxx,
 * pruning the search space from O(n) to O(k) where k is the average
 * x-overlap count.
 *
 * For same_input mode: thread i sweeps forward and emits upper-triangle
 * pairs (based on original row index) to avoid duplicates.
 *
 * For two-input mode: thread i sweeps forward and emits a pair whenever
 * the two elements come from different sides (left vs right).  Because
 * the sweep is directional (i < j in sorted order), each cross-side
 * pair is discovered exactly once -- by whichever element appears first.
 * The emitted pair is always (left_orig, right_orig) regardless of which
 * side the sweeping element belongs to.
 *
 * Two passes: pass 0 counts valid pairs per sorted element;
 *             pass 1 writes pairs using a prefix-sum offset array.
 */
extern "C" __global__ void sweep_sorted_mbr_overlap(
    const double* __restrict__ sorted_bounds,
    const int*    __restrict__ sorted_orig,
    const int*    __restrict__ sorted_side,
    int           total_count,
    int*          out_left,
    int*          out_right,
    int*          counts,
    int           pass_number,
    int           same_input,
    int           include_self
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_count) return;

  const int ibase = i * 4;
  const double ix0 = sorted_bounds[ibase + 0];
  const double iy0 = sorted_bounds[ibase + 1];
  const double ix1 = sorted_bounds[ibase + 2];
  const double iy1 = sorted_bounds[ibase + 3];
  const int i_orig = sorted_orig[i];
  const int i_side = sorted_side[i];

  int pair_count = 0;
  int write_offset = 0;
  if (pass_number == 1) write_offset = counts[i];

  /* Sweep forward: elements are sorted by minx, so once sorted_minx[j] > ix1
   * no further element can overlap in x with element i. */
  for (int j = i + 1; j < total_count; j++) {
    const int jbase = j * 4;
    const double jx0 = sorted_bounds[jbase + 0];
    /* Early exit: sorted by minx, so jx0 > ix1 means no more x-overlap */
    if (jx0 > ix1) break;

    const double jy0 = sorted_bounds[jbase + 1];
    const double jx1 = sorted_bounds[jbase + 2];
    const double jy1 = sorted_bounds[jbase + 3];

    /* Full y-axis overlap test (x-overlap guaranteed by sweep condition) */
    if (iy0 > jy1 || iy1 < jy0) continue;

    const int j_orig = sorted_orig[j];
    const int j_side = sorted_side[j];

    if (same_input) {
      /* Same-input mode: each pair (i,j) where i < j in sorted order is
       * discovered exactly once by the thread at position i.  Emit as
       * (min_orig, max_orig) for upper-triangle.  Skip self-pairs
       * (same original row index) unless include_self is set. */
      if (i_orig == j_orig && !include_self) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Emit as (smaller_orig, larger_orig) */
        if (i_orig <= j_orig) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    } else {
      /* Two-input mode: emit only cross-side pairs */
      if (i_side == j_side) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Always emit as (left_orig, right_orig) */
        if (i_side == 0) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    }
  }

  if (pass_number == 0) counts[i] = pair_count;
}
"""

_INDEXING_KERNEL_NAMES = ("morton_keys_from_bounds", "sweep_sorted_mbr_overlap")

# ---------------------------------------------------------------------------
# Segment MBR extraction kernels (Tier 1: geometry-specific inner loops)
# ---------------------------------------------------------------------------
# One kernel per geometry family.  Two-pass count/scatter:
#   Pass 0 (pass_number==0): each thread counts segments for one geometry row.
#   Pass 1 (pass_number==1): each thread scatters (row, seg_idx, minx,miny,maxx,maxy).
#
# The kernels receive coordinate buffers in SoA layout (separate x[], y[])
# plus the offset hierarchy for the family.
# ---------------------------------------------------------------------------

_SEGMENT_MBR_KERNEL_SOURCE = """
/* ---- LineString segment MBR ---- */
extern "C" __global__ void segment_mbr_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int c0 = geom_offsets[g];
    const int c1 = geom_offsets[g + 1];
    const int n_seg = c1 - c0 - 1;
    if (n_seg <= 0) {
      if (pass_number == 0) counts[g] = 0;
      continue;
    }
    if (pass_number == 0) {
      counts[g] = n_seg;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      for (int s = 0; s < n_seg; s++) {
        const int ci = c0 + s;
        const double x0 = x[ci], y0 = y[ci];
        const double x1 = x[ci + 1], y1 = y[ci + 1];
        const int idx = base + s;
        row_out[idx] = grow;
        seg_out[idx] = s;
        bounds_out[idx * 4 + 0] = fmin(x0, x1);
        bounds_out[idx * 4 + 1] = fmin(y0, y1);
        bounds_out[idx * 4 + 2] = fmax(x0, x1);
        bounds_out[idx * 4 + 3] = fmax(y0, y1);
      }
    }
  }
}

/* ---- Polygon segment MBR ---- */
extern "C" __global__ void segment_mbr_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int r0 = geom_offsets[g];
    const int r1 = geom_offsets[g + 1];
    int total = 0;
    for (int r = r0; r < r1; r++) {
      const int rc0 = ring_offsets[r];
      const int rc1 = ring_offsets[r + 1];
      total += rc1 - rc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        const int n_seg = rc1 - rc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = rc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiLineString segment MBR ---- */
extern "C" __global__ void segment_mbr_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int pc0 = part_offsets[p];
      const int pc1 = part_offsets[p + 1];
      total += pc1 - pc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int pc0 = part_offsets[p];
        const int pc1 = part_offsets[p + 1];
        const int n_seg = pc1 - pc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = pc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiPolygon segment MBR ---- */
extern "C" __global__ void segment_mbr_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int r0 = part_offsets[p];
      const int r1 = part_offsets[p + 1];
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        total += rc1 - rc0 - 1;
      }
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int r0 = part_offsets[p];
        const int r1 = part_offsets[p + 1];
        for (int r = r0; r < r1; r++) {
          const int rc0 = ring_offsets[r];
          const int rc1 = ring_offsets[r + 1];
          const int n_seg = rc1 - rc0 - 1;
          for (int s = 0; s < n_seg; s++) {
            const int ci = rc0 + s;
            const double x0 = x[ci], y0 = y[ci];
            const double x1 = x[ci + 1], y1 = y[ci + 1];
            const int idx = base + seg;
            row_out[idx] = grow;
            seg_out[idx] = seg;
            bounds_out[idx * 4 + 0] = fmin(x0, x1);
            bounds_out[idx * 4 + 1] = fmin(y0, y1);
            bounds_out[idx * 4 + 2] = fmax(x0, x1);
            bounds_out[idx * 4 + 3] = fmax(y0, y1);
            seg++;
          }
        }
      }
    }
  }
}
"""

_SEGMENT_MBR_KERNEL_NAMES = (
    "segment_mbr_linestring",
    "segment_mbr_polygon",
    "segment_mbr_multilinestring",
    "segment_mbr_multipolygon",
)
