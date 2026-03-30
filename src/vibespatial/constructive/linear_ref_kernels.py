"""NVRTC kernel sources for linear_ref."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU thresholds
# ---------------------------------------------------------------------------

LINEAR_REF_GPU_THRESHOLD = 10_000
# ---------------------------------------------------------------------------
# NVRTC kernel source: interpolate along LineString
#
# 1 thread per geometry.  Walks segments accumulating length, then
# linearly interpolates the target point on the matching segment.
# ---------------------------------------------------------------------------

_INTERPOLATE_LINESTRING_KERNEL_SOURCE = r"""
extern "C" __global__ void interpolate_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const double* __restrict__ distances,
    const int normalized,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int n = ce - cs;

    if (n == 0) {
        out_x[row] = 0.0 / 0.0;  /* NaN */
        out_y[row] = 0.0 / 0.0;
        return;
    }
    if (n == 1) {
        out_x[row] = x[cs];
        out_y[row] = y[cs];
        return;
    }

    /* Compute total length if normalized. */
    double total_len = 0.0;
    if (normalized) {
        for (int i = cs; i < ce - 1; i++) {
            double dx = x[i + 1] - x[i];
            double dy = y[i + 1] - y[i];
            total_len += sqrt(dx * dx + dy * dy);
        }
    }

    double target = distances[row];
    if (normalized) {
        target = target * total_len;
    }

    /* Clamp negative distances to start. */
    if (target <= 0.0) {
        out_x[row] = x[cs];
        out_y[row] = y[cs];
        return;
    }

    /* Walk segments. */
    double accum = 0.0;
    for (int i = cs; i < ce - 1; i++) {
        double dx = x[i + 1] - x[i];
        double dy = y[i + 1] - y[i];
        double seg_len = sqrt(dx * dx + dy * dy);
        if (accum + seg_len >= target) {
            /* Interpolate on this segment. */
            double frac = (seg_len > 1e-30) ? (target - accum) / seg_len : 0.0;
            out_x[row] = x[i] + frac * dx;
            out_y[row] = y[i] + frac * dy;
            return;
        }
        accum += seg_len;
    }

    /* Distance exceeds total length: clamp to end. */
    out_x[row] = x[ce - 1];
    out_y[row] = y[ce - 1];
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel source: interpolate along MultiLineString
#
# 1 thread per geometry.  Walks parts then segments, accumulating total
# length across all parts before interpolating.
# ---------------------------------------------------------------------------

_INTERPOLATE_MULTILINESTRING_KERNEL_SOURCE = r"""
extern "C" __global__ void interpolate_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const double* __restrict__ distances,
    const int normalized,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    if (first_part == last_part) {
        out_x[row] = 0.0 / 0.0;  /* NaN */
        out_y[row] = 0.0 / 0.0;
        return;
    }

    /* Compute total length across all parts. */
    double total_len = 0.0;
    for (int p = first_part; p < last_part; p++) {
        const int cs = part_offsets[p];
        const int ce = part_offsets[p + 1];
        for (int i = cs; i < ce - 1; i++) {
            double dx = x[i + 1] - x[i];
            double dy = y[i + 1] - y[i];
            total_len += sqrt(dx * dx + dy * dy);
        }
    }

    double target = distances[row];
    if (normalized) {
        target = target * total_len;
    }

    /* Clamp negative distances to start of first part. */
    if (target <= 0.0) {
        const int cs = part_offsets[first_part];
        out_x[row] = x[cs];
        out_y[row] = y[cs];
        return;
    }

    /* Walk parts, then segments within each part. */
    double accum = 0.0;
    for (int p = first_part; p < last_part; p++) {
        const int cs = part_offsets[p];
        const int ce = part_offsets[p + 1];
        for (int i = cs; i < ce - 1; i++) {
            double dx = x[i + 1] - x[i];
            double dy = y[i + 1] - y[i];
            double seg_len = sqrt(dx * dx + dy * dy);
            if (accum + seg_len >= target) {
                double frac = (seg_len > 1e-30) ? (target - accum) / seg_len : 0.0;
                out_x[row] = x[i] + frac * dx;
                out_y[row] = y[i] + frac * dy;
                return;
            }
            accum += seg_len;
        }
    }

    /* Distance exceeds total length: clamp to end of last part. */
    const int last_ce = part_offsets[last_part] - 1;
    out_x[row] = x[last_ce];
    out_y[row] = y[last_ce];
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel source: project point onto LineString
#
# 1 thread per geometry.  For each segment, compute the closest point
# on the segment to the query point, track the minimum distance and
# accumulate the along-line distance to the projection.
# ---------------------------------------------------------------------------

_PROJECT_LINESTRING_KERNEL_SOURCE = r"""
extern "C" __global__ void project_linestring(
    const double* __restrict__ line_x,
    const double* __restrict__ line_y,
    const int* __restrict__ geometry_offsets,
    const double* __restrict__ point_x,
    const double* __restrict__ point_y,
    double* __restrict__ out_dist,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int n = ce - cs;

    if (n == 0) {
        out_dist[row] = 0.0 / 0.0;  /* NaN */
        return;
    }
    if (n == 1) {
        out_dist[row] = 0.0;
        return;
    }

    const double px = point_x[row];
    const double py = point_y[row];

    double best_sq_dist = 1e300;
    double best_along = 0.0;
    double accum = 0.0;

    for (int i = cs; i < ce - 1; i++) {
        double ax = line_x[i];
        double ay = line_y[i];
        double bx = line_x[i + 1];
        double by = line_y[i + 1];
        double dx = bx - ax;
        double dy = by - ay;
        double seg_len_sq = dx * dx + dy * dy;
        double seg_len = sqrt(seg_len_sq);

        double t = 0.0;
        if (seg_len_sq > 1e-30) {
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq;
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
        }

        double proj_x = ax + t * dx;
        double proj_y = ay + t * dy;
        double d_sq = (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y);

        if (d_sq < best_sq_dist) {
            best_sq_dist = d_sq;
            best_along = accum + t * seg_len;
        }
        accum += seg_len;
    }

    out_dist[row] = best_along;
}
"""
# ---------------------------------------------------------------------------
# Kernel names (for NVRTC precompile registry)
# ---------------------------------------------------------------------------

_INTERPOLATE_LINESTRING_KERNEL_NAMES = ("interpolate_linestring",)
_INTERPOLATE_MULTILINESTRING_KERNEL_NAMES = ("interpolate_multilinestring",)
_PROJECT_LINESTRING_KERNEL_NAMES = ("project_linestring",)
