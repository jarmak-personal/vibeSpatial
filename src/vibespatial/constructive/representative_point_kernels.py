"""NVRTC kernel sources for representative_point."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVRTC kernel: combined PIP check + horizontal ray representative point
# ---------------------------------------------------------------------------
# One thread per polygon row.  For each polygon:
# 1. Compute whether centroid is inside using even-odd ray casting.
# 2. If inside, keep centroid.
# 3. If outside (concave polygon), cast horizontal ray at centroid_y through
#    all rings (exterior + holes).  Collect X-intersection values, sort them,
#    pair them into interior intervals (even-odd), and pick the midpoint of
#    the widest interval.
# ---------------------------------------------------------------------------

_REPRESENTATIVE_POINT_KERNEL_SOURCE = r"""
extern "C" __global__ void representative_point_polygon(
    const double* centroid_x,
    const double* centroid_y,
    const double* poly_x,
    const double* poly_y,
    const int* ring_offsets,
    const int* geom_offsets,
    const int* global_row_indices,
    const int* family_row_indices,
    double* out_x,
    double* out_y,
    int num_polygons,
    int max_intersections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_polygons) return;

    const int gr = global_row_indices[idx];
    const int fr = family_row_indices[idx];
    const double cx = centroid_x[gr];
    const double cy = centroid_y[gr];

    // Determine ring range for this polygon
    const int ring_start = geom_offsets[fr];
    const int ring_end = geom_offsets[fr + 1];

    if (ring_start >= ring_end) {
        // Empty polygon - keep centroid (will be NaN for null rows)
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 1: PIP check via even-odd ray casting (horizontal ray from cx,cy
    // going right)
    bool inside = false;
    bool on_boundary = false;
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ax = poly_x[c];
            const double ay = poly_y[c];
            const double bx = poly_x[c + 1];
            const double by = poly_y[c + 1];

            // Point-on-segment check
            double edge_dx = bx - ax;
            double edge_dy = by - ay;
            double cross = (cx - ax) * edge_dy - (cy - ay) * edge_dx;
            double scale = fabs(edge_dx) + fabs(edge_dy) + 1.0;
            if (fabs(cross) <= 1e-10 * scale) {
                double minx = ax < bx ? ax : bx;
                double maxx = ax > bx ? ax : bx;
                double miny = ay < by ? ay : by;
                double maxy = ay > by ? ay : by;
                if (cx >= minx - 1e-10 && cx <= maxx + 1e-10 &&
                    cy >= miny - 1e-10 && cy <= maxy + 1e-10) {
                    on_boundary = true;
                }
            }

            // Even-odd crossing test
            if (((ay > cy) != (by > cy)) &&
                (cx < (bx - ax) * (cy - ay) / (by - ay) + ax)) {
                inside = !inside;
            }
        }
    }

    if (inside || on_boundary) {
        // Centroid is inside - use it directly
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 2: Centroid is outside. Cast horizontal ray at y=cy through all
    // rings and collect X-intersection values.
    // We use shared memory for the intersection array, allocated per-thread.
    // Since shared memory is limited, we use a fixed maximum and fall back
    // to a simple strategy if exceeded.

    // Use dynamically allocated local array (stack).  max_intersections is
    // bounded by the caller based on the maximum ring vertex count.
    // Use register-based storage for small counts, device memory for large.
    // For simplicity and correctness, we use a two-pass approach:
    // Pass 1: count intersections.  Pass 2: store and sort.

    int n_intersections = 0;

    // Pass 1: count
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ay = poly_y[c];
            const double by = poly_y[c + 1];
            if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                n_intersections++;
            }
        }
    }

    if (n_intersections < 2) {
        // Degenerate - no valid interior interval.  Use centroid as best
        // effort (this should not happen for valid polygons).
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Cap to max_intersections to avoid buffer overrun
    int cap = n_intersections < max_intersections ? n_intersections : max_intersections;

    // Pass 2: collect X-intersection values into the output scratch area.
    // We use the per-thread slice of the scratch buffer passed in via
    // a separate device array (x_scratch, allocated by the host).
    // BUT to avoid a separate scratch buffer, we use a simple insertion-sort
    // approach: maintain a sorted list in local registers.  For typical
    // polygons, the intersection count is small (< 100 edges).

    // Use the per-polygon slice of the scratch buffer.
    // The caller provides x_scratch[idx * max_intersections ... ]
    extern __shared__ double shared_scratch[];
    double* x_vals = &shared_scratch[threadIdx.x * max_intersections];

    int stored = 0;
    for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
        const int coord_start = ring_offsets[ring_idx];
        const int coord_end = ring_offsets[ring_idx + 1];
        for (int c = coord_start; c < coord_end - 1; c++) {
            const double ax = poly_x[c];
            const double ay = poly_y[c];
            const double bx = poly_x[c + 1];
            const double by = poly_y[c + 1];
            if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                double x_int = ax + (cy - ay) / (by - ay) * (bx - ax);
                if (stored < cap) {
                    // Insertion sort: find position and shift
                    int pos = stored;
                    while (pos > 0 && x_vals[pos - 1] > x_int) {
                        x_vals[pos] = x_vals[pos - 1];
                        pos--;
                    }
                    x_vals[pos] = x_int;
                    stored++;
                }
            }
        }
    }

    // Step 3: Find widest interior interval.
    // By even-odd rule, intervals are (x_vals[0],x_vals[1]),
    // (x_vals[2],x_vals[3]), ...
    double best_mid_x = cx;  // fallback
    double best_width = -1.0;
    for (int i = 0; i + 1 < stored; i += 2) {
        double width = x_vals[i + 1] - x_vals[i];
        if (width > best_width) {
            best_width = width;
            best_mid_x = (x_vals[i] + x_vals[i + 1]) * 0.5;
        }
    }

    out_x[gr] = best_mid_x;
    out_y[gr] = cy;  // Y stays at centroid Y
}


extern "C" __global__ void representative_point_multipolygon(
    const double* centroid_x,
    const double* centroid_y,
    const double* poly_x,
    const double* poly_y,
    const int* ring_offsets,
    const int* part_offsets,
    const int* geom_offsets,
    const int* global_row_indices,
    const int* family_row_indices,
    double* out_x,
    double* out_y,
    int num_polygons,
    int max_intersections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_polygons) return;

    const int gr = global_row_indices[idx];
    const int fr = family_row_indices[idx];
    const double cx = centroid_x[gr];
    const double cy = centroid_y[gr];

    // MultiPolygon: geometry_offsets -> part_offsets -> ring_offsets -> coords
    const int part_start = geom_offsets[fr];
    const int part_end = geom_offsets[fr + 1];

    if (part_start >= part_end) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Collect all ring ranges across all parts
    // Step 1: PIP check across all rings of all parts
    bool inside = false;
    bool on_boundary = false;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ax = poly_x[c];
                const double ay = poly_y[c];
                const double bx = poly_x[c + 1];
                const double by = poly_y[c + 1];

                double edge_dx = bx - ax;
                double edge_dy = by - ay;
                double cross = (cx - ax) * edge_dy - (cy - ay) * edge_dx;
                double scale = fabs(edge_dx) + fabs(edge_dy) + 1.0;
                if (fabs(cross) <= 1e-10 * scale) {
                    double minx = ax < bx ? ax : bx;
                    double maxx = ax > bx ? ax : bx;
                    double miny = ay < by ? ay : by;
                    double maxy = ay > by ? ay : by;
                    if (cx >= minx - 1e-10 && cx <= maxx + 1e-10 &&
                        cy >= miny - 1e-10 && cy <= maxy + 1e-10) {
                        on_boundary = true;
                    }
                }

                if (((ay > cy) != (by > cy)) &&
                    (cx < (bx - ax) * (cy - ay) / (by - ay) + ax)) {
                    inside = !inside;
                }
            }
        }
    }

    if (inside || on_boundary) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    // Step 2: Horizontal ray intersection for fallback
    int n_intersections = 0;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ay = poly_y[c];
                const double by = poly_y[c + 1];
                if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                    n_intersections++;
                }
            }
        }
    }

    if (n_intersections < 2) {
        out_x[gr] = cx;
        out_y[gr] = cy;
        return;
    }

    int cap = n_intersections < max_intersections ? n_intersections : max_intersections;

    extern __shared__ double shared_scratch[];
    double* x_vals = &shared_scratch[threadIdx.x * max_intersections];

    int stored = 0;
    for (int part_idx = part_start; part_idx < part_end; part_idx++) {
        const int ring_start = part_offsets[part_idx];
        const int ring_end = part_offsets[part_idx + 1];
        for (int ring_idx = ring_start; ring_idx < ring_end; ring_idx++) {
            const int coord_start = ring_offsets[ring_idx];
            const int coord_end = ring_offsets[ring_idx + 1];
            for (int c = coord_start; c < coord_end - 1; c++) {
                const double ax = poly_x[c];
                const double ay = poly_y[c];
                const double bx = poly_x[c + 1];
                const double by = poly_y[c + 1];
                if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
                    double x_int = ax + (cy - ay) / (by - ay) * (bx - ax);
                    if (stored < cap) {
                        int pos = stored;
                        while (pos > 0 && x_vals[pos - 1] > x_int) {
                            x_vals[pos] = x_vals[pos - 1];
                            pos--;
                        }
                        x_vals[pos] = x_int;
                        stored++;
                    }
                }
            }
        }
    }

    double best_mid_x = cx;
    double best_width = -1.0;
    for (int i = 0; i + 1 < stored; i += 2) {
        double width = x_vals[i + 1] - x_vals[i];
        if (width > best_width) {
            best_width = width;
            best_mid_x = (x_vals[i] + x_vals[i + 1]) * 0.5;
        }
    }

    out_x[gr] = best_mid_x;
    out_y[gr] = cy;
}
"""
_REPRESENTATIVE_POINT_KERNEL_NAMES = (
    "representative_point_polygon",
    "representative_point_multipolygon",
)
# ---------------------------------------------------------------------------
# Maximum intersections per polygon for shared-memory sizing.
# Conservative default: typical polygons have < 200 edges.  For very complex
# polygons, the kernel caps the intersection count.
# ---------------------------------------------------------------------------
_MAX_INTERSECTIONS_DEFAULT = 256
