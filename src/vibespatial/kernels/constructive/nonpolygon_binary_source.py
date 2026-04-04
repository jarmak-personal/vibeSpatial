"""CUDA kernel source for non-polygon binary constructive operations.

Contains NVRTC kernel source strings for:
- Point-LineString: point-on-segment test
- LineString-Polygon: segment clipping against polygon boundary
- LineString-LineString: segment-segment intersection

Extracted from nonpolygon_binary.py -- dispatch logic remains there.
"""
from __future__ import annotations

from vibespatial.cuda.device_functions.point_in_ring import POINT_IN_RING_DEVICE

_POINT_LINESTRING_KERNEL_SOURCE = r"""
extern "C" __global__ __launch_bounds__(256, 4) void point_linestring_on_line(
    const double* __restrict__ pt_x,
    const double* __restrict__ pt_y,
    const int* __restrict__ pt_geom_offsets,
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const int* __restrict__ valid_mask,
    int* __restrict__ out_on_line,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) {
        out_on_line[idx] = 0;
        return;
    }

    /* Point coordinates */
    const int pt_start = pt_geom_offsets[idx];
    const int pt_end = pt_geom_offsets[idx + 1];
    if (pt_end <= pt_start) {
        out_on_line[idx] = 0;
        return;
    }
    const double px = pt_x[pt_start];
    const double py = pt_y[pt_start];

    /* LineString coordinates */
    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;
    if (seg_count < 1) {
        out_on_line[idx] = 0;
        return;
    }

    /* Check each segment for point-on-segment */
    const double tol = 1e-8;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        /* Cross product to check collinearity */
        const double dx_ab = bx - ax;
        const double dy_ab = by - ay;
        const double cross = (px - ax) * dy_ab - (py - ay) * dx_ab;
        const double seg_len_sq = dx_ab * dx_ab + dy_ab * dy_ab;

        if (seg_len_sq < 1e-30) {
            /* Degenerate segment (zero length): check if point == segment point */
            if ((px - ax) * (px - ax) + (py - ay) * (py - ay) < tol * tol) {
                out_on_line[idx] = 1;
                return;
            }
            continue;
        }

        /* Relative cross product magnitude */
        if (cross * cross > tol * tol * seg_len_sq) {
            continue;  /* not collinear */
        }

        /* Project point onto segment to check if within bounds */
        const double t = ((px - ax) * dx_ab + (py - ay) * dy_ab) / seg_len_sq;
        if (t >= -tol && t <= 1.0 + tol) {
            out_on_line[idx] = 1;
            return;
        }
    }

    out_on_line[idx] = 0;
}
"""

_LINESTRING_POLYGON_KERNEL_SOURCE = POINT_IN_RING_DEVICE + r"""
/* ------------------------------------------------------------------ */
/*  Device helper: segment-segment intersection                       */
/* ------------------------------------------------------------------ */
__device__ bool seg_seg_intersect(
    double ax, double ay, double bx, double by,
    double cx, double cy, double dx, double dy,
    double* ix, double* iy
) {
    const double d1x = bx - ax, d1y = by - ay;
    const double d2x = dx - cx, d2y = dy - cy;
    const double denom = d1x * d2y - d1y * d2x;
    if (fabs(denom) < 1e-15) return false;
    const double t = ((cx - ax) * d2y - (cy - ay) * d2x) / denom;
    const double u = ((cx - ax) * d1y - (cy - ay) * d1x) / denom;
    if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) return false;
    *ix = ax + t * d1x;
    *iy = ay + t * d1y;
    return true;
}

__device__ int collect_segment_polygon_hits(
    double ax, double ay, double bx, double by,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int ring_start,
    const int poly_n,
    double* hit0_t, double* hit0_x, double* hit0_y,
    double* hit1_t, double* hit1_x, double* hit1_y
) {
    const double tol_sq = 1e-18;
    int hit_count = 0;
    *hit0_t = 2.0;
    *hit1_t = 2.0;

    const double d1x = bx - ax;
    const double d1y = by - ay;

    for (int i = 0, j = poly_n - 1; i < poly_n; j = i++) {
        double ix, iy;
        if (!seg_seg_intersect(
                ax, ay, bx, by,
                poly_x[ring_start + j], poly_y[ring_start + j],
                poly_x[ring_start + i], poly_y[ring_start + i],
                &ix, &iy)) {
            continue;
        }

        if (hit_count >= 1) {
            const double dx0 = ix - *hit0_x;
            const double dy0 = iy - *hit0_y;
            if ((dx0 * dx0 + dy0 * dy0) <= tol_sq) {
                continue;
            }
        }
        if (hit_count >= 2) {
            const double dx1 = ix - *hit1_x;
            const double dy1 = iy - *hit1_y;
            if ((dx1 * dx1 + dy1 * dy1) <= tol_sq) {
                continue;
            }
        }

        double t = 0.0;
        if (fabs(d1x) >= fabs(d1y)) {
            if (fabs(d1x) < 1e-15) continue;
            t = (ix - ax) / d1x;
        } else {
            if (fabs(d1y) < 1e-15) continue;
            t = (iy - ay) / d1y;
        }

        if (hit_count == 0) {
            *hit0_t = t;
            *hit0_x = ix;
            *hit0_y = iy;
            hit_count = 1;
        } else if (hit_count == 1) {
            *hit1_t = t;
            *hit1_x = ix;
            *hit1_y = iy;
            if (*hit1_t < *hit0_t) {
                const double tmp_t = *hit0_t;
                const double tmp_x = *hit0_x;
                const double tmp_y = *hit0_y;
                *hit0_t = *hit1_t;
                *hit0_x = *hit1_x;
                *hit0_y = *hit1_y;
                *hit1_t = tmp_t;
                *hit1_x = tmp_x;
                *hit1_y = tmp_y;
            }
            hit_count = 2;
        }
    }

    return hit_count;
}

__device__ bool segment_has_extent(
    double ax, double ay,
    double bx, double by
) {
    const double dx = bx - ax;
    const double dy = by - ay;
    return (dx * dx + dy * dy) > 1e-18;
}

__device__ void count_linestring_fragment(
    double ax, double ay,
    double bx, double by,
    bool* prev_kept,
    int* coord_count,
    int* part_count
) {
    if (!segment_has_extent(ax, ay, bx, by)) return;
    if (!(*prev_kept)) {
        *coord_count += 2;
        *part_count += 1;
    } else {
        *coord_count += 1;
    }
    *prev_kept = true;
}

__device__ void scatter_linestring_fragment(
    double ax, double ay,
    double bx, double by,
    bool* prev_kept,
    int* coord_pos,
    int* part_pos,
    int* __restrict__ out_part_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y
) {
    if (!segment_has_extent(ax, ay, bx, by)) return;
    if (!(*prev_kept)) {
        out_part_offsets[*part_pos] = *coord_pos;
        (*part_pos)++;
        out_x[*coord_pos] = ax;
        out_y[*coord_pos] = ay;
        (*coord_pos)++;
    }
    out_x[*coord_pos] = bx;
    out_y[*coord_pos] = by;
    (*coord_pos)++;
    *prev_kept = true;
}

/* ------------------------------------------------------------------ */
/*  Count kernel: count output vertices per linestring-polygon pair.  */
/*                                                                     */
/*  For intersection: output segments that are inside the polygon.     */
/*  For difference:   output segments that are outside the polygon.    */
/*                                                                     */
/*  Each segment of the linestring is tested against the polygon:      */
/*  - If both endpoints inside:  keep whole segment (2 coords if       */
/*    starting a new output segment, 1 coord if continuing)            */
/*  - If one endpoint inside:    clip at intersection point            */
/*  - If both endpoints outside: check for edge crossings              */
/*                                                                     */
/*  mode: 0 = intersection (keep inside), 1 = difference (keep outside)*/
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void linestring_polygon_count(
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int* __restrict__ poly_ring_offsets,
    const int* __restrict__ poly_geom_offsets,
    const int* __restrict__ valid_mask,
    int* __restrict__ out_counts,
    const int mode,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) {
        out_counts[idx] = 0;
        return;
    }

    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;

    /* Get polygon exterior ring */
    const int first_ring = poly_geom_offsets[idx];
    const int ring_start = poly_ring_offsets[first_ring];
    const int ring_end = poly_ring_offsets[first_ring + 1];
    const int poly_n = ring_end - ring_start;

    if (seg_count < 1 || poly_n < 3) {
        out_counts[idx] = 0;
        return;
    }

    int count = 0;
    bool prev_kept = false;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        bool a_inside = vs_ring_contains_point(ax, ay, poly_x, poly_y, ring_start, ring_end);
        bool b_inside = vs_ring_contains_point(bx, by, poly_x, poly_y, ring_start, ring_end);

        /* For intersection mode (0): keep inside segments
           For difference mode (1): keep outside segments (invert) */
        if (mode == 1) {
            a_inside = !a_inside;
            b_inside = !b_inside;
        }

        if (a_inside && b_inside) {
            /* Both inside: keep full segment */
            if (!prev_kept) {
                count += 2;  /* start + end */
            } else {
                count += 1;  /* continue: just end point */
            }
            prev_kept = true;
        } else if (a_inside && !b_inside) {
            /* A inside, B outside: clip at exit point */
            if (!prev_kept) {
                count += 2;  /* start + clip point */
            } else {
                count += 1;  /* clip point */
            }
            prev_kept = false;
        } else if (!a_inside && b_inside) {
            /* A outside, B inside: clip at entry point */
            count += 2;  /* clip point + end */
            prev_kept = true;
        } else {
            /* Both outside: keep pass-through segments that cross the polygon. */
            double hit0_t = 2.0, hit0_x = 0.0, hit0_y = 0.0;
            double hit1_t = 2.0, hit1_x = 0.0, hit1_y = 0.0;
            const int hit_count = collect_segment_polygon_hits(
                ax, ay, bx, by,
                poly_x, poly_y, ring_start, poly_n,
                &hit0_t, &hit0_x, &hit0_y,
                &hit1_t, &hit1_x, &hit1_y
            );
            if (hit_count >= 2) {
                count += 2;
            } else if (hit_count == 1) {
                count += 1;
            }
            prev_kept = false;
        }
    }

    out_counts[idx] = count;
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel: write clipped linestring vertices at offsets.      */
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void linestring_polygon_scatter(
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int* __restrict__ poly_ring_offsets,
    const int* __restrict__ poly_geom_offsets,
    const int* __restrict__ valid_mask,
    const int* __restrict__ output_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int mode,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) return;

    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;

    const int first_ring = poly_geom_offsets[idx];
    const int ring_start = poly_ring_offsets[first_ring];
    const int ring_end = poly_ring_offsets[first_ring + 1];

    if (seg_count < 1 || (ring_end - ring_start) < 3) return;

    const int poly_n = ring_end - ring_start;
    int pos = output_offsets[idx];
    bool prev_kept = false;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        bool a_inside = vs_ring_contains_point(ax, ay, poly_x, poly_y, ring_start, ring_end);
        bool b_inside = vs_ring_contains_point(bx, by, poly_x, poly_y, ring_start, ring_end);

        if (mode == 1) {
            a_inside = !a_inside;
            b_inside = !b_inside;
        }

        if (a_inside && b_inside) {
            if (!prev_kept) {
                out_x[pos] = ax;
                out_y[pos] = ay;
                pos++;
            }
            out_x[pos] = bx;
            out_y[pos] = by;
            pos++;
            prev_kept = true;
        } else if (a_inside && !b_inside) {
            /* Find clip point on polygon boundary */
            double clip_x = bx, clip_y = by;
            for (int i = 0, j = poly_n - 1; i < poly_n; j = i++) {
                double ix, iy;
                if (seg_seg_intersect(
                        ax, ay, bx, by,
                        poly_x[ring_start + j], poly_y[ring_start + j],
                        poly_x[ring_start + i], poly_y[ring_start + i],
                        &ix, &iy)) {
                    clip_x = ix;
                    clip_y = iy;
                    break;
                }
            }
            if (!prev_kept) {
                out_x[pos] = ax;
                out_y[pos] = ay;
                pos++;
            }
            out_x[pos] = clip_x;
            out_y[pos] = clip_y;
            pos++;
            prev_kept = false;
        } else if (!a_inside && b_inside) {
            double clip_x = ax, clip_y = ay;
            for (int i = 0, j = poly_n - 1; i < poly_n; j = i++) {
                double ix, iy;
                if (seg_seg_intersect(
                        ax, ay, bx, by,
                        poly_x[ring_start + j], poly_y[ring_start + j],
                        poly_x[ring_start + i], poly_y[ring_start + i],
                        &ix, &iy)) {
                    clip_x = ix;
                    clip_y = iy;
                    break;
                }
            }
            out_x[pos] = clip_x;
            out_y[pos] = clip_y;
            pos++;
            out_x[pos] = bx;
            out_y[pos] = by;
            pos++;
            prev_kept = true;
        } else {
            double hit0_t = 2.0, hit0_x = 0.0, hit0_y = 0.0;
            double hit1_t = 2.0, hit1_x = 0.0, hit1_y = 0.0;
            const int hit_count = collect_segment_polygon_hits(
                ax, ay, bx, by,
                poly_x, poly_y, ring_start, poly_n,
                &hit0_t, &hit0_x, &hit0_y,
                &hit1_t, &hit1_x, &hit1_y
            );
            if (hit_count >= 1) {
                out_x[pos] = hit0_x;
                out_y[pos] = hit0_y;
                pos++;
            }
            if (hit_count >= 2) {
                out_x[pos] = hit1_x;
                out_y[pos] = hit1_y;
                pos++;
            }
            prev_kept = false;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void linestring_polygon_difference_count(
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int* __restrict__ poly_ring_offsets,
    const int* __restrict__ poly_geom_offsets,
    const int* __restrict__ valid_mask,
    int* __restrict__ out_coord_counts,
    int* __restrict__ out_part_counts,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) {
        out_coord_counts[idx] = 0;
        out_part_counts[idx] = 0;
        return;
    }

    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;

    const int first_ring = poly_geom_offsets[idx];
    const int ring_start = poly_ring_offsets[first_ring];
    const int ring_end = poly_ring_offsets[first_ring + 1];
    const int poly_n = ring_end - ring_start;

    if (seg_count < 1 || poly_n < 3) {
        out_coord_counts[idx] = 0;
        out_part_counts[idx] = 0;
        return;
    }

    int coord_count = 0;
    int part_count = 0;
    bool prev_kept = false;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        const bool a_inside = vs_ring_contains_point(ax, ay, poly_x, poly_y, ring_start, ring_end);
        const bool b_inside = vs_ring_contains_point(bx, by, poly_x, poly_y, ring_start, ring_end);

        double hit0_t = 2.0, hit0_x = 0.0, hit0_y = 0.0;
        double hit1_t = 2.0, hit1_x = 0.0, hit1_y = 0.0;
        const int hit_count = collect_segment_polygon_hits(
            ax, ay, bx, by,
            poly_x, poly_y, ring_start, poly_n,
            &hit0_t, &hit0_x, &hit0_y,
            &hit1_t, &hit1_x, &hit1_y
        );

        if (!a_inside && !b_inside) {
            if (hit_count == 0) {
                count_linestring_fragment(
                    ax, ay, bx, by,
                    &prev_kept, &coord_count, &part_count
                );
            } else if (hit_count == 1) {
                if (hit0_t <= 1e-12 || hit0_t >= (1.0 - 1e-12)) {
                    count_linestring_fragment(
                        ax, ay, bx, by,
                        &prev_kept, &coord_count, &part_count
                    );
                } else {
                    count_linestring_fragment(
                        ax, ay, hit0_x, hit0_y,
                        &prev_kept, &coord_count, &part_count
                    );
                    prev_kept = false;
                    count_linestring_fragment(
                        hit0_x, hit0_y, bx, by,
                        &prev_kept, &coord_count, &part_count
                    );
                }
            } else {
                count_linestring_fragment(
                    ax, ay, hit0_x, hit0_y,
                    &prev_kept, &coord_count, &part_count
                );
                prev_kept = false;
                count_linestring_fragment(
                    hit1_x, hit1_y, bx, by,
                    &prev_kept, &coord_count, &part_count
                );
            }
        } else if (!a_inside && b_inside) {
            if (hit_count >= 1) {
                count_linestring_fragment(
                    ax, ay, hit0_x, hit0_y,
                    &prev_kept, &coord_count, &part_count
                );
            }
            prev_kept = false;
        } else if (a_inside && !b_inside) {
            if (hit_count >= 1) {
                const double sx = (hit_count >= 2) ? hit1_x : hit0_x;
                const double sy = (hit_count >= 2) ? hit1_y : hit0_y;
                prev_kept = false;
                count_linestring_fragment(
                    sx, sy, bx, by,
                    &prev_kept, &coord_count, &part_count
                );
            } else {
                prev_kept = false;
            }
        } else {
            if (hit_count >= 2) {
                prev_kept = false;
                count_linestring_fragment(
                    hit0_x, hit0_y, hit1_x, hit1_y,
                    &prev_kept, &coord_count, &part_count
                );
            }
            prev_kept = false;
        }
    }

    out_coord_counts[idx] = coord_count;
    out_part_counts[idx] = part_count;
}

extern "C" __global__ __launch_bounds__(256, 2) void linestring_polygon_difference_scatter(
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const double* __restrict__ poly_x,
    const double* __restrict__ poly_y,
    const int* __restrict__ poly_ring_offsets,
    const int* __restrict__ poly_geom_offsets,
    const int* __restrict__ valid_mask,
    const int* __restrict__ coord_output_offsets,
    const int* __restrict__ part_output_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int* __restrict__ out_part_offsets,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) return;

    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;

    const int first_ring = poly_geom_offsets[idx];
    const int ring_start = poly_ring_offsets[first_ring];
    const int ring_end = poly_ring_offsets[first_ring + 1];
    const int poly_n = ring_end - ring_start;

    if (seg_count < 1 || poly_n < 3) return;

    int coord_pos = coord_output_offsets[idx];
    int part_pos = part_output_offsets[idx];
    bool prev_kept = false;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        const bool a_inside = vs_ring_contains_point(ax, ay, poly_x, poly_y, ring_start, ring_end);
        const bool b_inside = vs_ring_contains_point(bx, by, poly_x, poly_y, ring_start, ring_end);

        double hit0_t = 2.0, hit0_x = 0.0, hit0_y = 0.0;
        double hit1_t = 2.0, hit1_x = 0.0, hit1_y = 0.0;
        const int hit_count = collect_segment_polygon_hits(
            ax, ay, bx, by,
            poly_x, poly_y, ring_start, poly_n,
            &hit0_t, &hit0_x, &hit0_y,
            &hit1_t, &hit1_x, &hit1_y
        );

        if (!a_inside && !b_inside) {
            if (hit_count == 0) {
                scatter_linestring_fragment(
                    ax, ay, bx, by,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
            } else if (hit_count == 1) {
                if (hit0_t <= 1e-12 || hit0_t >= (1.0 - 1e-12)) {
                    scatter_linestring_fragment(
                        ax, ay, bx, by,
                        &prev_kept, &coord_pos, &part_pos,
                        out_part_offsets, out_x, out_y
                    );
                } else {
                    scatter_linestring_fragment(
                        ax, ay, hit0_x, hit0_y,
                        &prev_kept, &coord_pos, &part_pos,
                        out_part_offsets, out_x, out_y
                    );
                    prev_kept = false;
                    scatter_linestring_fragment(
                        hit0_x, hit0_y, bx, by,
                        &prev_kept, &coord_pos, &part_pos,
                        out_part_offsets, out_x, out_y
                    );
                }
            } else {
                scatter_linestring_fragment(
                    ax, ay, hit0_x, hit0_y,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
                prev_kept = false;
                scatter_linestring_fragment(
                    hit1_x, hit1_y, bx, by,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
            }
        } else if (!a_inside && b_inside) {
            if (hit_count >= 1) {
                scatter_linestring_fragment(
                    ax, ay, hit0_x, hit0_y,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
            }
            prev_kept = false;
        } else if (a_inside && !b_inside) {
            if (hit_count >= 1) {
                const double sx = (hit_count >= 2) ? hit1_x : hit0_x;
                const double sy = (hit_count >= 2) ? hit1_y : hit0_y;
                prev_kept = false;
                scatter_linestring_fragment(
                    sx, sy, bx, by,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
            } else {
                prev_kept = false;
            }
        } else {
            if (hit_count >= 2) {
                prev_kept = false;
                scatter_linestring_fragment(
                    hit0_x, hit0_y, hit1_x, hit1_y,
                    &prev_kept, &coord_pos, &part_pos,
                    out_part_offsets, out_x, out_y
                );
            }
            prev_kept = false;
        }
    }
}
"""

_LINESTRING_LINESTRING_KERNEL_SOURCE = r"""
/* ------------------------------------------------------------------ */
/*  Count kernel: count intersection points per linestring pair       */
/* ------------------------------------------------------------------ */
extern "C" __global__ __launch_bounds__(256, 4) void linestring_linestring_count(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_geom_offsets,
    const int* __restrict__ valid_mask,
    int* __restrict__ out_counts,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) {
        out_counts[idx] = 0;
        return;
    }

    const int l_start = left_geom_offsets[idx];
    const int l_end = left_geom_offsets[idx + 1];
    const int l_segs = l_end - l_start - 1;

    const int r_start = right_geom_offsets[idx];
    const int r_end = right_geom_offsets[idx + 1];
    const int r_segs = r_end - r_start - 1;

    if (l_segs < 1 || r_segs < 1) {
        out_counts[idx] = 0;
        return;
    }

    int count = 0;
    for (int i = 0; i < l_segs; i++) {
        const double ax = left_x[l_start + i];
        const double ay = left_y[l_start + i];
        const double bx = left_x[l_start + i + 1];
        const double by = left_y[l_start + i + 1];

        for (int j = 0; j < r_segs; j++) {
            const double cx = right_x[r_start + j];
            const double cy = right_y[r_start + j];
            const double dx = right_x[r_start + j + 1];
            const double dy = right_y[r_start + j + 1];

            const double d1x = bx - ax, d1y = by - ay;
            const double d2x = dx - cx, d2y = dy - cy;
            const double denom = d1x * d2y - d1y * d2x;
            if (fabs(denom) < 1e-15) continue;

            const double t = ((cx - ax) * d2y - (cy - ay) * d2x) / denom;
            const double u = ((cx - ax) * d1y - (cy - ay) * d1x) / denom;

            if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
                count++;
            }
        }
    }
    out_counts[idx] = count;
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel: write intersection points                          */
/* ------------------------------------------------------------------ */
extern "C" __global__ __launch_bounds__(256, 4) void linestring_linestring_scatter(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_geom_offsets,
    const int* __restrict__ valid_mask,
    const int* __restrict__ output_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) return;

    const int l_start = left_geom_offsets[idx];
    const int l_end = left_geom_offsets[idx + 1];
    const int l_segs = l_end - l_start - 1;

    const int r_start = right_geom_offsets[idx];
    const int r_end = right_geom_offsets[idx + 1];
    const int r_segs = r_end - r_start - 1;

    if (l_segs < 1 || r_segs < 1) return;

    int pos = output_offsets[idx];

    for (int i = 0; i < l_segs; i++) {
        const double ax = left_x[l_start + i];
        const double ay = left_y[l_start + i];
        const double bx = left_x[l_start + i + 1];
        const double by = left_y[l_start + i + 1];

        for (int j = 0; j < r_segs; j++) {
            const double cx = right_x[r_start + j];
            const double cy = right_y[r_start + j];
            const double dx = right_x[r_start + j + 1];
            const double dy = right_y[r_start + j + 1];

            const double d1x = bx - ax, d1y = by - ay;
            const double d2x = dx - cx, d2y = dy - cy;
            const double denom = d1x * d2y - d1y * d2x;
            if (fabs(denom) < 1e-15) continue;

            const double t = ((cx - ax) * d2y - (cy - ay) * d2x) / denom;
            const double u = ((cx - ax) * d1y - (cy - ay) * d1x) / denom;

            if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
                out_x[pos] = ax + t * d1x;
                out_y[pos] = ay + t * d1y;
                pos++;
            }
        }
    }
}
"""

_POINT_LINESTRING_KERNEL_NAMES = ("point_linestring_on_line",)
_LINESTRING_POLYGON_KERNEL_NAMES = (
    "linestring_polygon_count",
    "linestring_polygon_scatter",
    "linestring_polygon_difference_count",
    "linestring_polygon_difference_scatter",
)
_LINESTRING_LINESTRING_KERNEL_NAMES = (
    "linestring_linestring_count",
    "linestring_linestring_scatter",
)
