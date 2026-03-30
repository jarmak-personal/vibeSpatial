"""NVRTC kernel sources for linestring."""

from __future__ import annotations

_LINESTRING_BUFFER_KERNEL_SOURCE = r"""
#define PI 3.14159265358979323846
#define EPSILON 1e-12

/* Join/cap style constants — must match Python-side encoding */
#define JOIN_ROUND 0
#define JOIN_MITRE 1
#define JOIN_BEVEL 2
#define CAP_ROUND  0
#define CAP_FLAT   1
#define CAP_SQUARE 2

/* ------------------------------------------------------------------ */
/*  Device helpers                                                     */
/* ------------------------------------------------------------------ */

__device__ void normalize_2d(double dx, double dy,
                             double* nx, double* ny, double* len) {
    double l = sqrt(dx * dx + dy * dy);
    *len = l;
    if (l < EPSILON) { *nx = 0.0; *ny = 0.0; return; }
    *nx = dx / l;
    *ny = dy / l;
}

__device__ double cross_2d(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

__device__ int arc_vertex_count(double sweep_angle, int quad_segs) {
    double step = PI / (2.0 * quad_segs);
    int n = (int)ceil(fabs(sweep_angle) / step);
    return n < 1 ? 1 : n;
}

__device__ void emit_arc_verts(double cx, double cy, double radius,
                               double start_angle, double sweep,
                               int n_steps,
                               double* out_x, double* out_y, int* pos) {
    for (int i = 1; i <= n_steps; i++) {
        double angle = start_angle + sweep * ((double)i / (double)n_steps);
        out_x[*pos] = cx + radius * cos(angle);
        out_y[*pos] = cy + radius * sin(angle);
        (*pos)++;
    }
}

/* Count vertices for a convex join (outside of turn).
   Returns number of vertices to emit. */
__device__ int convex_join_count(
    double prev_nx, double prev_ny, double curr_nx, double curr_ny,
    double prev_dx, double prev_dy, double curr_dx, double curr_dy,
    int join_style, int quad_segs, double mitre_limit
) {
    if (join_style == JOIN_ROUND) {
        double start_a = atan2(prev_ny, prev_nx);
        double end_a   = atan2(curr_ny, curr_nx);
        double sweep = end_a - start_a;
        if (sweep > 0.0) sweep -= 2.0 * PI;
        return 1 + arc_vertex_count(sweep, quad_segs);  /* arrival + arc */
    }
    if (join_style == JOIN_MITRE) {
        double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
        /* mitre_ratio^2 = 2/(1-dot_d).  Exceeds limit → bevel. */
        if ((1.0 - dot_d) * mitre_limit * mitre_limit < 2.0) {
            return 2;  /* bevel fallback: arrival + departure */
        }
        return 1;  /* mitre intersection */
    }
    /* JOIN_BEVEL */
    return 2;  /* arrival + departure */
}

/* Count vertices for a cap.
   Round=2*q, Flat=0, Square=2. */
__device__ int cap_count(int cap_style, int quad_segs) {
    if (cap_style == CAP_ROUND)  return 2 * quad_segs;
    if (cap_style == CAP_SQUARE) return 2;
    return 0;  /* CAP_FLAT */
}

/* ------------------------------------------------------------------ */
/*  Count kernel                                                       */
/*                                                                     */
/*  Left turn  (cross > 0): left=concave, right=convex                */
/*  Right turn (cross < 0): left=convex,  right=concave               */
/* ------------------------------------------------------------------ */

extern "C" __global__ void linestring_buffer_count(
    const int* line_row_offsets,
    const int* line_geometry_offsets,
    const unsigned char* line_empty_mask,
    const double* line_x,
    const double* line_y,
    const double* distances,
    int quad_segs,
    int join_style,
    int cap_style,
    double mitre_limit,
    int* out_counts,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int line_row = line_row_offsets[row];
    if (line_row < 0 || line_empty_mask[line_row]) {
        out_counts[row] = 0;
        return;
    }

    const int coord_start = line_geometry_offsets[line_row];
    const int coord_end   = line_geometry_offsets[line_row + 1];
    const int N = coord_end - coord_start;

    if (N < 2) {
        if (N == 1) {
            out_counts[row] = 4 * quad_segs + 1;
        } else {
            out_counts[row] = 0;
        }
        return;
    }

    const int n_segs = N - 1;
    const int q = quad_segs;
    double prev_nx, prev_ny, curr_nx, curr_ny;
    double seg_len;
    double prev_dx, prev_dy, curr_dx, curr_dy;

    int total = 0;

    /* === Left side forward === */
    total += 1;  /* start vertex */

    normalize_2d(line_x[coord_start + 1] - line_x[coord_start],
                 line_y[coord_start + 1] - line_y[coord_start],
                 &prev_dx, &prev_dy, &seg_len);
    prev_nx = -prev_dy;
    prev_ny =  prev_dx;

    for (int j = 1; j < n_segs; j++) {
        int ci = coord_start + j;
        normalize_2d(line_x[ci + 1] - line_x[ci],
                     line_y[ci + 1] - line_y[ci],
                     &curr_dx, &curr_dy, &seg_len);
        curr_nx = -curr_dy;
        curr_ny =  curr_dx;

        double cross = cross_2d(prev_dx, prev_dy, curr_dx, curr_dy);
        if (cross < -EPSILON) {
            /* Right turn → convex on left */
            total += convex_join_count(
                prev_nx, prev_ny, curr_nx, curr_ny,
                prev_dx, prev_dy, curr_dx, curr_dy,
                join_style, q, mitre_limit);
        } else {
            /* Left turn (concave) or collinear → single point */
            total += 1;
        }

        prev_dx = curr_dx; prev_dy = curr_dy;
        prev_nx = curr_nx; prev_ny = curr_ny;
    }

    total += 1;  /* left side end vertex */

    /* === Right cap === */
    total += cap_count(cap_style, q);
    total += 1;  /* right side start vertex */

    /* === Right side backward === */
    normalize_2d(line_x[coord_end - 1] - line_x[coord_end - 2],
                 line_y[coord_end - 1] - line_y[coord_end - 2],
                 &prev_dx, &prev_dy, &seg_len);
    prev_nx = -prev_dy;
    prev_ny =  prev_dx;

    for (int j = n_segs - 1; j >= 1; j--) {
        int ci = coord_start + j;
        normalize_2d(line_x[ci] - line_x[ci - 1],
                     line_y[ci] - line_y[ci - 1],
                     &curr_dx, &curr_dy, &seg_len);
        curr_nx = -curr_dy;
        curr_ny =  curr_dx;

        /* cross = forward cross at vertex j */
        double cross = cross_2d(curr_dx, curr_dy, prev_dx, prev_dy);
        if (cross > EPSILON) {
            /* Left turn → convex on right.
               Arc sweep magnitude is the same for left/right normals,
               and mitre/bevel don't use normals, so we can pass left normals. */
            total += convex_join_count(
                prev_nx, prev_ny, curr_nx, curr_ny,
                prev_dx, prev_dy, curr_dx, curr_dy,
                join_style, q, mitre_limit);
        } else {
            /* Right turn (concave) or collinear → single point */
            total += 1;
        }

        prev_dx = curr_dx; prev_dy = curr_dy;
    }

    total += 1;  /* right side end vertex */

    /* === Left cap === */
    total += cap_count(cap_style, q);

    /* === Closing vertex === */
    total += 1;

    out_counts[row] = total;
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel                                                     */
/* ------------------------------------------------------------------ */

/* Emit vertices for a convex join on the left side (offset = +d*n). */
__device__ void emit_convex_join_left(
    double vx, double vy, double d,
    double prev_nx, double prev_ny, double curr_nx, double curr_ny,
    double prev_dx, double prev_dy, double curr_dx, double curr_dy,
    double cross,
    int join_style, int quad_segs, double mitre_limit,
    double* out_x, double* out_y, int* pos
) {
    if (join_style == JOIN_ROUND) {
        /* arrival + clockwise arc */
        out_x[*pos] = vx + d * prev_nx;
        out_y[*pos] = vy + d * prev_ny;
        (*pos)++;
        double start_a = atan2(prev_ny, prev_nx);
        double end_a   = atan2(curr_ny, curr_nx);
        double sweep = end_a - start_a;
        if (sweep > 0.0) sweep -= 2.0 * PI;
        int n_steps = arc_vertex_count(sweep, quad_segs);
        emit_arc_verts(vx, vy, d, start_a, sweep, n_steps,
                       out_x, out_y, pos);
        return;
    }
    if (join_style == JOIN_MITRE) {
        double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
        if ((1.0 - dot_d) * mitre_limit * mitre_limit < 2.0) {
            /* Mitre limit exceeded → bevel */
            out_x[*pos] = vx + d * prev_nx;
            out_y[*pos] = vy + d * prev_ny;
            (*pos)++;
            out_x[*pos] = vx + d * curr_nx;
            out_y[*pos] = vy + d * curr_ny;
            (*pos)++;
            return;
        }
        /* Mitre intersection */
        double t = d * (dot_d - 1.0) / cross;
        out_x[*pos] = vx + d * prev_nx + t * prev_dx;
        out_y[*pos] = vy + d * prev_ny + t * prev_dy;
        (*pos)++;
        return;
    }
    /* JOIN_BEVEL: arrival + departure */
    out_x[*pos] = vx + d * prev_nx;
    out_y[*pos] = vy + d * prev_ny;
    (*pos)++;
    out_x[*pos] = vx + d * curr_nx;
    out_y[*pos] = vy + d * curr_ny;
    (*pos)++;
}

/* Emit vertices for a convex join on the right side (offset = -d*n). */
__device__ void emit_convex_join_right(
    double vx, double vy, double d,
    double prev_nx, double prev_ny, double curr_nx, double curr_ny,
    double prev_dx, double prev_dy, double curr_dx, double curr_dy,
    double cross,
    int join_style, int quad_segs, double mitre_limit,
    double* out_x, double* out_y, int* pos
) {
    if (join_style == JOIN_ROUND) {
        /* arrival + clockwise arc using negated normals */
        out_x[*pos] = vx - d * prev_nx;
        out_y[*pos] = vy - d * prev_ny;
        (*pos)++;
        double start_a = atan2(-prev_ny, -prev_nx);
        double end_a   = atan2(-curr_ny, -curr_nx);
        double sweep = end_a - start_a;
        if (sweep > 0.0) sweep -= 2.0 * PI;
        int n_steps = arc_vertex_count(sweep, quad_segs);
        emit_arc_verts(vx, vy, d, start_a, sweep, n_steps,
                       out_x, out_y, pos);
        return;
    }
    if (join_style == JOIN_MITRE) {
        double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
        if ((1.0 - dot_d) * mitre_limit * mitre_limit < 2.0) {
            /* Mitre limit exceeded → bevel */
            out_x[*pos] = vx - d * prev_nx;
            out_y[*pos] = vy - d * prev_ny;
            (*pos)++;
            out_x[*pos] = vx - d * curr_nx;
            out_y[*pos] = vy - d * curr_ny;
            (*pos)++;
            return;
        }
        /* Mitre intersection (right side: offset is -d*n) */
        double t = d * (dot_d - 1.0) / cross;
        out_x[*pos] = vx - d * prev_nx + t * prev_dx;
        out_y[*pos] = vy - d * prev_ny + t * prev_dy;
        (*pos)++;
        return;
    }
    /* JOIN_BEVEL: arrival + departure */
    out_x[*pos] = vx - d * prev_nx;
    out_y[*pos] = vy - d * prev_ny;
    (*pos)++;
    out_x[*pos] = vx - d * curr_nx;
    out_y[*pos] = vy - d * curr_ny;
    (*pos)++;
}

/* Emit cap vertices at an endpoint. */
__device__ void emit_cap(
    double vx, double vy, double d,
    double dir_x, double dir_y,          /* segment direction at this end */
    double left_nx, double left_ny,      /* left normal */
    int cap_style, int quad_segs, int is_right_cap,
    double* out_x, double* out_y, int* pos
) {
    if (cap_style == CAP_ROUND) {
        double start_a = atan2(left_ny, left_nx);
        if (is_right_cap) {
            /* Clockwise semicircle from left normal to right normal */
        } else {
            /* Clockwise semicircle from right normal to left normal */
            start_a = atan2(-left_ny, -left_nx);
        }
        emit_arc_verts(vx, vy, d, start_a, -PI, 2 * quad_segs,
                       out_x, out_y, pos);
        return;
    }
    if (cap_style == CAP_SQUARE) {
        /* Extend left/right endpoints by d along the segment direction.
           For right cap: extend in +dir direction; order = left, right.
           For left cap: extend in -dir direction; order = right, left
           (we arrive from right side and go to left side). */
        double ext_x, ext_y;
        if (is_right_cap) {
            ext_x = dir_x;  ext_y = dir_y;
            out_x[*pos] = vx + d * left_nx + d * ext_x;
            out_y[*pos] = vy + d * left_ny + d * ext_y;
            (*pos)++;
            out_x[*pos] = vx - d * left_nx + d * ext_x;
            out_y[*pos] = vy - d * left_ny + d * ext_y;
            (*pos)++;
        } else {
            ext_x = -dir_x; ext_y = -dir_y;
            out_x[*pos] = vx - d * left_nx + d * ext_x;
            out_y[*pos] = vy - d * left_ny + d * ext_y;
            (*pos)++;
            out_x[*pos] = vx + d * left_nx + d * ext_x;
            out_y[*pos] = vy + d * left_ny + d * ext_y;
            (*pos)++;
        }
        return;
    }
    /* CAP_FLAT: no vertices */
}

extern "C" __global__ void linestring_buffer_scatter(
    const int* line_row_offsets,
    const int* line_geometry_offsets,
    const unsigned char* line_empty_mask,
    const double* line_x,
    const double* line_y,
    const double* distances,
    int quad_segs,
    int join_style,
    int cap_style,
    double mitre_limit,
    const int* output_offsets,
    double* out_x,
    double* out_y,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int line_row = line_row_offsets[row];
    if (line_row < 0 || line_empty_mask[line_row]) return;

    const int coord_start = line_geometry_offsets[line_row];
    const int coord_end   = line_geometry_offsets[line_row + 1];
    const int N = coord_end - coord_start;
    const double d = distances[row];
    const int q = quad_segs;

    int pos = output_offsets[row];

    if (N < 2) {
        if (N == 1) {
            double px = line_x[coord_start];
            double py = line_y[coord_start];
            int n_arc = 4 * q;
            double step = -2.0 * PI / (double)n_arc;
            for (int i = 0; i < n_arc; i++) {
                double angle = (double)i * step;
                out_x[pos] = px + d * cos(angle);
                out_y[pos] = py + d * sin(angle);
                pos++;
            }
            out_x[pos] = out_x[output_offsets[row]];
            out_y[pos] = out_y[output_offsets[row]];
            pos++;
        }
        return;
    }

    const int n_segs = N - 1;
    double prev_dx, prev_dy, prev_nx, prev_ny;
    double curr_dx, curr_dy, curr_nx, curr_ny;
    double seg_len;

    /* --- First segment --- */
    normalize_2d(line_x[coord_start + 1] - line_x[coord_start],
                 line_y[coord_start + 1] - line_y[coord_start],
                 &prev_dx, &prev_dy, &seg_len);
    prev_nx = -prev_dy;
    prev_ny =  prev_dx;

    double first_dx = prev_dx, first_dy = prev_dy;
    double first_nx = prev_nx, first_ny = prev_ny;

    /* === Left side forward === */
    out_x[pos] = line_x[coord_start] + d * prev_nx;
    out_y[pos] = line_y[coord_start] + d * prev_ny;
    pos++;

    double last_nx = prev_nx, last_ny = prev_ny;

    for (int j = 1; j < n_segs; j++) {
        int ci = coord_start + j;
        normalize_2d(line_x[ci + 1] - line_x[ci],
                     line_y[ci + 1] - line_y[ci],
                     &curr_dx, &curr_dy, &seg_len);
        curr_nx = -curr_dy;
        curr_ny =  curr_dx;

        double vx = line_x[ci];
        double vy = line_y[ci];
        double cross = cross_2d(prev_dx, prev_dy, curr_dx, curr_dy);

        if (cross < -EPSILON) {
            /* Right turn → convex on left */
            emit_convex_join_left(vx, vy, d,
                prev_nx, prev_ny, curr_nx, curr_ny,
                prev_dx, prev_dy, curr_dx, curr_dy,
                cross, join_style, q, mitre_limit,
                out_x, out_y, &pos);
        } else if (cross > EPSILON) {
            /* Left turn → concave on left: offset line intersection */
            double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
            double t = d * (dot_d - 1.0) / cross;
            out_x[pos] = vx + d * prev_nx + t * prev_dx;
            out_y[pos] = vy + d * prev_ny + t * prev_dy;
            pos++;
        } else {
            /* Collinear */
            out_x[pos] = vx + d * prev_nx;
            out_y[pos] = vy + d * prev_ny;
            pos++;
        }

        prev_dx = curr_dx; prev_dy = curr_dy;
        prev_nx = curr_nx; prev_ny = curr_ny;
    }

    last_nx = prev_nx;
    last_ny = prev_ny;
    double last_dx = prev_dx, last_dy = prev_dy;

    /* Left side end */
    out_x[pos] = line_x[coord_end - 1] + d * last_nx;
    out_y[pos] = line_y[coord_end - 1] + d * last_ny;
    pos++;

    /* === Right cap at v[N-1] === */
    emit_cap(line_x[coord_end - 1], line_y[coord_end - 1], d,
             last_dx, last_dy, last_nx, last_ny,
             cap_style, q, 1,
             out_x, out_y, &pos);

    /* Right side start: right offset of v[N-1] */
    out_x[pos] = line_x[coord_end - 1] - d * last_nx;
    out_y[pos] = line_y[coord_end - 1] - d * last_ny;
    pos++;

    /* === Right side backward === */
    normalize_2d(line_x[coord_end - 1] - line_x[coord_end - 2],
                 line_y[coord_end - 1] - line_y[coord_end - 2],
                 &prev_dx, &prev_dy, &seg_len);
    prev_nx = -prev_dy;
    prev_ny =  prev_dx;

    for (int j = n_segs - 1; j >= 1; j--) {
        int ci = coord_start + j;
        normalize_2d(line_x[ci] - line_x[ci - 1],
                     line_y[ci] - line_y[ci - 1],
                     &curr_dx, &curr_dy, &seg_len);
        curr_nx = -curr_dy;
        curr_ny =  curr_dx;

        double vx = line_x[ci];
        double vy = line_y[ci];

        /* cross = forward cross at vertex j */
        double cross = cross_2d(curr_dx, curr_dy, prev_dx, prev_dy);

        if (cross > EPSILON) {
            /* Left turn → convex on right */
            emit_convex_join_right(vx, vy, d,
                prev_nx, prev_ny, curr_nx, curr_ny,
                prev_dx, prev_dy, curr_dx, curr_dy,
                cross, join_style, q, mitre_limit,
                out_x, out_y, &pos);
        } else if (cross < -EPSILON) {
            /* Right turn → concave on right: offset line intersection */
            double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
            double t = d * (dot_d - 1.0) / cross;
            out_x[pos] = vx - d * prev_nx + t * prev_dx;
            out_y[pos] = vy - d * prev_ny + t * prev_dy;
            pos++;
        } else {
            /* Collinear */
            out_x[pos] = vx - d * prev_nx;
            out_y[pos] = vy - d * prev_ny;
            pos++;
        }

        prev_dx = curr_dx; prev_dy = curr_dy;
        prev_nx = curr_nx; prev_ny = curr_ny;
    }

    /* Right side end */
    out_x[pos] = line_x[coord_start] - d * first_nx;
    out_y[pos] = line_y[coord_start] - d * first_ny;
    pos++;

    /* === Left cap at v[0] === */
    emit_cap(line_x[coord_start], line_y[coord_start], d,
             first_dx, first_dy, first_nx, first_ny,
             cap_style, q, 0,
             out_x, out_y, &pos);

    /* === Closing vertex === */
    out_x[pos] = out_x[output_offsets[row]];
    out_y[pos] = out_y[output_offsets[row]];
    pos++;
}
"""
LINESTRING_BUFFER_GPU_THRESHOLD = 5_000
_LINESTRING_BUFFER_KERNEL_NAMES = ("linestring_buffer_count", "linestring_buffer_scatter")
