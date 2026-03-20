from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total_with_transfer,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, TransferTrigger

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

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("linestring-buffer", _LINESTRING_BUFFER_KERNEL_SOURCE, _LINESTRING_BUFFER_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


def _linestring_buffer_kernels():
    return compile_kernel_group("linestring-buffer", _LINESTRING_BUFFER_KERNEL_SOURCE, _LINESTRING_BUFFER_KERNEL_NAMES)


def _build_device_backed_polygon_output_variable(
    device_x,
    device_y,
    *,
    row_count: int,
    ring_offsets: np.ndarray,
    bounds: np.ndarray | None = None,
) -> OwnedGeometryArray:
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=bounds,
        host_materialized=False,
    )
    runtime = get_cuda_runtime()
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None if bounds is None else runtime.from_host(bounds),
                )
            },
        ),
    )


_CAP_STYLE_MAP = {"round": 0, "flat": 1, "square": 2}
_JOIN_STYLE_MAP = {"round": 0, "mitre": 1, "bevel": 2}


def linestring_buffer_owned_array(
    lines: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 8,
    cap_style: str = "round",
    join_style: str = "round",
    mitre_limit: float = 5.0,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    if GeometryFamily.LINESTRING not in lines.families or len(lines.families) != 1:
        raise ValueError("linestring_buffer_owned_array requires a linestring-only OwnedGeometryArray")
    if not np.all(lines.validity):
        raise ValueError("linestring_buffer_owned_array requires non-null rows only")
    if np.any(lines.tags != FAMILY_TAGS[GeometryFamily.LINESTRING]):
        raise ValueError("linestring_buffer_owned_array requires linestring-only rows")

    line_buffer = lines.families[GeometryFamily.LINESTRING]
    if np.any(line_buffer.empty_mask):
        raise ValueError("linestring_buffer_owned_array requires non-empty rows only")

    radii = (
        np.full(lines.row_count, float(distance), dtype=np.float64)
        if np.isscalar(distance)
        else np.asarray(distance, dtype=np.float64)
    )
    if radii.shape != (lines.row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    selected_mode = plan_dispatch_selection(
        kernel_name="linestring_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=lines.row_count,
        requested_mode=dispatch_mode,
    ).selected

    cap_int = _CAP_STYLE_MAP.get(cap_style, 0)
    join_int = _JOIN_STYLE_MAP.get(join_style, 0)

    if selected_mode is not ExecutionMode.GPU:
        return _build_linestring_buffers_cpu(
            lines, radii, quad_segs=quad_segs,
            cap_style=cap_style, join_style=join_style, mitre_limit=mitre_limit,
        )

    return _build_linestring_buffers_gpu(
        lines, radii, quad_segs=quad_segs,
        cap_style=cap_int, join_style=join_int, mitre_limit=mitre_limit,
    )


def _build_linestring_buffers_cpu(
    lines: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    cap_style: str = "round",
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    import shapely
    lines._ensure_host_state()
    shapely_geoms = lines.to_shapely()
    result_geoms = shapely.buffer(
        np.asarray(shapely_geoms, dtype=object),
        radii,
        quad_segs=quad_segs,
        cap_style=cap_style,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
    from vibespatial.geometry.owned import from_shapely_geometries
    return from_shapely_geometries(list(result_geoms))


def _build_linestring_buffers_gpu(
    lines: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    cap_style: int = 0,
    join_style: int = 0,
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    lines.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="linestring_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = lines._ensure_device_state()
    line_buf = state.families[GeometryFamily.LINESTRING]

    device_radii = runtime.from_host(radii)
    device_counts = runtime.allocate((lines.row_count,), np.int32)
    device_offsets = None
    device_x = None
    device_y = None
    success = False

    try:
        kernels = _linestring_buffer_kernels()
        ptr = runtime.pointer

        # Pass 1: count output vertices per row
        count_params = (
            (
                ptr(state.family_row_offsets),
                ptr(line_buf.geometry_offsets),
                ptr(line_buf.empty_mask),
                ptr(line_buf.x),
                ptr(line_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                cap_style,
                mitre_limit,
                ptr(device_counts),
                lines.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(kernels["linestring_buffer_count"], lines.row_count)
        runtime.launch(kernels["linestring_buffer_count"],
                       grid=count_grid, block=count_block, params=count_params)

        # Compute exclusive prefix sum for scatter offsets
        device_offsets = exclusive_sum(device_counts)

        # Get total and start async full-counts D2H transfer.
        if lines.row_count > 0:
            total_verts, xfer_stream, pinned_counts = count_scatter_total_with_transfer(
                runtime, device_counts, device_offsets,
            )
        else:
            total_verts = 0
            xfer_stream = None
            pinned_counts = None

        if total_verts == 0:
            if xfer_stream is not None:
                xfer_stream.synchronize()
                runtime.destroy_stream(xfer_stream)
            device_x = runtime.allocate((0,), np.float64)
            device_y = runtime.allocate((0,), np.float64)
            ring_offsets = np.zeros(lines.row_count + 1, dtype=np.int32)
            success = True
            return _build_device_backed_polygon_output_variable(
                device_x, device_y,
                row_count=lines.row_count,
                ring_offsets=ring_offsets,
            )

        # Allocate output coordinate arrays
        device_x = runtime.allocate((total_verts,), np.float64)
        device_y = runtime.allocate((total_verts,), np.float64)

        # Pass 2: scatter vertices
        scatter_params = (
            (
                ptr(state.family_row_offsets),
                ptr(line_buf.geometry_offsets),
                ptr(line_buf.empty_mask),
                ptr(line_buf.x),
                ptr(line_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                cap_style,
                mitre_limit,
                ptr(device_offsets),
                ptr(device_x),
                ptr(device_y),
                lines.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["linestring_buffer_scatter"], lines.row_count)
        runtime.launch(kernels["linestring_buffer_scatter"],
                       grid=scatter_grid, block=scatter_block, params=scatter_params)
        runtime.synchronize()

        # Counts D2H transfer was started before the scatter kernel;
        # wait for it now (should already be done).
        xfer_stream.synchronize()
        runtime.destroy_stream(xfer_stream)
        host_counts = pinned_counts
        host_offsets = runtime.copy_device_to_host(device_offsets)
        ring_offsets = np.empty(lines.row_count + 1, dtype=np.int32)
        ring_offsets[0] = 0
        ring_offsets[1:] = host_offsets[: lines.row_count] + host_counts[: lines.row_count]

        success = True
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=lines.row_count,
            ring_offsets=ring_offsets,
        )
    finally:
        runtime.free(device_radii)
        runtime.free(device_counts)
        if device_offsets is not None:
            runtime.free(device_offsets)
        if not success:
            if device_x is not None:
                runtime.free(device_x)
            if device_y is not None:
                runtime.free(device_y)
