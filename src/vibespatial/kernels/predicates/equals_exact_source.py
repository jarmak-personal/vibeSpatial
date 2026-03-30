"""CUDA kernel source for geometry equals_exact predicate.

Contains the NVRTC kernel source template (parameterized by compute_type)
and kernel name tuples for per-family coordinate comparison.

Extracted from equals_exact.py -- dispatch logic remains there.
"""
from __future__ import annotations

from vibespatial.geometry.buffers import GeometryFamily

_EQUALS_EXACT_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

/* ------------------------------------------------------------------ */
/* Point / MultiPoint: geometry_offsets -> coordinates                 */
/* ------------------------------------------------------------------ */
extern "C" __global__ void equals_exact_point(
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int*    __restrict__ a_geom_offsets,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const int*    __restrict__ b_geom_offsets,
    const int*    __restrict__ row_map,
    const int*    __restrict__ a_fro,
    const int*    __restrict__ b_fro,
    double tolerance,
    int*   __restrict__ out,
    int n
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int afr = a_fro[row_map[idx]];
    const int bfr = b_fro[row_map[idx]];

    const int a_start = a_geom_offsets[afr];
    const int a_end   = a_geom_offsets[afr + 1];
    const int b_start = b_geom_offsets[bfr];
    const int b_end   = b_geom_offsets[bfr + 1];

    const int a_count = a_end - a_start;
    const int b_count = b_end - b_start;

    if (a_count != b_count) {{
        out[idx] = 0;
        return;
    }}

    const compute_t tol = (compute_t)tolerance;
    for (int i = 0; i < a_count; ++i) {{
        const compute_t dx = (compute_t)a_x[a_start + i] - (compute_t)b_x[b_start + i];
        const compute_t dy = (compute_t)a_y[a_start + i] - (compute_t)b_y[b_start + i];
        if (dx > tol || dx < -tol || dy > tol || dy < -tol) {{
            out[idx] = 0;
            return;
        }}
    }}
    out[idx] = 1;
}}

/* ------------------------------------------------------------------ */
/* LineString: geometry_offsets -> coordinates (same as point)         */
/* ------------------------------------------------------------------ */
extern "C" __global__ void equals_exact_linestring(
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int*    __restrict__ a_geom_offsets,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const int*    __restrict__ b_geom_offsets,
    const int*    __restrict__ row_map,
    const int*    __restrict__ a_fro,
    const int*    __restrict__ b_fro,
    double tolerance,
    int*   __restrict__ out,
    int n
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int afr = a_fro[row_map[idx]];
    const int bfr = b_fro[row_map[idx]];

    const int a_start = a_geom_offsets[afr];
    const int a_end   = a_geom_offsets[afr + 1];
    const int b_start = b_geom_offsets[bfr];
    const int b_end   = b_geom_offsets[bfr + 1];

    const int a_count = a_end - a_start;
    const int b_count = b_end - b_start;

    if (a_count != b_count) {{
        out[idx] = 0;
        return;
    }}

    const compute_t tol = (compute_t)tolerance;
    for (int i = 0; i < a_count; ++i) {{
        const compute_t dx = (compute_t)a_x[a_start + i] - (compute_t)b_x[b_start + i];
        const compute_t dy = (compute_t)a_y[a_start + i] - (compute_t)b_y[b_start + i];
        if (dx > tol || dx < -tol || dy > tol || dy < -tol) {{
            out[idx] = 0;
            return;
        }}
    }}
    out[idx] = 1;
}}

/* ------------------------------------------------------------------ */
/* MultiLineString: geometry_offsets -> part_offsets -> coordinates    */
/* ------------------------------------------------------------------ */
extern "C" __global__ void equals_exact_multilinestring(
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int*    __restrict__ a_geom_offsets,
    const int*    __restrict__ a_part_offsets,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const int*    __restrict__ b_geom_offsets,
    const int*    __restrict__ b_part_offsets,
    const int*    __restrict__ row_map,
    const int*    __restrict__ a_fro,
    const int*    __restrict__ b_fro,
    double tolerance,
    int*   __restrict__ out,
    int n
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int afr = a_fro[row_map[idx]];
    const int bfr = b_fro[row_map[idx]];

    /* Check part count */
    const int a_part_start = a_geom_offsets[afr];
    const int a_part_end   = a_geom_offsets[afr + 1];
    const int b_part_start = b_geom_offsets[bfr];
    const int b_part_end   = b_geom_offsets[bfr + 1];

    if ((a_part_end - a_part_start) != (b_part_end - b_part_start)) {{
        out[idx] = 0;
        return;
    }}

    /* Check per-part coordinate counts match */
    const int part_count = a_part_end - a_part_start;
    for (int p = 0; p < part_count; ++p) {{
        const int a_cs = a_part_offsets[a_part_start + p];
        const int a_ce = a_part_offsets[a_part_start + p + 1];
        const int b_cs = b_part_offsets[b_part_start + p];
        const int b_ce = b_part_offsets[b_part_start + p + 1];
        if ((a_ce - a_cs) != (b_ce - b_cs)) {{
            out[idx] = 0;
            return;
        }}
    }}

    /* Compare all coordinates contiguously */
    const int a_start = a_part_offsets[a_part_start];
    const int a_end   = a_part_offsets[a_part_end];
    const int b_start = b_part_offsets[b_part_start];

    const compute_t tol = (compute_t)tolerance;
    const int count = a_end - a_start;
    for (int i = 0; i < count; ++i) {{
        const compute_t dx = (compute_t)a_x[a_start + i] - (compute_t)b_x[b_start + i];
        const compute_t dy = (compute_t)a_y[a_start + i] - (compute_t)b_y[b_start + i];
        if (dx > tol || dx < -tol || dy > tol || dy < -tol) {{
            out[idx] = 0;
            return;
        }}
    }}
    out[idx] = 1;
}}

/* ------------------------------------------------------------------ */
/* Polygon: geometry_offsets -> ring_offsets -> coordinates            */
/* ------------------------------------------------------------------ */
extern "C" __global__ void equals_exact_polygon(
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int*    __restrict__ a_geom_offsets,
    const int*    __restrict__ a_ring_offsets,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const int*    __restrict__ b_geom_offsets,
    const int*    __restrict__ b_ring_offsets,
    const int*    __restrict__ row_map,
    const int*    __restrict__ a_fro,
    const int*    __restrict__ b_fro,
    double tolerance,
    int*   __restrict__ out,
    int n
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int afr = a_fro[row_map[idx]];
    const int bfr = b_fro[row_map[idx]];

    /* Check ring count */
    const int a_ring_start = a_geom_offsets[afr];
    const int a_ring_end   = a_geom_offsets[afr + 1];
    const int b_ring_start = b_geom_offsets[bfr];
    const int b_ring_end   = b_geom_offsets[bfr + 1];

    if ((a_ring_end - a_ring_start) != (b_ring_end - b_ring_start)) {{
        out[idx] = 0;
        return;
    }}

    /* Check per-ring coordinate counts match */
    const int ring_count = a_ring_end - a_ring_start;
    for (int r = 0; r < ring_count; ++r) {{
        const int a_cs = a_ring_offsets[a_ring_start + r];
        const int a_ce = a_ring_offsets[a_ring_start + r + 1];
        const int b_cs = b_ring_offsets[b_ring_start + r];
        const int b_ce = b_ring_offsets[b_ring_start + r + 1];
        if ((a_ce - a_cs) != (b_ce - b_cs)) {{
            out[idx] = 0;
            return;
        }}
    }}

    /* Compare all coordinates contiguously */
    const int a_start = a_ring_offsets[a_ring_start];
    const int a_end   = a_ring_offsets[a_ring_end];
    const int b_start = b_ring_offsets[b_ring_start];

    const compute_t tol = (compute_t)tolerance;
    const int count = a_end - a_start;
    for (int i = 0; i < count; ++i) {{
        const compute_t dx = (compute_t)a_x[a_start + i] - (compute_t)b_x[b_start + i];
        const compute_t dy = (compute_t)a_y[a_start + i] - (compute_t)b_y[b_start + i];
        if (dx > tol || dx < -tol || dy > tol || dy < -tol) {{
            out[idx] = 0;
            return;
        }}
    }}
    out[idx] = 1;
}}

/* ------------------------------------------------------------------ */
/* MultiPolygon: geom_offsets -> part_offsets -> ring_offsets -> coords */
/* ------------------------------------------------------------------ */
extern "C" __global__ void equals_exact_multipolygon(
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int*    __restrict__ a_geom_offsets,
    const int*    __restrict__ a_part_offsets,
    const int*    __restrict__ a_ring_offsets,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const int*    __restrict__ b_geom_offsets,
    const int*    __restrict__ b_part_offsets,
    const int*    __restrict__ b_ring_offsets,
    const int*    __restrict__ row_map,
    const int*    __restrict__ a_fro,
    const int*    __restrict__ b_fro,
    double tolerance,
    int*   __restrict__ out,
    int n
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int afr = a_fro[row_map[idx]];
    const int bfr = b_fro[row_map[idx]];

    /* Check part (polygon) count */
    const int a_part_start = a_geom_offsets[afr];
    const int a_part_end   = a_geom_offsets[afr + 1];
    const int b_part_start = b_geom_offsets[bfr];
    const int b_part_end   = b_geom_offsets[bfr + 1];

    if ((a_part_end - a_part_start) != (b_part_end - b_part_start)) {{
        out[idx] = 0;
        return;
    }}

    /* Check per-polygon ring counts match */
    const int part_count = a_part_end - a_part_start;
    for (int p = 0; p < part_count; ++p) {{
        const int a_rs = a_part_offsets[a_part_start + p];
        const int a_re = a_part_offsets[a_part_start + p + 1];
        const int b_rs = b_part_offsets[b_part_start + p];
        const int b_re = b_part_offsets[b_part_start + p + 1];
        if ((a_re - a_rs) != (b_re - b_rs)) {{
            out[idx] = 0;
            return;
        }}
        /* Check per-ring coordinate counts match */
        const int ring_count = a_re - a_rs;
        for (int r = 0; r < ring_count; ++r) {{
            const int a_cs = a_ring_offsets[a_rs + r];
            const int a_ce = a_ring_offsets[a_rs + r + 1];
            const int b_cs = b_ring_offsets[b_rs + r];
            const int b_ce = b_ring_offsets[b_rs + r + 1];
            if ((a_ce - a_cs) != (b_ce - b_cs)) {{
                out[idx] = 0;
                return;
            }}
        }}
    }}

    /* Compare all coordinates contiguously */
    const int a_ring_start = a_part_offsets[a_part_start];
    const int a_ring_end   = a_part_offsets[a_part_end];
    const int b_ring_start = b_part_offsets[b_part_start];

    const int a_start = a_ring_offsets[a_ring_start];
    const int a_end   = a_ring_offsets[a_ring_end];
    const int b_start = b_ring_offsets[b_ring_start];

    const compute_t tol = (compute_t)tolerance;
    const int count = a_end - a_start;
    for (int i = 0; i < count; ++i) {{
        const compute_t dx = (compute_t)a_x[a_start + i] - (compute_t)b_x[b_start + i];
        const compute_t dy = (compute_t)a_y[a_start + i] - (compute_t)b_y[b_start + i];
        if (dx > tol || dx < -tol || dy > tol || dy < -tol) {{
            out[idx] = 0;
            return;
        }}
    }}
    out[idx] = 1;
}}
"""

_KERNEL_NAMES = (
    "equals_exact_point",
    "equals_exact_linestring",
    "equals_exact_multilinestring",
    "equals_exact_polygon",
    "equals_exact_multipolygon",
)

# Map family -> kernel entry point name.
_FAMILY_KERNEL_NAME: dict[GeometryFamily, str] = {
    GeometryFamily.POINT: "equals_exact_point",
    GeometryFamily.MULTIPOINT: "equals_exact_point",  # same offset structure
    GeometryFamily.LINESTRING: "equals_exact_linestring",
    GeometryFamily.MULTILINESTRING: "equals_exact_multilinestring",
    GeometryFamily.POLYGON: "equals_exact_polygon",
    GeometryFamily.MULTIPOLYGON: "equals_exact_multipolygon",
}


def _format_kernel_source(compute_type: str = "double") -> str:
    return _EQUALS_EXACT_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)
