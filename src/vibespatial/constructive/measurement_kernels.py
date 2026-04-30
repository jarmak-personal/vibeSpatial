"""NVRTC kernel sources for measurement."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE
from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# Backward-compat alias for any external consumers
_PRECISION_PREAMBLE = PRECISION_PREAMBLE
# ---------------------------------------------------------------------------
# Cooperative area kernel: 1 block per geometry (for complex polygons)
# ---------------------------------------------------------------------------

_POLYGON_AREA_COOPERATIVE_KERNEL_SOURCE = PRECISION_PREAMBLE + STRIP_CLOSURE_DEVICE + r"""
extern "C" __global__ __launch_bounds__(256, 4)
void polygon_area_cooperative(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_area,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x;
    if (row >= row_count) return;

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];

    /* Shared memory for inter-warp Kahan reduction (up to 256/32 = 8 warps) */
    __shared__ compute_t warp_sums[8];
    __shared__ compute_t warp_comp[8];

    compute_t total_area = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int ring = first_ring; ring < last_ring; ring++) {{
        const int cs = ring_offsets[ring];
        const int ce = ring_offsets[ring + 1];
        int n = ce - cs;

        /* Strip closure vertex if present. */
        n = vs_strip_closure(x, y, cs, ce, n, 1e-24);
        if (n < 3) {{
            __syncthreads();  /* keep block synchronized even on skip */
            continue;
        }}
        const double ring_center_x = x[cs];
        const double ring_center_y = y[cs];

        /* Each thread accumulates partial cross product sum with Kahan */
        compute_t partial_sum = (compute_t)0.0;
        compute_t partial_c = (compute_t)0.0;

        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {{
            const int cur = cs + i;
            const int nxt = cs + ((i + 1) % n);
            const compute_t xi  = (compute_t)(x[cur] - ring_center_x);
            const compute_t yi  = (compute_t)(y[cur] - ring_center_y);
            const compute_t xi1 = (compute_t)(x[nxt] - ring_center_x);
            const compute_t yi1 = (compute_t)(y[nxt] - ring_center_y);
            KAHAN_ADD(partial_sum, xi * yi1 - xi1 * yi, partial_c);
        }}

        /* Warp-level reduction via shared Kahan shuffle macro */
        WARP_KAHAN_REDUCE(partial_sum, partial_c);

        /* Block-level reduction via shared memory */
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        if (lane_id == 0) {{
            warp_sums[warp_id] = partial_sum;
            warp_comp[warp_id] = partial_c;
        }}
        __syncthreads();

        /* Thread 0 combines across warps with Kahan for this ring */
        if (threadIdx.x == 0) {{
            compute_t ring_sum = (compute_t)0.0;
            compute_t ring_c = (compute_t)0.0;
            const int num_warps = ((int)blockDim.x + 31) >> 5;
            for (int w = 0; w < num_warps; ++w) {{
                /* Incorporate warp compensation into value before adding */
                KAHAN_ADD(ring_sum, warp_sums[w] - warp_comp[w], ring_c);
            }}

            compute_t ring_area = ring_sum * (compute_t)0.5;
            if (ring == first_ring) {{
                /* Exterior ring: positive area. */
                KAHAN_ADD(total_area, fabs(ring_area), c_total);
            }} else {{
                /* Interior ring (hole): subtract. */
                KAHAN_ADD(total_area, -fabs(ring_area), c_total);
            }}
        }}
        __syncthreads();
    }}

    if (threadIdx.x == 0) {{
        out_area[row] = (double)total_area;
    }}
}}
"""
# ---------------------------------------------------------------------------
# Area kernel: Polygon (also used by MultiPolygon per-polygon-part)
# ---------------------------------------------------------------------------

_POLYGON_AREA_KERNEL_SOURCE = PRECISION_PREAMBLE + STRIP_CLOSURE_DEVICE + r"""
extern "C" __global__ void polygon_area(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* geometry_offsets,
    double* out_area,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];

    compute_t total_area = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int ring = first_ring; ring < last_ring; ring++) {{
        const int cs = ring_offsets[ring];
        const int ce = ring_offsets[ring + 1];
        int n = ce - cs;

        /* Strip closure vertex if present. */
        n = vs_strip_closure(x, y, cs, ce, n, 1e-24);
        if (n < 3) continue;
        const double ring_center_x = x[cs];
        const double ring_center_y = y[cs];

        compute_t sum_cross = (compute_t)0.0;
        compute_t c_cross = (compute_t)0.0;

        for (int i = 0; i < n; i++) {{
            const int cur = cs + i;
            const int nxt = cs + ((i + 1) % n);
            const compute_t xi  = (compute_t)(x[cur] - ring_center_x);
            const compute_t yi  = (compute_t)(y[cur] - ring_center_y);
            const compute_t xi1 = (compute_t)(x[nxt] - ring_center_x);
            const compute_t yi1 = (compute_t)(y[nxt] - ring_center_y);
            KAHAN_ADD(sum_cross, xi * yi1 - xi1 * yi, c_cross);
        }}

        compute_t ring_area = sum_cross * (compute_t)0.5;
        if (ring == first_ring) {{
            /* Exterior ring: positive area. */
            KAHAN_ADD(total_area, fabs(ring_area), c_total);
        }} else {{
            /* Interior ring (hole): subtract. */
            KAHAN_ADD(total_area, -fabs(ring_area), c_total);
        }}
    }}
    out_area[row] = (double)total_area;
}}
"""
# ---------------------------------------------------------------------------
# Area kernel: MultiPolygon (triple indirection: geom -> part -> ring -> coord)
# ---------------------------------------------------------------------------

_MULTIPOLYGON_AREA_KERNEL_SOURCE = PRECISION_PREAMBLE + STRIP_CLOSURE_DEVICE + r"""
extern "C" __global__ void multipolygon_area(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* part_offsets,
    const int* geometry_offsets,
    double* out_area,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    compute_t total_area = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int part = first_part; part < last_part; part++) {{
        const int first_ring = part_offsets[part];
        const int last_ring = part_offsets[part + 1];

        for (int ring = first_ring; ring < last_ring; ring++) {{
            const int cs = ring_offsets[ring];
            const int ce = ring_offsets[ring + 1];
            int n = ce - cs;

            /* Strip closure vertex if present. */
            n = vs_strip_closure(x, y, cs, ce, n, 1e-24);
            if (n < 3) continue;
            const double ring_center_x = x[cs];
            const double ring_center_y = y[cs];

            compute_t sum_cross = (compute_t)0.0;
            compute_t c_cross = (compute_t)0.0;

            for (int i = 0; i < n; i++) {{
                const int cur = cs + i;
                const int nxt = cs + ((i + 1) % n);
                const compute_t xi  = (compute_t)(x[cur] - ring_center_x);
                const compute_t yi  = (compute_t)(y[cur] - ring_center_y);
                const compute_t xi1 = (compute_t)(x[nxt] - ring_center_x);
                const compute_t yi1 = (compute_t)(y[nxt] - ring_center_y);
                KAHAN_ADD(sum_cross, xi * yi1 - xi1 * yi, c_cross);
            }}

            compute_t ring_area = sum_cross * (compute_t)0.5;
            if (ring == first_ring) {{
                /* Exterior ring of this polygon part: positive area. */
                KAHAN_ADD(total_area, fabs(ring_area), c_total);
            }} else {{
                /* Interior ring (hole): subtract. */
                KAHAN_ADD(total_area, -fabs(ring_area), c_total);
            }}
        }}
    }}
    out_area[row] = (double)total_area;
}}
"""
# ---------------------------------------------------------------------------
# Length kernel: shared device helper for segment distance
# ---------------------------------------------------------------------------

_LENGTH_DEVICE_HELPER = r"""
/* Accumulate segment lengths for a coordinate span [cs, cs+n).
   Uses Kahan summation for the running total.  Centering reduces
   absolute magnitude before the difference, improving fp32 accuracy
   for large-offset coordinates (e.g. UTM eastings ~500,000). */
__device__ void accumulate_segment_lengths(
    const double* x, const double* y,
    int cs, int n,
    double center_x, double center_y,
    compute_t* total, compute_t* c_total
) {{
    for (int i = 0; i < n - 1; i++) {{
        const int cur = cs + i;
        const int nxt = cs + i + 1;
        const compute_t dx = CX(x[nxt]) - CX(x[cur]);
        const compute_t dy = CY(y[nxt]) - CY(y[cur]);
        const compute_t seg_len = sqrt(dx * dx + dy * dy);
        KAHAN_ADD(*total, seg_len, *c_total);
    }}
}}
"""
# ---------------------------------------------------------------------------
# Length kernel: Polygon (all rings — exterior + holes)
# ---------------------------------------------------------------------------

_POLYGON_LENGTH_KERNEL_SOURCE = PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
extern "C" __global__ void polygon_length(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* geometry_offsets,
    double* out_length,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];

    compute_t total_length = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int ring = first_ring; ring < last_ring; ring++) {{
        const int cs = ring_offsets[ring];
        const int ce = ring_offsets[ring + 1];
        const int n = ce - cs;
        if (n < 2) continue;
        accumulate_segment_lengths(x, y, cs, n, center_x, center_y,
                                   &total_length, &c_total);
    }}
    out_length[row] = (double)total_length;
}}
"""
# ---------------------------------------------------------------------------
# Length kernel: MultiPolygon (all rings of all polygon parts)
# ---------------------------------------------------------------------------

_MULTIPOLYGON_LENGTH_KERNEL_SOURCE = PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
extern "C" __global__ void multipolygon_length(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* part_offsets,
    const int* geometry_offsets,
    double* out_length,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    compute_t total_length = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int part = first_part; part < last_part; part++) {{
        const int first_ring = part_offsets[part];
        const int last_ring = part_offsets[part + 1];
        for (int ring = first_ring; ring < last_ring; ring++) {{
            const int cs = ring_offsets[ring];
            const int ce = ring_offsets[ring + 1];
            const int n = ce - cs;
            if (n < 2) continue;
            accumulate_segment_lengths(x, y, cs, n, center_x, center_y,
                                       &total_length, &c_total);
        }}
    }}
    out_length[row] = (double)total_length;
}}
"""
# ---------------------------------------------------------------------------
# Length kernel: LineString (geometry_offsets -> coords)
# ---------------------------------------------------------------------------

_LINESTRING_LENGTH_KERNEL_SOURCE = PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
extern "C" __global__ void linestring_length(
    const double* x,
    const double* y,
    const int* geometry_offsets,
    double* out_length,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int n = ce - cs;

    if (n < 2) {{
        out_length[row] = 0.0;
        return;
    }}

    compute_t total_length = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;
    accumulate_segment_lengths(x, y, cs, n, center_x, center_y,
                               &total_length, &c_total);
    out_length[row] = (double)total_length;
}}
"""
# ---------------------------------------------------------------------------
# Length kernel: MultiLineString (geometry_offsets -> part_offsets -> coords)
# ---------------------------------------------------------------------------

_MULTILINESTRING_LENGTH_KERNEL_SOURCE = PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
extern "C" __global__ void multilinestring_length(
    const double* x,
    const double* y,
    const int* part_offsets,
    const int* geometry_offsets,
    double* out_length,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    compute_t total_length = (compute_t)0.0;
    compute_t c_total = (compute_t)0.0;

    for (int part = first_part; part < last_part; part++) {{
        const int cs = part_offsets[part];
        const int ce = part_offsets[part + 1];
        const int n = ce - cs;
        if (n < 2) continue;
        accumulate_segment_lengths(x, y, cs, n, center_x, center_y,
                                   &total_length, &c_total);
    }}
    out_length[row] = (double)total_length;
}}
"""
# ---------------------------------------------------------------------------
# Kernel names and precompiled source variants
# ---------------------------------------------------------------------------

_POLYGON_AREA_NAMES = ("polygon_area",)
_POLYGON_AREA_COOPERATIVE_NAMES = ("polygon_area_cooperative",)
_MULTIPOLYGON_AREA_NAMES = ("multipolygon_area",)
_POLYGON_LENGTH_NAMES = ("polygon_length",)
_MULTIPOLYGON_LENGTH_NAMES = ("multipolygon_length",)
_LINESTRING_LENGTH_NAMES = ("linestring_length",)
_MULTILINESTRING_LENGTH_NAMES = ("multilinestring_length",)
_POLYGON_AREA_FP64 = _POLYGON_AREA_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_AREA_FP32 = _POLYGON_AREA_KERNEL_SOURCE.format(compute_type="float")
_POLYGON_AREA_COOPERATIVE_FP64 = _POLYGON_AREA_COOPERATIVE_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_AREA_COOPERATIVE_FP32 = _POLYGON_AREA_COOPERATIVE_KERNEL_SOURCE.format(compute_type="float")
_MULTIPOLYGON_AREA_FP64 = _MULTIPOLYGON_AREA_KERNEL_SOURCE.format(compute_type="double")
_MULTIPOLYGON_AREA_FP32 = _MULTIPOLYGON_AREA_KERNEL_SOURCE.format(compute_type="float")
_POLYGON_LENGTH_FP64 = _POLYGON_LENGTH_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_LENGTH_FP32 = _POLYGON_LENGTH_KERNEL_SOURCE.format(compute_type="float")
_MULTIPOLYGON_LENGTH_FP64 = _MULTIPOLYGON_LENGTH_KERNEL_SOURCE.format(compute_type="double")
_MULTIPOLYGON_LENGTH_FP32 = _MULTIPOLYGON_LENGTH_KERNEL_SOURCE.format(compute_type="float")
_LINESTRING_LENGTH_FP64 = _LINESTRING_LENGTH_KERNEL_SOURCE.format(compute_type="double")
_LINESTRING_LENGTH_FP32 = _LINESTRING_LENGTH_KERNEL_SOURCE.format(compute_type="float")
_MULTILINESTRING_LENGTH_FP64 = _MULTILINESTRING_LENGTH_KERNEL_SOURCE.format(compute_type="double")
_MULTILINESTRING_LENGTH_FP32 = _MULTILINESTRING_LENGTH_KERNEL_SOURCE.format(compute_type="float")
