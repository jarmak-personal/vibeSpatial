"""GPU-accelerated area and length measurement kernels.

Tier 1 NVRTC kernels (ADR-0033) for computing geometric area and length
directly from OwnedGeometryArray coordinate buffers.  ADR-0002 METRIC class
precision dispatch: fp32 + Kahan + coordinate centering on consumer GPUs,
native fp64 on datacenter GPUs.

Zero host/device transfers mid-process.  When data is already device-resident
(vibeFrame path), kernels read directly from DeviceFamilyGeometryBuffer
pointers with no copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan

# ---------------------------------------------------------------------------
# Shared CUDA macros (identical to polygon_constructive.py centroid kernel)
# ---------------------------------------------------------------------------

_PRECISION_PREAMBLE = r"""
typedef {compute_type} compute_t;

/* Centered coordinate read: subtract center in fp64, then cast to compute_t.
   When compute_t is double, this is a no-op identity.  When compute_t is float,
   centering reduces absolute magnitude before the lossy cast. */
#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

/* Kahan summation helper -- add `val` to `sum` with compensation `c`. */
#define KAHAN_ADD(sum, val, c) do {{ \
    const compute_t _y = (val) - (c); \
    const compute_t _t = (sum) + _y; \
    (c) = (_t - (sum)) - _y; \
    (sum) = _t; \
}} while(0)
"""

# ---------------------------------------------------------------------------
# Cooperative area kernel: 1 block per geometry (for complex polygons)
# ---------------------------------------------------------------------------

_POLYGON_AREA_COOPERATIVE_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
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
        if (n >= 2) {{
            double dx = x[cs] - x[ce - 1];
            double dy = y[cs] - y[ce - 1];
            if (dx * dx + dy * dy < 1e-24) n--;
        }}
        if (n < 3) {{
            __syncthreads();  /* keep block synchronized even on skip */
            continue;
        }}

        /* Each thread accumulates partial cross product sum with Kahan */
        compute_t partial_sum = (compute_t)0.0;
        compute_t partial_c = (compute_t)0.0;

        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {{
            const int cur = cs + i;
            const int nxt = cs + ((i + 1) % n);
            KAHAN_ADD(partial_sum, CX(x[cur]) * CY(y[nxt]) - CX(x[nxt]) * CY(y[cur]), partial_c);
        }}

        /* Warp-level reduction via __shfl_down_sync */
        const unsigned int FULL_MASK = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset >>= 1) {{
            compute_t shfl_sum = __shfl_down_sync(FULL_MASK, partial_sum, offset);
            compute_t shfl_c = __shfl_down_sync(FULL_MASK, partial_c, offset);
            /* Kahan add the shuffled value */
            KAHAN_ADD(partial_sum, shfl_sum - shfl_c, partial_c);
        }}

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

_POLYGON_AREA_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
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
        if (n >= 2) {{
            double dx = x[cs] - x[ce - 1];
            double dy = y[cs] - y[ce - 1];
            if (dx * dx + dy * dy < 1e-24) n--;
        }}
        if (n < 3) continue;

        compute_t sum_cross = (compute_t)0.0;
        compute_t c_cross = (compute_t)0.0;

        for (int i = 0; i < n; i++) {{
            const int cur = cs + i;
            const int nxt = cs + ((i + 1) % n);
            const compute_t xi  = CX(x[cur]);
            const compute_t yi  = CY(y[cur]);
            const compute_t xi1 = CX(x[nxt]);
            const compute_t yi1 = CY(y[nxt]);
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

_MULTIPOLYGON_AREA_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
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
            if (n >= 2) {{
                double dx = x[cs] - x[ce - 1];
                double dy = y[cs] - y[ce - 1];
                if (dx * dx + dy * dy < 1e-24) n--;
            }}
            if (n < 3) continue;

            compute_t sum_cross = (compute_t)0.0;
            compute_t c_cross = (compute_t)0.0;

            for (int i = 0; i < n; i++) {{
                const int cur = cs + i;
                const int nxt = cs + ((i + 1) % n);
                const compute_t xi  = CX(x[cur]);
                const compute_t yi  = CY(y[cur]);
                const compute_t xi1 = CX(x[nxt]);
                const compute_t yi1 = CY(y[nxt]);
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

_POLYGON_LENGTH_KERNEL_SOURCE = _PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
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

_MULTIPOLYGON_LENGTH_KERNEL_SOURCE = _PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
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

_LINESTRING_LENGTH_KERNEL_SOURCE = _PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
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

_MULTILINESTRING_LENGTH_KERNEL_SOURCE = _PRECISION_PREAMBLE + _LENGTH_DEVICE_HELPER + r"""
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

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("polygon-area-fp64", _POLYGON_AREA_FP64, _POLYGON_AREA_NAMES),
    ("polygon-area-fp32", _POLYGON_AREA_FP32, _POLYGON_AREA_NAMES),
    ("polygon-area-cooperative-fp64", _POLYGON_AREA_COOPERATIVE_FP64, _POLYGON_AREA_COOPERATIVE_NAMES),
    ("polygon-area-cooperative-fp32", _POLYGON_AREA_COOPERATIVE_FP32, _POLYGON_AREA_COOPERATIVE_NAMES),
    ("multipolygon-area-fp64", _MULTIPOLYGON_AREA_FP64, _MULTIPOLYGON_AREA_NAMES),
    ("multipolygon-area-fp32", _MULTIPOLYGON_AREA_FP32, _MULTIPOLYGON_AREA_NAMES),
    ("polygon-length-fp64", _POLYGON_LENGTH_FP64, _POLYGON_LENGTH_NAMES),
    ("polygon-length-fp32", _POLYGON_LENGTH_FP32, _POLYGON_LENGTH_NAMES),
    ("multipolygon-length-fp64", _MULTIPOLYGON_LENGTH_FP64, _MULTIPOLYGON_LENGTH_NAMES),
    ("multipolygon-length-fp32", _MULTIPOLYGON_LENGTH_FP32, _MULTIPOLYGON_LENGTH_NAMES),
    ("linestring-length-fp64", _LINESTRING_LENGTH_FP64, _LINESTRING_LENGTH_NAMES),
    ("linestring-length-fp32", _LINESTRING_LENGTH_FP32, _LINESTRING_LENGTH_NAMES),
    ("multilinestring-length-fp64", _MULTILINESTRING_LENGTH_FP64, _MULTILINESTRING_LENGTH_NAMES),
    ("multilinestring-length-fp32", _MULTILINESTRING_LENGTH_FP32, _MULTILINESTRING_LENGTH_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _compile_kernel(name_prefix: str, fp64_source: str, fp32_source: str,
                    kernel_names: tuple[str, ...], compute_type: str = "double"):
    source = fp64_source if compute_type == "double" else fp32_source
    suffix = "fp64" if compute_type == "double" else "fp32"
    return compile_kernel_group(f"{name_prefix}-{suffix}", source, kernel_names)


# ---------------------------------------------------------------------------
# Shared helpers: coordinate statistics from OwnedGeometryArray
# ---------------------------------------------------------------------------

def _fp32_center_coords(
    owned: OwnedGeometryArray,
) -> tuple[float, float]:
    """Return ``(center_x, center_y)`` for coordinate centering.

    Scans the first non-empty family in *owned* and computes the midpoint of
    the bounding box.  When device buffers are available the four CuPy
    reductions (min_x, max_x, min_y, max_y) are packed into a single device
    array so that only **one** ``.get()`` call (and therefore one implicit D2H
    sync) is issued instead of four.
    """
    import cupy as _cp

    for fam, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        ds = owned.device_state
        if ds is not None and fam in ds.families:
            d_buf = ds.families[fam]
            if int(d_buf.x.size) > 0:
                d_x = _cp.asarray(d_buf.x)
                d_y = _cp.asarray(d_buf.y)
                # Batch 4 reductions into 1 D2H transfer
                stats = _cp.array([
                    _cp.min(d_x), _cp.max(d_x),
                    _cp.min(d_y), _cp.max(d_y),
                ])
                s = stats.get()  # single sync
                center_x = float((s[0] + s[1]) * 0.5)
                center_y = float((s[2] + s[3]) * 0.5)
                return center_x, center_y
        elif buf.x.size > 0:
            center_x = float((buf.x.min() + buf.x.max()) * 0.5)
            center_y = float((buf.y.min() + buf.y.max()) * 0.5)
            return center_x, center_y
    return 0.0, 0.0


def _coord_stats_from_owned(
    owned: OwnedGeometryArray,
) -> tuple[float, float, float]:
    """Return ``(max_abs, coord_min, coord_max)`` across all families.

    When device buffers are available the six CuPy reductions per family
    (abs_max_x, abs_max_y, min_x, min_y, max_x, max_y) are packed into a
    single device array so that only **one** ``.get()`` call is issued per
    family instead of six.
    """
    import cupy as _cp

    max_abs: float = 0.0
    coord_min: float = float("inf")
    coord_max: float = float("-inf")

    for fam, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        ds = owned.device_state
        if ds is not None and fam in ds.families:
            d_buf = ds.families[fam]
            if int(d_buf.x.size) > 0:
                d_x = _cp.asarray(d_buf.x)
                d_y = _cp.asarray(d_buf.y)
                # Batch 6 reductions into 1 D2H transfer
                stats = _cp.array([
                    _cp.max(_cp.abs(d_x)), _cp.max(_cp.abs(d_y)),
                    _cp.min(d_x), _cp.min(d_y),
                    _cp.max(d_x), _cp.max(d_y),
                ])
                s = stats.get()  # single sync
                max_abs = max(max_abs, float(s[0]), float(s[1]))
                coord_min = min(coord_min, float(s[2]), float(s[3]))
                coord_max = max(coord_max, float(s[4]), float(s[5]))
        elif buf.x.size > 0:
            max_abs = max(max_abs, float(np.abs(buf.x).max()), float(np.abs(buf.y).max()))
            coord_min = min(coord_min, float(buf.x.min()), float(buf.y.min()))
            coord_max = max(coord_max, float(buf.x.max()), float(buf.y.max()))

    return max_abs, coord_min, coord_max


# ---------------------------------------------------------------------------
# GPU implementation: Area
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_area",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "area", "kahan", "centered"),
)
def _area_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray:
    """GPU-accelerated area computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = _fp32_center_coords(owned)

    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # --- Polygon family ---
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    poly_mask = tags == poly_tag
    if np.any(poly_mask) and owned.family_has_rows(GeometryFamily.POLYGON):
        buf = owned.families[GeometryFamily.POLYGON]
        n = buf.row_count

        # Choose cooperative vs simple kernel based on avg vertex count.
        # When device_state has the family, read vertex count from device
        # buffers since host stubs may have empty x arrays.
        if device_state is not None and GeometryFamily.POLYGON in (device_state.families if device_state else {}):
            avg_verts = int(device_state.families[GeometryFamily.POLYGON].x.size) / max(n, 1) if n > 0 else 0
        else:
            avg_verts = buf.x.size / max(n, 1) if n > 0 else 0
        use_cooperative = avg_verts >= 64

        if use_cooperative:
            coop_kernels = _compile_kernel(
                "polygon-area-cooperative",
                _POLYGON_AREA_COOPERATIVE_FP64, _POLYGON_AREA_COOPERATIVE_FP32,
                _POLYGON_AREA_COOPERATIVE_NAMES, compute_type,
            )
            kernel = coop_kernels["polygon_area_cooperative"]
        else:
            kernels = _compile_kernel("polygon-area", _POLYGON_AREA_FP64, _POLYGON_AREA_FP32,
                                      _POLYGON_AREA_NAMES, compute_type)
            kernel = kernels["polygon_area"]

        global_rows = np.flatnonzero(poly_mask)
        family_rows = family_row_offsets[global_rows]

        # Zero-copy: use device pointers if already resident
        needs_free = device_state is None or GeometryFamily.POLYGON not in (device_state.families if device_state else {})
        if not needs_free:
            ds = device_state.families[GeometryFamily.POLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
                 ptr(d_out), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
            )
            if use_cooperative:
                # 1 block per geometry; fixed at 256 to match __launch_bounds__(256, 4)
                # and shared memory sized for 8 warps (256 / 32).
                grid = (n, 1, 1)
                block = (256, 1, 1)
            else:
                grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(d_out)
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_geom)

    # --- MultiPolygon family ---
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    mpoly_mask = tags == mpoly_tag
    if np.any(mpoly_mask) and owned.family_has_rows(GeometryFamily.MULTIPOLYGON):
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        kernels = _compile_kernel("multipolygon-area", _MULTIPOLYGON_AREA_FP64, _MULTIPOLYGON_AREA_FP32,
                                  _MULTIPOLYGON_AREA_NAMES, compute_type)
        kernel = kernels["multipolygon_area"]
        global_rows = np.flatnonzero(mpoly_mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or GeometryFamily.MULTIPOLYGON not in (device_state.families if device_state else {})
        if not needs_free:
            ds = device_state.families[GeometryFamily.MULTIPOLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_part = ds.part_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
                 ptr(d_out), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32),
            )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(d_out)
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_part)
                runtime.free(d_geom)

    # Points, LineStrings, MultiPoints, MultiLineStrings: area = 0.0 (already zero-initialized)
    return result


# ---------------------------------------------------------------------------
# GPU implementation: Length
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_length",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "length", "kahan", "centered"),
)
def _length_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray:
    """GPU-accelerated length computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = _fp32_center_coords(owned)

    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    def _launch_ring_length(family: GeometryFamily, kernel_name: str, source_fp64: str,
                            source_fp32: str, names: tuple[str, ...], prefix: str,
                            has_part_offsets: bool):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or not owned.family_has_rows(family):
            return
        buf = owned.families[family]

        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or family not in (device_state.families if device_state else {})
        allocated = []
        if not needs_free:
            ds = device_state.families[family]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_geom = ds.geometry_offsets
            d_part = ds.part_offsets if has_part_offsets else None
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
            allocated.extend([d_x, d_y, d_ring, d_geom])
            if has_part_offsets:
                d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
                allocated.append(d_part)
            else:
                d_part = None

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                     KERNEL_PARAM_I32),
                )
            else:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(d_out)
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            for d in allocated:
                runtime.free(d)

    def _launch_line_length(family: GeometryFamily, kernel_name: str, source_fp64: str,
                            source_fp32: str, names: tuple[str, ...], prefix: str,
                            has_part_offsets: bool):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or not owned.family_has_rows(family):
            return
        buf = owned.families[family]

        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or family not in (device_state.families if device_state else {})
        allocated = []
        if not needs_free:
            ds = device_state.families[family]
            d_x, d_y = ds.x, ds.y
            d_geom = ds.geometry_offsets
            d_part = ds.part_offsets if has_part_offsets else None
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
            allocated.extend([d_x, d_y, d_geom])
            if has_part_offsets:
                d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
                allocated.append(d_part)
            else:
                d_part = None

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
                )
            else:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(d_out)
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            for d in allocated:
                runtime.free(d)

    # Polygon length (all rings)
    _launch_ring_length(
        GeometryFamily.POLYGON, "polygon_length",
        _POLYGON_LENGTH_FP64, _POLYGON_LENGTH_FP32,
        _POLYGON_LENGTH_NAMES, "polygon-length", has_part_offsets=False,
    )

    # MultiPolygon length (all rings of all polygon parts)
    _launch_ring_length(
        GeometryFamily.MULTIPOLYGON, "multipolygon_length",
        _MULTIPOLYGON_LENGTH_FP64, _MULTIPOLYGON_LENGTH_FP32,
        _MULTIPOLYGON_LENGTH_NAMES, "multipolygon-length", has_part_offsets=True,
    )

    # LineString length
    _launch_line_length(
        GeometryFamily.LINESTRING, "linestring_length",
        _LINESTRING_LENGTH_FP64, _LINESTRING_LENGTH_FP32,
        _LINESTRING_LENGTH_NAMES, "linestring-length", has_part_offsets=False,
    )

    # MultiLineString length
    _launch_line_length(
        GeometryFamily.MULTILINESTRING, "multilinestring_length",
        _MULTILINESTRING_LENGTH_FP64, _MULTILINESTRING_LENGTH_FP32,
        _MULTILINESTRING_LENGTH_NAMES, "multilinestring-length", has_part_offsets=True,
    )

    # Points and MultiPoints: length = 0.0 (already zero-initialized)
    return result


# ---------------------------------------------------------------------------
# CPU fallback: Area (NumPy, NO Shapely)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_area",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "area"),
)
def _area_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU area computation using NumPy — no Shapely dependency."""
    # Materialize host buffers from device if needed (stubs have empty x/y
    # and None ring_offsets when host_materialized=False).
    owned._ensure_host_state()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or family not in owned.families:
            continue
        buf = owned.families[family]
        if buf.row_count == 0 or buf.ring_offsets is None:
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        x, y = buf.x, buf.y
        ring_offsets = buf.ring_offsets
        geom_offsets = buf.geometry_offsets
        part_offsets = buf.part_offsets
        is_multi = family is GeometryFamily.MULTIPOLYGON

        for gi, fr in zip(global_rows, family_rows):
            if is_multi:
                first_part = geom_offsets[fr]
                last_part = geom_offsets[fr + 1]
                total = 0.0
                for part in range(first_part, last_part):
                    first_ring = part_offsets[part]
                    last_ring = part_offsets[part + 1]
                    total += _rings_area(x, y, ring_offsets, first_ring, last_ring)
                result[gi] = total
            else:
                first_ring = geom_offsets[fr]
                last_ring = geom_offsets[fr + 1]
                result[gi] = _rings_area(x, y, ring_offsets, first_ring, last_ring)

    return result


def _rings_area(x, y, ring_offsets, first_ring, last_ring):
    """Compute area for a set of rings (exterior + holes)."""
    total = 0.0
    for ring in range(first_ring, last_ring):
        cs = ring_offsets[ring]
        ce = ring_offsets[ring + 1]
        n = ce - cs
        if n < 3:
            continue
        # Strip closure vertex
        if n >= 2:
            dx = x[cs] - x[ce - 1]
            dy = y[cs] - y[ce - 1]
            if dx * dx + dy * dy < 1e-24:
                n -= 1
        if n < 3:
            continue

        rx = x[cs:cs + n]
        ry = y[cs:cs + n]
        rx1 = np.roll(rx, -1)
        ry1 = np.roll(ry, -1)
        signed_area = np.sum(rx * ry1 - rx1 * ry) * 0.5

        if ring == first_ring:
            total += abs(signed_area)
        else:
            total -= abs(signed_area)
    return total


# ---------------------------------------------------------------------------
# CPU fallback: Length (NumPy, NO Shapely)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_length",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "length"),
)
def _length_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU length computation using NumPy — no Shapely dependency."""
    # Materialize host buffers from device if needed (stubs have empty x/y
    # and None ring_offsets when host_materialized=False).
    owned._ensure_host_state()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # LineString
    _length_cpu_lines(owned, result, tags, family_row_offsets,
                      GeometryFamily.LINESTRING, multi=False)
    # MultiLineString
    _length_cpu_lines(owned, result, tags, family_row_offsets,
                      GeometryFamily.MULTILINESTRING, multi=True)
    # Polygon (all rings)
    _length_cpu_rings(owned, result, tags, family_row_offsets,
                      GeometryFamily.POLYGON, multi=False)
    # MultiPolygon (all rings of all polygon parts)
    _length_cpu_rings(owned, result, tags, family_row_offsets,
                      GeometryFamily.MULTIPOLYGON, multi=True)

    return result


def _length_cpu_lines(owned, result, tags, family_row_offsets,
                      family: GeometryFamily, multi: bool):
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    x, y = buf.x, buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for gi, fr in zip(global_rows, family_rows):
        if multi:
            fp = geom_offsets[fr]
            lp = geom_offsets[fr + 1]
            total = 0.0
            for p in range(fp, lp):
                cs = part_offsets[p]
                ce = part_offsets[p + 1]
                total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total
        else:
            cs = geom_offsets[fr]
            ce = geom_offsets[fr + 1]
            result[gi] = _segment_length_sum(x, y, cs, ce)


def _length_cpu_rings(owned, result, tags, family_row_offsets,
                      family: GeometryFamily, multi: bool):
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0 or buf.ring_offsets is None:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    x, y = buf.x, buf.y
    ring_offsets = buf.ring_offsets
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for gi, fr in zip(global_rows, family_rows):
        if multi:
            fp = geom_offsets[fr]
            lp = geom_offsets[fr + 1]
            total = 0.0
            for p in range(fp, lp):
                fring = part_offsets[p]
                lring = part_offsets[p + 1]
                for ring in range(fring, lring):
                    cs = ring_offsets[ring]
                    ce = ring_offsets[ring + 1]
                    total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total
        else:
            fring = geom_offsets[fr]
            lring = geom_offsets[fr + 1]
            total = 0.0
            for ring in range(fring, lring):
                cs = ring_offsets[ring]
                ce = ring_offsets[ring + 1]
                total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total


def _segment_length_sum(x, y, cs, ce):
    """Sum of Euclidean segment lengths for coords[cs:ce]."""
    n = ce - cs
    if n < 2:
        return 0.0
    dx = np.diff(x[cs:ce])
    dy = np.diff(y[cs:ce])
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def area_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute area directly from OwnedGeometryArray coordinate buffers.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import CoordinateStats, select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
        span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="geometry_area GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        try:
            result = _area_gpu(owned, precision_plan=precision_plan)
            result[~owned.validity] = np.nan
        except Exception:
            pass  # fall through to CPU
        else:
            record_dispatch_event(
                surface="geopandas.array.area",
                operation="area",
                implementation="gpu_nvrtc_shoelace",
                reason="GPU NVRTC area kernel",
                detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
                selected=ExecutionMode.GPU,
            )
            return result

    record_dispatch_event(
        surface="geopandas.array.area",
        operation="area",
        implementation="numpy",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    result = _area_cpu(owned)
    result[~owned.validity] = np.nan
    return result


def length_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute length directly from OwnedGeometryArray coordinate buffers.

    For Polygons, measures the perimeter (all rings including holes).
    For LineStrings, measures total segment length.
    Points return 0.0.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import CoordinateStats, select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="geometry_length",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
        span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="geometry_length GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        try:
            result = _length_gpu(owned, precision_plan=precision_plan)
            result[~owned.validity] = np.nan
        except Exception:
            pass  # fall through to CPU
        else:
            record_dispatch_event(
                surface="geopandas.array.length",
                operation="length",
                implementation="gpu_nvrtc_segment_length",
                reason="GPU NVRTC length kernel",
                detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
                selected=ExecutionMode.GPU,
            )
            return result

    record_dispatch_event(
        surface="geopandas.array.length",
        operation="length",
        implementation="numpy",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    result = _length_cpu(owned)
    result[~owned.validity] = np.nan
    return result
