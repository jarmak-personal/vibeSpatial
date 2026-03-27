"""GPU NVRTC kernel for geometry equals_exact predicate.

Tier 1 (ADR-0033): custom NVRTC kernel for per-pair coordinate comparison
with tolerance.  One thread per geometry pair, loops over coordinates,
early exits on structure mismatch.

ADR-0002: PREDICATE class, dual fp32/fp64 via PrecisionPlan.
ADR-0034: registered for NVRTC precompilation warmup.

Supported families: Point, LineString, Polygon, MultiPoint,
MultiLineString, MultiPolygon.
"""

from __future__ import annotations

import logging

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NVRTC kernel source template
# ---------------------------------------------------------------------------
# Each family has a separate kernel entry point because the offset
# indirection structure differs (geometry_offsets only for point/linestring,
# ring_offsets for polygon, part_offsets+ring_offsets for multi*).
#
# Storage is always fp64.  Compute is templated on compute_t so the
# coordinate diff + tolerance comparison can run in fp32 on consumer GPUs.
# ---------------------------------------------------------------------------

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


def _compile_equals_exact_kernels(compute_type: str = "double"):
    """Compile the equals_exact kernel source for the given compute precision."""
    source = _format_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        f"equals-exact-{compute_type}", source
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# ADR-0034: register for NVRTC background precompilation
# ---------------------------------------------------------------------------
_WARMUP_SOURCE_FP64 = _format_kernel_source("double")
_WARMUP_SOURCE_FP32 = _format_kernel_source("float")

request_nvrtc_warmup([
    ("equals-exact-double", _WARMUP_SOURCE_FP64, _KERNEL_NAMES),
    ("equals-exact-float", _WARMUP_SOURCE_FP32, _KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel variant registration
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geom_equals_exact",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=("gpu",),
    geometry_families=(
        "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon",
    ),
    supports_mixed=True,
    preferred_residency=Residency.DEVICE,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    tags=("equals-exact", "coordinate-comparison", "cuda-python"),
)
def launch_equals_exact_family(
    left_state: OwnedGeometryDeviceState,
    right_state: OwnedGeometryDeviceState,
    left_buf: DeviceFamilyGeometryBuffer,
    right_buf: DeviceFamilyGeometryBuffer,
    family: GeometryFamily,
    row_indices_device,  # DeviceArray int32 — indices into the global row arrays
    tolerance: float,
    compute_type: str = "double",
) -> None:
    """Launch equals_exact kernel for a single geometry family.

    Writes results (0/1) into ``out_device`` at positions corresponding to
    ``row_indices_device``.  The caller is responsible for allocating the
    output buffer and scattering results back.

    Parameters
    ----------
    left_state, right_state : OwnedGeometryDeviceState
        Device-resident routing metadata (tags, validity, family_row_offsets).
    left_buf, right_buf : DeviceFamilyGeometryBuffer
        Per-family device buffers (x, y, geometry_offsets, part_offsets, ring_offsets).
    family : GeometryFamily
        Which geometry family these rows belong to.
    row_indices_device : DeviceArray[int32]
        Global row indices for the rows in this family batch.
    tolerance : float
        Coordinate tolerance for equality.
    compute_type : str
        "float" or "double" — from PrecisionPlan.

    Returns
    -------
    DeviceArray[int32]
        Per-row result (0 or 1) for each row in row_indices_device.
    """
    runtime = get_cuda_runtime()
    kernels = _compile_equals_exact_kernels(compute_type)
    kernel_name = _FAMILY_KERNEL_NAME[family]
    kernel = kernels[kernel_name]

    n = int(row_indices_device.shape[0])
    if n == 0:
        return runtime.allocate((0,), np.int32)

    # Allocate output buffer (zero-filled so default is False)
    d_out = runtime.allocate((n,), np.int32, zero=True)

    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)

    # Build parameter list depending on family offset structure
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT,
                  GeometryFamily.LINESTRING):
        # Simple: geometry_offsets -> coordinates
        params = (
            (
                ptr(left_buf.x), ptr(left_buf.y), ptr(left_buf.geometry_offsets),
                ptr(right_buf.x), ptr(right_buf.y), ptr(right_buf.geometry_offsets),
                ptr(row_indices_device),
                ptr(left_state.family_row_offsets), ptr(right_state.family_row_offsets),
                tolerance,
                ptr(d_out),
                n,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )

    elif family == GeometryFamily.MULTILINESTRING:
        # geometry_offsets -> part_offsets -> coordinates
        params = (
            (
                ptr(left_buf.x), ptr(left_buf.y),
                ptr(left_buf.geometry_offsets), ptr(left_buf.part_offsets),
                ptr(right_buf.x), ptr(right_buf.y),
                ptr(right_buf.geometry_offsets), ptr(right_buf.part_offsets),
                ptr(row_indices_device),
                ptr(left_state.family_row_offsets), ptr(right_state.family_row_offsets),
                tolerance,
                ptr(d_out),
                n,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )

    elif family == GeometryFamily.POLYGON:
        # geometry_offsets -> ring_offsets -> coordinates
        params = (
            (
                ptr(left_buf.x), ptr(left_buf.y),
                ptr(left_buf.geometry_offsets), ptr(left_buf.ring_offsets),
                ptr(right_buf.x), ptr(right_buf.y),
                ptr(right_buf.geometry_offsets), ptr(right_buf.ring_offsets),
                ptr(row_indices_device),
                ptr(left_state.family_row_offsets), ptr(right_state.family_row_offsets),
                tolerance,
                ptr(d_out),
                n,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )

    elif family == GeometryFamily.MULTIPOLYGON:
        # geometry_offsets -> part_offsets -> ring_offsets -> coordinates
        params = (
            (
                ptr(left_buf.x), ptr(left_buf.y),
                ptr(left_buf.geometry_offsets), ptr(left_buf.part_offsets),
                ptr(left_buf.ring_offsets),
                ptr(right_buf.x), ptr(right_buf.y),
                ptr(right_buf.geometry_offsets), ptr(right_buf.part_offsets),
                ptr(right_buf.ring_offsets),
                ptr(row_indices_device),
                ptr(left_state.family_row_offsets), ptr(right_state.family_row_offsets),
                tolerance,
                ptr(d_out),
                n,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_out
