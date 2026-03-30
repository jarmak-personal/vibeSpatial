"""Hausdorff and discrete Frechet distance on OwnedGeometryArray pairs.

CPU brute-force implementations plus GPU NVRTC kernels that read directly
from device-resident coordinate buffers, avoiding ``_ensure_host_state()``.
Only the small result array (N floats) transfers D->H at the end.

METRIC kernel class per ADR-0002.
ADR-0033: Tier 1 NVRTC -- geometry-specific inner loops (min-of-max,
DP coupling) require custom kernels.
"""

from __future__ import annotations

import logging
import math

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray, tile_single_row
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry families that carry meaningful coordinate sequences.
# ---------------------------------------------------------------------------
_LINESTRING_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.LINESTRING,
    GeometryFamily.MULTILINESTRING,
})

_COORD_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.POINT,
    GeometryFamily.LINESTRING,
    GeometryFamily.POLYGON,
    GeometryFamily.MULTIPOINT,
    GeometryFamily.MULTILINESTRING,
    GeometryFamily.MULTIPOLYGON,
})

# Tag values for linestring families (used for Frechet type check).
_LINESTRING_TAG = FAMILY_TAGS[GeometryFamily.LINESTRING]

# GPU families supported by the flat-offset builder (no 3-level indirection).
_GPU_SUPPORTED_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.POINT,
    GeometryFamily.MULTIPOINT,
    GeometryFamily.LINESTRING,
    GeometryFamily.MULTILINESTRING,
    GeometryFamily.POLYGON,
})


# ---------------------------------------------------------------------------
# Internal: extract coordinate arrays for a single row
# ---------------------------------------------------------------------------

def _coords_for_row(
    owned: OwnedGeometryArray,
    row: int,
) -> np.ndarray | None:
    """Return an (N, 2) float64 array of coordinates for *row*, or None.

    Works across all geometry families by reading directly from the packed
    FamilyGeometryBuffer coordinate arrays.  Returns None for null or empty
    rows.
    """
    if not owned.validity[row]:
        return None

    tag = int(owned.tags[row])
    family = None
    for fam, fam_tag in FAMILY_TAGS.items():
        if fam_tag == tag:
            family = fam
            break
    if family is None:
        return None

    if family not in owned.families:
        return None

    buf = owned.families[family]
    fro = owned.family_row_offsets[row]
    if fro < 0:
        return None
    if buf.empty_mask[fro]:
        return None

    # Gather all coordinates for this geometry row.
    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING):
        # geometry_offsets[fro] .. geometry_offsets[fro+1]
        cs = int(buf.geometry_offsets[fro])
        ce = int(buf.geometry_offsets[fro + 1])
    elif family in (GeometryFamily.MULTIPOINT, GeometryFamily.MULTILINESTRING):
        # Gather across parts.
        gs = int(buf.geometry_offsets[fro])
        ge = int(buf.geometry_offsets[fro + 1])
        if gs >= ge:
            return None
        cs = int(buf.part_offsets[gs])
        ce = int(buf.part_offsets[ge - 1 + 1]) if ge > gs else cs
        # Simpler: first part start to last part end
        cs = int(buf.part_offsets[gs])
        ce = int(buf.part_offsets[ge])
    elif family == GeometryFamily.POLYGON:
        # Gather across rings.
        rs = int(buf.geometry_offsets[fro])
        re_ = int(buf.geometry_offsets[fro + 1])
        if rs >= re_:
            return None
        cs = int(buf.ring_offsets[rs])
        ce = int(buf.ring_offsets[re_])
    elif family == GeometryFamily.MULTIPOLYGON:
        # Gather across polygons -> rings.
        ps = int(buf.geometry_offsets[fro])
        pe = int(buf.geometry_offsets[fro + 1])
        if ps >= pe:
            return None
        first_ring = int(buf.part_offsets[ps])
        last_ring = int(buf.part_offsets[pe])
        if first_ring >= last_ring:
            return None
        cs = int(buf.ring_offsets[first_ring])
        ce = int(buf.ring_offsets[last_ring])
    else:
        return None

    if ce <= cs:
        return None

    x = buf.x[cs:ce]
    y = buf.y[cs:ce]
    return np.column_stack((x, y)).astype(np.float64, copy=False)


def _linestring_coords_for_row(
    owned: OwnedGeometryArray,
    row: int,
) -> np.ndarray | None:
    """Return ordered (N, 2) coordinates for a LineString row, or None.

    Returns None if the row is not a simple LineString (MultiLineString,
    Polygon, etc. are rejected).
    """
    if not owned.validity[row]:
        return None
    tag = int(owned.tags[row])
    if tag != _LINESTRING_TAG:
        return None

    family = GeometryFamily.LINESTRING
    if family not in owned.families:
        return None

    buf = owned.families[family]
    fro = owned.family_row_offsets[row]
    if fro < 0:
        return None
    if buf.empty_mask[fro]:
        return None

    cs = int(buf.geometry_offsets[fro])
    ce = int(buf.geometry_offsets[fro + 1])
    if ce <= cs:
        return None

    x = buf.x[cs:ce]
    y = buf.y[cs:ce]
    return np.column_stack((x, y)).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Euclidean distance helpers
# ---------------------------------------------------------------------------

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two (2,) coordinate arrays."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


_MIN_DIST_CHUNK = 1024


def _min_distances_vectorized(
    pts_from: np.ndarray,
    pts_to: np.ndarray,
) -> np.ndarray:
    """For each point in *pts_from*, compute the min distance to any point in *pts_to*.

    Both inputs are (M, 2) and (K, 2) float64 arrays.
    Returns (M,) float64 array.

    Processes in chunks of _MIN_DIST_CHUNK rows to cap memory at
    ~16MB per chunk regardless of geometry size.
    """
    m = pts_from.shape[0]
    result = np.empty(m, dtype=np.float64)
    for start in range(0, m, _MIN_DIST_CHUNK):
        end = min(start + _MIN_DIST_CHUNK, m)
        chunk = pts_from[start:end]
        diff = chunk[:, np.newaxis, :] - pts_to[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        result[start:end] = np.min(dists, axis=1)
    return result


# ===========================================================================
# GPU NVRTC Kernels (ADR-0033 Tier 1)
# ===========================================================================

from vibespatial.cuda.preamble import PRECISION_PREAMBLE  # noqa: E402

# ---------------------------------------------------------------------------
# Hausdorff kernel: 1 block per pair, shared-memory max reduction
# ---------------------------------------------------------------------------

_HAUSDORFF_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ __launch_bounds__(256, 4)
void hausdorff_distance(
    const double* __restrict__ x_a,
    const double* __restrict__ y_a,
    const int* __restrict__ offsets_a,
    const double* __restrict__ x_b,
    const double* __restrict__ y_b,
    const int* __restrict__ offsets_b,
    double* __restrict__ result,
    double center_x,
    double center_y,
    int pair_count
) {{
    const int pair = blockIdx.x;
    if (pair >= pair_count) return;

    const int a_start = offsets_a[pair];
    const int a_end   = offsets_a[pair + 1];
    const int b_start = offsets_b[pair];
    const int b_end   = offsets_b[pair + 1];
    const int na = a_end - a_start;
    const int nb = b_end - b_start;

    if (na == 0 || nb == 0) {{
        if (threadIdx.x == 0) result[pair] = 0.0 / 0.0;  /* NaN */
        return;
    }}

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    /* Forward direction: max over A of {{ min over B of dist(a,b) }} */
    compute_t local_max_forward = (compute_t)0.0;
    for (int i = tid; i < na; i += stride) {{
        const compute_t ax = CX(x_a[a_start + i]);
        const compute_t ay = CY(y_a[a_start + i]);
        compute_t min_dist_sq = (compute_t)1e38;
        for (int j = 0; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const compute_t d_sq = dx * dx + dy * dy;
            if (d_sq < min_dist_sq) min_dist_sq = d_sq;
        }}
        const compute_t min_dist = sqrt((double)min_dist_sq);
        if (min_dist > local_max_forward) local_max_forward = min_dist;
    }}

    /* Backward direction: max over B of {{ min over A of dist(b,a) }} */
    compute_t local_max_backward = (compute_t)0.0;
    for (int j = tid; j < nb; j += stride) {{
        const compute_t bx = CX(x_b[b_start + j]);
        const compute_t by = CY(y_b[b_start + j]);
        compute_t min_dist_sq = (compute_t)1e38;
        for (int i = 0; i < na; i++) {{
            const compute_t dx = bx - CX(x_a[a_start + i]);
            const compute_t dy = by - CY(y_a[a_start + i]);
            const compute_t d_sq = dx * dx + dy * dy;
            if (d_sq < min_dist_sq) min_dist_sq = d_sq;
        }}
        const compute_t min_dist = sqrt((double)min_dist_sq);
        if (min_dist > local_max_backward) local_max_backward = min_dist;
    }}

    double local_max = (double)((local_max_forward > local_max_backward)
                                ? local_max_forward : local_max_backward);

    /* Block-wide max reduction via shared memory */
    __shared__ double sdata[256];
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s] > sdata[tid]) {{
            sdata[tid] = sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) result[pair] = sdata[0];
}}
"""

_HAUSDORFF_KERNEL_NAMES = ("hausdorff_distance",)

# ---------------------------------------------------------------------------
# Frechet kernel: 1 thread per pair, rolling-row DP in local memory
# ---------------------------------------------------------------------------

_MAX_FRECHET_B = 2048

_FRECHET_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
#define MAX_FRECHET_B {max_frechet_b}

extern "C" __global__
void frechet_distance(
    const double* __restrict__ x_a,
    const double* __restrict__ y_a,
    const int* __restrict__ offsets_a,
    const double* __restrict__ x_b,
    const double* __restrict__ y_b,
    const int* __restrict__ offsets_b,
    double* __restrict__ result,
    double center_x,
    double center_y,
    int pair_count,
    int max_b_len
) {{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;

    const int a_start = offsets_a[pair];
    const int a_end   = offsets_a[pair + 1];
    const int b_start = offsets_b[pair];
    const int b_end   = offsets_b[pair + 1];
    const int na = a_end - a_start;
    const int nb = b_end - b_start;

    if (na == 0 || nb == 0) {{
        result[pair] = 0.0 / 0.0;  /* NaN */
        return;
    }}

    if (nb > MAX_FRECHET_B) {{
        /* Too large for local arrays -- return NaN as safe fallback */
        result[pair] = 0.0 / 0.0;
        return;
    }}

    /* Rolling DP: previous and current rows, each of length nb.
       For large nb these spill to local memory (L1/L2 cached). */
    double prev_row[MAX_FRECHET_B];
    double curr_row[MAX_FRECHET_B];

    /* Initialize row 0: ca[0][j] = max(ca[0][j-1], d(a_0, b_j)) */
    {{
        const compute_t ax = CX(x_a[a_start]);
        const compute_t ay = CY(y_a[a_start]);
        {{
            const compute_t dx = ax - CX(x_b[b_start]);
            const compute_t dy = ay - CY(y_b[b_start]);
            prev_row[0] = sqrt((double)(dx * dx + dy * dy));
        }}
        for (int j = 1; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const double d = sqrt((double)(dx * dx + dy * dy));
            prev_row[j] = (d > prev_row[j - 1]) ? d : prev_row[j - 1];
        }}
    }}

    /* Fill rows 1..na-1 */
    for (int i = 1; i < na; i++) {{
        const compute_t ax = CX(x_a[a_start + i]);
        const compute_t ay = CY(y_a[a_start + i]);

        /* curr_row[0] = max(prev_row[0], d(a_i, b_0)) */
        {{
            const compute_t dx = ax - CX(x_b[b_start]);
            const compute_t dy = ay - CY(y_b[b_start]);
            const double d = sqrt((double)(dx * dx + dy * dy));
            curr_row[0] = (d > prev_row[0]) ? d : prev_row[0];
        }}

        for (int j = 1; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const double d = sqrt((double)(dx * dx + dy * dy));

            double min_prev = prev_row[j];          /* ca[i-1][j]   */
            if (curr_row[j - 1] < min_prev) min_prev = curr_row[j - 1]; /* ca[i][j-1]   */
            if (prev_row[j - 1] < min_prev) min_prev = prev_row[j - 1]; /* ca[i-1][j-1] */

            curr_row[j] = (d > min_prev) ? d : min_prev;
        }}

        /* Swap: prev_row = curr_row */
        for (int j = 0; j < nb; j++) prev_row[j] = curr_row[j];
    }}

    result[pair] = prev_row[nb - 1];
}}
"""

_FRECHET_KERNEL_NAMES = ("frechet_distance",)

# ---------------------------------------------------------------------------
# Pre-format kernel sources for fp64 and fp32 variants
# ---------------------------------------------------------------------------

_HAUSDORFF_FP64 = _HAUSDORFF_KERNEL_SOURCE.format(compute_type="double")
_HAUSDORFF_FP32 = _HAUSDORFF_KERNEL_SOURCE.format(compute_type="float")
_FRECHET_FP64 = _FRECHET_KERNEL_SOURCE.format(
    compute_type="double", max_frechet_b=_MAX_FRECHET_B,
)
_FRECHET_FP32 = _FRECHET_KERNEL_SOURCE.format(
    compute_type="float", max_frechet_b=_MAX_FRECHET_B,
)

# ---------------------------------------------------------------------------
# ADR-0034 background precompilation
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("hausdorff-fp64", _HAUSDORFF_FP64, _HAUSDORFF_KERNEL_NAMES),
    ("hausdorff-fp32", _HAUSDORFF_FP32, _HAUSDORFF_KERNEL_NAMES),
    ("frechet-fp64", _FRECHET_FP64, _FRECHET_KERNEL_NAMES),
    ("frechet-fp32", _FRECHET_FP32, _FRECHET_KERNEL_NAMES),
])

# ---------------------------------------------------------------------------
# GPU infrastructure imports (guarded)
# ---------------------------------------------------------------------------

from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.runtime.dispatch import record_dispatch_event  # noqa: E402
from vibespatial.runtime.kernel_registry import register_kernel_variant  # noqa: E402

# ---------------------------------------------------------------------------
# Build flat coordinate offsets on device (per-geometry coord range)
# ---------------------------------------------------------------------------

def _build_flat_coord_offsets(device_buf, family):
    """Build a device array mapping each geometry to its flat coord range.

    Returns a CuPy int32 array of shape ``(row_count + 1,)`` where
    ``offsets[g] .. offsets[g+1]`` spans all coordinates of geometry ``g``.

    Returns ``None`` for unsupported families (MultiPolygon).
    """
    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT,
                  GeometryFamily.LINESTRING):
        # geometry_offsets directly index into coords
        return cp.asarray(device_buf.geometry_offsets)

    if family is GeometryFamily.POLYGON:
        # Polygon: geometry_offsets -> ring indices, ring_offsets -> coord indices
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_ring_off = cp.asarray(device_buf.ring_offsets)
        return d_ring_off[d_geom_off]

    if family is GeometryFamily.MULTILINESTRING:
        # MultiLineString: geometry_offsets -> part indices, part_offsets -> coord indices
        d_geom_off = cp.asarray(device_buf.geometry_offsets)
        d_part_off = cp.asarray(device_buf.part_offsets)
        return d_part_off[d_geom_off]

    # MultiPolygon: 3-level indirection -- not supported on GPU path
    return None


# ---------------------------------------------------------------------------
# Kernel compilation helper
# ---------------------------------------------------------------------------

def _compile_distance_kernel(name_prefix, fp64_source, fp32_source,
                             kernel_names, compute_type="double"):
    """Compile and cache the fp64 or fp32 variant of a distance kernel."""
    source = fp64_source if compute_type == "double" else fp32_source
    suffix = "fp64" if compute_type == "double" else "fp32"
    return compile_kernel_group(f"{name_prefix}-{suffix}", source, kernel_names)


# ---------------------------------------------------------------------------
# GPU dispatch: Hausdorff distance
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "hausdorff_distance",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "multipoint", "linestring",
                       "multilinestring", "polygon"),
    supports_mixed=False,
    tags=("cuda-python", "metric", "hausdorff"),
)
def _hausdorff_gpu(owned_a, owned_b):
    """GPU Hausdorff distance.  Returns ``np.ndarray`` of shape ``(n,)``."""
    runtime = get_cuda_runtime()
    d_state_a = owned_a._ensure_device_state()
    d_state_b = owned_b._ensure_device_state()

    # Must be single-family, non-MultiPolygon
    if len(d_state_a.families) != 1 or len(d_state_b.families) != 1:
        raise NotImplementedError("GPU hausdorff requires single-family inputs")

    family_a = next(iter(d_state_a.families))
    family_b = next(iter(d_state_b.families))

    if family_a not in _GPU_SUPPORTED_FAMILIES or family_b not in _GPU_SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"GPU hausdorff does not support {family_a.name}/{family_b.name}"
        )

    buf_a = d_state_a.families[family_a]
    buf_b = d_state_b.families[family_b]

    d_offsets_a = _build_flat_coord_offsets(buf_a, family_a)
    d_offsets_b = _build_flat_coord_offsets(buf_b, family_b)
    if d_offsets_a is None or d_offsets_b is None:
        raise NotImplementedError("Could not build flat coord offsets")

    n = owned_a.row_count

    # Broadcast-right: the tiled array shares the 1-row family buffer so
    # d_offsets_b has 2 entries (1 geometry).  Build a constant-repeat
    # offset array on device and tile the coordinates so the kernel can
    # use standard CSR indexing.
    d_bcast_x = None
    d_bcast_y = None
    if int(d_offsets_b.size) == 2 and n > 1:
        k = int(d_offsets_b[1] - d_offsets_b[0])
        d_offsets_b = cp.arange(0, (n + 1) * k, k, dtype=np.int32)
        d_bcast_x = cp.tile(cp.asarray(buf_b.x), n)
        d_bcast_y = cp.tile(cp.asarray(buf_b.y), n)

    # Precision plan: METRIC class, default fp64 (wire for observability)
    compute_type = "double"
    center_x, center_y = 0.0, 0.0

    d_result = runtime.allocate((n,), np.float64)

    kernels = _compile_distance_kernel(
        "hausdorff", _HAUSDORFF_FP64, _HAUSDORFF_FP32,
        _HAUSDORFF_KERNEL_NAMES, compute_type,
    )
    kernel = kernels["hausdorff_distance"]

    # Launch: 1 block per pair, 256 threads per block (matches __launch_bounds__)
    grid = (n, 1, 1)
    block = (256, 1, 1)

    ptr = runtime.pointer
    bx = d_bcast_x if d_bcast_x is not None else buf_b.x
    by = d_bcast_y if d_bcast_y is not None else buf_b.y
    params = (
        (ptr(buf_a.x), ptr(buf_a.y), ptr(d_offsets_a),
         ptr(bx), ptr(by), ptr(d_offsets_b),
         ptr(d_result), center_x, center_y, n),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
         KERNEL_PARAM_I32),
    )
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # Single D->H transfer of the small result array
    return cp.asnumpy(d_result)


# ---------------------------------------------------------------------------
# GPU dispatch: Frechet distance
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "frechet_distance",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring",),
    supports_mixed=False,
    tags=("cuda-python", "metric", "frechet"),
)
def _frechet_gpu(owned_a, owned_b):
    """GPU discrete Frechet distance.  Returns ``np.ndarray`` of shape ``(n,)``."""
    runtime = get_cuda_runtime()
    d_state_a = owned_a._ensure_device_state()
    d_state_b = owned_b._ensure_device_state()

    # Frechet requires LineString family
    if GeometryFamily.LINESTRING not in d_state_a.families:
        raise NotImplementedError("GPU Frechet requires LineString in A")
    if GeometryFamily.LINESTRING not in d_state_b.families:
        raise NotImplementedError("GPU Frechet requires LineString in B")

    buf_a = d_state_a.families[GeometryFamily.LINESTRING]
    buf_b = d_state_b.families[GeometryFamily.LINESTRING]

    d_offsets_a = cp.asarray(buf_a.geometry_offsets)
    d_offsets_b = cp.asarray(buf_b.geometry_offsets)

    n = owned_a.row_count

    # Broadcast-right: the tiled array shares the 1-row family buffer so
    # d_offsets_b has 2 entries (1 geometry).  Tile the coordinates on
    # device and build monotonically increasing offsets.
    d_bcast_x = None
    d_bcast_y = None
    if int(d_offsets_b.size) == 2 and n > 1:
        k = int(d_offsets_b[1] - d_offsets_b[0])
        d_offsets_b = cp.arange(0, (n + 1) * k, k, dtype=np.int32)
        d_bcast_x = cp.tile(cp.asarray(buf_b.x), n)
        d_bcast_y = cp.tile(cp.asarray(buf_b.y), n)

    # Compute max B geometry coord count on device, transfer single scalar
    if n > 0:
        d_lens_b = d_offsets_b[1:] - d_offsets_b[:-1]
        max_b_len = int(cp.max(d_lens_b))
    else:
        max_b_len = 0

    if max_b_len > _MAX_FRECHET_B:
        raise NotImplementedError(
            f"Frechet B-sequence too long for GPU kernel "
            f"(max_b_len={max_b_len} > {_MAX_FRECHET_B})"
        )

    # Precision plan: METRIC class, default fp64
    compute_type = "double"
    center_x, center_y = 0.0, 0.0

    d_result = runtime.allocate((n,), np.float64)

    kernels = _compile_distance_kernel(
        "frechet", _FRECHET_FP64, _FRECHET_FP32,
        _FRECHET_KERNEL_NAMES, compute_type,
    )
    kernel = kernels["frechet_distance"]

    # 1 thread per pair -- use occupancy-based launch config
    grid, block = runtime.launch_config(kernel, n)

    ptr = runtime.pointer
    bx = d_bcast_x if d_bcast_x is not None else buf_b.x
    by = d_bcast_y if d_bcast_y is not None else buf_b.y
    params = (
        (ptr(buf_a.x), ptr(buf_a.y), ptr(d_offsets_a),
         ptr(bx), ptr(by), ptr(d_offsets_b),
         ptr(d_result), center_x, center_y, n, max_b_len),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
         KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # Single D->H transfer of the small result array
    return cp.asnumpy(d_result)


# ===========================================================================
# CPU implementations
# ===========================================================================

# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------

def _hausdorff_pair(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Brute-force Hausdorff distance between two coordinate arrays.

    coords_a, coords_b: (N, 2) and (M, 2) float64 arrays.
    Returns max(max_min(A->B), max_min(B->A)).
    """
    # For each point in A, min distance to B; take max
    forward = _min_distances_vectorized(coords_a, coords_b)
    # For each point in B, min distance to A; take max
    backward = _min_distances_vectorized(coords_b, coords_a)
    return float(max(np.max(forward), np.max(backward)))


@register_kernel_variant(
    "hausdorff_distance",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("point", "multipoint", "linestring",
                       "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cpu", "metric", "hausdorff"),
)
def _hausdorff_cpu(owned_a, owned_b, n):
    """CPU Hausdorff fallback.  Returns ``np.ndarray`` of shape ``(n,)``."""
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        coords_a = _coords_for_row(owned_a, i)
        coords_b = _coords_for_row(owned_b, i)
        if coords_a is None or coords_b is None:
            continue
        if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
            continue
        result[i] = _hausdorff_pair(coords_a, coords_b)
    return result


# ---------------------------------------------------------------------------
# Discrete Frechet distance
# ---------------------------------------------------------------------------

def _frechet_pair(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Discrete Frechet distance between two ordered coordinate sequences.

    Uses the classic DP coupling matrix:
      ca[i][j] = max(d(a_i, b_j), min(ca[i-1][j], ca[i][j-1], ca[i-1][j-1]))

    coords_a, coords_b: (N, 2) and (M, 2) float64 arrays.
    """
    na = coords_a.shape[0]
    nb = coords_b.shape[0]

    # Build pairwise distance matrix.
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))  # (na, nb)

    # DP coupling matrix.
    ca = np.empty((na, nb), dtype=np.float64)

    ca[0, 0] = d[0, 0]

    # First column.
    for i in range(1, na):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])

    # First row.
    for j in range(1, nb):
        ca[0, j] = max(ca[0, j - 1], d[0, j])

    # Fill rest.
    for i in range(1, na):
        for j in range(1, nb):
            ca[i, j] = max(
                d[i, j],
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
            )

    return float(ca[na - 1, nb - 1])


@register_kernel_variant(
    "frechet_distance",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring",),
    supports_mixed=False,
    tags=("cpu", "metric", "frechet"),
)
def _frechet_cpu(owned_a, owned_b, n):
    """CPU Frechet fallback.  Returns ``np.ndarray`` of shape ``(n,)``."""
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        coords_a = _linestring_coords_for_row(owned_a, i)
        coords_b = _linestring_coords_for_row(owned_b, i)
        if coords_a is None or coords_b is None:
            continue
        if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
            continue
        result[i] = _frechet_pair(coords_a, coords_b)
    return result


# ===========================================================================
# Public dispatch APIs
# ===========================================================================

def hausdorff_distance_owned(
    owned_a: OwnedGeometryArray,
    owned_b: OwnedGeometryArray,
    densify: float | None = None,
) -> np.ndarray:
    """Per-pair Hausdorff distance between two OwnedGeometryArrays.

    Returns a float64 array of length ``owned_a.row_count``.  NaN for null
    or empty geometry rows.

    The *densify* parameter is accepted for API compatibility but is not
    currently used by the CPU brute-force implementation (reserved for
    future GPU kernel densification).

    Works directly on OwnedGeometryArray coordinate buffers -- no Shapely
    round-trip.
    """
    from vibespatial.runtime.workload import WorkloadShape, detect_workload_shape

    n = owned_a.row_count
    workload = detect_workload_shape(n, owned_b.row_count)

    # Broadcast-right: tile the 1-row right to match left.
    if workload is WorkloadShape.BROADCAST_RIGHT:
        owned_b = tile_single_row(owned_b, n)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="hausdorff_distance",
        kernel_class=KernelClass.METRIC,
        row_count=n,
        requested_mode=ExecutionMode.AUTO,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _hausdorff_gpu(owned_a, owned_b)
        except (NotImplementedError, RuntimeError) as exc:
            logger.debug("Hausdorff GPU dispatch failed, falling back to CPU: %s", exc)
        else:
            record_dispatch_event(
                surface="geopandas.array.hausdorff_distance",
                operation="hausdorff_distance",
                implementation="hausdorff_gpu_nvrtc",
                reason=selection.reason,
                detail=f"rows={n}, workload={workload.value}",
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _hausdorff_cpu(owned_a, owned_b, n)
    record_dispatch_event(
        surface="geopandas.array.hausdorff_distance",
        operation="hausdorff_distance",
        implementation="hausdorff_cpu_brute_force",
        reason="CPU fallback",
        detail=f"rows={n}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def frechet_distance_owned(
    owned_a: OwnedGeometryArray,
    owned_b: OwnedGeometryArray,
    densify: float | None = None,
) -> np.ndarray:
    """Per-pair discrete Frechet distance between two OwnedGeometryArrays.

    Only works on LineString pairs.  Other geometry types produce NaN.
    Returns a float64 array of length ``owned_a.row_count``.  NaN for null
    rows or non-LineString geometry types.

    The *densify* parameter is accepted for API compatibility but is not
    currently used by the CPU implementation.

    Works directly on OwnedGeometryArray coordinate buffers -- no Shapely
    round-trip.
    """
    from vibespatial.runtime.workload import WorkloadShape, detect_workload_shape

    n = owned_a.row_count
    workload = detect_workload_shape(n, owned_b.row_count)

    # Broadcast-right: tile the 1-row right to match left.
    if workload is WorkloadShape.BROADCAST_RIGHT:
        owned_b = tile_single_row(owned_b, n)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="frechet_distance",
        kernel_class=KernelClass.METRIC,
        row_count=n,
        requested_mode=ExecutionMode.AUTO,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _frechet_gpu(owned_a, owned_b)
        except (NotImplementedError, RuntimeError) as exc:
            logger.debug("Frechet GPU dispatch failed, falling back to CPU: %s", exc)
        else:
            record_dispatch_event(
                surface="geopandas.array.frechet_distance",
                operation="frechet_distance",
                implementation="frechet_gpu_nvrtc",
                reason=selection.reason,
                detail=f"rows={n}, workload={workload.value}",
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _frechet_cpu(owned_a, owned_b, n)
    record_dispatch_event(
        surface="geopandas.array.frechet_distance",
        operation="frechet_distance",
        implementation="frechet_cpu_dp",
        reason="CPU fallback",
        detail=f"rows={n}, workload={workload.value}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
