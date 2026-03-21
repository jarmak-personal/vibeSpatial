"""GPU-accelerated linear referencing operations.

Provides ``interpolate`` (point at a distance along a line) and ``project``
(distance along a line to the nearest point) for LineString and
MultiLineString geometries, operating directly on OwnedGeometryArray
coordinate buffers.

ADR-0033: Tier 1 NVRTC kernels launched via CudaDriverRuntime with
CPU fallback when GPU is unavailable.
ADR-0002: CONSTRUCTIVE class — output coordinates are deterministic
linear combinations of input coordinates.

CPU fallback works on coordinate buffers directly (no Shapely round-trip
for the hot path).
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

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

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("interpolate-linestring", _INTERPOLATE_LINESTRING_KERNEL_SOURCE, _INTERPOLATE_LINESTRING_KERNEL_NAMES),
    ("interpolate-multilinestring", _INTERPOLATE_MULTILINESTRING_KERNEL_SOURCE, _INTERPOLATE_MULTILINESTRING_KERNEL_NAMES),
    ("project-linestring", _PROJECT_LINESTRING_KERNEL_SOURCE, _PROJECT_LINESTRING_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# CPU fallback: interpolate along LineString coordinate buffer
# ---------------------------------------------------------------------------

def _interpolate_linestring_cpu(
    buf: FamilyGeometryBuffer,
    family_rows: np.ndarray,
    distances: np.ndarray,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate points along LineString coordinate spans.

    Returns (out_x, out_y) arrays aligned 1:1 with *family_rows*.
    """
    n = family_rows.size
    out_x = np.full(n, np.nan, dtype=np.float64)
    out_y = np.full(n, np.nan, dtype=np.float64)
    geom_offsets = buf.geometry_offsets

    for idx in range(n):
        fr = family_rows[idx]
        cs = geom_offsets[fr]
        ce = geom_offsets[fr + 1]
        npts = ce - cs
        if npts == 0:
            continue
        if npts == 1:
            out_x[idx] = buf.x[cs]
            out_y[idx] = buf.y[cs]
            continue

        seg_x = buf.x[cs:ce]
        seg_y = buf.y[cs:ce]
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        seg_lens = np.sqrt(dx * dx + dy * dy)
        total_len = float(np.sum(seg_lens))

        target = float(distances[idx])
        if normalized:
            target = target * total_len

        if target <= 0.0:
            out_x[idx] = seg_x[0]
            out_y[idx] = seg_y[0]
            continue

        accum = 0.0
        found = False
        for si in range(len(seg_lens)):
            sl = seg_lens[si]
            if accum + sl >= target:
                frac = (target - accum) / sl if sl > 1e-30 else 0.0
                out_x[idx] = seg_x[si] + frac * dx[si]
                out_y[idx] = seg_y[si] + frac * dy[si]
                found = True
                break
            accum += sl

        if not found:
            # Clamp to end
            out_x[idx] = seg_x[-1]
            out_y[idx] = seg_y[-1]

    return out_x, out_y


# ---------------------------------------------------------------------------
# CPU fallback: interpolate along MultiLineString coordinate buffer
# ---------------------------------------------------------------------------

def _interpolate_multilinestring_cpu(
    buf: FamilyGeometryBuffer,
    family_rows: np.ndarray,
    distances: np.ndarray,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate points along MultiLineString coordinate spans.

    Treats the MultiLineString as a single path by concatenating parts
    in order. Returns (out_x, out_y) aligned 1:1 with *family_rows*.
    """
    n = family_rows.size
    out_x = np.full(n, np.nan, dtype=np.float64)
    out_y = np.full(n, np.nan, dtype=np.float64)
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for idx in range(n):
        fr = family_rows[idx]
        fp = geom_offsets[fr]
        lp = geom_offsets[fr + 1]
        if fp == lp:
            continue

        # Compute total length across all parts
        total_len = 0.0
        for p in range(fp, lp):
            cs = part_offsets[p]
            ce = part_offsets[p + 1]
            if ce - cs < 2:
                continue
            seg_x = buf.x[cs:ce]
            seg_y = buf.y[cs:ce]
            dx = np.diff(seg_x)
            dy = np.diff(seg_y)
            total_len += float(np.sum(np.sqrt(dx * dx + dy * dy)))

        target = float(distances[idx])
        if normalized:
            target = target * total_len

        # Clamp to start
        if target <= 0.0:
            cs0 = part_offsets[fp]
            out_x[idx] = buf.x[cs0]
            out_y[idx] = buf.y[cs0]
            continue

        # Walk parts and segments
        accum = 0.0
        found = False
        for p in range(fp, lp):
            cs = part_offsets[p]
            ce = part_offsets[p + 1]
            if ce - cs < 2:
                continue
            seg_x = buf.x[cs:ce]
            seg_y = buf.y[cs:ce]
            dxs = np.diff(seg_x)
            dys = np.diff(seg_y)
            seg_lens = np.sqrt(dxs * dxs + dys * dys)
            for si in range(len(seg_lens)):
                sl = seg_lens[si]
                if accum + sl >= target:
                    frac = (target - accum) / sl if sl > 1e-30 else 0.0
                    out_x[idx] = seg_x[si] + frac * dxs[si]
                    out_y[idx] = seg_y[si] + frac * dys[si]
                    found = True
                    break
                accum += sl
            if found:
                break

        if not found:
            # Clamp to end of last part
            last_ce = part_offsets[lp] - 1
            out_x[idx] = buf.x[last_ce]
            out_y[idx] = buf.y[last_ce]

    return out_x, out_y


# ---------------------------------------------------------------------------
# CPU fallback: project point onto LineString
# ---------------------------------------------------------------------------

def _project_linestring_cpu(
    line_buf: FamilyGeometryBuffer,
    line_family_rows: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> np.ndarray:
    """Project points onto LineStrings, returning cumulative distance to closest point.

    Returns float64 array aligned 1:1 with *line_family_rows*.
    """
    n = line_family_rows.size
    out = np.full(n, np.nan, dtype=np.float64)
    geom_offsets = line_buf.geometry_offsets

    for idx in range(n):
        fr = line_family_rows[idx]
        cs = geom_offsets[fr]
        ce = geom_offsets[fr + 1]
        npts = ce - cs

        if npts == 0:
            continue
        if npts == 1:
            out[idx] = 0.0
            continue

        px = float(point_x[idx])
        py = float(point_y[idx])

        best_sq_dist = np.inf
        best_along = 0.0
        accum = 0.0

        lx = line_buf.x[cs:ce]
        ly = line_buf.y[cs:ce]

        for si in range(npts - 1):
            ax, ay = float(lx[si]), float(ly[si])
            bx, by = float(lx[si + 1]), float(ly[si + 1])
            dx = bx - ax
            dy = by - ay
            seg_len_sq = dx * dx + dy * dy
            seg_len = np.sqrt(seg_len_sq)

            t = 0.0
            if seg_len_sq > 1e-30:
                t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))

            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

            if d_sq < best_sq_dist:
                best_sq_dist = d_sq
                best_along = accum + t * seg_len

            accum += seg_len

        out[idx] = best_along

    return out


# ---------------------------------------------------------------------------
# GPU implementation: interpolate
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "interpolate_linear_ref",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "linear-ref", "interpolate"),
)
def _interpolate_gpu(
    owned: OwnedGeometryArray,
    dist_arr: np.ndarray,
    normalized: bool,
) -> OwnedGeometryArray:
    """GPU interpolate -- returns device-resident Point OwnedGeometryArray."""
    from vibespatial.constructive.point import _build_device_backed_point_output

    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count

    # Allocate global output arrays (NaN = invalid/missing)
    d_out_x = cp.full(row_count, cp.nan, dtype=cp.float64)
    d_out_y = cp.full(row_count, cp.nan, dtype=cp.float64)

    normalized_int = 1 if normalized else 0
    ptr = runtime.pointer

    # Retrieve host tags for family routing (small array, once)
    h_tags = owned.tags

    # --- LineString family ---
    if GeometryFamily.LINESTRING in d_state.families:
        dbuf = d_state.families[GeometryFamily.LINESTRING]
        fam_row_count = int(dbuf.geometry_offsets.shape[0]) - 1
        if fam_row_count > 0:
            kernels = compile_kernel_group(
                "interpolate-linestring",
                _INTERPOLATE_LINESTRING_KERNEL_SOURCE,
                _INTERPOLATE_LINESTRING_KERNEL_NAMES,
            )
            kernel = kernels["interpolate_linestring"]

            # Find global rows belonging to this family
            ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
            global_rows = np.flatnonzero(h_tags == ls_tag)
            fam_distances = cp.asarray(dist_arr[global_rows])

            d_fam_out_x = runtime.allocate((fam_row_count,), np.float64)
            d_fam_out_y = runtime.allocate((fam_row_count,), np.float64)

            params = (
                (
                    ptr(dbuf.x),
                    ptr(dbuf.y),
                    ptr(dbuf.geometry_offsets),
                    ptr(fam_distances),
                    normalized_int,
                    ptr(d_fam_out_x),
                    ptr(d_fam_out_y),
                    fam_row_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, fam_row_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)

            # Scatter family-local results into global output
            d_global_rows = cp.asarray(global_rows)
            d_out_x[d_global_rows] = d_fam_out_x
            d_out_y[d_global_rows] = d_fam_out_y

    # --- MultiLineString family ---
    if GeometryFamily.MULTILINESTRING in d_state.families:
        dbuf = d_state.families[GeometryFamily.MULTILINESTRING]
        fam_row_count = int(dbuf.geometry_offsets.shape[0]) - 1
        if fam_row_count > 0:
            kernels = compile_kernel_group(
                "interpolate-multilinestring",
                _INTERPOLATE_MULTILINESTRING_KERNEL_SOURCE,
                _INTERPOLATE_MULTILINESTRING_KERNEL_NAMES,
            )
            kernel = kernels["interpolate_multilinestring"]

            mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
            global_rows = np.flatnonzero(h_tags == mls_tag)
            fam_distances = cp.asarray(dist_arr[global_rows])

            d_fam_out_x = runtime.allocate((fam_row_count,), np.float64)
            d_fam_out_y = runtime.allocate((fam_row_count,), np.float64)

            params = (
                (
                    ptr(dbuf.x),
                    ptr(dbuf.y),
                    ptr(dbuf.part_offsets),
                    ptr(dbuf.geometry_offsets),
                    ptr(fam_distances),
                    normalized_int,
                    ptr(d_fam_out_x),
                    ptr(d_fam_out_y),
                    fam_row_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, fam_row_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)

            d_global_rows = cp.asarray(global_rows)
            d_out_x[d_global_rows] = d_fam_out_x
            d_out_y[d_global_rows] = d_fam_out_y

    # Mark invalid rows as NaN on device
    d_validity = cp.asarray(owned.validity)
    d_out_x[~d_validity] = cp.nan
    d_out_y[~d_validity] = cp.nan

    return _build_device_backed_point_output(d_out_x, d_out_y, row_count=row_count)


# ---------------------------------------------------------------------------
# GPU implementation: project
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "project_linear_ref",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring",),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "linear-ref", "project"),
)
def _project_gpu(
    owned: OwnedGeometryArray,
    point_owned: OwnedGeometryArray,
) -> np.ndarray:
    """GPU project -- returns np.ndarray of float64 distances."""
    runtime = get_cuda_runtime()
    d_line_state = owned._ensure_device_state()
    d_point_state = point_owned._ensure_device_state()

    row_count = owned.row_count

    line_dbuf = d_line_state.families[GeometryFamily.LINESTRING]
    point_dbuf = d_point_state.families[GeometryFamily.POINT]

    # Extract point coordinates: each Point has geometry_offsets[row] as
    # the coordinate index.  For a Point-only array with 1 coord per row,
    # point_dbuf.x and point_dbuf.y are already contiguous coordinate arrays.
    # We need per-row x,y: coord_idx = geometry_offsets[0..row_count-1].
    d_point_coord_idx = point_dbuf.geometry_offsets[:-1]  # CuPy slice on device
    d_point_x = point_dbuf.x[d_point_coord_idx]
    d_point_y = point_dbuf.y[d_point_coord_idx]

    d_out_dist = runtime.allocate((row_count,), np.float64)

    kernels = compile_kernel_group(
        "project-linestring",
        _PROJECT_LINESTRING_KERNEL_SOURCE,
        _PROJECT_LINESTRING_KERNEL_NAMES,
    )
    kernel = kernels["project_linestring"]

    ptr = runtime.pointer
    params = (
        (
            ptr(line_dbuf.x),
            ptr(line_dbuf.y),
            ptr(line_dbuf.geometry_offsets),
            ptr(d_point_x),
            ptr(d_point_y),
            ptr(d_out_dist),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    return cp.asnumpy(d_out_dist)


# ---------------------------------------------------------------------------
# Public API: interpolate_owned
# ---------------------------------------------------------------------------

def interpolate_owned(
    owned: OwnedGeometryArray,
    distance: float | np.ndarray,
    normalized: bool = False,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Interpolate a point at *distance* along each geometry.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Must contain only LineString and/or MultiLineString families.
    distance : float or array-like
        Distance(s) along the line.  Scalar is broadcast.
    normalized : bool
        If True, *distance* is a fraction of total line length [0, 1].
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO, GPU, CPU).

    Returns
    -------
    OwnedGeometryArray
        Point-family OwnedGeometryArray with one point per input geometry.

    Raises
    ------
    ValueError
        If *owned* contains non-line families (Point, Polygon, etc.).
    """
    # GPU path (future): use _build_device_backed_point_output from
    # vibespatial.constructive.point to return device-resident points.
    # For now, all paths use CPU fallback and return via point_owned_from_xy.

    # Validate families
    allowed = {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}
    present = set(owned.families.keys())
    if not present:
        raise ValueError("interpolate_owned requires a non-empty OwnedGeometryArray")
    if not present.issubset(allowed):
        bad = present - allowed
        raise ValueError(
            f"interpolate_owned only supports LineString/MultiLineString, "
            f"got: {', '.join(str(f) for f in bad)}"
        )

    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.constructive.point import _empty_point_output
        return _empty_point_output()

    # Prepare distance array
    if np.isscalar(distance):
        dist_arr = np.full(row_count, float(distance), dtype=np.float64)
    else:
        dist_arr = np.asarray(distance, dtype=np.float64)
    if dist_arr.shape != (row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    # Dispatch selection
    selection = plan_dispatch_selection(
        kernel_name="interpolate_linear_ref",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            return _interpolate_gpu(owned, dist_arr, normalized)
        except Exception:
            pass

    # CPU fallback
    owned._ensure_host_state()

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    out_x = np.full(row_count, np.nan, dtype=np.float64)
    out_y = np.full(row_count, np.nan, dtype=np.float64)

    # --- LineString family ---
    ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    ls_mask = tags == ls_tag
    if np.any(ls_mask) and GeometryFamily.LINESTRING in owned.families:
        buf = owned.families[GeometryFamily.LINESTRING]
        if buf.row_count > 0:
            global_rows = np.flatnonzero(ls_mask & owned.validity)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows]
                row_dists = dist_arr[global_rows]
                ix, iy = _interpolate_linestring_cpu(buf, family_rows, row_dists, normalized)
                out_x[global_rows] = ix
                out_y[global_rows] = iy

    # --- MultiLineString family ---
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    mls_mask = tags == mls_tag
    if np.any(mls_mask) and GeometryFamily.MULTILINESTRING in owned.families:
        buf = owned.families[GeometryFamily.MULTILINESTRING]
        if buf.row_count > 0 and buf.part_offsets is not None:
            global_rows = np.flatnonzero(mls_mask & owned.validity)
            if global_rows.size > 0:
                family_rows = family_row_offsets[global_rows]
                row_dists = dist_arr[global_rows]
                ix, iy = _interpolate_multilinestring_cpu(buf, family_rows, row_dists, normalized)
                out_x[global_rows] = ix
                out_y[global_rows] = iy

    # Mark null rows as NaN
    out_x[~owned.validity] = np.nan
    out_y[~owned.validity] = np.nan

    # Build Point OwnedGeometryArray from CPU results
    from vibespatial.constructive.point import point_owned_from_xy
    return point_owned_from_xy(out_x, out_y)


# ---------------------------------------------------------------------------
# Public API: project_owned
# ---------------------------------------------------------------------------

def project_owned(
    owned: OwnedGeometryArray,
    point_owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> np.ndarray:
    """Project each point onto the corresponding LineString.

    Returns the cumulative distance along the LineString to the point
    on the line nearest to the query point.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Must contain only LineString family (MultiLineString not supported
        for project).
    point_owned : OwnedGeometryArray
        Point-family OwnedGeometryArray, same row count as *owned*.
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO, GPU, CPU).

    Returns
    -------
    np.ndarray
        float64 array of projected distances, one per row.

    Raises
    ------
    ValueError
        If *owned* is not LineString-only or *point_owned* is not Point-only,
        or row counts do not match.
    """
    # Validate line input
    if GeometryFamily.LINESTRING not in owned.families or len(owned.families) != 1:
        raise ValueError("project_owned requires a LineString-only OwnedGeometryArray")

    # Validate point input
    if GeometryFamily.POINT not in point_owned.families or len(point_owned.families) != 1:
        raise ValueError("project_owned requires a Point-only OwnedGeometryArray for point_owned")

    row_count = owned.row_count
    if point_owned.row_count != row_count:
        raise ValueError(
            f"owned and point_owned must have the same row count, "
            f"got {row_count} vs {point_owned.row_count}"
        )

    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    # Dispatch selection
    selection = plan_dispatch_selection(
        kernel_name="project_linear_ref",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            return _project_gpu(owned, point_owned)
        except Exception:
            pass

    # CPU fallback
    owned._ensure_host_state()
    point_owned._ensure_host_state()

    line_buf = owned.families[GeometryFamily.LINESTRING]
    point_buf = point_owned.families[GeometryFamily.POINT]

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    point_tags = point_owned.tags
    point_family_row_offsets = point_owned.family_row_offsets

    out = np.full(row_count, np.nan, dtype=np.float64)

    # Select valid rows where both line and point are valid
    valid_mask = owned.validity & point_owned.validity
    ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    pt_tag = FAMILY_TAGS[GeometryFamily.POINT]
    type_mask = (tags == ls_tag) & (point_tags == pt_tag)
    work_mask = valid_mask & type_mask
    global_rows = np.flatnonzero(work_mask)

    if global_rows.size > 0:
        line_family_rows = family_row_offsets[global_rows]
        point_family_rows = point_family_row_offsets[global_rows]

        # Extract point coordinates
        pt_coord_indices = point_buf.geometry_offsets[point_family_rows]
        px = point_buf.x[pt_coord_indices]
        py = point_buf.y[pt_coord_indices]

        result = _project_linestring_cpu(line_buf, line_family_rows, px, py)
        out[global_rows] = result

    return out
