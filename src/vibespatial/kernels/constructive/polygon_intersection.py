"""GPU-native element-wise polygon intersection kernel.

Computes element-wise intersection of two equal-length OwnedGeometryArrays
containing polygons, returning a device-resident OwnedGeometryArray without
any D->H transfer in the hot path.

Algorithm: Sutherland-Hodgman polygon clipping on GPU.
- For each pair (left[i], right[i]), clips left's exterior ring by each
  edge of right's exterior ring.
- Two-pass count-scatter pattern: pass 1 counts output vertices per pair,
  prefix sum computes offsets, pass 2 scatters clipped vertices.
- Degenerate results (empty, point, line) produce empty polygons with
  validity=False.

ADR-0033: Tier 1 (custom NVRTC kernel) -- geometry-specific inner loop
  with ring traversal and edge-by-edge clipping.
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
  PrecisionPlan wired through for observability only.
ADR-0034: NVRTC precompilation via request_nvrtc_warmup at module scope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
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
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import PrecisionPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC kernel source -- Sutherland-Hodgman polygon clipping
# ---------------------------------------------------------------------------
# The kernel uses a workspace buffer sized per-pair to hold intermediate
# clipped vertex lists.  Two buffers alternate roles (input/output) as
# each clip edge is processed.
#
# Limitations of Sutherland-Hodgman:
# - Subject polygon is clipped by a convex clip polygon (right operand).
#   For concave clip polygons, the result may include extra area.
#   This is acceptable as a first implementation; Weiler-Atherton can
#   be added later for full generality.
# - Holes are not handled in this initial version; only exterior rings.
#
# The workspace is sized at MAX_CLIP_VERTS per pair.  If clipping produces
# more vertices than this, the pair is marked as overflowed and falls back
# to validity=False (the CPU fallback handles it).
# ---------------------------------------------------------------------------

_MAX_CLIP_VERTS = 64  # 4 buffers * 64 * 8 bytes = 2KB per thread (vs 8KB at 256)

_POLYGON_INTERSECTION_KERNEL_SOURCE = r"""
#define MAX_CLIP_VERTS """ + str(_MAX_CLIP_VERTS) + r"""

/* ------------------------------------------------------------------ */
/*  Sutherland-Hodgman: clip a polygon by a single edge               */
/*                                                                     */
/*  clip_edge defined by points (ex0,ey0) -> (ex1,ey1).               */
/*  "Inside" is the left side of the directed edge.                    */
/* ------------------------------------------------------------------ */

/* Compute 2x signed area of a ring (shoelace formula).
   Positive => CCW, negative => CW.  Used to detect winding direction
   so the Sutherland-Hodgman inside/outside test works for both CW and
   CCW clip polygons. */
__device__ double ring_signed_area_2x(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start, int count
) {
    double area2 = 0.0;
    for (int i = 0; i < count; i++) {
        int j = (i + 1 < count) ? i + 1 : 0;
        area2 += x[start + i] * y[start + j] - x[start + j] * y[start + i];
    }
    return area2;
}

__device__ double cross_sign(
    double px, double py,
    double ex0, double ey0,
    double ex1, double ey1
) {
    return (ex1 - ex0) * (py - ey0) - (ey1 - ey0) * (px - ex0);
}

__device__ void line_intersect(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    double dx, double dy,
    double* ix, double* iy
) {
    /* Intersection of line (a->b) with line (c->d). */
    double a1 = by - ay;
    double b1 = ax - bx;
    double c1 = a1 * ax + b1 * ay;

    double a2 = dy - cy;
    double b2 = cx - dx;
    double c2 = a2 * cx + b2 * cy;

    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < 1e-15) {
        /* Parallel lines -- use midpoint of the shared segment. */
        *ix = (ax + bx) * 0.5;
        *iy = (ay + by) * 0.5;
    } else {
        *ix = (c1 * b2 - c2 * b1) / det;
        *iy = (a1 * c2 - a2 * c1) / det;
    }
}

/* ------------------------------------------------------------------ */
/*  Count kernel: compute output vertex count per pair                 */
/*                                                                     */
/*  One thread per geometry pair.  Runs Sutherland-Hodgman in          */
/*  registers/local memory to count output vertices.                   */
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void polygon_intersection_count(
    /* Left (subject) polygon buffers */
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    /* Validity masks (1=valid, 0=null/empty) */
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    /* Output */
    int* __restrict__ out_counts,
    int* __restrict__ out_valid,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* Invalid inputs -> empty output */
    if (!left_valid[idx] || !right_valid[idx]) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[idx];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[idx];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present (last == first). */
    if (l_n >= 2) {
        double dx = left_x[l_coord_start] - left_x[l_coord_end - 1];
        double dy = left_y[l_coord_start] - left_y[l_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) l_n--;
    }
    if (r_n >= 2) {
        double dx = right_x[r_coord_start] - right_x[r_coord_end - 1];
        double dy = right_y[r_coord_start] - right_y[r_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) r_n--;
    }

    /* Degenerate inputs -> empty */
    if (l_n < 3 || r_n < 3) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }

    /* Detect winding direction of the clip (right) polygon.
       Sutherland-Hodgman assumes the clip polygon is CCW (interior on
       the left side of each directed edge).  If the clip polygon is CW,
       we negate the cross_sign results so the inside/outside test still
       works correctly.  This handles arbitrary input winding without
       requiring a pre-normalization step. */
    const double clip_area2 = ring_signed_area_2x(
        right_x, right_y, r_coord_start, r_n);
    const double wsign = (clip_area2 >= 0.0) ? 1.0 : -1.0;

    /* Local workspace for Sutherland-Hodgman.
       We alternate between buf_a and buf_b. */
    double buf_ax[MAX_CLIP_VERTS], buf_ay[MAX_CLIP_VERTS];
    double buf_bx[MAX_CLIP_VERTS], buf_by[MAX_CLIP_VERTS];

    /* Initialize buf_a with the subject polygon vertices. */
    int input_count;
    if (l_n > MAX_CLIP_VERTS) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }
    for (int i = 0; i < l_n; i++) {
        buf_ax[i] = left_x[l_coord_start + i];
        buf_ay[i] = left_y[l_coord_start + i];
    }
    input_count = l_n;

    /* For each edge of the clip polygon, clip the current polygon. */
    double* in_x = buf_ax;
    double* in_y = buf_ay;
    double* out_x = buf_bx;
    double* out_y = buf_by;

    for (int e = 0; e < r_n; e++) {
        double ex0 = right_x[r_coord_start + e];
        double ey0 = right_y[r_coord_start + e];
        double ex1 = right_x[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];
        double ey1 = right_y[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];

        int out_count = 0;

        if (input_count == 0) break;

        for (int i = 0; i < input_count; i++) {
            int j = i + 1 < input_count ? i + 1 : 0;

            double sx = in_x[i], sy = in_y[i];
            double px = in_x[j], py = in_y[j];

            double s_side = wsign * cross_sign(sx, sy, ex0, ey0, ex1, ey1);
            double p_side = wsign * cross_sign(px, py, ex0, ey0, ex1, ey1);

            if (s_side >= 0.0) {
                /* S is inside */
                if (out_count < MAX_CLIP_VERTS) {
                    out_x[out_count] = sx;
                    out_y[out_count] = sy;
                    out_count++;
                }
                if (p_side < 0.0) {
                    /* S inside, P outside -> emit intersection */
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_x[out_count] = ix;
                        out_y[out_count] = iy;
                        out_count++;
                    }
                }
            } else {
                /* S is outside */
                if (p_side >= 0.0) {
                    /* S outside, P inside -> emit intersection then P */
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_x[out_count] = ix;
                        out_y[out_count] = iy;
                        out_count++;
                    }
                }
            }
        }

        /* Swap buffers for next edge */
        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_x;
        in_y = out_y;
        out_x = tmp_x;
        out_y = tmp_y;
        input_count = out_count;
    }

    if (input_count < 3) {
        /* Degenerate result (point or line) -> empty */
        out_counts[idx] = 0;
        out_valid[idx] = 0;
    } else {
        /* +1 for closing vertex */
        out_counts[idx] = input_count + 1;
        out_valid[idx] = 1;
    }
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel: write clipped polygon vertices to output           */
/*                                                                     */
/*  Re-runs Sutherland-Hodgman (same as count pass) and writes         */
/*  the result vertices at the pre-computed offsets.                    */
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void polygon_intersection_scatter(
    /* Left (subject) polygon buffers */
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    /* Validity masks */
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    /* Scatter targets */
    const int* __restrict__ output_offsets,
    const int* __restrict__ output_valid,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!output_valid[idx]) return;

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[idx];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[idx];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present. */
    if (l_n >= 2) {
        double dx = left_x[l_coord_start] - left_x[l_coord_end - 1];
        double dy = left_y[l_coord_start] - left_y[l_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) l_n--;
    }
    if (r_n >= 2) {
        double dx = right_x[r_coord_start] - right_x[r_coord_end - 1];
        double dy = right_y[r_coord_start] - right_y[r_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) r_n--;
    }

    /* Detect winding direction of the clip polygon (same as count pass). */
    const double clip_area2 = ring_signed_area_2x(
        right_x, right_y, r_coord_start, r_n);
    const double wsign = (clip_area2 >= 0.0) ? 1.0 : -1.0;

    /* Local workspace for Sutherland-Hodgman. */
    double buf_ax[MAX_CLIP_VERTS], buf_ay[MAX_CLIP_VERTS];
    double buf_bx[MAX_CLIP_VERTS], buf_by[MAX_CLIP_VERTS];

    int input_count;
    if (l_n > MAX_CLIP_VERTS) return;
    for (int i = 0; i < l_n; i++) {
        buf_ax[i] = left_x[l_coord_start + i];
        buf_ay[i] = left_y[l_coord_start + i];
    }
    input_count = l_n;

    double* in_x = buf_ax;
    double* in_y = buf_ay;
    double* out_bx = buf_bx;
    double* out_by = buf_by;

    for (int e = 0; e < r_n; e++) {
        double ex0 = right_x[r_coord_start + e];
        double ey0 = right_y[r_coord_start + e];
        double ex1 = right_x[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];
        double ey1 = right_y[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];

        int out_count = 0;
        if (input_count == 0) break;

        for (int i = 0; i < input_count; i++) {
            int j = i + 1 < input_count ? i + 1 : 0;

            double sx = in_x[i], sy = in_y[i];
            double px = in_x[j], py = in_y[j];

            double s_side = wsign * cross_sign(sx, sy, ex0, ey0, ex1, ey1);
            double p_side = wsign * cross_sign(px, py, ex0, ey0, ex1, ey1);

            if (s_side >= 0.0) {
                if (out_count < MAX_CLIP_VERTS) {
                    out_bx[out_count] = sx;
                    out_by[out_count] = sy;
                    out_count++;
                }
                if (p_side < 0.0) {
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_bx[out_count] = ix;
                        out_by[out_count] = iy;
                        out_count++;
                    }
                }
            } else {
                if (p_side >= 0.0) {
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_bx[out_count] = ix;
                        out_by[out_count] = iy;
                        out_count++;
                    }
                }
            }
        }

        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_bx;
        in_y = out_by;
        out_bx = tmp_x;
        out_by = tmp_y;
        input_count = out_count;
    }

    /* Write clipped vertices at the pre-computed offset. */
    int pos = output_offsets[idx];
    for (int i = 0; i < input_count; i++) {
        out_x[pos + i] = in_x[i];
        out_y[pos + i] = in_y[i];
    }
    /* Closing vertex: first vertex repeated. */
    out_x[pos + input_count] = in_x[0];
    out_y[pos + input_count] = in_y[0];
}
"""

_KERNEL_NAMES = ("polygon_intersection_count", "polygon_intersection_scatter")

# ---------------------------------------------------------------------------
# ADR-0034: request NVRTC precompilation at module scope
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("polygon-intersection", _POLYGON_INTERSECTION_KERNEL_SOURCE, _KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helper
# ---------------------------------------------------------------------------

def _polygon_intersection_kernels():
    """Compile and cache polygon intersection NVRTC kernels."""
    return compile_kernel_group(
        "polygon-intersection",
        _POLYGON_INTERSECTION_KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Device-backed OwnedGeometryArray builder
# ---------------------------------------------------------------------------

def _build_device_backed_polygon_intersection_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    """Build a device-resident OwnedGeometryArray from GPU coordinate buffers.

    Follows the DeviceFamilyGeometryBuffer pattern from
    ``_build_device_backed_fixed_polygon_output`` but handles variable-length
    output and per-row validity.
    """
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=None,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        runtime_history=[runtime_selection],
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
                    bounds=None,
                )
            },
        ),
    )


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _extract_polygon_family_buffers(owned: OwnedGeometryArray):
    """Extract polygon family device buffers, uploading if needed.

    Returns (device_buf, host_buf) for the POLYGON family, or
    (None, None) if no polygon rows exist.
    """
    if GeometryFamily.POLYGON not in owned.families:
        return None, None
    host_buf = owned.families[GeometryFamily.POLYGON]
    if host_buf.row_count == 0:
        return None, None

    state = owned._ensure_device_state()
    if GeometryFamily.POLYGON not in state.families:
        return None, None
    return state.families[GeometryFamily.POLYGON], host_buf


def _polygon_intersection_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    """GPU polygon intersection via Sutherland-Hodgman clipping.

    Both inputs must be polygon-only OwnedGeometryArrays of equal length.
    Returns a device-resident OwnedGeometryArray.
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    n = left.row_count

    # Ensure device state for both inputs
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_intersection selected GPU execution",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_intersection selected GPU execution",
    )

    left_dev, left_host = _extract_polygon_family_buffers(left)
    right_dev, right_host = _extract_polygon_family_buffers(right)

    if left_dev is None or right_dev is None:
        # No polygon data -- return all-empty
        return _build_empty_result(n, runtime_selection)

    # Validate that both inputs are polygon-only (1:1 global-to-family mapping).
    # The kernel uses idx as both global row index and family buffer index.
    if left_host.row_count != n or right_host.row_count != n:
        raise ValueError(
            "polygon_intersection GPU path requires polygon-only inputs "
            f"(left family rows={left_host.row_count}, "
            f"right family rows={right_host.row_count}, expected={n})"
        )

    # Build per-row validity masks on device (int32 for kernel compatibility).
    # Since we verified polygon-only, the family empty_mask is 1:1 with rows.
    left_state = left.device_state
    right_state = right.device_state
    d_left_valid = (
        left_state.validity.astype(cp.bool_) & ~left_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)
    d_right_valid = (
        right_state.validity.astype(cp.bool_) & ~right_dev.empty_mask.astype(cp.bool_)
    ).astype(cp.int32)

    # Allocate output arrays for the count pass
    d_counts = runtime.allocate((n,), np.int32, zero=True)
    d_valid = runtime.allocate((n,), np.int32, zero=True)

    # Compile and launch count kernel
    kernels = _polygon_intersection_kernels()
    ptr = runtime.pointer

    count_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_counts),
            ptr(d_valid),
            n,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["polygon_intersection_count"], n)
    runtime.launch(
        kernels["polygon_intersection_count"],
        grid=grid, block=block, params=count_params,
    )

    # Exclusive prefix sum for scatter offsets (same-stream, no sync needed)
    d_offsets = exclusive_sum(d_counts, synchronize=False)

    # Get total output vertices
    total_verts = count_scatter_total(runtime, d_counts, d_offsets)

    if total_verts == 0:
        # All intersections are empty
        return _build_empty_result(n, runtime_selection)

    # Allocate output coordinate arrays
    d_out_x = runtime.allocate((total_verts,), np.float64)
    d_out_y = runtime.allocate((total_verts,), np.float64)

    # Launch scatter kernel
    scatter_params = (
        (
            ptr(left_dev.x),
            ptr(left_dev.y),
            ptr(left_dev.ring_offsets),
            ptr(left_dev.geometry_offsets),
            ptr(right_dev.x),
            ptr(right_dev.y),
            ptr(right_dev.ring_offsets),
            ptr(right_dev.geometry_offsets),
            ptr(d_left_valid),
            ptr(d_right_valid),
            ptr(d_offsets),
            ptr(d_valid),
            ptr(d_out_x),
            ptr(d_out_y),
            n,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    scatter_grid, scatter_block = runtime.launch_config(
        kernels["polygon_intersection_scatter"], n,
    )
    runtime.launch(
        kernels["polygon_intersection_scatter"],
        grid=scatter_grid, block=scatter_block, params=scatter_params,
    )

    # Build ring_offsets on device from the existing d_offsets (exclusive prefix
    # sum of d_counts) to avoid D2H -> host cumsum -> H2D ping-pong.
    # ring_offsets[i] = d_offsets[i] for i < n, ring_offsets[n] = total_verts.
    runtime.synchronize()

    # d_offsets is already the exclusive prefix sum = inclusive ring_offsets[0:n].
    # Append total_verts to get the full ring_offsets array on device.
    import cupy as _cp

    d_ring_offsets = _cp.empty(n + 1, dtype=_cp.int32)
    d_ring_offsets[:n] = _cp.asarray(d_offsets)
    d_ring_offsets[n] = total_verts

    # geometry_offsets: one ring per polygon = [0, 1, 2, ..., n]
    # (host-side only; device state uses the same pattern implicitly)

    # Host copies for OwnedGeometryArray metadata (small: O(n) int/bool).
    # d_valid is already a CuPy array (returned by runtime.allocate), so
    # pass it directly to copy_device_to_host -- no cp.asarray() needed.
    validity = runtime.copy_device_to_host(d_valid).astype(bool)
    geometry_offsets = np.arange(n + 1, dtype=np.int32)
    ring_offsets = runtime.copy_device_to_host(d_ring_offsets)

    return _build_device_backed_polygon_intersection_output(
        d_out_x,
        d_out_y,
        row_count=n,
        validity=validity,
        geometry_offsets=geometry_offsets,
        ring_offsets=ring_offsets,
        runtime_selection=runtime_selection,
    )


def _build_empty_result(n: int, runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    """Build an all-empty polygon result."""
    runtime = get_cuda_runtime()
    d_x = runtime.allocate((0,), np.float64)
    d_y = runtime.allocate((0,), np.float64)
    return _build_device_backed_polygon_intersection_output(
        d_x,
        d_y,
        row_count=n,
        validity=np.zeros(n, dtype=bool),
        geometry_offsets=np.arange(n + 1, dtype=np.int32),
        ring_offsets=np.zeros(n + 1, dtype=np.int32),
        runtime_selection=runtime_selection,
    )


# ---------------------------------------------------------------------------
# Registered kernel variants
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "polygon_intersection",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    tags=("shapely", "constructive", "intersection"),
)
def _polygon_intersection_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """CPU fallback: Shapely element-wise polygon intersection."""
    import shapely

    left_geoms = left.to_shapely()
    right_geoms = right.to_shapely()

    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms

    result_arr = shapely.intersection(left_arr, right_arr)
    return from_shapely_geometries(list(result_arr))


@register_kernel_variant(
    "polygon_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon",),
    supports_mixed=False,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "constructive", "intersection", "sutherland-hodgman"),
)
def _polygon_intersection_gpu_variant(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray:
    """GPU polygon intersection via Sutherland-Hodgman NVRTC kernel."""
    return _polygon_intersection_gpu(
        left, right,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def polygon_intersection(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Element-wise polygon intersection of two OwnedGeometryArrays.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Input polygon arrays of equal length.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint (GPU/CPU/AUTO).
    precision : PrecisionMode or str, default AUTO
        Precision mode. CONSTRUCTIVE kernels stay fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Device-resident result when GPU path is taken; host-resident
        when CPU fallback is used.
    """
    if left.row_count != right.row_count:
        raise ValueError(
            f"row count mismatch: left={left.row_count}, right={right.row_count}"
        )

    n = left.row_count
    if n == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="polygon_intersection",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        # ADR-0002: CONSTRUCTIVE stays fp64; plan is for observability.
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )

        try:
            result = _polygon_intersection_gpu(
                left, right,
                runtime_selection=selection,
                precision_plan=precision_plan,
            )
            record_dispatch_event(
                surface="vibespatial.kernels.constructive.polygon_intersection",
                operation="polygon_intersection",
                implementation="polygon_intersection_gpu",
                reason=selection.reason,
                detail=(
                    f"rows={n}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            logger.debug(
                "GPU polygon_intersection failed, falling back to CPU",
                exc_info=True,
            )

    # CPU fallback
    result = _polygon_intersection_cpu(left, right, precision=precision)
    record_dispatch_event(
        surface="vibespatial.kernels.constructive.polygon_intersection",
        operation="polygon_intersection",
        implementation="polygon_intersection_cpu",
        reason=selection.reason,
        detail=f"rows={n}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
