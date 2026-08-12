"""Module to clip vector data using GeoPandas."""

import logging
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pandas.api.types
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_results import (
    GeometryNativeResult,
    LeftConstructiveResult,
    NativeTabularResult,
    NativeTabularSelection,
    _clip_constructive_parts_to_native_tabular_result,
    _clip_native_tabular_result_from_rowset,
    _concat_native_tabular_selections,
    _spatial_to_native_tabular_result,
)
from vibespatial.api._native_results import (
    _geometry_composition_from_owned_parts_at_capacity as _clip_geometry_composition_at_capacity,
)
from vibespatial.api._native_rowset import NativeDeviceSelection
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.api.geometry_array import (
    from_shapely as _geometryarray_from_shapely,
)
from vibespatial.constructive.grouped_mixed_union import (
    pack_line_part_capacity_device as _clip_pack_line_boundary_part_capacity_device,
)
from vibespatial.constructive.grouped_mixed_union import (
    pack_point_part_capacity_device as _clip_pack_point_boundary_part_capacity_device,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime._runtime import has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_physical_work_from_owned,
)
from vibespatial.runtime.fallbacks import (
    StrictNativeFallbackError,
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

_DEVICE_CLIP_GEOM_TYPES = POINT_GEOM_TYPES | LINE_GEOM_TYPES | POLYGON_GEOM_TYPES
logger = logging.getLogger(__name__)

_CLIP_ROWSET_KERNEL_NAMES = (
    "bbox_candidate_rows_i64_kernel",
    "bbox_candidate_mask_u8_kernel",
    "bool_mask_to_rows_i64_kernel",
)

_CLIP_ROWSET_KERNEL_SOURCE = r"""
extern "C" __global__ void bbox_candidate_rows_i64_kernel(
    const double* __restrict__ bounds,
    long long n,
    const double* __restrict__ query_bounds,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int use_query_bounds,
    long long* __restrict__ rows,
    int* __restrict__ count
) {
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double qxmin = use_query_bounds ? query_bounds[0] : xmin;
    const double qymin = use_query_bounds ? query_bounds[1] : ymin;
    const double qxmax = use_query_bounds ? query_bounds[2] : xmax;
    const double qymax = use_query_bounds ? query_bounds[3] : ymax;
    const double* b = bounds + (4 * i);
    const bool hit = (
        b[0] <= qxmax && b[2] >= qxmin &&
        b[1] <= qymax && b[3] >= qymin
    );
    if (!hit) return;
    const int pos = atomicAdd(count, 1);
    rows[pos] = i;
}

extern "C" __global__ void bbox_candidate_mask_u8_kernel(
    const double* __restrict__ bounds,
    long long n,
    const double* __restrict__ query_bounds,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int use_query_bounds,
    unsigned char* __restrict__ mask
) {
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double qxmin = use_query_bounds ? query_bounds[0] : xmin;
    const double qymin = use_query_bounds ? query_bounds[1] : ymin;
    const double qxmax = use_query_bounds ? query_bounds[2] : xmax;
    const double qymax = use_query_bounds ? query_bounds[3] : ymax;
    const double* b = bounds + (4 * i);
    mask[i] = (
        b[0] <= qxmax && b[2] >= qxmin &&
        b[1] <= qymax && b[3] >= qymin
    ) ? 1u : 0u;
}

extern "C" __global__ void bool_mask_to_rows_i64_kernel(
    const unsigned char* __restrict__ mask,
    long long n,
    long long* __restrict__ rows,
    int* __restrict__ count
) {
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask[i] == 0u) return;
    const int pos = atomicAdd(count, 1);
    rows[pos] = i;
}
"""

_CLIP_POINT_CONTACT_SLIVER_KERNEL_NAMES = ("point_contact_sliver_rows_kernel",)

_CLIP_POINT_CONTACT_SLIVER_KERNEL_SOURCE = r"""
extern "C" __global__ void point_contact_sliver_rows_kernel(
    const double* __restrict__ mask_x,
    const double* __restrict__ mask_y,
    const int* __restrict__ mask_geometry_offsets,
    const int* __restrict__ mask_ring_offsets,
    const int* __restrict__ mask_family_rows,
    const unsigned char* __restrict__ mask_single_ring,
    const double* __restrict__ point_x,
    const double* __restrict__ point_y,
    const unsigned char* __restrict__ active_rows,
    const double* __restrict__ row_bounds,
    long long n,
    double eps,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned char* __restrict__ out_valid
) {
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (active_rows[i] == 0u || mask_single_ring[0] == 0u) return;

    const int mask_family_row = mask_family_rows[0];
    if (mask_family_row < 0) return;
    const int first_ring = mask_geometry_offsets[mask_family_row];
    const int next_ring = mask_geometry_offsets[mask_family_row + 1];
    if (next_ring - first_ring != 1) return;
    const int coord_start = mask_ring_offsets[first_ring];
    const int coord_stop = mask_ring_offsets[next_ring];
    const long long vertex_count =
        (long long)coord_stop - (long long)coord_start - 1LL;
    if (vertex_count < 3LL || vertex_count > 256LL) return;

    const double px = point_x[i];
    const double py = point_y[i];
    const double xmin = row_bounds[4 * i + 0];
    const double ymin = row_bounds[4 * i + 1];
    const double xmax = row_bounds[4 * i + 2];
    const double ymax = row_bounds[4 * i + 3];
    if (
        !isfinite(px) || !isfinite(py) ||
        !isfinite(xmin) || !isfinite(ymin) ||
        !isfinite(xmax) || !isfinite(ymax) ||
        vertex_count <= 0
    ) {
        return;
    }

    long long vertex_index = 0;
    int has_vertex = 0;
    for (long long j = 0; j < vertex_count; ++j) {
        if (
            fabs(px - mask_x[coord_start + j]) <= eps &&
            fabs(py - mask_y[coord_start + j]) <= eps
        ) {
            vertex_index = j;
            has_vertex = 1;
            break;
        }
    }
    if (!has_vertex) return;

    const double vx = mask_x[coord_start + vertex_index];
    const double vy = mask_y[coord_start + vertex_index];
    const int on_left = fabs(px - xmin) <= eps;
    const int on_right = fabs(px - xmax) <= eps;
    const int on_bottom = fabs(py - ymin) <= eps;
    const int on_top = fabs(py - ymax) <= eps;
    const int side_count = on_left + on_right + on_bottom + on_top;
    const int side_interior = (
        ((on_left || on_right) && (py > ymin + eps) && (py < ymax - eps)) ||
        ((on_bottom || on_top) && (px > xmin + eps) && (px < xmax - eps))
    );
    const int vertex_inside = (
        (
            (on_right && (vx < xmax) && (vx >= xmin - eps)) ||
            (on_left && (vx > xmin) && (vx <= xmax + eps)) ||
            (on_top && (vy < ymax) && (vy >= ymin - eps)) ||
            (on_bottom && (vy > ymin) && (vy <= ymax + eps))
        ) &&
        (vx >= xmin - eps) && (vx <= xmax + eps) &&
        (vy >= ymin - eps) && (vy <= ymax + eps)
    );
    if (!(vertex_inside && side_count == 1 && side_interior)) return;

    const long long prev_vertex = (vertex_index + vertex_count - 1) % vertex_count;
    const long long next_vertex = (vertex_index + 1) % vertex_count;
    const double pvx = mask_x[coord_start + prev_vertex];
    const double pvy = mask_y[coord_start + prev_vertex];
    const double nvx = mask_x[coord_start + next_vertex];
    const double nvy = mask_y[coord_start + next_vertex];
    const int vertical = on_left || on_right;
    const double side_x = on_left ? xmin : xmax;
    const double side_y = on_bottom ? ymin : ymax;

    const double prev_dx = pvx - vx;
    const double next_dx = nvx - vx;
    const double prev_dy = pvy - vy;
    const double next_dy = nvy - vy;
    const double prev_t_x = (fabs(prev_dx) > eps) ? ((side_x - vx) / prev_dx) : 0.0;
    const double next_t_x = (fabs(next_dx) > eps) ? ((side_x - vx) / next_dx) : 0.0;
    const double prev_t_y = (fabs(prev_dy) > eps) ? ((side_y - vy) / prev_dy) : 0.0;
    const double next_t_y = (fabs(next_dy) > eps) ? ((side_y - vy) / next_dy) : 0.0;

    double pix = vertical ? side_x : (vx + prev_t_y * prev_dx);
    double piy = vertical ? (vy + prev_t_x * prev_dy) : side_y;
    double nix = vertical ? side_x : (vx + next_t_y * next_dx);
    double niy = vertical ? (vy + next_t_x * next_dy) : side_y;
    if (!isfinite(pix)) pix = px;
    if (!isfinite(piy)) piy = py;
    if (!isfinite(nix)) nix = px;
    if (!isfinite(niy)) niy = py;
    pix = fmin(fmax(pix, xmin), xmax);
    nix = fmin(fmax(nix, xmin), xmax);
    piy = fmin(fmax(piy, ymin), ymax);
    niy = fmin(fmax(niy, ymin), ymax);

    const long long base = i * 4LL;
    out_x[base + 0] = pix;
    out_y[base + 0] = piy;
    out_x[base + 1] = vx;
    out_y[base + 1] = vy;
    out_x[base + 2] = nix;
    out_y[base + 2] = niy;
    out_x[base + 3] = pix;
    out_y[base + 3] = piy;
    out_valid[i] = 1u;
}
"""

request_nvrtc_warmup(
    [
        ("clip-rowset", _CLIP_ROWSET_KERNEL_SOURCE, _CLIP_ROWSET_KERNEL_NAMES),
        (
            "clip-point-contact-sliver",
            _CLIP_POINT_CONTACT_SLIVER_KERNEL_SOURCE,
            _CLIP_POINT_CONTACT_SLIVER_KERNEL_NAMES,
        ),
    ]
)


def _clip_rowset_kernels():
    return compile_kernel_group(
        "clip-rowset",
        _CLIP_ROWSET_KERNEL_SOURCE,
        _CLIP_ROWSET_KERNEL_NAMES,
    )


def _clip_point_contact_sliver_kernels():
    return compile_kernel_group(
        "clip-point-contact-sliver",
        _CLIP_POINT_CONTACT_SLIVER_KERNEL_SOURCE,
        _CLIP_POINT_CONTACT_SLIVER_KERNEL_NAMES,
    )


@dataclass(frozen=True)
class _ClipFamilyMasks:
    point: np.ndarray
    line: np.ndarray
    multiline: np.ndarray
    simple_line: np.ndarray
    polygon: np.ndarray
    non_point: np.ndarray
    generic: np.ndarray
    all_point: bool
    all_polygon: bool
    all_polygonal: bool


@dataclass(frozen=True)
class _ClipCandidateRows:
    rows: np.ndarray
    device_rows: object | None = None
    spatially_ordered: bool = False


@dataclass(frozen=True)
class _ClipDeviceCandidateRows:
    device_rows: object
    spatially_ordered: bool = False


def _clip_spatially_order_device_rows(
    device_rows,
    device_bounds,
    *,
    active_mask=None,
):
    """Order candidate rows by exact SoA bounds without a stacked key matrix."""
    import cupy as cp

    from vibespatial.overlay.graph import (
        _fp64_radix_keys,
        _stable_radix_order_pass,
    )

    d_rows = cp.asarray(device_rows, dtype=cp.int64)
    if int(d_rows.size) <= 1:
        return d_rows
    candidate_bounds = cp.asarray(device_bounds, dtype=cp.float64).reshape(-1, 4)[d_rows]
    order = cp.arange(int(d_rows.size), dtype=cp.int32)
    order = _stable_radix_order_pass(order, d_rows)
    for coordinate_column in (3, 2, 1, 0):
        order = _stable_radix_order_pass(
            order,
            _fp64_radix_keys(candidate_bounds[:, coordinate_column]),
        )
    if active_mask is not None:
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != int(d_rows.size):
            raise ValueError("clip candidate active mask must match row capacity")
        order = _stable_radix_order_pass(
            order,
            (~d_active).astype(cp.int32, copy=False),
        )
    return d_rows[order]


@dataclass(frozen=True)
class _ClipPositiveRows:
    mask: np.ndarray
    rows: np.ndarray
    device_rows: object | None = None


def _clip_take_candidate_device_rows(candidate_device_rows, local_mask: np.ndarray):
    """Return device source rows for a candidate subpartition when available."""
    if candidate_device_rows is None or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    mask = np.asarray(local_mask, dtype=bool)
    if mask.size == 0:
        return cp.asarray([], dtype=cp.int64)
    d_rows = cp.asarray(candidate_device_rows, dtype=cp.int64)
    if bool(mask.all()):
        return d_rows
    local_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    if local_rows.size == 0:
        return cp.asarray([], dtype=cp.int64)
    return d_rows[cp.asarray(local_rows, dtype=cp.int64)]


def _spatial_geometry(spatial):
    return spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial


def _spatial_values(spatial):
    return _spatial_geometry(spatial).values


def _spatial_owned(spatial):
    values = _spatial_values(spatial)
    return getattr(values, "_owned", None) or getattr(values, "owned", None)


def _clip_device_to_host(device_array: object, *, reason: str) -> np.ndarray:
    return np.asarray(get_cuda_runtime().copy_device_to_host(device_array, reason=reason))


def _clip_bool_scalar(device_value: object, *, reason: str) -> bool:
    value = _clip_device_to_host(device_value, reason=reason)
    return bool(np.asarray(value, dtype=bool).reshape(-1)[0])


def _clip_device_mask_to_rows(d_mask):
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import compact_indices

    d_mask = cp.asarray(d_mask, dtype=cp.bool_)
    n = int(d_mask.size)
    if n == 0:
        return cp.empty(0, dtype=cp.int64)

    compacted = compact_indices(d_mask.astype(cp.uint8, copy=False))
    return cp.asarray(compacted.values, dtype=cp.int64)


def _clip_bbox_candidate_rows_device(
    d_bounds,
    *,
    d_query_bounds=None,
    query_bounds=None,
):
    import cupy as cp

    d_bounds = cp.asarray(d_bounds, dtype=cp.float64).reshape(-1, 4)
    n = int(d_bounds.shape[0])
    if n == 0:
        return cp.empty(0, dtype=cp.int64)

    if d_query_bounds is not None:
        d_query = cp.asarray(d_query_bounds, dtype=cp.float64).reshape(-1)
        if int(d_query.size) < 4:
            return None
        xmin = ymin = xmax = ymax = 0.0
        use_query_bounds = 1
    else:
        if query_bounds is None:
            return None
        xmin, ymin, xmax, ymax = (float(value) for value in query_bounds)
        d_query = None
        use_query_bounds = 0

    runtime = get_cuda_runtime()
    from vibespatial.cuda.cccl_primitives import compact_indices

    d_mask = cp.empty(n, dtype=cp.uint8)
    kernel = _clip_rowset_kernels()["bbox_candidate_mask_u8_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_bounds),
                int(n),
                ptr(d_query),
                xmin,
                ymin,
                xmax,
                ymax,
                use_query_bounds,
                ptr(d_mask),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    compacted = compact_indices(d_mask)
    return cp.asarray(compacted.values, dtype=cp.int64)


def _clip_source_rows_identity_hint(source_rows: object, source_row_count: int) -> bool:
    """Return whether source-row positions are proven full identity order."""
    row_count = int(source_row_count)
    size = int(getattr(source_rows, "size", len(source_rows)))
    if size != row_count:
        return False
    if row_count <= 1:
        return True
    if hasattr(source_rows, "__cuda_array_interface__"):
        return False
    rows = np.asarray(source_rows, dtype=np.int64)
    return np.array_equal(rows, np.arange(row_count, dtype=rows.dtype))


def _clip_bounds_filter_work_estimate(row_count: int, owned=None) -> PhysicalWorkEstimate:
    """Estimate scalar-mask candidate filter work without geometry materialization."""
    if owned is None:
        return PhysicalWorkEstimate.from_rows(row_count)
    estimate = estimate_physical_work_from_owned(owned)
    dispatch_units = max(
        int(row_count),
        int(estimate.coordinate_count),
        int(estimate.segment_count),
    )
    return PhysicalWorkEstimate(
        row_count=int(row_count),
        coordinate_count=int(estimate.coordinate_count),
        segment_count=int(estimate.segment_count),
        ring_count=int(estimate.ring_count),
        primary_unit_count=dispatch_units,
        primary_unit_name="bbox-coordinate",
    )


def _clip_collective_grouped_mask_work_estimates(
    source_owned,
    mask_source_owned,
    *,
    relation_pair_count: int,
    collective_source_count: int,
    positive_area_pair_count: int | None = None,
    mask_rectangle_strip_admissible: bool = False,
) -> tuple[PhysicalWorkEstimate, PhysicalWorkEstimate]:
    """Estimate relation-first and union-first collective clip shapes.

    The relation plan repeats only mask members that actually intersect each
    source row, then reduces polygon fragments by source. The union plan pays
    once for the complete grouped mask and intersects every collective source
    row with that output. Both plans stay device-native; this estimate chooses
    the physical shape without inspecting geometry objects or relation rows on
    the host.
    """
    pair_count = int(relation_pair_count)
    source_count = int(collective_source_count)
    source_shape = estimate_physical_work_from_owned(source_owned)
    mask_shape = estimate_physical_work_from_owned(mask_source_owned)

    source_rows = max(int(source_shape.row_count), 1)
    mask_rows = max(int(mask_shape.row_count), 1)
    source_units = max(
        int(source_shape.coordinate_count),
        int(source_shape.segment_count),
        source_rows,
    )
    mask_units = max(
        int(mask_shape.coordinate_count),
        int(mask_shape.segment_count),
        mask_rows,
    )
    source_units_per_row = (source_units + source_rows - 1) // source_rows
    mask_units_per_row = (mask_units + mask_rows - 1) // mask_rows

    source_pair_units = source_units_per_row
    reduced_mask_units_per_source = mask_units
    mask_is_axis_rectangles = False
    if source_owned.residency is Residency.DEVICE and source_owned.device_state is not None:
        from vibespatial.geometry.buffers import GeometryFamily

        source_state = source_owned._ensure_device_state(preserve_indexed_view=True)
        polygon_buffer = source_state.families.get(GeometryFamily.POLYGON)
        if (
            polygon_buffer is not None
            and len(source_state.families) == 1
            and polygon_buffer.axis_aligned_rectangles is True
        ):
            source_pair_units = 0
    if (
        mask_source_owned.residency is Residency.DEVICE
        and mask_source_owned.device_state is not None
    ):
        from vibespatial.geometry.buffers import GeometryFamily

        mask_state = mask_source_owned._ensure_device_state(preserve_indexed_view=True)
        mask_polygon_buffer = mask_state.families.get(GeometryFamily.POLYGON)
        if (
            mask_polygon_buffer is not None
            and len(mask_state.families) == 1
            and mask_polygon_buffer.axis_aligned_rectangles is True
        ):
            mask_is_axis_rectangles = True
            if mask_rectangle_strip_admissible:
                # A connected same-span strip reduces to one boundary-shaped
                # carrier for each subsequent source intersection.
                reduced_mask_units_per_source = mask_units_per_row

    positive_pairs = (
        pair_count
        if positive_area_pair_count is None
        else min(max(int(positive_area_pair_count), 0), pair_count)
    )
    boundary_pairs = pair_count - positive_pairs
    boundary_pair_units = 1 if mask_is_axis_rectangles else mask_units_per_row
    pair_constructive_units = positive_pairs * (
        source_pair_units + mask_units_per_row
    ) + boundary_pairs * (source_pair_units + boundary_pair_units)
    # Relation reduction must consume every emitted fragment. Use the mask
    # member shape as a conservative lower bound for each fragment's topology.
    relation_reduce_units = (
        positive_pairs * mask_units_per_row + boundary_pairs * boundary_pair_units
    )
    relation_units = pair_constructive_units + relation_reduce_units
    relation_output_bytes = (
        pair_count
        * max(
            source_units_per_row,
            mask_units_per_row,
            1,
        )
        * 16
    )
    relation_temporary_bytes = pair_count * 32 + relation_output_bytes * 2 + source_count * 16
    relation_estimate = PhysicalWorkEstimate(
        row_count=source_count,
        coordinate_count=pair_constructive_units,
        segment_count=pair_constructive_units,
        relation_pair_count=pair_count,
        group_count=source_count,
        output_row_count=source_count,
        output_byte_count=relation_output_bytes,
        temporary_byte_count=relation_temporary_bytes,
        primary_unit_count=max(
            relation_units,
            relation_output_bytes // 64,
            relation_temporary_bytes // 128,
        ),
        primary_unit_name="collective-relation-unit",
    )

    # Exact grouped union is at least a topology-discovery pass followed by
    # constructive assembly. Intersections with the reduced carrier are then
    # priced separately by its physical boundary shape.
    union_reduction_passes = 1 if mask_rectangle_strip_admissible else 2
    union_constructive_units = union_reduction_passes * mask_units + source_count * (
        source_pair_units + reduced_mask_units_per_source
    )
    union_output_bytes = (
        mask_units
        + source_count * max(source_units_per_row, reduced_mask_units_per_source)
    ) * 16
    union_temporary_bytes = mask_units * 32 + source_count * 48 + union_output_bytes
    union_estimate = PhysicalWorkEstimate(
        row_count=source_count,
        coordinate_count=union_constructive_units,
        segment_count=union_constructive_units,
        relation_pair_count=pair_count,
        group_count=1,
        output_row_count=source_count,
        output_byte_count=union_output_bytes,
        temporary_byte_count=union_temporary_bytes,
        primary_unit_count=max(
            union_constructive_units,
            union_output_bytes // 64,
            union_temporary_bytes // 128,
        ),
        primary_unit_name="collective-union-unit",
    )
    return relation_estimate, union_estimate


def _clip_collective_grouped_mask_prefers_relation(
    source_owned,
    mask_source_owned,
    *,
    relation_pair_count: int,
    collective_source_count: int,
    positive_area_pair_count: int | None = None,
    mask_rectangle_strip_admissible: bool = False,
    available_device_bytes: int | None = None,
) -> tuple[bool, PhysicalWorkEstimate, PhysicalWorkEstimate]:
    relation_estimate, union_estimate = _clip_collective_grouped_mask_work_estimates(
        source_owned,
        mask_source_owned,
        relation_pair_count=relation_pair_count,
        collective_source_count=collective_source_count,
        positive_area_pair_count=positive_area_pair_count,
        mask_rectangle_strip_admissible=mask_rectangle_strip_admissible,
    )
    return (
        relation_estimate.dispatch_unit_count() <= union_estimate.dispatch_unit_count(),
        relation_estimate,
        union_estimate,
    )


def _clip_available_device_bytes() -> int | None:
    """Return bytes available to the active device allocator without a sync."""
    if not has_gpu_runtime():
        return None
    import cupy as cp

    runtime = get_cuda_runtime()
    cuda_free_bytes = int(cp.cuda.Device().mem_info[0])
    pool_free_bytes = int(runtime.memory_pool_stats().get("free_bytes", 0))
    return cuda_free_bytes + pool_free_bytes


def _clip_bounds_filter_selects_device(row_count: int, owned=None) -> bool:
    """Return True when scalar-mask candidate filtering should stay device-shaped."""
    if not has_gpu_runtime():
        return False
    if strict_native_mode_enabled():
        return True
    plan = plan_dispatch_selection(
        kernel_name="clip_scalar_mask_bounds_filter",
        kernel_class=KernelClass.COARSE,
        row_count=int(row_count),
        requested_mode=ExecutionMode.AUTO,
        gpu_available=True,
        current_residency=(
            owned.residency if owned is not None and hasattr(owned, "residency") else Residency.HOST
        ),
        work_estimate=_clip_bounds_filter_work_estimate(row_count, owned),
    )
    return plan.selected is ExecutionMode.GPU


def _clip_source_nonmissing_rowset(
    source_values,
    *,
    source_token: str,
    prefer_device: bool,
):
    """Return valid, non-empty source rows without crossing the device boundary."""
    from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet

    owned = getattr(source_values, "_owned", None) or getattr(source_values, "owned", None)
    row_count = int(getattr(owned, "row_count", len(source_values)))
    if (
        prefer_device
        and owned is not None
        and owned.residency is Residency.DEVICE
        and owned.device_state is not None
        and has_gpu_runtime()
    ):
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.api._native_rowset import NativeDeviceSelection
            from vibespatial.geometry.owned import FAMILY_TAGS

            device_state = owned.device_state
            if (
                device_state.trusted_all_valid is True
                and device_state.trusted_all_non_empty is True
            ):
                return NativeRowSet.from_positions(
                    cp.arange(row_count, dtype=cp.int64),
                    source_token=source_token,
                    source_row_count=row_count,
                    ordered=True,
                    unique=True,
                    identity=True,
                    trusted_all_valid_rows=True,
                )

            d_valid = cp.asarray(device_state.validity).astype(cp.bool_, copy=False)
            d_nonmissing = d_valid.copy()
            d_tags = cp.asarray(device_state.tags)
            d_family_row_offsets = cp.asarray(device_state.family_row_offsets)
            for family, device_buffer in device_state.families.items():
                d_family_mask = d_valid & (d_tags == cp.int8(FAMILY_TAGS[family]))
                d_empty = cp.asarray(device_buffer.empty_mask, dtype=cp.bool_)
                if int(d_empty.size) == 0:
                    d_nonmissing &= ~d_family_mask
                    continue
                d_local_rows = d_family_row_offsets.astype(cp.int64, copy=False)
                d_local_valid = (
                    d_family_mask & (d_local_rows >= 0) & (d_local_rows < int(d_empty.size))
                )
                d_safe_local = cp.where(d_local_valid, d_local_rows, 0)
                d_family_nonempty = d_local_valid & ~d_empty[d_safe_local]
                d_nonmissing &= ~d_family_mask | d_family_nonempty

            return NativeDeviceSelection.from_mask(
                d_nonmissing,
                source_token=source_token,
                source_row_count=row_count,
            )

    missing = np.asarray(source_values.isna() | source_values.is_empty, dtype=bool)
    positions = np.flatnonzero(~missing).astype(np.int64, copy=False)
    return NativeRowSet.from_positions(
        positions,
        source_token=source_token,
        source_row_count=row_count,
        ordered=True,
        unique=True,
        identity=int(positions.size) == row_count,
        trusted_all_valid_rows=True,
    )


def _clip_source_nonmissing_rows_for_compatibility(
    source_rowset,
    *,
    surface: str,
) -> np.ndarray:
    """Export sparse source positions only at an explicit compatibility tail."""
    return np.asarray(
        source_rowset.to_host_positions(
            surface=surface,
            strict_disallowed=False,
        ),
        dtype=np.intp,
    )


def _spatial_total_bounds_private(spatial) -> np.ndarray:
    """Return total bounds without recording a public bounds export."""
    values = _spatial_values(spatial)
    if (
        isinstance(values, GeometryArray | DeviceGeometryArray)
        or getattr(values, "_owned", None) is not None
    ):
        return np.asarray(values.total_bounds, dtype=np.float64)
    return np.asarray(_spatial_geometry(spatial).total_bounds, dtype=np.float64)


def _spatial_prefers_device_bounds_private(spatial) -> bool:
    owned = _spatial_owned(spatial)
    return bool(
        owned is not None
        and (
            owned.residency is Residency.DEVICE or getattr(owned, "device_state", None) is not None
        )
    )


def _spatial_device_row_bounds_private(spatial):
    """Return per-row bounds on device when geometry already has device state."""
    if not has_gpu_runtime():
        return None
    owned = _spatial_owned(spatial)
    if owned is None:
        return None
    if owned.residency is not Residency.DEVICE and getattr(owned, "device_state", None) is None:
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )

    return cp.asarray(
        compute_geometry_bounds_device(owned, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(len(spatial), 4)


def _single_row_polygon_mask_owned_private(spatial):
    """Return a resident one-row polygon mask carrier without public export."""
    if not isinstance(spatial, GeoDataFrame | GeoSeries) or len(spatial) != 1:
        return None
    if not has_gpu_runtime():
        return None
    owned = _spatial_owned(spatial)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.geometry.buffers import GeometryFamily

    if not _owned_active_family_subset(
        owned,
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON},
    ):
        return None
    if int(owned.row_count) == 1:
        return owned
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None
    return owned.device_take(cp.asarray([0], dtype=cp.int64))


def _lazy_grouped_union_mask_owned_private(spatial):
    """Return a lazy one-row grouped-union mask without materializing it."""
    if not isinstance(spatial, GeoDataFrame | GeoSeries) or len(spatial) != 1:
        return None
    if not has_gpu_runtime():
        return None
    owned = _spatial_owned(spatial)
    if (
        owned is None
        or not getattr(owned, "_is_lazy_grouped_union_owned", False)
        or owned.residency is not Residency.DEVICE
        or int(getattr(owned, "row_count", -1)) != 1
    ):
        return None
    source_owned = getattr(owned, "_source_owned", None)
    if source_owned is None or source_owned.residency is not Residency.DEVICE:
        return None
    return owned


def _owned_supported_clip_families(owned) -> bool:
    from vibespatial.geometry.buffers import GeometryFamily

    supported = {
        GeometryFamily.POINT,
        GeometryFamily.MULTIPOINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
    return set(owned.families).issubset(supported)


def _owned_active_family_subset(owned, allowed_families) -> bool:
    """Return whether active owned rows belong only to ``allowed_families``.

    Indexed and gathered device carriers may retain inactive family buffers
    from an earlier mixed output. Their producer must carry a trusted family
    domain; admission never synchronizes device tags to rediscover that proof.
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    allowed_family_set = set(allowed_families)
    device_state = getattr(owned, "device_state", None)
    if device_state is not None:
        trusted_family = getattr(device_state, "trusted_homogeneous_family", None)
        if trusted_family is not None:
            return trusted_family in allowed_family_set
        if getattr(device_state, "trusted_polygonal_only", None) is True:
            polygonal_families = {
                GeometryFamily.POLYGON,
                GeometryFamily.MULTIPOLYGON,
            }
            if polygonal_families.issubset(allowed_family_set):
                return True
            if allowed_family_set.isdisjoint(polygonal_families):
                return False

    allowed_tags = {np.int8(FAMILY_TAGS[family]) for family in allowed_family_set}
    carrier_families = set(owned.families)
    if carrier_families.issubset(allowed_family_set):
        return True
    if carrier_families.isdisjoint(allowed_family_set):
        return False
    if int(getattr(owned, "row_count", 0)) == 0:
        return True

    host_tags = getattr(owned, "_tags", None)
    if host_tags is not None:
        tags = np.asarray(host_tags, dtype=np.int8)
        return bool(np.isin(tags, list(allowed_tags)).all())

    if owned.residency is Residency.DEVICE and owned.device_state is not None:
        return False

    tags = np.asarray(owned.tags, dtype=np.int8)
    return bool(np.isin(tags, list(allowed_tags)).all())


def _clip_family_masks(spatial) -> _ClipFamilyMasks:
    """Return clip routing masks from private owned tags when available."""
    owned = _spatial_owned(spatial)
    if owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        tags = np.asarray(owned.tags)
        row_count = int(tags.size)
        point = tags == FAMILY_TAGS[GeometryFamily.POINT]
        line = (tags == FAMILY_TAGS[GeometryFamily.LINESTRING]) | (
            tags == FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
        )
        multiline = tags == FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
        polygon = (tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
            tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
        )
        simple_line = line & ~multiline
        non_point = ~point
        generic = non_point & ~(simple_line | multiline | polygon)
        return _ClipFamilyMasks(
            point=point,
            line=line,
            multiline=multiline,
            simple_line=simple_line,
            polygon=polygon,
            non_point=non_point,
            generic=generic,
            all_point=bool(row_count == 0 or np.all(point)),
            all_polygon=bool(row_count > 0 and np.all(tags == FAMILY_TAGS[GeometryFamily.POLYGON])),
            all_polygonal=bool(row_count == 0 or np.all(polygon)),
        )

    geom_types = _spatial_geometry(spatial).geom_type
    point = np.asarray(geom_types == "Point", dtype=bool)
    line = np.asarray(geom_types.isin(LINE_GEOM_TYPES), dtype=bool)
    multiline = np.asarray(geom_types == "MultiLineString", dtype=bool)
    polygon = np.asarray(geom_types.isin(POLYGON_GEOM_TYPES), dtype=bool)
    simple_line = line & ~multiline
    non_point = ~point
    generic = non_point & ~(simple_line | multiline | polygon)
    return _ClipFamilyMasks(
        point=point,
        line=line,
        multiline=multiline,
        simple_line=simple_line,
        polygon=polygon,
        non_point=non_point,
        generic=generic,
        all_point=bool(point.size == 0 or np.all(point)),
        all_polygon=bool(
            point.size > 0 and np.all(np.asarray(geom_types == "Polygon", dtype=bool))
        ),
        all_polygonal=bool(polygon.size == 0 or np.all(polygon)),
    )


def _spatial_all_polygonal_private(spatial) -> bool:
    owned = _spatial_owned(spatial)
    if owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily

        return _owned_active_family_subset(
            owned,
            {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON},
        )
    return _clip_family_masks(spatial).all_polygonal


def _maybe_seed_polygon_validity_cache(spatial) -> None:
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None:
        return

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    state = getattr(owned, "device_state", None)
    if state is not None:
        trusted_family = getattr(state, "trusted_homogeneous_family", None)
        if trusted_family is not None and trusted_family not in polygonal_families:
            return
        if (
            trusted_family in polygonal_families
            or getattr(
                state,
                "trusted_polygonal_only",
                None,
            )
            is True
        ):
            pass
        else:
            host_tags = getattr(owned, "_tags", None)
            if host_tags is None:
                if not set(owned.families).issubset(polygonal_families):
                    return
            else:
                polygonal_tags = {
                    np.int8(FAMILY_TAGS[GeometryFamily.POLYGON]),
                    np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]),
                }
                if not bool(np.isin(host_tags, list(polygonal_tags)).all()):
                    return
    elif not _owned_active_family_subset(owned, polygonal_families):
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _mask_is_list_like_rectangle(mask):
    """
    Check if the input mask is list-like and not an instance of
    specific geometric types.

    Parameters
    ----------
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.

    Returns
    -------
    bool
        True if `mask` is list-like and not an instance of `GeoDataFrame`,
        `GeoSeries`, `Polygon`, or `MultiPolygon`, otherwise False.
    """
    return pandas.api.types.is_list_like(mask) and not isinstance(
        mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon
    )


def _rectangle_bounds_from_mask(mask):
    """Return rectangle bounds for axis-aligned rectangle masks, else None."""
    if _mask_is_list_like_rectangle(mask):
        return tuple(float(v) for v in mask)
    if isinstance(mask, MultiPolygon):
        return None
    if not isinstance(mask, Polygon) or mask.is_empty or len(mask.interiors) != 0:
        return None
    coords = np.asarray(mask.exterior.coords)
    if len(coords) < 5:
        return None
    body = coords[:-1]
    xmin = float(np.min(body[:, 0]))
    xmax = float(np.max(body[:, 0]))
    ymin = float(np.min(body[:, 1]))
    ymax = float(np.max(body[:, 1]))
    if not (xmin < xmax and ymin < ymax):
        return None
    eps = max(float(SPATIAL_EPSILON), max(abs(xmax - xmin), abs(ymax - ymin)) * 1e-12)
    x = body[:, 0]
    y = body[:, 1]
    on_left = np.abs(x - xmin) <= eps
    on_right = np.abs(x - xmax) <= eps
    on_bottom = np.abs(y - ymin) <= eps
    on_top = np.abs(y - ymax) <= eps
    if not np.all(on_left | on_right | on_bottom | on_top):
        return None
    if not (np.any(on_left) and np.any(on_right) and np.any(on_bottom) and np.any(on_top)):
        return None
    x_closed = coords[:, 0]
    y_closed = coords[:, 1]
    area2 = abs(float(np.dot(x_closed[:-1], y_closed[1:]) - np.dot(y_closed[:-1], x_closed[1:])))
    box_area2 = 2.0 * (xmax - xmin) * (ymax - ymin)
    if abs(area2 - box_area2) > max(eps, box_area2 * 1e-12):
        return None
    return (xmin, ymin, xmax, ymax)


def _device_rectangle_owned_from_bounds(
    rectangle_bounds: tuple[float, float, float, float],
    *,
    residency: Residency,
):
    """Build an owned rectangle mask from scalar bounds without Shapely rewrap."""
    if residency is not Residency.DEVICE or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds

    d_bounds = cp.asarray([rectangle_bounds], dtype=cp.float64)
    return _build_device_boxes_from_bounds(d_bounds, row_count=1)


def _bbox_candidate_rows_for_scalar_clip_mask(
    gdf,
    query_input,
    *,
    sort: bool = False,
) -> np.ndarray | None:
    result = _bbox_candidate_rows_for_scalar_clip_mask_result(
        gdf,
        query_input,
        sort=sort,
    )
    return None if result is None else result.rows


def _bbox_device_candidate_rows_for_scalar_clip_mask_result(
    gdf,
    query_input,
    *,
    sort: bool = False,
) -> _ClipDeviceCandidateRows | None:
    """Return device bbox candidate rows for native scalar-mask clip paths."""
    if len(gdf) == 0 or not has_gpu_runtime():
        return None

    if isinstance(query_input, GeoDataFrame | GeoSeries):
        if len(query_input) != 1:
            return None
        d_query_bounds = _spatial_device_row_bounds_private(query_input)
        query_bounds = (
            None if d_query_bounds is not None else _spatial_total_bounds_private(query_input)
        )
    elif isinstance(query_input, Polygon | MultiPolygon):
        d_query_bounds = None
        query_bounds = np.asarray(query_input.bounds, dtype=np.float64)
    else:
        return None

    if query_bounds is not None and query_bounds.shape != (4,):
        return None

    geometry = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if not _clip_partition_supports_device_promotion(gdf):
        return None
    if owned is None and hasattr(values, "to_owned"):
        owned = values.to_owned()
    if owned is None:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )
    from vibespatial.runtime.residency import TransferTrigger

    if owned.residency is not Residency.DEVICE:
        owned = owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=(
                "clip scalar-mask bbox candidate query promoted supported source geometry to device"
            ),
        )

    d_bounds = cp.asarray(
        compute_geometry_bounds_device(
            owned,
            preserve_indexed_view=True,
        ),
        dtype=cp.float64,
    ).reshape(len(gdf), 4)
    if d_query_bounds is not None:
        d_rows = _clip_bbox_candidate_rows_device(
            d_bounds,
            d_query_bounds=d_query_bounds,
        )
    else:
        assert query_bounds is not None
        d_rows = _clip_bbox_candidate_rows_device(
            d_bounds,
            query_bounds=query_bounds,
        )
    if d_rows is None:
        return None
    if not sort and int(d_rows.size) > 1:
        d_rows = _clip_spatially_order_device_rows(d_rows, d_bounds)
    return _ClipDeviceCandidateRows(
        d_rows,
        spatially_ordered=not sort,
    )


def _bbox_candidate_rows_for_scalar_clip_mask_result(
    gdf,
    query_input,
    *,
    sort: bool = False,
) -> _ClipCandidateRows | None:
    """Return bbox candidate rows for scalar clip masks without building an sindex.

    The exact clip stage still performs the real geometric intersection. This
    helper only replaces the candidate query for the common scalar-mask cases
    where building/querying an index is more expensive than one vectorized
    bounds overlap pass. When the source geometry is already device-backed, the
    bounds overlap pass stays on the device instead of round-tripping through
    the generic candidate-query path.
    """
    if len(gdf) == 0:
        return _ClipCandidateRows(np.empty(0, dtype=np.int32), spatially_ordered=True)

    if isinstance(query_input, GeoDataFrame | GeoSeries):
        if len(query_input) != 1:
            return None
        d_query_bounds = _spatial_device_row_bounds_private(query_input)
        query_bounds = (
            None if d_query_bounds is not None else _spatial_total_bounds_private(query_input)
        )
    elif isinstance(query_input, Polygon | MultiPolygon):
        query_bounds = np.asarray(query_input.bounds, dtype=np.float64)
    else:
        return None

    geometry = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
    values = geometry.values
    owned = getattr(values, "_owned", None)
    device_promotion_supported = has_gpu_runtime() and _clip_partition_supports_device_promotion(
        gdf
    )
    if (
        owned is None
        and hasattr(values, "to_owned")
        and (
            strict_native_mode_enabled()
            or (device_promotion_supported and _clip_bounds_filter_selects_device(len(gdf), None))
        )
    ):
        owned = values.to_owned()

    device_bounds_selected = device_promotion_supported and owned is not None
    use_device_bounds = (
        device_promotion_supported
        and owned is not None
        and (strict_native_mode_enabled() or device_bounds_selected)
    )
    if use_device_bounds:
        return None

    if query_bounds is None or query_bounds.shape != (4,):
        return None

    # Avoid O(n) bounds filtering for genuinely large clip workloads where the
    # flat spatial index amortizes its build/query cost better.
    if _clip_bounds_filter_selects_device(len(gdf), owned):
        return None

    values = geometry.values
    if (
        isinstance(values, GeometryArray | DeviceGeometryArray)
        or getattr(values, "_owned", None) is not None
    ):
        source_bounds = np.asarray(values.bounds, dtype=np.float64)
    else:
        source_bounds = np.asarray(geometry.bounds, dtype=np.float64)
    if source_bounds.ndim != 2 or source_bounds.shape[1] != 4:
        return None

    xmin, ymin, xmax, ymax = query_bounds
    overlap_mask = (
        (source_bounds[:, 0] <= xmax)
        & (source_bounds[:, 2] >= xmin)
        & (source_bounds[:, 1] <= ymax)
        & (source_bounds[:, 3] >= ymin)
    )
    rows = np.flatnonzero(overlap_mask).astype(np.int32, copy=False)
    if rows.size <= 1:
        return _ClipCandidateRows(rows, spatially_ordered=not sort)

    if sort:
        order = np.argsort(np.asarray(gdf.index.take(rows)), kind="stable")
        return _ClipCandidateRows(rows[order].astype(np.int32, copy=False))

    # Match the public "unsorted" contract by returning a deterministic spatial
    # encounter order rather than monotonic source-index order.
    candidate_bounds = source_bounds[rows]
    order = np.lexsort(
        (
            rows,
            candidate_bounds[:, 3],
            candidate_bounds[:, 2],
            candidate_bounds[:, 1],
            candidate_bounds[:, 0],
        )
    )
    return _ClipCandidateRows(
        rows[order].astype(np.int32, copy=False),
        spatially_ordered=True,
    )


def _clip_device_candidate_rows_from_native_relation_result(
    gdf,
    query_input,
    *,
    sort: bool,
) -> _ClipDeviceCandidateRows | None:
    """Return device-backed clip candidates from a native spatial relation."""
    if sort or query_input is None or not has_gpu_runtime():
        return None

    geometry = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    query_geometry = query_input.geometry if isinstance(query_input, GeoDataFrame) else query_input
    query_values = (
        query_geometry.values if isinstance(query_geometry, GeoSeries) else query_geometry
    )
    query_owned = getattr(query_values, "_owned", None)
    query_relation_input = query_owned if query_owned is not None else query_geometry
    query_row_count = int(query_owned.row_count) if query_owned is not None else None
    geometry_name = (
        gdf._geometry_column_name
        if isinstance(gdf, GeoDataFrame)
        else getattr(gdf, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(gdf, geometry_name)
    source_token = None if source_state is None else source_state.lineage_token
    rowset, _execution = gdf.sindex.query_right_semijoin(
        query_relation_input,
        predicate="intersects",
        source_token=source_token,
        query_row_count=query_row_count,
    )
    if not rowset.is_device:
        return None

    if isinstance(rowset, NativeDeviceSelection):
        d_rows = cp.asarray(
            rowset.partition_capacity_positions(),
            dtype=cp.int64,
        )
        d_active = rowset.active_capacity_mask()
    else:
        d_rows = cp.asarray(rowset.positions, dtype=cp.int64)
        d_active = None
    if int(d_rows.size) > 1:
        source_bounds = cp.asarray(
            compute_geometry_bounds_device(
                owned,
                preserve_indexed_view=True,
            ),
            dtype=cp.float64,
        ).reshape(owned.row_count, 4)
        d_rows = _clip_spatially_order_device_rows(
            d_rows,
            source_bounds,
            active_mask=d_active,
        )

    if isinstance(rowset, NativeDeviceSelection):
        rowset = replace(
            rowset,
            positions=d_rows,
            ordered=True,
            full_selection_implies_identity=False,
        )
        return _ClipDeviceCandidateRows(rowset, spatially_ordered=True)

    return _ClipDeviceCandidateRows(d_rows, spatially_ordered=True)


def _clip_polygon_single_mask_candidate_predicates_device(
    source_owned,
    mask_owned,
    d_candidate_rows,
    *,
    d_candidate_active=None,
    source_token: str,
    operation_prefix: str,
    mask_bounds_are_exact: bool = False,
):
    """Evaluate polygon-mask clip predicates on a device source-rowset.

    The device-candidate clip path already has source row ids from the spatial
    index.  Keep that physical shape for predicate refinement instead of first
    materializing a candidate ``OwnedGeometryArray`` solely to run
    ``intersects`` and ``covered_by`` against a single mask row.
    """
    if not has_gpu_runtime() or mask_owned is None:
        return None
    if (
        source_owned.residency is not Residency.DEVICE
        or mask_owned.residency is not Residency.DEVICE
    ):
        return None
    if int(mask_owned.row_count) != 1:
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    from vibespatial.api._native_expression import NativeExpression
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_rectangle_polygon_mask_and_bounds,
        device_trusted_rectangle_bounds_matrix,
    )
    from vibespatial.predicates.binary import (
        _polygonal_single_right_candidate_predicates_device,
    )
    from vibespatial.predicates.polygon import (
        compute_rect_bounds_polygon_mask_predicates_gpu,
    )

    d_candidate_rows = cp.asarray(d_candidate_rows, dtype=cp.int64)
    candidate_count = int(d_candidate_rows.size)
    if candidate_count == 0:
        empty = cp.empty(0, dtype=cp.bool_)
        return {
            name: NativeExpression(
                operation=f"{operation_prefix}.{name}",
                values=empty.copy(),
                source_token=source_token,
                source_row_count=0,
                dtype="bool",
                precision="predicate",
            )
            for name in ("intersects", "covered_by")
        }

    if d_candidate_active is None:
        d_active = cp.ones(candidate_count, dtype=cp.bool_)
    else:
        d_active = cp.asarray(d_candidate_active, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != candidate_count:
            raise ValueError("clip candidate activity must match candidate capacity")

    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    mask_families = [family for family in polygonal_families if family in mask_state.families]
    if len(mask_families) != 1:
        return None
    mask_family = mask_families[0]

    d_rectangle_active = cp.zeros(candidate_count, dtype=cp.bool_)
    candidate_rectangle_bounds = None
    rectangle_values = None
    if mask_family is GeometryFamily.POLYGON and not mask_bounds_are_exact:
        source_state = source_owned._ensure_device_state(preserve_indexed_view=True)
        d_safe_rows = cp.where(d_active, d_candidate_rows, cp.int64(0))
        rect_bounds_matrix = device_trusted_rectangle_bounds_matrix(source_owned)
        if rect_bounds_matrix is not None:
            candidate_rectangle_bounds = cp.asarray(
                rect_bounds_matrix,
                dtype=cp.float64,
            ).reshape(source_owned.row_count, 4)[d_safe_rows]
            d_rectangle_active = (
                d_active
                & cp.asarray(source_state.validity, dtype=cp.bool_)[d_safe_rows]
                & (
                    cp.asarray(source_state.tags, dtype=cp.int8)[d_safe_rows]
                    == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
                )
                & cp.all(cp.isfinite(candidate_rectangle_bounds), axis=1)
            )
        else:
            rect_info = device_rectangle_polygon_mask_and_bounds(source_owned)
            if rect_info is not None:
                d_rect_rows, rect_bounds_matrix = rect_info
                candidate_rectangle_bounds = cp.asarray(
                    rect_bounds_matrix,
                    dtype=cp.float64,
                ).reshape(source_owned.row_count, 4)[d_safe_rows]
                d_rectangle_active = (
                    d_active
                    & cp.asarray(
                        d_rect_rows,
                        dtype=cp.bool_,
                    )[d_safe_rows]
                )

        if candidate_rectangle_bounds is not None:
            rectangle_values = compute_rect_bounds_polygon_mask_predicates_gpu(
                mask_owned,
                candidate_rectangle_bounds,
                mask_family=mask_family,
                return_device=True,
            )
            if rectangle_values is None:
                d_rectangle_active = cp.zeros(candidate_count, dtype=cp.bool_)

    device_values = _polygonal_single_right_candidate_predicates_device(
        ("intersects", "covered_by"),
        source_owned,
        mask_owned,
        d_candidate_rows,
        d_candidate_active=d_active & ~d_rectangle_active,
        right_bounds_are_exact=mask_bounds_are_exact,
    )
    if device_values is None:
        return None

    if rectangle_values is not None:
        d_rect_intersects, d_rect_covered_by = rectangle_values
        device_values["intersects"] = cp.where(
            d_rectangle_active,
            cp.asarray(d_rect_intersects, dtype=cp.bool_),
            cp.asarray(device_values["intersects"], dtype=cp.bool_),
        )
        device_values["covered_by"] = cp.where(
            d_rectangle_active,
            cp.asarray(d_rect_covered_by, dtype=cp.bool_),
            cp.asarray(device_values["covered_by"], dtype=cp.bool_),
        )

    expressions = {}
    for name, values in device_values.items():
        expressions[name] = NativeExpression(
            operation=f"{operation_prefix}.{name}",
            values=values,
            source_token=source_token,
            source_row_count=candidate_count,
            dtype="bool",
            precision="predicate",
        )
    return expressions


def _clip_candidate_rows_from_native_relation(
    gdf,
    query_input,
    *,
    sort: bool,
) -> _ClipCandidateRows | None:
    """Decline host-visible export of native relation clip candidates."""
    device_result = _clip_device_candidate_rows_from_native_relation_result(
        gdf,
        query_input,
        sort=sort,
    )
    if device_result is None:
        return None
    return None


def _geometry_series_from_values(values, *, index, crs, name=None):
    """Build a GeoSeries-like object without demoting extension-backed geometry."""
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        result = pd.Series(values, index=index, copy=False, name=name)
        result.__class__ = GeoSeries
        result._crs = crs
        return result
    return GeoSeries(values, index=index, crs=crs, name=name)


def _geometry_column_series(values, *, index, crs, name):
    """Build a DataFrame geometry column while preserving extension backing."""
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        return pd.Series(values, index=index, copy=False, name=name)
    return GeoSeries(values, index=index, crs=crs, name=name)


def _clip_partition_supports_device_promotion(partition) -> bool:
    if not isinstance(partition, GeoDataFrame | GeoSeries):
        return False
    owned = _spatial_owned(partition)
    if owned is not None:
        return _owned_supported_clip_families(owned)
    geom_types = partition.geom_type
    supported_mask = geom_types.isna() | geom_types.isin(_DEVICE_CLIP_GEOM_TYPES)
    return bool(np.asarray(supported_mask, dtype=bool).all())


def _clip_native_state_for_source(source, geometry_name: str):
    """Return or attach a private native state for device-backed clip sources."""
    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.api._native_state import (
        attach_native_state_from_native_tabular_result,
        get_native_state,
    )

    state = get_native_state(source)
    if state is not None:
        return state

    if not isinstance(source, GeoDataFrame | GeoSeries):
        return None
    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    if getattr(geometry, "name", None) not in {None, geometry_name}:
        return None
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None:
        return None

    if isinstance(source, GeoDataFrame):
        if source._geometry_column_name != geometry_name:
            return None
        attributes = NativeAttributeTable(
            dataframe=source.drop(columns=[geometry_name]).copy(deep=False),
        )
        column_order = tuple(source.columns)
    else:
        attributes = NativeAttributeTable(dataframe=pd.DataFrame(index=source.index))
        column_order = (geometry_name,)

    source_attrs = getattr(source, "attrs", None)
    payload = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(owned, crs=source.crs),
        geometry_name=geometry_name,
        column_order=column_order,
        attrs=source_attrs.copy() if isinstance(source_attrs, dict) and source_attrs else None,
    )
    attach_native_state_from_native_tabular_result(source, payload)
    return get_native_state(source)


def _replace_geometry_column(frame, values):
    """Replace the active geometry column while preserving extension backing."""
    if isinstance(frame, GeoDataFrame):
        geom_name = frame._geometry_column_name
        geometry_series = _geometry_column_series(
            values,
            index=frame.index,
            crs=frame.crs,
            name=geom_name,
        )
        data_columns = {
            column_name: (geometry_series if column_name == geom_name else frame[column_name])
            for column_name in frame.columns
        }
        rebuilt = pd.DataFrame(data_columns, index=frame.index, copy=False)
        rebuilt.__class__ = GeoDataFrame
        rebuilt._geometry_column_name = geom_name
        frame_attrs = getattr(frame, "attrs", None)
        rebuilt.attrs = frame_attrs.copy() if isinstance(frame_attrs, dict) else {}
        return rebuilt

    return _geometry_series_from_values(
        values,
        index=frame.index,
        crs=frame.crs,
        name=getattr(frame, "name", None),
    )


def _take_spatial_rows(spatial, keep_mask):
    """Filter rows by position without forcing geometry object materialization."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if keep_mask.all():
        if owned is None:
            return spatial
        if owned.residency is Residency.DEVICE and isinstance(values, DeviceGeometryArray):
            return spatial
        if owned.residency is not Residency.DEVICE and isinstance(values, GeometryArray):
            return spatial
        return _replace_geometry_column(
            spatial.copy(deep=not PANDAS_GE_30),
            _geometry_values_from_owned(owned, crs=getattr(spatial, "crs", None)),
        )
    keep_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    filtered = spatial.iloc[keep_rows].copy(deep=not PANDAS_GE_30)
    if owned is None:
        return filtered

    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            taken_owned = owned.device_take(
                cp.asarray(keep_rows, dtype=cp.int64),
                host_indices_for_sizing=np.asarray(keep_rows, dtype=np.int64),
            )
        else:
            taken_owned = owned.take(keep_rows)
    else:
        taken_owned = owned.take(keep_rows)
    taken_values = _geometry_values_from_owned(
        taken_owned,
        crs=getattr(spatial, "crs", None),
    )
    return _replace_geometry_column(filtered, taken_values)


def _record_clip_host_cleanup_fallback(*, detail: str, pipeline: str) -> None:
    raise StrictNativeFallbackError(
        f"clip native semantic cleanup declined before host materialization: {pipeline}: {detail}"
    )


def _geometry_values_from_owned(owned, *, crs):
    from vibespatial.runtime.residency import Residency

    if owned.residency is Residency.DEVICE:
        return DeviceGeometryArray._from_owned(owned, crs=crs)
    return GeometryArray.from_owned(owned, crs=crs)


def _take_geometry_object_values(values, rows) -> np.ndarray:
    """Materialize only the selected rows as host geometry objects."""
    row_size = getattr(rows, "size", None)
    row_count = int(row_size) if row_size is not None else len(rows)
    if row_count == 0:
        return np.empty(0, dtype=object)
    rows_on_device = hasattr(rows, "__cuda_array_interface__")
    owned = getattr(values, "_owned", None)
    if owned is not None:
        from vibespatial.geometry.host_bridge import owned_to_shapely
        from vibespatial.runtime.materialization import (
            NativeExportBoundary,
            record_native_export_boundary,
        )

        host_rows = None
        if rows_on_device:
            if owned.residency is not Residency.DEVICE or not has_gpu_runtime():
                host_rows = _clip_device_to_host(
                    rows,
                    reason="clip selected geometry rows host export",
                ).astype(np.int64, copy=False)
        else:
            host_rows = np.asarray(rows, dtype=np.int64)
        taken_owned = None
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
            cp = None
        if owned.residency is Residency.DEVICE and has_gpu_runtime() and cp is not None:
            d_rows = (
                cp.asarray(rows, dtype=cp.int64)
                if rows_on_device
                else cp.asarray(host_rows, dtype=cp.int64)
            )
            taken_owned = owned.device_take(
                d_rows,
                host_indices_for_sizing=host_rows,
            )
        else:
            if host_rows is None:
                host_rows = np.asarray(rows, dtype=np.int64)
            taken_owned = owned.take(host_rows)
        record_native_export_boundary(
            NativeExportBoundary(
                surface="vibespatial.api.tools.clip._take_geometry_object_values",
                operation="clip_selected_geometry_rows_to_shapely",
                target="shapely",
                reason=(
                    "clip terminal exact typing exported selected geometry rows to Shapely objects"
                ),
                detail=(
                    "residency="
                    f"{getattr(getattr(taken_owned, 'residency', None), 'value', 'unknown')}"
                ),
                row_count=int(taken_owned.row_count),
                d2h_transfer=taken_owned.device_state is not None,
            )
        )
        return np.asarray(owned_to_shapely(taken_owned), dtype=object)
    if rows_on_device:
        rows = _clip_device_to_host(
            rows,
            reason="clip selected geometry rows host export",
        )
    return np.asarray(values.take(rows), dtype=object)


def _raise_for_invalid_polygon_clip_candidates(
    source,
    mask,
    candidate_rows,
    *,
    clipping_by_rectangle: bool,
) -> None:
    """Preserve GEOS invalid-input errors before native polygon clip shortcuts."""
    if clipping_by_rectangle or not isinstance(mask, Polygon | MultiPolygon):
        return
    row_count = int(getattr(candidate_rows, "size", len(candidate_rows)))
    if row_count == 0:
        return

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    owned = getattr(getattr(geometry, "values", None), "_owned", None)
    if owned is not None and owned.residency is Residency.DEVICE and owned.device_state is not None:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.constructive.validity import validity_expression_owned

            d_rows = cp.asarray(candidate_rows, dtype=cp.int64)
            candidate_owned = (
                owned if int(d_rows.size) == owned.row_count else owned.device_take(d_rows)
            )
            d_validity = cp.asarray(
                candidate_owned._ensure_device_state(
                    preserve_indexed_view=True,
                ).validity,
                dtype=cp.bool_,
            )
            d_valid_geometry = cp.asarray(
                validity_expression_owned(candidate_owned).values,
                dtype=cp.bool_,
            )
            d_invalid_local_rows = cp.flatnonzero(d_validity & ~d_valid_geometry).astype(
                cp.int64,
                copy=False,
            )
            if int(d_invalid_local_rows.size) == 0:
                return

            invalid_source_rows = d_rows[d_invalid_local_rows]
            invalid_values = _take_geometry_object_values(
                geometry.values,
                invalid_source_rows,
            )
            repeated_mask = np.empty(invalid_values.size, dtype=object)
            repeated_mask[:] = mask
            # GEOS is the public compatibility contract for invalid source rows.
            shapely.intersection(invalid_values, repeated_mask)
            return

    candidate_values = _take_geometry_object_values(geometry.values, candidate_rows)
    invalid = ~np.asarray(shapely.is_missing(candidate_values), dtype=bool) & ~np.asarray(
        shapely.is_valid(candidate_values),
        dtype=bool,
    )
    if not bool(np.any(invalid)):
        return

    invalid_values = candidate_values[invalid]
    repeated_mask = np.empty(invalid_values.size, dtype=object)
    repeated_mask[:] = mask
    # GEOS is the public compatibility contract for invalid source rows. This
    # probe intentionally lets TopologyException propagate when exact clip
    # semantics would fail, rather than silently returning a native passthrough.
    shapely.intersection(invalid_values, repeated_mask)


def _is_axis_aligned_rectangle_polygon(geom) -> bool:
    if geom is None or getattr(geom, "is_empty", False):
        return False
    if getattr(geom, "geom_type", None) != "Polygon" or len(geom.interiors) != 0:
        return False
    if len(geom.exterior.coords) != 5:
        return False
    return bool(geom.equals(geom.envelope))


def _all_axis_aligned_rectangle_polygons(values) -> bool:
    """Return True when every row is an axis-aligned rectangle polygon.

    Prefer the owned/native polygon metadata when available so rectangle-heavy
    parcel clips do not pay a per-row Shapely envelope check just to prove a
    shape that is already explicit in the coordinate buffers.
    """
    if len(values) == 0:
        return True

    owned = getattr(values, "_owned", None)
    if owned is None and hasattr(values, "to_owned"):
        owned = values.to_owned()

    if owned is not None:
        from vibespatial.geometry.owned import (
            host_owned_axis_aligned_rectangle_batch,
        )

        owned_classification = host_owned_axis_aligned_rectangle_batch(owned)
        if owned_classification is not None:
            return owned_classification

    first_geom = values[0]
    if first_geom is not None and not _is_axis_aligned_rectangle_polygon(first_geom):
        return False

    boundary_geoms = np.asarray(values, dtype=object)
    return all(_is_axis_aligned_rectangle_polygon(geom) for geom in boundary_geoms)


def _seed_rectangle_clip_validity_cache_if_safe(result_owned, source_values) -> None:
    """Mark rectangle clip output valid when the source rows are valid boxes."""
    if result_owned is None:
        return

    source_owned = getattr(source_values, "_owned", None)
    if source_owned is not None and source_owned.residency is Residency.DEVICE:
        import cupy as cp

        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            _device_rectangle_bounds,
        )

        device_polygon_buf = (
            source_owned.device_state.families.get(GeometryFamily.POLYGON)
            if source_owned.device_state is not None
            else None
        )
        device_bounds = _device_rectangle_bounds(
            device_polygon_buf,
            source_owned.row_count,
        )
        if device_bounds is None:
            return
        xmin, ymin, xmax, ymax = device_bounds
        valid_rectangles = _clip_bool_scalar(
            cp.all(((xmax - xmin) > SPATIAL_EPSILON) & ((ymax - ymin) > SPATIAL_EPSILON)),
            reason="clip rectangle validity-cache span admission scalar fence",
        )
        if not valid_rectangles:
            return
    elif not _all_axis_aligned_rectangle_polygons(source_values):
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(result_owned)


def _exact_rectangle_clip_boundary_rows(
    boundary_values,
    boundary_bounds: np.ndarray,
    rectangle_bounds: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Return exact box-vs-box clip output when every boundary row is a rectangle."""
    if len(boundary_bounds) == 0:
        return np.empty(0, dtype=object)

    if not _all_axis_aligned_rectangle_polygons(boundary_values):
        return None

    rxmin, rymin, rxmax, rymax = rectangle_bounds
    result = np.empty(len(boundary_bounds), dtype=object)
    result[:] = None

    for row_index, bounds in enumerate(boundary_bounds):
        xmin = max(float(bounds[0]), rxmin)
        ymin = max(float(bounds[1]), rymin)
        xmax = min(float(bounds[2]), rxmax)
        ymax = min(float(bounds[3]), rymax)
        dx = xmax - xmin
        dy = ymax - ymin

        if dx < -SPATIAL_EPSILON or dy < -SPATIAL_EPSILON:
            continue
        if dx > SPATIAL_EPSILON and dy > SPATIAL_EPSILON:
            result[row_index] = box(xmin, ymin, xmax, ymax)
            continue
        if abs(dx) <= SPATIAL_EPSILON and abs(dy) <= SPATIAL_EPSILON:
            result[row_index] = Point(xmin, ymin)
            continue
        if abs(dx) <= SPATIAL_EPSILON:
            result[row_index] = LineString([(xmin, ymin), (xmin, ymax)])
            continue
        result[row_index] = LineString([(xmin, ymin), (xmax, ymin)])

    return result


@dataclass(frozen=True)
class ClipNativeResult:
    """Deferred clip export that preserves native geometry results until the boundary."""

    source: GeoDataFrame | GeoSeries
    parts: tuple[LeftConstructiveResult, ...]
    ordered_index: pd.Index
    ordered_row_positions: np.ndarray
    clipping_by_rectangle: bool
    has_non_point_candidates: bool
    keep_geom_type: bool

    def _materialize_parts(self):
        if not self.parts:
            return self.source.iloc[:0]

        if isinstance(self.source, GeoDataFrame):
            parts = [part.to_geodataframe(self.source) for part in self.parts]
        else:
            parts = [part.to_geoseries(self.source) for part in self.parts]

        if len(parts) == 1:
            return parts[0]

        concatenated = pd.concat(parts)
        all_row_positions = np.concatenate(
            [np.asarray(part.row_positions, dtype=np.intp) for part in self.parts]
        )
        sorter = np.argsort(all_row_positions, kind="stable")
        order = sorter[
            np.searchsorted(
                all_row_positions[sorter],
                self.ordered_row_positions,
            )
        ]
        reordered = concatenated.iloc[order].copy(deep=not PANDAS_GE_30)
        reordered.index = self.ordered_index
        return reordered

    def _normalize_geometry_backing(self, clipped):
        from vibespatial.geometry.owned import from_shapely_geometries

        def _coerce_host_geometry_values(values):
            shapely_values = np.asarray(values, dtype=object)
            try:
                owned = from_shapely_geometries(
                    shapely_values,
                    residency=Residency.HOST,
                )
            except NotImplementedError:
                return _geometryarray_from_shapely(
                    shapely_values,
                    crs=self.source.crs,
                )
            return _geometry_values_from_owned(owned, crs=self.source.crs)

        if isinstance(clipped, GeoDataFrame):
            geom_name = clipped._geometry_column_name
            geom_values = clipped[geom_name].values
            if isinstance(geom_values, DeviceGeometryArray) or (
                isinstance(geom_values, GeometryArray) and geom_values._owned is not None
            ):
                return _replace_geometry_column(
                    clipped.copy(deep=not PANDAS_GE_30),
                    geom_values,
                )
            return _replace_geometry_column(
                clipped.copy(deep=not PANDAS_GE_30),
                _coerce_host_geometry_values(clipped[geom_name]),
            )

        values = clipped.values
        if isinstance(values, DeviceGeometryArray) or (
            isinstance(values, GeometryArray) and values._owned is not None
        ):
            return _replace_geometry_column(clipped, values)
        return _replace_geometry_column(
            clipped,
            _coerce_host_geometry_values(clipped),
        )

    def _filter_result(self, clipped):
        clipped_geometry = clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
        clipped_values = clipped_geometry.values

        def _coerce_owned_for_rows(row_mask, *, full_array: bool = False):
            current_owned = getattr(clipped_values, "_owned", None)
            if current_owned is not None:
                if current_owned.residency is not Residency.DEVICE:
                    if not has_gpu_runtime():
                        return None
                    from vibespatial.runtime.residency import TransferTrigger

                    current_owned = current_owned.move_to(
                        Residency.DEVICE,
                        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason=("clip cleanup promoted representable owned result back to device"),
                    )
                if full_array:
                    return current_owned
                row_ids = np.flatnonzero(np.asarray(row_mask, dtype=bool)).astype(
                    np.intp,
                    copy=False,
                )
                if current_owned.residency is Residency.DEVICE and has_gpu_runtime():
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                        cp = None
                    if cp is not None:
                        return current_owned.device_take(
                            cp.asarray(row_ids, dtype=cp.int64),
                            host_indices_for_sizing=np.asarray(row_ids, dtype=np.int64),
                        )
                return current_owned.take(row_ids)

            if not strict_native_mode_enabled():
                return None

            from vibespatial.geometry.owned import from_shapely_geometries

            if full_array:
                values = np.asarray(clipped_geometry, dtype=object)
            else:
                values = np.asarray(clipped_geometry, dtype=object)[
                    np.asarray(row_mask, dtype=bool)
                ]
            if values.size == 0:
                return None
            try:
                return from_shapely_geometries(
                    values.tolist(),
                    residency=(
                        Residency.DEVICE
                        if has_gpu_runtime() and strict_native_mode_enabled()
                        else Residency.HOST
                    ),
                )
            except NotImplementedError:
                return None

        if self.clipping_by_rectangle:
            keep_mask = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
            return _take_spatial_rows(clipped, keep_mask)

        keep = ~clipped_geometry.isna() & ~clipped_geometry.is_empty
        if self.has_non_point_candidates:
            current_owned = getattr(clipped_values, "_owned", None)

            def _cleanup_would_cross_device_boundary() -> bool:
                return bool(
                    isinstance(clipped_values, DeviceGeometryArray)
                    or (current_owned is not None and current_owned.residency is Residency.DEVICE)
                )

            poly_rows = clipped.geom_type.isin(POLYGON_GEOM_TYPES)
            if poly_rows.any():
                poly_mask = np.asarray(poly_rows, dtype=bool)
                poly_owned = _coerce_owned_for_rows(poly_mask)
                if poly_owned is not None:
                    from vibespatial.constructive.measurement import area_owned

                    nonpositive_area = np.asarray(area_owned(poly_owned), dtype=np.float64) <= 0.0
                else:
                    if (
                        strict_native_mode_enabled()
                        and current_owned is not None
                        and current_owned.residency is not Residency.DEVICE
                    ):
                        _record_clip_host_cleanup_fallback(
                            detail=(
                                "host polygon cleanup encountered host-backed "
                                "geometry values under strict native mode"
                            ),
                            pipeline="clip.to_spatial",
                        )
                    if _cleanup_would_cross_device_boundary():
                        _record_clip_host_cleanup_fallback(
                            detail=(
                                "host polygon cleanup would materialize Shapely "
                                "objects for area filtering"
                            ),
                            pipeline="clip.to_spatial",
                        )
                    poly_values = np.asarray(clipped_geometry, dtype=object)[poly_mask]
                    nonpositive_area = (
                        np.asarray(
                            shapely.area(poly_values),
                            dtype=np.float64,
                        )
                        <= 0.0
                    )
                if np.any(nonpositive_area):
                    poly_bounds = np.asarray(clipped_geometry.bounds, dtype=np.float64)[poly_mask]
                    pointlike_zero_area = (
                        nonpositive_area
                        & (np.abs(poly_bounds[:, 2] - poly_bounds[:, 0]) <= SPATIAL_EPSILON)
                        & (np.abs(poly_bounds[:, 3] - poly_bounds[:, 1]) <= SPATIAL_EPSILON)
                    )
                    poly_keep = np.ones(poly_mask.sum(), dtype=bool)
                    poly_keep[nonpositive_area & ~pointlike_zero_area] = False
                    keep_array = np.array(keep, dtype=bool, copy=True)
                    keep_array[np.flatnonzero(poly_mask)] &= poly_keep
                    keep = keep_array

            line_rows = clipped.geom_type.isin(LINE_GEOM_TYPES)
            if line_rows.any():
                line_mask = np.asarray(line_rows, dtype=bool)
                line_owned = _coerce_owned_for_rows(line_mask)
                if line_owned is not None:
                    from vibespatial.constructive.measurement import length_owned

                    degenerate_lines = np.asarray(length_owned(line_owned), dtype=np.float64) == 0.0
                else:
                    if (
                        strict_native_mode_enabled()
                        and current_owned is not None
                        and current_owned.residency is not Residency.DEVICE
                    ):
                        multiline_rows = np.asarray(
                            clipped.geom_type == "MultiLineString",
                            dtype=bool,
                        )
                        if multiline_rows.any():
                            from vibespatial.constructive.measurement import length_owned

                            line_owned = current_owned.take(
                                np.flatnonzero(line_mask).astype(np.intp, copy=False)
                            )
                            degenerate_lines = (
                                np.asarray(length_owned(line_owned), dtype=np.float64) == 0.0
                            )
                        else:
                            _record_clip_host_cleanup_fallback(
                                detail=(
                                    "line cleanup encountered host-backed geometry "
                                    "values under strict native mode"
                                ),
                                pipeline="clip.to_spatial",
                            )
                            degenerate_lines = np.zeros(line_mask.sum(), dtype=bool)
                    else:
                        line_values = np.asarray(clipped_geometry, dtype=object)[line_mask]
                        degenerate_lines = (
                            np.asarray(
                                shapely.length(line_values),
                                dtype=np.float64,
                            )
                            == 0.0
                        )
                if np.any(degenerate_lines):
                    full_owned = _coerce_owned_for_rows(line_mask, full_array=True)
                    if full_owned is not None:
                        from vibespatial.constructive.centroid import centroid_owned
                        from vibespatial.constructive.extract_unique_points import (
                            extract_unique_points_owned,
                        )
                        from vibespatial.geometry.owned import (
                            concat_owned_scatter,
                            device_concat_owned_scatter,
                        )

                        degenerate_rows = np.flatnonzero(line_mask)[degenerate_lines].astype(
                            np.intp,
                            copy=False,
                        )
                        if full_owned.residency is Residency.DEVICE and has_gpu_runtime():
                            try:
                                import cupy as cp
                            except ModuleNotFoundError:  # pragma: no cover
                                cp = None
                        else:
                            cp = None
                        if cp is not None:
                            d_degenerate_rows = cp.asarray(
                                degenerate_rows,
                                dtype=cp.int64,
                            )
                            degenerate_owned = full_owned.device_take(
                                d_degenerate_rows,
                                host_indices_for_sizing=np.asarray(
                                    degenerate_rows,
                                    dtype=np.int64,
                                ),
                            )
                        else:
                            d_degenerate_rows = None
                            degenerate_owned = full_owned.take(degenerate_rows)
                        repaired_owned = centroid_owned(
                            extract_unique_points_owned(
                                degenerate_owned,
                                dispatch_mode=(
                                    ExecutionMode.GPU
                                    if degenerate_owned.residency is Residency.DEVICE
                                    else ExecutionMode.AUTO
                                ),
                            ),
                            dispatch_mode=(
                                ExecutionMode.GPU
                                if full_owned.residency is Residency.DEVICE
                                else ExecutionMode.AUTO
                            ),
                        )
                        if d_degenerate_rows is not None:
                            full_owned = device_concat_owned_scatter(
                                full_owned,
                                repaired_owned,
                                d_degenerate_rows,
                            )
                        else:
                            full_owned = concat_owned_scatter(
                                full_owned,
                                repaired_owned,
                                degenerate_rows,
                            )
                        clipped = _replace_geometry_column(
                            clipped.copy(deep=not PANDAS_GE_30),
                            _geometry_values_from_owned(full_owned, crs=self.source.crs),
                        )
                    else:
                        if _cleanup_would_cross_device_boundary():
                            _record_clip_host_cleanup_fallback(
                                detail=(
                                    "line cleanup would materialize Shapely objects "
                                    "for validity repair"
                                ),
                                pipeline="clip.to_spatial",
                            )
                        line_values = np.asarray(clipped_geometry, dtype=object)[line_mask]
                        repaired_values = np.asarray(clipped_geometry, dtype=object).copy()
                        repaired_values[np.flatnonzero(line_mask)[degenerate_lines]] = (
                            shapely.make_valid(line_values[degenerate_lines])
                        )
                        clipped = _replace_geometry_column(
                            clipped.copy(deep=not PANDAS_GE_30),
                            _geometryarray_from_shapely(repaired_values, crs=self.source.crs),
                        )
                    clipped_geometry = (
                        clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
                    )
                    clipped_values = clipped_geometry.values
        return _take_spatial_rows(clipped, keep)

    def _apply_keep_geom_type(self, clipped):
        return _apply_clip_keep_geom_type_terminal(
            self.source,
            clipped,
            keep_geom_type=self.keep_geom_type,
        )

    def to_spatial(self):
        if self.parts and all(
            part.geometry.owned is not None and part.geometry.owned.residency is Residency.DEVICE
            for part in self.parts
        ):
            native_result = _clip_constructive_parts_to_native_tabular_result(
                source=self.source,
                parts=self.parts,
                ordered_row_positions=self.ordered_row_positions,
                clipping_by_rectangle=self.clipping_by_rectangle,
                has_non_point_candidates=self.has_non_point_candidates,
                keep_geom_type=self.keep_geom_type,
            )
            clipped = _clip_native_tabular_to_spatial(
                native_result,
                source=self.source,
                keep_geom_type=self.keep_geom_type,
            )
            _maybe_seed_polygon_validity_cache(clipped)
            return clipped
        clipped = self._materialize_parts()
        clipped = self._normalize_geometry_backing(clipped)
        clipped = self._filter_result(clipped)
        clipped = self._apply_keep_geom_type(clipped)
        _maybe_seed_polygon_validity_cache(clipped)
        return clipped

    def to_geodataframe(self) -> GeoDataFrame:
        clipped = self.to_spatial()
        if not isinstance(clipped, GeoDataFrame):
            raise TypeError("ClipNativeResult source is not a GeoDataFrame")
        return clipped

    def to_geoseries(self) -> GeoSeries:
        clipped = self.to_spatial()
        if not isinstance(clipped, GeoSeries):
            raise TypeError("ClipNativeResult source is not a GeoSeries")
        return clipped


@dataclass(frozen=True)
class _ClipPartitionOutput:
    geometry_values: object
    local_rows: np.ndarray
    local_rows_device: object | None = None


@dataclass(frozen=True)
class _ClipDevicePartitionOutput:
    geometry: GeometryNativeResult
    local_rows_device: object
    selection: object | None = None


def _clip_source_rowset_for_positions(
    source,
    row_positions,
    local_rows=None,
    *,
    device_row_positions=None,
    device_local_rows=None,
):
    from vibespatial.api._native_rowset import NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if isinstance(source, GeoDataFrame)
        else getattr(source, "name", None) or "geometry"
    )
    state = _clip_native_state_for_source(source, geometry_name)
    if state is None:
        return None
    rows = None if row_positions is None else np.asarray(row_positions, dtype=np.int64)
    d_rows = None
    if local_rows is not None and rows is not None:
        local_rows = np.asarray(local_rows, dtype=np.int64)
        if local_rows.size == rows.size and np.array_equal(
            local_rows,
            np.arange(rows.size, dtype=local_rows.dtype),
        ):
            # Identity sparse outputs should not force a local-row H2D upload.
            local_rows = None
        else:
            rows = rows[local_rows]
    identity = False if rows is None else _clip_source_rows_identity_hint(rows, state.row_count)
    if device_row_positions is not None and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            d_rows = cp.asarray(device_row_positions, dtype=cp.int64)
            if device_local_rows is not None or local_rows is not None:
                d_local_rows = (
                    cp.asarray(device_local_rows, dtype=cp.int64)
                    if device_local_rows is not None
                    else cp.asarray(local_rows, dtype=cp.int64)
                )
                d_rows = d_rows[d_local_rows]
            if rows is None and state.row_count == 1 and int(d_rows.size) == 1:
                identity = True
    if not has_gpu_runtime():
        if rows is None:
            return None
        return NativeRowSet.from_positions(
            rows,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=True,
            unique=True,
            identity=identity,
        )
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    if d_rows is None:
        if rows is None:
            return None
        d_rows = cp.asarray(rows, dtype=cp.int64)
    return NativeRowSet.from_positions(
        d_rows,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
        identity=identity,
    )


def _clip_native_part(
    source,
    row_positions: np.ndarray,
    geometry_values,
    *,
    rowset=None,
) -> LeftConstructiveResult:
    geometry = (
        geometry_values
        if isinstance(geometry_values, GeometryNativeResult)
        else GeometryNativeResult.from_values(geometry_values, crs=source.crs)
    )
    return LeftConstructiveResult(
        geometry=geometry.with_crs(source.crs),
        row_positions=np.asarray(row_positions, dtype=np.intp),
        rowset=rowset,
    )


def _as_geometry_values(values, *, crs):
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        return values
    return _geometryarray_from_shapely(np.asarray(values, dtype=object), crs=crs)


def _promote_geometry_backing_to_device(frame, *, reason):
    """Rebuild a public geometry container with device-backed owned storage."""
    if not has_gpu_runtime() or not isinstance(frame, GeoDataFrame | GeoSeries):
        return frame

    values = frame.geometry.values if isinstance(frame, GeoDataFrame) else frame.values
    if isinstance(values, DeviceGeometryArray):
        return frame
    if not isinstance(values, GeometryArray):
        return frame

    from vibespatial.runtime.residency import Residency, TransferTrigger

    owned = values.to_owned()
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=reason,
    )
    return _replace_geometry_column(
        frame.copy(deep=not PANDAS_GE_30),
        DeviceGeometryArray._from_owned(owned, crs=frame.crs),
    )


def _clip_polygon_area_intersection_owned(
    left_owned,
    mask_owned,
    *,
    preserve_lower_dimensional: bool = False,
):
    """Intersect polygon rows with one mask using device-capacity carriers."""
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
        broadcast_right_polygon_intersection_capacity_gpu,
    )
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.residency import Residency

    if (
        not preserve_lower_dimensional
        and has_gpu_runtime()
        and left_owned.residency is Residency.DEVICE
        and mask_owned.row_count == 1
    ):
        cell_mask_result = _clip_polygon_rect_cell_mask_intersection_owned(
            left_owned,
            mask_owned,
            None,
        )
        if cell_mask_result is not None:
            return cell_mask_result

    result = (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
            left_owned,
            mask_owned,
            dispatch_mode=ExecutionMode.GPU,
        )
        if preserve_lower_dimensional
        else broadcast_right_polygon_intersection_capacity_gpu(
            left_owned,
            mask_owned,
            right_row=0,
            dispatch_mode=ExecutionMode.GPU,
        )
    )
    if result is None:
        raise StrictNativeFallbackError(
            "polygon-mask intersection declined its canonical broadcast-right "
            "device-capacity carrier"
        )
    record_dispatch_event(
        surface="geopandas.clip",
        operation="intersection",
        implementation="polygon_mask_broadcast_right_capacity_gpu",
        reason=(
            "polygon-mask intersection consumed exact broadcast-right topology "
            "for mixed-dimensional output"
            if preserve_lower_dimensional
            else "polygon-mask intersection consumed the canonical rectangle, SH, "
            "swapped-SH, and exact device-capacity partitioner"
        ),
        detail=(
            f"rows={left_owned.row_count}; physical_shape=row_indirected_broadcast_right_capacity"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result


def _clip_polygon_rect_cell_mask_intersection_owned(
    left_owned,
    mask_owned,
    tiled_mask_owned=None,
):
    """Clip rectangle-cell source rows by a single-ring polygon mask on device.

    The raw polygon-rectangle kernel is fast but can encode disconnected cell
    intersections as one repeated-boundary polygon ring.  The boundary split
    carrier repairs only those rows into MultiPolygons and leaves ordinary
    single-component rows untouched, preserving the exact row-aligned clip
    shape without reopening the full overlay pipeline.
    """
    if left_owned.row_count == 0:
        return None
    if mask_owned.row_count != 1:
        return None
    if not has_gpu_runtime() or left_owned.residency is not Residency.DEVICE:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover
        return None

    from vibespatial.geometry.owned import (
        device_select_owned_capacity_partitions,
        tile_single_row,
    )
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_trusted_rectangle_bounds_matrix,
        device_trusted_single_ring_polygon_batch,
        polygon_rect_intersection,
        polygon_rect_intersection_can_handle,
        polygon_rect_intersection_from_bounds,
        polygon_rect_split_boundary_component_replacements,
        polygon_rect_split_boundary_component_replacements_from_bounds,
    )
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.residency import TransferTrigger

    def _repair_boundary_splits(fast_owned, rect_bounds):
        boundary_overlap = getattr(fast_owned, "_polygon_rect_boundary_overlap", None)
        if boundary_overlap is None:
            return fast_owned, 0
        d_boundary_overlap = cp.asarray(boundary_overlap, dtype=cp.bool_)
        if int(d_boundary_overlap.size) != fast_owned.row_count:
            return None, 0
        split_replacements = polygon_rect_split_boundary_component_replacements_from_bounds(
            fast_owned,
            rect_bounds,
            d_boundary_overlap,
        )
        if split_replacements is None:
            return None, fast_owned.row_count
        split_owned, d_split_mask = split_replacements
        if split_owned.row_count != fast_owned.row_count:
            return None, fast_owned.row_count
        fast_owned = device_select_owned_capacity_partitions(
            fast_owned,
            [(split_owned, d_split_mask)],
        )
        fast_owned._polygon_rect_boundary_overlap = cp.zeros(
            fast_owned.row_count,
            dtype=cp.bool_,
        )
        return fast_owned, fast_owned.row_count

    if mask_owned.residency is not Residency.DEVICE or mask_owned.device_state is None:
        mask_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="polygon rect cell mask clip selected row-indirected GPU mask",
        )

    rect_bounds = device_trusted_rectangle_bounds_matrix(left_owned)
    if rect_bounds is not None:
        logical_mask = (
            mask_owned
            if left_owned.row_count == 1
            else tile_single_row(mask_owned, left_owned.row_count)
        )
        if device_trusted_single_ring_polygon_batch(logical_mask):
            fast_owned = polygon_rect_intersection_from_bounds(
                logical_mask,
                rect_bounds,
                dispatch_mode=ExecutionMode.GPU,
            )
            if fast_owned.row_count == left_owned.row_count:
                fast_owned, split_count = _repair_boundary_splits(
                    fast_owned,
                    rect_bounds,
                )
                if fast_owned is not None:
                    record_dispatch_event(
                        surface="geopandas.clip",
                        operation="clip",
                        implementation="polygon_rect_cell_mask_split_gpu",
                        reason=(
                            "single-ring polygon mask clipped rectangle-cell "
                            "source rows with native boundary split repair"
                        ),
                        detail=(
                            f"rows={left_owned.row_count}; "
                            f"split_candidate_capacity={split_count}; "
                            "physical_shape="
                            "row_indirected_polygon_mask_device_rectangle_bounds"
                        ),
                        requested=ExecutionMode.GPU,
                        selected=ExecutionMode.GPU,
                    )
                    fast_owned._clip_polygon_positive_rows_from_validity = True
                    return fast_owned

    if tiled_mask_owned is None or tiled_mask_owned.row_count != left_owned.row_count:
        return None

    if not polygon_rect_intersection_can_handle(tiled_mask_owned, left_owned):
        return None

    fast_owned = polygon_rect_intersection(
        tiled_mask_owned,
        left_owned,
        dispatch_mode=ExecutionMode.GPU,
    )
    if fast_owned.row_count != left_owned.row_count:
        return None

    rect_bounds = device_trusted_rectangle_bounds_matrix(left_owned)
    if rect_bounds is not None:
        fast_owned, split_count = _repair_boundary_splits(fast_owned, rect_bounds)
        if fast_owned is None:
            return None
    else:
        boundary_overlap = getattr(fast_owned, "_polygon_rect_boundary_overlap", None)
        split_count = 0
        if boundary_overlap is not None:
            d_boundary_overlap = cp.asarray(boundary_overlap, dtype=cp.bool_)
            if (
                d_boundary_overlap is not None
                and int(d_boundary_overlap.size) == fast_owned.row_count
            ):
                split_count = fast_owned.row_count
        if split_count > 0:
            split_replacements = polygon_rect_split_boundary_component_replacements(
                fast_owned,
                left_owned,
                eligible_mask=d_boundary_overlap,
            )
            if split_replacements is None:
                return None
            split_owned, d_split_mask = split_replacements
            if split_owned.row_count != fast_owned.row_count:
                return None
            fast_owned = device_select_owned_capacity_partitions(
                fast_owned,
                [(split_owned, d_split_mask)],
            )
            fast_owned._polygon_rect_boundary_overlap = cp.zeros(
                fast_owned.row_count,
                dtype=cp.bool_,
            )

    record_dispatch_event(
        surface="geopandas.clip",
        operation="clip",
        implementation="polygon_rect_cell_mask_split_gpu",
        reason=(
            "single-ring polygon mask clipped rectangle-cell source rows with "
            "native boundary split repair"
        ),
        detail=(
            f"rows={left_owned.row_count}; split_candidate_capacity={split_count}; "
            "physical_shape=polygon-mask-rectangle-cell"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    fast_owned._clip_polygon_positive_rows_from_validity = True
    return fast_owned


def _clip_polygon_single_pair_containment_owned(left_owned, mask_owned):
    """Return a device-native scalar polygon clip result for trivial containment.

    For the common ``1x1`` polygon clip shape, full exact overlay is wasted
    when one polygon wholly contains the other. Reuse the existing GPU
    containment bypass kernels in both directions:

    - ``left inside mask`` -> clip result is ``left``
    - ``mask inside left`` -> clip result is ``mask``

    Return ``None`` when neither bypass applies so the caller can continue to
    the exact constructive path.
    """
    from vibespatial.overlay.bypass import _containment_bypass_gpu
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.residency import Residency

    if (
        left_owned.row_count != 1
        or mask_owned.row_count != 1
        or not has_gpu_runtime()
        or left_owned.residency is not Residency.DEVICE
        or mask_owned.residency is not Residency.DEVICE
    ):
        return None

    left_inside_mask, left_remainder = _containment_bypass_gpu(
        left_owned,
        mask_owned,
        "intersection",
    )
    if left_inside_mask is not None and left_remainder is None:
        record_dispatch_event(
            surface="geopandas.clip",
            operation="intersection",
            implementation="clip_polygon_single_pair_containment_bypass",
            reason="single-row polygon clip returned the source polygon via GPU containment bypass",
            detail="left_inside_mask",
            selected=ExecutionMode.GPU,
        )
        return left_inside_mask

    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        polygon_rect_intersection_can_handle,
    )

    if polygon_rect_intersection_can_handle(mask_owned, left_owned):
        mask_inside_left, mask_remainder = _containment_bypass_gpu(
            mask_owned,
            left_owned,
            "intersection",
        )
        if mask_inside_left is not None and mask_remainder is None:
            record_dispatch_event(
                surface="geopandas.clip",
                operation="intersection",
                implementation="clip_polygon_single_pair_containment_bypass",
                reason=(
                    "single-row polygon clip returned the mask polygon via "
                    "GPU containment bypass against a rectangular source"
                ),
                detail="mask_inside_rectangular_left",
                selected=ExecutionMode.GPU,
            )
            return mask_inside_left
    return None


def _host_polygonal_area_intersection_owned(left_owned, right_owned):
    """Host exact intersection for clip's polygonal area-only contract."""
    from vibespatial.api.tools.overlay import _strip_non_polygon_collection_parts
    from vibespatial.geometry.owned import from_shapely_geometries

    left_values = np.asarray(left_owned.to_shapely(), dtype=object)
    if right_owned.row_count == 1 and left_values.size > 1:
        right_geom = right_owned.to_shapely()[0]
        right_values = np.full(left_values.size, right_geom, dtype=object)
    else:
        right_values = np.asarray(right_owned.to_shapely(), dtype=object)

    raw = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    polygonal = _strip_non_polygon_collection_parts(raw)
    area_only = np.asarray(
        [
            geom
            if (
                geom is not None
                and getattr(geom, "geom_type", None) in POLYGON_GEOM_TYPES
                and not getattr(geom, "is_empty", False)
            )
            else None
            for geom in polygonal
        ],
        dtype=object,
    )
    return from_shapely_geometries(area_only.tolist(), residency=left_owned.residency)


def _clip_polygon_rectangle_area_intersection_owned(
    left_owned,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Compute polygon-only rectangle intersections at source-row capacity."""
    from vibespatial.constructive.binary_constructive import (
        broadcast_right_polygon_intersection_capacity_gpu,
    )
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        device_mask_owned_capacity,
        from_shapely_geometries,
    )

    rectangle_mask = box(*rectangle_bounds)
    rectangle_owned = _device_rectangle_owned_from_bounds(
        rectangle_bounds,
        residency=left_owned.residency,
    )
    if rectangle_owned is None:
        rectangle_owned = from_shapely_geometries(
            [rectangle_mask],
            residency=left_owned.residency,
        )

    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        result = broadcast_right_polygon_intersection_capacity_gpu(
            left_owned,
            rectangle_owned,
            right_row=0,
            dispatch_mode=ExecutionMode.GPU,
        )
        if result is None:
            raise StrictNativeFallbackError(
                "polygon-rectangle clip declined its canonical broadcast-right "
                "device-capacity carrier"
            )
        d_positive = _owned_nonempty_polygon_device_mask(result)
        if d_positive is None:
            raise StrictNativeFallbackError(
                "polygon-rectangle clip could not derive its device positive-area mask"
            )
        return device_mask_owned_capacity(result, d_positive)

    record_fallback_event(
        surface="geopandas.clip",
        reason="polygon-rectangle clip used CPU compatibility boundary without GPU runtime",
        detail=f"rows={left_owned.row_count}",
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
        pipeline="_clip_polygon_rectangle_area_intersection_owned",
        d2h_transfer=left_owned.residency is Residency.DEVICE,
    )
    result = _host_polygonal_area_intersection_owned(
        left_owned,
        rectangle_owned,
    )
    positive_rows = np.flatnonzero(_owned_nonempty_polygon_mask(result)).astype(
        np.intp,
        copy=False,
    )
    if positive_rows.size == result.row_count:
        return result
    if positive_rows.size == 0:
        return build_null_owned_array(
            left_owned.row_count,
            residency=result.residency,
        )
    return concat_owned_scatter(
        build_null_owned_array(
            left_owned.row_count,
            residency=result.residency,
        ),
        result.take(positive_rows),
        positive_rows,
    )


def _clip_multipolygon_rectangle_keep_geom_type_owned(
    left_owned,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Recover polygonal rectangle clip output for multipolygon rows on device.

    Rectangle `keep_geom_type=True` only needs the polygonal area portion of
    the public intersection result. For multipolygon rows, the full-row exact
    intersection path can degrade into mixed GeometryCollection semantics that
    are hard to preserve in the owned family model. Explode the multipolygon
    into polygon parts on device, clip each part through the polygon-rectangle
    area path, then regroup the surviving polygon parts back to the original
    row ids without leaving the device.
    """
    if not has_gpu_runtime() or left_owned.row_count == 0:
        return None
    if left_owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
        _pack_disjoint_multipart_intersection_parts_gpu,
    )

    polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(
        left_owned,
    )

    if polygon_parts is None or polygon_parts.capacity == 0:
        return None

    part_result = _clip_polygon_rectangle_area_intersection_owned(
        polygon_parts.geometry,
        rectangle_bounds,
    )

    d_positive = _owned_nonempty_polygon_device_mask(part_result)
    if d_positive is None:
        return None
    d_positive &= polygon_parts.selection.active_capacity_mask()
    return _pack_disjoint_multipart_intersection_parts_gpu(
        part_result,
        polygon_parts.source_rows,
        output_row_count=left_owned.row_count,
        assume_disjoint=True,
        d_valid_rows_mask=d_positive,
    )


def _clip_polygonal_rectangle_keep_geom_type_device_owned(
    left_owned,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Select polygonal rectangle intersections at input row capacity."""
    if (
        not has_gpu_runtime()
        or left_owned.residency is not Residency.DEVICE
        or left_owned.row_count == 0
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        build_null_owned_array,
        device_mask_owned_capacity,
        device_select_owned_capacity_partitions,
    )

    state = left_owned._ensure_device_state(preserve_indexed_view=True)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    trusted_family_domain = (
        None if state.trusted_family_domain is None else frozenset(state.trusted_family_domain)
    )
    partitions = (
        (
            GeometryFamily.POLYGON,
            _clip_polygon_rectangle_area_intersection_owned,
        ),
        (
            GeometryFamily.MULTIPOLYGON,
            _clip_multipolygon_rectangle_keep_geom_type_owned,
        ),
    )
    capacity_partitions = []
    for family, constructive in partitions:
        if trusted_family_domain is not None and family not in trusted_family_domain:
            continue
        d_family_mask = d_validity & (d_tags == cp.int8(FAMILY_TAGS[family]))
        family_result = constructive(
            device_mask_owned_capacity(left_owned, d_family_mask),
            rectangle_bounds,
        )
        if (
            family_result is None
            or family_result.residency is not Residency.DEVICE
            or family_result.row_count != left_owned.row_count
        ):
            return None
        d_positive = _owned_nonempty_polygon_device_mask(family_result)
        if d_positive is None:
            return None
        d_positive &= d_family_mask
        capacity_partitions.append(
            (
                device_mask_owned_capacity(family_result, d_positive),
                d_positive,
            )
        )

    if not capacity_partitions:
        return build_null_owned_array(
            left_owned.row_count,
            residency=left_owned.residency,
        )

    return device_select_owned_capacity_partitions(
        build_null_owned_array(
            left_owned.row_count,
            residency=left_owned.residency,
        ),
        capacity_partitions,
    )


def _clip_polygon_boundary_touch_mask(
    source_values,
    boundary_rows: np.ndarray,
    *,
    mask,
) -> np.ndarray:
    """Return host compatibility intersections for clip boundary rows."""
    if boundary_rows.size == 0:
        return np.empty(0, dtype=bool)
    return np.asarray(source_values.take(boundary_rows).intersects(mask), dtype=bool)


def _owned_nonempty_polygon_mask(owned) -> np.ndarray:
    """Return rows backed by polygonal output with strictly positive area."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.constructive.measurement import _area_gpu_device_fp64
            from vibespatial.cuda._runtime import get_cuda_runtime

            device_state = owned._ensure_device_state(preserve_indexed_view=True)
            d_tags = cp.asarray(device_state.tags)
            d_validity = cp.asarray(device_state.validity)
            d_polygonal_mask = (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
                d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            )
            d_areas = _area_gpu_device_fp64(owned)
            d_keep = d_validity & d_polygonal_mask & cp.isfinite(d_areas) & (d_areas > 0.0)
            d_rows = cp.flatnonzero(d_keep).astype(cp.int64, copy=False)
            row_count = int(owned.row_count)
            positive_count = int(d_rows.size)
            if positive_count == 0:
                return np.zeros(row_count, dtype=bool)
            if positive_count == row_count:
                return np.ones(row_count, dtype=bool)
            if positive_count * np.dtype(np.int64).itemsize < row_count:
                rows = np.asarray(
                    get_cuda_runtime().copy_device_to_host(
                        d_rows,
                        reason=(
                            "clip keep-geometry-type polygonal positive-area terminal rows export"
                        ),
                    ),
                    dtype=np.intp,
                )
                mask = np.zeros(row_count, dtype=bool)
                mask[rows] = True
                return mask
            return np.asarray(
                get_cuda_runtime().copy_device_to_host(
                    d_keep,
                    reason=("clip keep-geometry-type polygonal positive-area terminal mask export"),
                ),
                dtype=bool,
            )

    from vibespatial.constructive.measurement import area_owned

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=tags.dtype if tags.size else np.int8,
    )
    polygonal_mask = validity & np.isin(tags, polygon_tags)
    if not polygonal_mask.any():
        return np.zeros(len(tags), dtype=bool)

    areas = np.asarray(area_owned(owned), dtype=np.float64)
    if areas.size != len(tags):
        return np.zeros(len(tags), dtype=bool)
    return polygonal_mask & np.isfinite(areas) & (areas > 0.0)


def _owned_nonempty_polygon_rows(
    owned,
    *,
    keep_pointlike_zero_area: bool = False,
) -> _ClipPositiveRows:
    """Return positive polygonal rows and preserve device row positions."""
    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            from vibespatial.constructive.measurement import _area_gpu_device_fp64
            from vibespatial.cuda._runtime import get_cuda_runtime
            from vibespatial.geometry.buffers import GeometryFamily
            from vibespatial.geometry.owned import FAMILY_TAGS

            device_state = owned._ensure_device_state(preserve_indexed_view=True)
            d_tags = cp.asarray(device_state.tags)
            d_validity = cp.asarray(device_state.validity)
            d_polygonal_mask = (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
                d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            )
            d_areas = _area_gpu_device_fp64(owned)
            d_positive = d_validity & d_polygonal_mask & cp.isfinite(d_areas) & (d_areas > 0.0)
            if keep_pointlike_zero_area:
                from vibespatial.kernels.core.geometry_analysis import (
                    compute_geometry_bounds_device,
                )

                d_nonpositive = (
                    d_validity & d_polygonal_mask & ~(cp.isfinite(d_areas) & (d_areas > 0.0))
                )
                d_bounds = compute_geometry_bounds_device(
                    owned,
                    preserve_indexed_view=True,
                )
                d_width = cp.abs(d_bounds[:, 2] - d_bounds[:, 0])
                d_height = cp.abs(d_bounds[:, 3] - d_bounds[:, 1])
                d_keep = d_positive | (
                    d_nonpositive & (d_width <= SPATIAL_EPSILON) & (d_height <= SPATIAL_EPSILON)
                )
            else:
                d_keep = d_positive
            d_rows = cp.flatnonzero(d_keep).astype(cp.int64, copy=False)
            row_count = int(owned.row_count)
            positive_count = int(d_rows.size)
            if positive_count == 0:
                mask = np.zeros(row_count, dtype=bool)
                rows = np.empty(0, dtype=np.intp)
            elif positive_count == row_count:
                mask = np.ones(row_count, dtype=bool)
                rows = np.arange(row_count, dtype=np.intp)
            elif positive_count * np.dtype(np.int64).itemsize < row_count:
                rows = np.asarray(
                    get_cuda_runtime().copy_device_to_host(
                        d_rows,
                        reason=(
                            "clip keep-geometry-type polygonal positive-area terminal rows export"
                        ),
                    ),
                    dtype=np.intp,
                )
                mask = np.zeros(row_count, dtype=bool)
                mask[rows] = True
            else:
                mask = np.asarray(
                    get_cuda_runtime().copy_device_to_host(
                        d_keep,
                        reason=(
                            "clip keep-geometry-type polygonal positive-area terminal mask export"
                        ),
                    ),
                    dtype=bool,
                )
                rows = np.flatnonzero(mask).astype(np.intp, copy=False)
            return _ClipPositiveRows(
                mask=mask,
                rows=rows,
                device_rows=d_rows,
            )

    mask = _owned_nonempty_polygon_mask(owned)
    return _ClipPositiveRows(
        mask=mask,
        rows=np.flatnonzero(mask).astype(np.intp, copy=False),
    )


def _owned_nonempty_polygon_device_mask(
    owned,
    *,
    keep_pointlike_zero_area: bool = False,
):
    """Return a device mask for positive polygonal rows, if available."""
    if owned.residency is not Residency.DEVICE or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    if getattr(owned, "_clip_polygon_positive_rows_from_validity", False):
        device_state = owned._ensure_device_state(preserve_indexed_view=True)
        return cp.asarray(device_state.validity, dtype=cp.bool_)

    from vibespatial.constructive.measurement import _area_gpu_device_fp64
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    device_state = owned._ensure_device_state(preserve_indexed_view=True)
    d_tags = cp.asarray(device_state.tags)
    d_validity = cp.asarray(device_state.validity)
    d_polygonal_mask = (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]) | (
        d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    )
    d_areas = _area_gpu_device_fp64(owned)
    d_positive = d_validity & d_polygonal_mask & cp.isfinite(d_areas) & (d_areas > 0.0)
    if keep_pointlike_zero_area:
        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )

        d_nonpositive = d_validity & d_polygonal_mask & ~(cp.isfinite(d_areas) & (d_areas > 0.0))
        d_bounds = compute_geometry_bounds_device(
            owned,
            preserve_indexed_view=True,
        )
        d_width = cp.abs(d_bounds[:, 2] - d_bounds[:, 0])
        d_height = cp.abs(d_bounds[:, 3] - d_bounds[:, 1])
        d_pointlike = d_nonpositive & (d_width <= SPATIAL_EPSILON) & (d_height <= SPATIAL_EPSILON)
        d_keep = d_positive | d_pointlike
    else:
        d_keep = d_positive
    return d_keep


def _owned_nonempty_polygon_device_rows(owned, *, keep_pointlike_zero_area: bool = False):
    """Return device row positions for positive polygonal rows, if available."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    d_keep = _owned_nonempty_polygon_device_mask(
        owned,
        keep_pointlike_zero_area=keep_pointlike_zero_area,
    )
    if d_keep is None:
        return None
    return cp.flatnonzero(d_keep).astype(cp.int64, copy=False)


def _owned_valid_nonempty_device_mask(owned):
    """Return a device bool mask for valid non-empty rows, if available."""
    if owned.residency is not Residency.DEVICE or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    from vibespatial.geometry.owned import device_valid_nonempty_mask

    return cp.asarray(device_valid_nonempty_mask(owned), dtype=cp.bool_)


def _owned_valid_nonempty_device_rows(owned):
    """Return device row positions for valid non-empty rows in an owned result."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    d_keep = _owned_valid_nonempty_device_mask(owned)
    if d_keep is None:
        return None
    return cp.flatnonzero(d_keep).astype(cp.int64, copy=False)


def _exact_polygon_clip_boundary_rows(
    left_values,
    right_values,
) -> np.ndarray:
    """Return the exact host boundary rows for polygon-mask clip semantics."""
    return np.asarray(
        shapely.intersection(
            np.asarray(left_values, dtype=object),
            np.asarray(right_values, dtype=object),
        ),
        dtype=object,
    )


def _clip_partition_degenerate_line_part_capacity_device(
    part_selection,
    d_part_output_rows,
):
    """Split zero-length line parts into point capacity without compaction."""
    import cupy as cp

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        LinePartCapacitySelection,
        PointPartCapacitySelection,
    )
    from vibespatial.constructive.point import _build_device_backed_point_output
    from vibespatial.geometry.buffers import GeometryFamily

    capacity = int(part_selection.capacity)
    d_output_rows = cp.asarray(d_part_output_rows, dtype=cp.int64)
    if int(d_output_rows.size) != capacity:
        raise ValueError("line-part output rows must match part capacity")
    d_active = part_selection.selection.active_capacity_mask()
    state = part_selection.geometry._ensure_device_state(
        preserve_indexed_view=True,
    )
    line_buffer = state.families.get(GeometryFamily.LINESTRING)
    if line_buffer is None:
        return None
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    d_safe_family_rows = cp.where(d_active, d_family_rows, cp.int64(0))
    d_offsets = cp.asarray(line_buffer.geometry_offsets, dtype=cp.int64)
    d_starts = d_offsets[d_safe_family_rows]
    d_stops = d_offsets[d_safe_family_rows + 1]
    d_nonempty = d_active & (d_stops > d_starts)
    d_safe_ends = cp.maximum(d_stops - 1, d_starts)
    d_x = cp.asarray(line_buffer.x, dtype=cp.float64)
    d_y = cp.asarray(line_buffer.y, dtype=cp.float64)
    if int(d_x.size) == 0:
        d_degenerate = cp.zeros(capacity, dtype=cp.bool_)
        d_point_x = cp.zeros(capacity, dtype=cp.float64)
        d_point_y = cp.zeros(capacity, dtype=cp.float64)
    else:
        d_degenerate = (
            d_nonempty
            & (cp.abs(d_x[d_starts] - d_x[d_safe_ends]) <= SPATIAL_EPSILON)
            & (cp.abs(d_y[d_starts] - d_y[d_safe_ends]) <= SPATIAL_EPSILON)
        )
        d_point_x = d_x[d_starts].copy()
        d_point_y = d_y[d_starts].copy()

    def _gather_capacity(raw_geometry, d_mask, selection_type):
        selection = NativeDeviceSelection.from_mask(d_mask)
        from vibespatial.geometry.owned import device_take_owned_capacity_selection

        gathered_geometry = device_take_owned_capacity_selection(
            raw_geometry,
            selection,
        )
        selection_kwargs = {}
        if selection_type is LinePartCapacitySelection:
            selection_kwargs["coord_capacity"] = part_selection.coord_capacity
        return (
            selection_type(
                geometry=gathered_geometry,
                source_rows=selection.gather_capacity(
                    part_selection.source_rows,
                    fill_value=0,
                ).astype(cp.int32, copy=False),
                selection=selection.as_capacity_prefix(),
                **selection_kwargs,
            ),
            selection.gather_capacity(
                d_output_rows,
                fill_value=0,
            ).astype(cp.int64, copy=False),
        )

    line_selection, d_line_output_rows = _gather_capacity(
        part_selection.geometry,
        d_active & ~d_degenerate,
        LinePartCapacitySelection,
    )
    point_geometry = _build_device_backed_point_output(
        d_point_x,
        d_point_y,
        row_count=capacity,
    )
    point_selection, d_point_output_rows = _gather_capacity(
        point_geometry,
        d_degenerate,
        PointPartCapacitySelection,
    )
    return (
        line_selection,
        d_line_output_rows,
        point_selection,
        d_point_output_rows,
    )


def _clip_concat_point_part_capacities_device(point_parts):
    """Concatenate point-part capacities and restore one active prefix."""
    import cupy as cp

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import PointPartCapacitySelection
    from vibespatial.geometry.owned import OwnedGeometryArray

    parts = tuple(point_parts)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    raw_geometry = OwnedGeometryArray.concat([part.geometry for part, _d_output_rows in parts])
    d_raw_source_rows = cp.concatenate(
        [cp.asarray(part.source_rows, dtype=cp.int32) for part, _d_output_rows in parts]
    )
    d_raw_output_rows = cp.concatenate(
        [cp.asarray(d_output_rows, dtype=cp.int64) for _part, d_output_rows in parts]
    )
    d_raw_active = cp.concatenate(
        [part.selection.active_capacity_mask() for part, _d_output_rows in parts]
    )
    selection = NativeDeviceSelection.from_mask(d_raw_active)
    from vibespatial.geometry.owned import device_take_owned_capacity_selection

    geometry = device_take_owned_capacity_selection(
        raw_geometry,
        selection,
    )
    return (
        PointPartCapacitySelection(
            geometry=geometry,
            source_rows=selection.gather_capacity(
                d_raw_source_rows,
                fill_value=0,
            ).astype(cp.int32, copy=False),
            selection=selection.as_capacity_prefix(),
        ),
        selection.gather_capacity(
            d_raw_output_rows,
            fill_value=0,
        ).astype(cp.int64, copy=False),
    )


def _clip_polygon_boundary_intersection_device_capacity_parts(
    candidate_owned,
    mask_owned,
    d_candidate_rows,
    *,
    d_candidate_active=None,
    d_polygon_area_active=None,
):
    """Return row-aligned lineal/pointlike boundary parts at source capacity."""
    if candidate_owned.residency is not Residency.DEVICE or not has_gpu_runtime():
        return None
    if mask_owned.residency is not Residency.DEVICE:
        return None
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _explode_lineal_rows_to_line_capacity_gpu,
        _explode_point_rows_to_point_capacity_gpu,
    )
    from vibespatial.constructive.boundary import boundary_owned
    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        device_physicalize_owned_row_selection_capacity,
        tile_single_row,
    )
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_polygon_intersection,
    )

    d_candidate_rows = cp.asarray(d_candidate_rows, dtype=cp.int64)
    output_row_count = int(d_candidate_rows.size)
    if output_row_count == 0:
        return ()
    if d_candidate_active is None:
        d_candidate_active = cp.ones(output_row_count, dtype=cp.bool_)
    else:
        d_candidate_active = cp.asarray(d_candidate_active, dtype=cp.bool_)
        if int(d_candidate_active.size) != output_row_count:
            raise ValueError("boundary candidate activity must match row capacity")
    if d_polygon_area_active is None:
        d_polygon_area_active = cp.zeros(output_row_count, dtype=cp.bool_)
    else:
        d_polygon_area_active = cp.asarray(d_polygon_area_active, dtype=cp.bool_)
        if int(d_polygon_area_active.size) != output_row_count:
            raise ValueError("polygon area activity must match boundary row capacity")

    candidate_owned = candidate_owned.device_take_capacity(
        d_candidate_rows,
        d_candidate_active,
    )
    d_candidate_rows = cp.arange(output_row_count, dtype=cp.int64)
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_rectangle_polygon_mask_and_bounds,
        device_trusted_rectangle_bounds_matrix,
        polygon_rect_boundary_contacts_from_bounds,
        rectangle_rectangle_boundary_intersections_from_bounds,
    )

    rect_bounds = device_trusted_rectangle_bounds_matrix(candidate_owned)
    rectangle_rows_proven = rect_bounds is not None
    if rect_bounds is None:
        d_rectangle_rows, rect_bounds = device_rectangle_polygon_mask_and_bounds(
            candidate_owned,
        )
        d_rectangle_rows = (
            cp.asarray(d_rectangle_rows, dtype=cp.bool_) & d_candidate_active
        )
    else:
        rect_bounds = cp.asarray(rect_bounds, dtype=cp.float64)
        d_rectangle_rows = d_candidate_active & cp.all(cp.isfinite(rect_bounds), axis=1)
    mask_rect_bounds = device_trusted_rectangle_bounds_matrix(mask_owned)
    specialized_parts = []
    d_unresolved = d_candidate_active.copy()
    if (
        int(mask_owned.row_count) == 1
        and rect_bounds is not None
        and mask_rect_bounds is not None
    ):
        rectangle_boundary = rectangle_rectangle_boundary_intersections_from_bounds(
            cp.asarray(rect_bounds, dtype=cp.float64)[d_candidate_rows],
            cp.asarray(mask_rect_bounds, dtype=cp.float64).reshape(1, 4)[0],
            active_mask=d_rectangle_rows,
            dispatch_mode=ExecutionMode.GPU,
        )
        if rectangle_boundary is not None:
            specialized_parts.append(rectangle_boundary)
            d_unresolved &= ~d_rectangle_rows
            if rectangle_rows_proven:
                return tuple(specialized_parts)

    elif int(mask_owned.row_count) == 1 and rect_bounds is not None:
        boundary_contacts = polygon_rect_boundary_contacts_from_bounds(
            mask_owned,
            cp.asarray(rect_bounds, dtype=cp.float64)[d_candidate_rows],
            dispatch_mode=ExecutionMode.GPU,
        )
        if boundary_contacts is not None:
            (
                point_owned,
                d_point_mask,
                line_owned,
                d_line_mask,
                d_unsupported_mask,
            ) = boundary_contacts
            d_point_mask = cp.asarray(d_point_mask, dtype=cp.bool_) & d_rectangle_rows
            d_line_mask = cp.asarray(d_line_mask, dtype=cp.bool_) & d_rectangle_rows
            d_unsupported_mask = (
                cp.asarray(d_unsupported_mask, dtype=cp.bool_)
                & d_rectangle_rows
            )
            d_unresolved = (
                (d_candidate_active & ~d_rectangle_rows)
                | d_unsupported_mask
            )
            point_owned = device_mask_owned_capacity(
                point_owned,
                d_point_mask,
            )
            line_owned = device_mask_owned_capacity(
                line_owned,
                d_line_mask,
            )
            from vibespatial.runtime.dispatch import record_dispatch_event

            record_dispatch_event(
                surface="geopandas.clip",
                operation="clip",
                implementation="polygon_rect_boundary_contacts_gpu",
                reason=(
                    "rectangle-cell polygon-mask boundary point and line contacts "
                    "used a direct device bounds carrier before generic boundary "
                    "reconstruction"
                ),
                detail=(
                    f"candidate_rows={output_row_count}; "
                    f"output_capacity={point_owned.row_count}; "
                    "physical_shape=rowset_rectangle_boundary_capacity"
                ),
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
            )
            specialized_parts.extend((line_owned, point_owned))
            if rectangle_rows_proven:
                return tuple(specialized_parts)
    boundary_input = device_physicalize_owned_row_selection_capacity(
        candidate_owned,
        d_unresolved,
    )
    boundary_result = boundary_owned(
        boundary_input,
        dispatch_mode=ExecutionMode.GPU,
    )
    line_parts = _explode_lineal_rows_to_line_capacity_gpu(boundary_result)
    if line_parts is None or line_parts.capacity == 0:
        return tuple(specialized_parts)

    d_line_source_rows = cp.asarray(line_parts.source_rows, dtype=cp.int64)
    d_line_active = line_parts.selection.active_capacity_mask()
    if int(mask_owned.row_count) == 1:
        line_masks = device_mask_owned_capacity(
            tile_single_row(mask_owned, line_parts.capacity),
            d_line_active,
        )
    elif int(mask_owned.row_count) == int(candidate_owned.row_count):
        d_mask_rows = d_candidate_rows[cp.where(d_line_active, d_line_source_rows, cp.int64(0))]
        line_masks = mask_owned.device_take_capacity(d_mask_rows, d_line_active)
    else:
        return None

    intersection_native = linestring_polygon_intersection(
        line_parts.geometry,
        line_masks,
    )
    if intersection_native is None:
        return None
    if intersection_native.owned is not None:
        concrete_intersections = (
            (
                intersection_native.owned,
                cp.arange(intersection_native.row_count, dtype=cp.int64),
            ),
        )
    elif intersection_native.composition is not None:
        concrete_intersections = tuple(
            (part.geometry.owned, cp.asarray(part.output_rows, dtype=cp.int64))
            for part in intersection_native.composition.parts
            if part.geometry.owned is not None
        )
    else:
        return None

    packed_parts = []
    point_part_capacities = []
    for intersections, d_composition_rows in concrete_intersections:
        line_intersections = _explode_lineal_rows_to_line_capacity_gpu(intersections)
        if line_intersections is not None:
            d_intersection_rows = cp.asarray(
                line_intersections.source_rows,
                dtype=cp.int64,
            )
            d_pair_rows = d_composition_rows[d_intersection_rows]
            d_output_rows = d_line_source_rows[d_pair_rows]
            line_partitions = _clip_partition_degenerate_line_part_capacity_device(
                line_intersections,
                d_output_rows,
            )
            if line_partitions is None:
                return None
            (
                line_intersections,
                d_output_rows,
                degenerate_points,
                d_degenerate_output_rows,
            ) = line_partitions
            packed_line = _clip_pack_line_boundary_part_capacity_device(
                line_intersections,
                d_output_rows,
                output_row_count=output_row_count,
            )
            if packed_line is not None:
                packed_parts.append(packed_line)
            point_part_capacities.append(
                (degenerate_points, d_degenerate_output_rows),
            )

        point_intersections = _explode_point_rows_to_point_capacity_gpu(intersections)
        if point_intersections is not None:
            d_intersection_rows = cp.asarray(
                point_intersections.source_rows,
                dtype=cp.int64,
            )
            d_pair_rows = d_composition_rows[d_intersection_rows]
            d_output_rows = d_line_source_rows[d_pair_rows]
            point_part_capacities.append(
                (point_intersections, d_output_rows),
            )
    combined_points = _clip_concat_point_part_capacities_device(
        point_part_capacities,
    )
    if combined_points is not None:
        point_intersections, d_output_rows = combined_points
        packed_point = _clip_pack_point_boundary_part_capacity_device(
            point_intersections,
            d_output_rows,
            output_row_count=output_row_count,
        )
        if packed_point is not None:
            packed_parts.append(packed_point)
    return tuple((*specialized_parts, *packed_parts))


def _clip_point_contact_sliver_polygons_device(
    candidate_owned,
    mask_owned,
    boundary_owned,
    d_boundary_output_rows,
):
    """Build collapsed polygon rows for GEOS-compatible point-contact slivers.

    Scalar polygon-mask clips can produce a public GEOS Polygon with zero area
    when a mask vertex lies infinitesimally inside an axis-aligned source edge.
    The boundary intersection carrier correctly sees the point contact; this
    helper keeps the compatibility repair native by expanding only those proven
    vertex/edge contacts into collapsed polygon rings.
    """
    if (
        candidate_owned.residency is not Residency.DEVICE
        or mask_owned.residency is not Residency.DEVICE
        or boundary_owned.residency is not Residency.DEVICE
        or not has_gpu_runtime()
    ):
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by residency
        return None

    from vibespatial.constructive.point import _build_device_backed_polygon_output
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS, device_mask_owned_capacity
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_single_ring_polygon_mask,
        device_trusted_rectangle_bounds_matrix,
    )
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )

    row_count = int(boundary_owned.row_count)
    if row_count == 0:
        return None
    d_boundary_output_rows = cp.asarray(d_boundary_output_rows, dtype=cp.int64)
    if int(d_boundary_output_rows.size) != row_count:
        return None

    boundary_state = boundary_owned._ensure_device_state(preserve_indexed_view=True)
    point_buffer = boundary_state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        return None
    d_tags = cp.asarray(boundary_state.tags)
    d_validity = cp.asarray(boundary_state.validity, dtype=cp.bool_)
    point_physical_rows = max(int(point_buffer.geometry_offsets.size) - 1, 0)
    if point_physical_rows == 0:
        return None
    d_point_family_rows = cp.asarray(
        boundary_state.family_row_offsets,
        dtype=cp.int32,
    )
    d_point_active = (
        d_validity
        & (d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.POINT]))
        & (d_point_family_rows >= 0)
        & (d_point_family_rows < point_physical_rows)
    )
    d_safe_point_family_rows = cp.clip(
        d_point_family_rows,
        cp.int32(0),
        cp.int32(point_physical_rows - 1),
    ).astype(cp.int64, copy=False)

    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    if int(mask_owned.row_count) != 1:
        return None
    mask_polygon = mask_state.families.get(GeometryFamily.POLYGON)
    if mask_polygon is None or mask_polygon.ring_offsets is None:
        return None
    d_mask_single_ring = device_single_ring_polygon_mask(mask_owned)
    if d_mask_single_ring is None:
        return None
    d_mask_x = cp.asarray(mask_polygon.x, dtype=cp.float64)
    d_mask_y = cp.asarray(mask_polygon.y, dtype=cp.float64)
    d_mask_geometry_offsets = cp.asarray(
        mask_polygon.geometry_offsets,
        dtype=cp.int32,
    )
    d_mask_ring_offsets = cp.asarray(mask_polygon.ring_offsets, dtype=cp.int32)
    d_mask_family_rows = cp.asarray(mask_state.family_row_offsets, dtype=cp.int32)

    d_point_coord_rows = cp.asarray(
        point_buffer.geometry_offsets,
        dtype=cp.int32,
    )[d_safe_point_family_rows]
    d_px = cp.asarray(point_buffer.x, dtype=cp.float64)[d_point_coord_rows]
    d_py = cp.asarray(point_buffer.y, dtype=cp.float64)[d_point_coord_rows]

    if candidate_owned.row_count <= 0:
        return None
    d_candidate_rows = d_boundary_output_rows
    d_candidate_active = (d_candidate_rows >= 0) & (d_candidate_rows < candidate_owned.row_count)
    d_point_active &= d_candidate_active
    d_safe_candidate_rows = cp.clip(
        d_candidate_rows,
        cp.int64(0),
        cp.int64(candidate_owned.row_count - 1),
    )
    d_rect_bounds = device_trusted_rectangle_bounds_matrix(candidate_owned)
    if d_rect_bounds is not None:
        d_row_bounds = cp.asarray(d_rect_bounds, dtype=cp.float64)[d_safe_candidate_rows]
    else:
        d_bounds = cp.asarray(
            compute_geometry_bounds_device(candidate_owned, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(candidate_owned.row_count, 4)
        d_row_bounds = d_bounds[d_safe_candidate_rows]
    d_out_x = cp.empty(row_count * 4, dtype=cp.float64)
    d_out_y = cp.empty(row_count * 4, dtype=cp.float64)
    d_out_valid = cp.zeros(row_count, dtype=cp.uint8)
    d_row_bounds = cp.ascontiguousarray(
        cp.asarray(d_row_bounds, dtype=cp.float64).reshape(row_count, 4),
    )
    runtime = get_cuda_runtime()
    kernel = _clip_point_contact_sliver_kernels()["point_contact_sliver_rows_kernel"]
    ptr = runtime.pointer
    params = (
        (
            ptr(d_mask_x),
            ptr(d_mask_y),
            ptr(d_mask_geometry_offsets),
            ptr(d_mask_ring_offsets),
            ptr(d_mask_family_rows),
            ptr(d_mask_single_ring),
            ptr(d_px),
            ptr(d_py),
            ptr(d_point_active),
            ptr(d_row_bounds),
            row_count,
            float(SPATIAL_EPSILON),
            ptr(d_out_x),
            ptr(d_out_y),
            ptr(d_out_valid),
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
            KERNEL_PARAM_I64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    capacity_owned = _build_device_backed_polygon_output(
        d_out_x,
        d_out_y,
        row_count=row_count,
        bounds=None,
        verts_per_ring=4,
    )
    d_valid_mask = d_out_valid.astype(cp.bool_, copy=False)
    replacement_owned = device_mask_owned_capacity(capacity_owned, d_valid_mask)
    return replacement_owned, d_valid_mask


def _clip_boundary_row_matches_area(assembled_geom, area_geom) -> bool:
    """Return True when area-only output already matches public clip semantics.

    Topological equality is too weak here: degenerate polygon outputs can be
    point-set-equal to lower-dimensional artifacts, and near-equal polygonal
    rows can still drift numerically from the exact host boundary result. Keep
    the cheaper area result only when it is the same public geometry after
    normalization.
    """
    if assembled_geom is None and area_geom is None:
        return True
    if assembled_geom is None or area_geom is None:
        return False
    if getattr(assembled_geom, "geom_type", None) != getattr(area_geom, "geom_type", None):
        return False
    return bool(
        shapely.equals_exact(
            assembled_geom,
            area_geom,
            tolerance=0.0,
            normalize=True,
        )
    )


def _clip_polygon_partition_with_rectangle_mask(
    partition,
    rectangle_bounds: tuple[float, float, float, float],
    *,
    keep_geom_type_only: bool = False,
):
    """Clip polygon rows to a rectangle mask while preserving sliver leftovers.

    Rows fully inside the rectangle are pass-through. Only rows that cross or
    touch the rectangle boundary require exact area intersection to
    preserve lower-dimensional public clip semantics.
    """
    partition = _promote_geometry_backing_to_device(
        partition,
        reason="clip rectangle-mask polygon partition selected GPU-native constructive path",
    )
    xmin, ymin, xmax, ymax = rectangle_bounds
    from vibespatial.api.tools.overlay import (
        _assemble_polygon_intersection_rows_with_lower_dim,
    )
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        from_shapely_geometries,
    )
    from vibespatial.runtime.residency import Residency

    rectangle_mask = box(xmin, ymin, xmax, ymax)
    source_values = (
        partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    )
    source_is_native = (
        isinstance(source_values, DeviceGeometryArray)
        or getattr(source_values, "_owned", None) is not None
    )
    source_owned = source_values.to_owned() if source_is_native else None
    if keep_geom_type_only and source_owned is None:
        source_owned = source_values.to_owned()
    if (
        not keep_geom_type_only
        and source_owned is not None
        and source_owned.residency is Residency.DEVICE
        and has_gpu_runtime()
    ):
        import cupy as cp

        mask_owned = _device_rectangle_owned_from_bounds(
            rectangle_bounds,
            residency=source_owned.residency,
        )
        native_result = _clip_homogeneous_polygon_device_candidates_native(
            partition,
            rectangle_mask,
            cp.arange(source_owned.row_count, dtype=cp.int64),
            mask_owned=mask_owned,
            clipping_by_rectangle=False,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=False,
        )
        if native_result is None:
            raise StrictNativeFallbackError(
                "device polygon-rectangle partition declined its canonical "
                "candidate-capacity executor after GPU admission"
            )
        geometry_result = (
            native_result.capacity_result.geometry
            if isinstance(native_result, NativeTabularSelection)
            else native_result.geometry
        )
        if geometry_result.row_count != source_owned.row_count:
            raise RuntimeError("device polygon-rectangle capacity result violated row alignment")
        return geometry_result
    if (
        keep_geom_type_only
        and source_owned is not None
        and source_owned.residency is Residency.DEVICE
        and has_gpu_runtime()
    ):
        capacity_owned = _clip_polygonal_rectangle_keep_geom_type_device_owned(
            source_owned,
            rectangle_bounds,
        )
        if capacity_owned is None:
            raise StrictNativeFallbackError(
                "polygonal rectangle keep_geom_type device assembly declined "
                "before host residual reconstruction"
            )
        from vibespatial.runtime.dispatch import record_dispatch_event

        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation="polygon_rectangle_keep_geom_type_rowset_gpu",
            reason=(
                "polygonal rectangle keep-geometry-type output selected "
                "Polygon and MultiPolygon capacity partitions on device"
            ),
            detail=(
                f"input_rows={source_owned.row_count}; "
                f"output_capacity={capacity_owned.row_count}; "
                "physical_shape=polygon_family_partition_device_capacity"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        return _geometry_values_from_owned(capacity_owned, crs=partition.crs)
    assembled = np.empty(len(partition), dtype=object)
    assembled[:] = None

    source_rowset = _clip_source_nonmissing_rowset(
        source_values,
        source_token=f"clip-rectangle-mask:{id(source_values)}",
        prefer_device=False,
    )
    nonmissing_rows = _clip_source_nonmissing_rows_for_compatibility(
        source_rowset,
        surface="vibespatial.api.tools.clip.rectangle_mask_compatibility_rows",
    )
    source_bounds = np.asarray(source_values.bounds, dtype=np.float64)
    nonmissing_bounds = source_bounds[nonmissing_rows]
    fully_inside_nonmissing = (
        (nonmissing_bounds[:, 0] >= xmin)
        & (nonmissing_bounds[:, 1] >= ymin)
        & (nonmissing_bounds[:, 2] <= xmax)
    )
    fully_inside_nonmissing &= nonmissing_bounds[:, 3] <= ymax

    inside_rows = nonmissing_rows[fully_inside_nonmissing].astype(
        np.intp,
        copy=False,
    )
    if inside_rows.size > 0 and source_owned is None:
        assembled[inside_rows] = np.asarray(
            source_values.take(inside_rows),
            dtype=object,
        )

    boundary_rows = nonmissing_rows[~fully_inside_nonmissing].astype(
        np.intp,
        copy=False,
    )
    if keep_geom_type_only:
        from vibespatial.api.tools.overlay import (
            _strip_non_polygon_collection_parts,
        )

        host_values = np.asarray(source_values, dtype=object)
        repeated_mask = np.empty(host_values.size, dtype=object)
        repeated_mask[:] = rectangle_mask
        polygonal = _strip_non_polygon_collection_parts(
            _exact_polygon_clip_boundary_rows(host_values, repeated_mask),
        )
        keep = ~(shapely.is_missing(polygonal) | shapely.is_empty(polygonal))
        keep &= np.asarray(shapely.area(polygonal), dtype=np.float64) > 0.0
        polygonal[~keep] = None
        return _as_geometry_values(polygonal, crs=partition.crs)

    if boundary_rows.size > 0:
        boundary_index = partition.index.take(boundary_rows)
        boundary_values = source_values.take(boundary_rows)
        boundary_bounds = source_bounds[boundary_rows]
        rectangle_boundary = _exact_rectangle_clip_boundary_rows(
            boundary_values,
            boundary_bounds,
            rectangle_bounds,
        )
        if rectangle_boundary is not None:
            if source_owned is not None:
                result_owned = build_null_owned_array(
                    len(partition),
                    residency=source_owned.residency,
                )
                if inside_rows.size > 0:
                    result_owned = concat_owned_scatter(
                        result_owned,
                        source_owned.take(inside_rows),
                        inside_rows,
                    )
                replacement_owned = from_shapely_geometries(
                    rectangle_boundary.tolist(),
                    residency=result_owned.residency,
                )
                result_owned = concat_owned_scatter(
                    result_owned,
                    replacement_owned,
                    boundary_rows,
                )
                _seed_rectangle_clip_validity_cache_if_safe(
                    result_owned,
                    source_values,
                )
                result_values = (
                    DeviceGeometryArray._from_owned(result_owned, crs=partition.crs)
                    if result_owned.residency is Residency.DEVICE
                    else GeometryArray.from_owned(result_owned, crs=partition.crs)
                )
                return result_values

            assembled[boundary_rows] = rectangle_boundary
            return _as_geometry_values(assembled, crs=partition.crs)

        left_pairs = GeoSeries(
            boundary_values,
            index=boundary_index,
            crs=partition.crs,
        )
        boundary_owned = boundary_values.to_owned()
        area_owned = _clip_polygon_rectangle_area_intersection_owned(
            boundary_owned,
            rectangle_bounds,
        )
        area_pairs = GeoSeries(
            _geometry_values_from_owned(area_owned, crs=partition.crs),
            index=boundary_index,
            crs=partition.crs,
        )
        repeated_mask = np.empty(boundary_rows.size, dtype=object)
        repeated_mask[:] = rectangle_mask
        right_pairs = GeoSeries(
            repeated_mask,
            index=boundary_index,
            crs=partition.crs,
        )
        assembled[boundary_rows] = np.asarray(
            _assemble_polygon_intersection_rows_with_lower_dim(
                left_pairs,
                right_pairs,
                area_pairs,
            ),
            dtype=object,
        )
        changed_boundary_geoms = assembled[boundary_rows]
        contains_collection = any(
            geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
            for geom in changed_boundary_geoms
        )
        if not contains_collection and source_owned is not None:
            result_owned = concat_owned_scatter(
                source_owned,
                area_owned,
                boundary_rows,
            )
            area_objects = np.asarray(area_pairs, dtype=object)
            changed_mask = np.ones(boundary_rows.size, dtype=bool)
            for row_index, (assembled_geom, area_geom) in enumerate(
                zip(assembled[boundary_rows], area_objects, strict=True)
            ):
                if assembled_geom is None and area_geom is None:
                    changed_mask[row_index] = False
                    continue
                if assembled_geom is None or area_geom is None:
                    continue
                if bool(shapely.equals(assembled_geom, area_geom)):
                    changed_mask[row_index] = False
            changed_rows = boundary_rows[changed_mask]
            if changed_rows.size > 0:
                replacement_owned = from_shapely_geometries(
                    assembled[changed_rows].tolist(),
                    residency=result_owned.residency,
                )
                result_owned = concat_owned_scatter(
                    result_owned,
                    replacement_owned,
                    changed_rows,
                )

            result_values = (
                DeviceGeometryArray._from_owned(result_owned, crs=partition.crs)
                if result_owned.residency is Residency.DEVICE
                else GeometryArray.from_owned(result_owned, crs=partition.crs)
            )
            return result_values

    if source_owned is not None and boundary_rows.size == 0:
        result_owned = build_null_owned_array(
            len(partition),
            residency=source_owned.residency,
        )
        if inside_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                source_owned.take(inside_rows),
                inside_rows,
            )
        _seed_rectangle_clip_validity_cache_if_safe(result_owned, source_values)
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    if source_owned is not None and inside_rows.size > 0:
        assembled[inside_rows] = np.asarray(
            source_owned.take(inside_rows).to_shapely(),
            dtype=object,
        )
    return _as_geometry_values(assembled, crs=partition.crs)


def _clip_polygon_partition_with_polygon_mask(
    partition,
    mask,
    *,
    keep_geom_type_only: bool = False,
):
    """Clip polygon rows to a polygon mask while preserving owned backing.

    The bulk polygon area result stays on the owned/device path. Only rows
    without positive-area output pay the boundary reconstruction cost needed
    to preserve lower-dimensional public clip semantics.
    """
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        concat_owned_scatter,
        from_shapely_geometries,
    )

    partition = _promote_geometry_backing_to_device(
        partition,
        reason="clip polygon-mask partition selected GPU-native constructive path",
    )
    source_values = (
        partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    )
    left_owned = source_values.to_owned()
    mask_owned = from_shapely_geometries([mask], residency=left_owned.residency)

    scalar_bypass_owned = _clip_polygon_single_pair_containment_owned(
        left_owned,
        mask_owned,
    )
    if scalar_bypass_owned is not None:
        return _geometry_values_from_owned(scalar_bypass_owned, crs=partition.crs)

    # Device-backed polygon clip benefits from a cheap exact predicate refine
    # before exact intersection: bbox candidates include both false positives
    # and polygons already fully covered by the mask, and paying full exact
    # intersection for those rows dominates 10K-scale workflows.
    if has_gpu_runtime() and left_owned.residency is Residency.DEVICE:
        import cupy as cp

        native_result = _clip_homogeneous_polygon_device_candidates_native(
            partition,
            mask,
            cp.arange(left_owned.row_count, dtype=cp.int64),
            mask_owned=mask_owned,
            clipping_by_rectangle=False,
            rectangle_bounds=None,
            keep_geom_type=keep_geom_type_only,
        )
        if native_result is None:
            raise StrictNativeFallbackError(
                "device polygon-mask partition declined its canonical "
                "candidate-capacity executor after GPU admission"
            )
        geometry_result = (
            native_result.capacity_result.geometry
            if isinstance(native_result, NativeTabularSelection)
            else native_result.geometry
        )
        if geometry_result.row_count != left_owned.row_count:
            raise RuntimeError("device polygon-mask capacity result violated row alignment")
        return geometry_result
    source_rowset = _clip_source_nonmissing_rowset(
        source_values,
        source_token=f"clip-polygon-mask:{id(left_owned)}",
        prefer_device=False,
    )
    source_all_nonmissing = source_rowset.identity and len(source_rowset) == left_owned.row_count
    nonmissing_rows = _clip_source_nonmissing_rows_for_compatibility(
        source_rowset,
        surface="vibespatial.api.tools.clip.polygon_mask_compatibility_rows",
    )
    record_fallback_event(
        surface="geopandas.clip",
        reason="polygon-mask clip used CPU compatibility boundary without GPU runtime",
        detail=f"rows={left_owned.row_count}",
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
        pipeline="_clip_polygon_partition_with_polygon_mask",
        d2h_transfer=False,
    )
    area_owned = _host_polygonal_area_intersection_owned(
        left_owned,
        mask_owned,
    )

    area_values = _geometry_values_from_owned(area_owned, crs=partition.crs)
    area_nonempty = _owned_nonempty_polygon_mask(area_owned)
    positive_nonmissing = area_nonempty[nonmissing_rows]
    positive_rows = nonmissing_rows[positive_nonmissing].astype(
        np.int64,
        copy=False,
    )
    if keep_geom_type_only:
        if positive_rows.size == 0:
            return _geometry_values_from_owned(
                build_null_owned_array(
                    len(partition),
                    residency=left_owned.residency,
                ),
                crs=partition.crs,
            )
        result_owned = concat_owned_scatter(
            build_null_owned_array(
                len(partition),
                residency=left_owned.residency,
            ),
            area_owned.take(positive_rows),
            positive_rows,
        )
        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    boundary_rows = nonmissing_rows[~positive_nonmissing].astype(
        np.intp,
        copy=False,
    )
    if boundary_rows.size == 0:
        return area_values

    touch_boundary_mask = _clip_polygon_boundary_touch_mask(
        source_values,
        boundary_rows,
        mask=mask,
    )
    boundary_rows = boundary_rows[touch_boundary_mask]
    if boundary_rows.size == 0:
        if positive_rows.size == 0:
            return _geometry_values_from_owned(
                build_null_owned_array(
                    len(partition),
                    residency=left_owned.residency,
                ),
                crs=partition.crs,
            )
        sparse_result = _build_sparse_owned_clip_output(
            partition_crs=partition.crs,
            left_owned=left_owned,
            inside_rows=np.empty(0, dtype=np.intp),
            exact_area_owned=area_owned,
            positive_local_rows=positive_rows.astype(np.intp, copy=False),
            positive_rows=positive_rows.astype(np.intp, copy=False),
        )
        if sparse_result.local_rows.size == len(partition) and source_all_nonmissing:
            return sparse_result.geometry_values
        return sparse_result

    left_pairs = _take_geometry_object_values(source_values, boundary_rows)
    area_objects = _take_geometry_object_values(area_values, boundary_rows)
    repeated_mask = np.empty(boundary_rows.size, dtype=object)
    repeated_mask[:] = mask
    assembled_boundary = _exact_polygon_clip_boundary_rows(
        left_pairs,
        repeated_mask,
    )

    contains_collection = any(
        geom is not None and getattr(geom, "geom_type", None) == "GeometryCollection"
        for geom in assembled_boundary
    )
    if not contains_collection:
        result_owned = build_null_owned_array(
            len(partition),
            residency=left_owned.residency,
        )
        if positive_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                area_owned.take(positive_rows),
                positive_rows,
            )

        preserve_mask = np.zeros(boundary_rows.size, dtype=bool)
        changed_mask = np.zeros(boundary_rows.size, dtype=bool)
        for row_index, (assembled_geom, area_geom) in enumerate(
            zip(assembled_boundary, area_objects, strict=True)
        ):
            if _clip_boundary_row_matches_area(assembled_geom, area_geom):
                preserve_mask[row_index] = area_geom is not None and not getattr(
                    area_geom, "is_empty", False
                )
            else:
                changed_mask[row_index] = True
        preserved_rows = boundary_rows[preserve_mask]
        if preserved_rows.size > 0:
            result_owned = concat_owned_scatter(
                result_owned,
                area_owned.take(preserved_rows),
                preserved_rows,
            )
        changed_rows = boundary_rows[changed_mask]
        if changed_rows.size > 0:
            replacement_owned = from_shapely_geometries(
                assembled_boundary[changed_mask].tolist(),
                residency=result_owned.residency,
            )
            result_owned = concat_owned_scatter(
                result_owned,
                replacement_owned,
                changed_rows,
            )

        return _geometry_values_from_owned(result_owned, crs=partition.crs)

    assembled = np.empty(len(partition), dtype=object)
    assembled[:] = None
    if positive_rows.size > 0:
        assembled[positive_rows] = np.asarray(area_values.take(positive_rows), dtype=object)
    assembled[boundary_rows] = assembled_boundary
    return _as_geometry_values(assembled, crs=partition.crs)


def _clip_complex_polygon_partition_with_rectangle_mask(
    partition,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Preserve area and lower-dimensional remnants for complex rectangle clip rows."""
    partition = _promote_geometry_backing_to_device(
        partition,
        reason=(
            "clip complex rectangle-mask polygon partition selected GPU-native constructive path"
        ),
    )
    rectangle_mask = box(*rectangle_bounds)
    source_values = (
        partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    )
    source_owned = getattr(source_values, "_owned", None)
    if (
        source_owned is not None
        and source_owned.residency is Residency.DEVICE
        and has_gpu_runtime()
    ):
        import cupy as cp

        native_result = _clip_mixed_device_candidates_native(
            partition,
            rectangle_mask,
            cp.arange(source_owned.row_count, dtype=cp.int64),
            clipping_by_rectangle=True,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=False,
        )
        if native_result is None:
            raise StrictNativeFallbackError(
                "device complex polygon-rectangle partition declined its "
                "canonical candidate-capacity executor after GPU admission"
            )
        geometry_result = (
            native_result.capacity_result.geometry
            if isinstance(native_result, NativeTabularSelection)
            else native_result.geometry
        )
        if geometry_result.row_count != source_owned.row_count:
            raise RuntimeError(
                "device complex polygon-rectangle capacity result violated row alignment"
            )
        return geometry_result
    area_values = _clip_polygon_partition_with_polygon_mask(partition, rectangle_mask)
    area_objects = np.asarray(area_values, dtype=object)
    source_objects = np.asarray(source_values, dtype=object)
    rectangle_boundary = shapely.boundary(rectangle_mask)
    boundary_objects = np.asarray(
        shapely.intersection(shapely.boundary(source_objects), rectangle_boundary),
        dtype=object,
    )
    combined = np.empty(len(partition), dtype=object)

    for row_index, (area_geom, boundary_geom) in enumerate(
        zip(area_objects, boundary_objects, strict=True)
    ):
        if area_geom is None or getattr(area_geom, "is_empty", False):
            combined[row_index] = boundary_geom
            continue
        if boundary_geom is None or getattr(boundary_geom, "is_empty", False):
            combined[row_index] = area_geom
            continue
        combined[row_index] = GeometryCollection([area_geom, boundary_geom])

    return _as_geometry_values(combined, crs=partition.crs)


def _build_clip_partition_result(
    source,
    row_positions,
    geometry_values,
    *,
    row_positions_device=None,
):
    """Create a native row-preserving clip fragment without frame assembly."""
    rowset = None
    if isinstance(geometry_values, _ClipPartitionOutput):
        local_rows = np.asarray(geometry_values.local_rows, dtype=np.intp)
        rowset = _clip_source_rowset_for_positions(
            source,
            row_positions,
            local_rows,
            device_row_positions=row_positions_device,
            device_local_rows=geometry_values.local_rows_device,
        )
        row_positions = np.asarray(row_positions, dtype=np.intp)[local_rows]
        geometry_values = geometry_values.geometry_values
    else:
        rowset = _clip_source_rowset_for_positions(
            source,
            row_positions,
            device_row_positions=row_positions_device,
        )
    return _clip_native_part(
        source,
        row_positions,
        (
            geometry_values
            if isinstance(geometry_values, GeometryNativeResult)
            else _as_geometry_values(
                geometry_values,
                crs=source.crs,
            )
        ),
        rowset=rowset,
    )


def _clip_point_polygon_mask_expression(owned, mask):
    if owned is not None and owned.row_count == 0:
        return None
    if owned is not None and owned.residency is Residency.DEVICE and has_gpu_runtime():
        from vibespatial.geometry.owned import from_shapely_geometries
        from vibespatial.predicates.binary import binary_predicate_expression

        mask_owned = from_shapely_geometries([mask], residency=owned.residency)
        expression = binary_predicate_expression(
            "intersects",
            owned,
            mask_owned,
            source_token=f"clip-point-polygon-mask:{id(owned)}",
            operation="clip.point_polygon_mask.intersects",
        )
        return expression
    return None


def _clip_point_owned_with_polygon_mask_device(
    owned,
    mask,
    *,
    crs,
) -> _ClipDevicePartitionOutput | None:
    expression = _clip_point_polygon_mask_expression(owned, mask)
    if expression is None:
        return None
    selection = expression.equal_to_selection(True)
    return _ClipDevicePartitionOutput(
        geometry=GeometryNativeResult.from_owned(owned, crs=crs),
        local_rows_device=selection.positions,
        selection=selection,
    )


def _clip_point_owned_with_polygon_mask(owned, mask, *, crs) -> _ClipPartitionOutput | None:
    expression = _clip_point_polygon_mask_expression(owned, mask)
    if expression is not None:
        hit_rowset = expression.equal_to(True)
        keep_rows = (
            np.empty(0, dtype=np.intp)
            if len(hit_rowset) == 0
            else hit_rowset.to_host_positions(
                surface="vibespatial.api.tools.clip.point_polygon_mask_rows",
                strict_disallowed=False,
            ).astype(np.intp, copy=False)
        )
        if hit_rowset.is_device:
            kept_owned = owned.device_take(
                hit_rowset.positions,
                host_indices_for_sizing=np.asarray(keep_rows, dtype=np.int64),
            )
            return _ClipPartitionOutput(
                geometry_values=DeviceGeometryArray._from_owned(
                    kept_owned,
                    crs=crs,
                ),
                local_rows=keep_rows,
                local_rows_device=hit_rowset.positions,
            )
    return None


def _clip_point_partition_with_polygon_mask(partition, mask):
    """Filter point candidates by exact polygon intersection without re-clipping."""
    geometry = partition.geometry if isinstance(partition, GeoDataFrame) else partition
    values = geometry.values
    owned = getattr(values, "_owned", None)
    native_result = _clip_point_owned_with_polygon_mask(owned, mask, crs=geometry.crs)
    if native_result is not None:
        return native_result

    keep_rows = np.flatnonzero(np.asarray(geometry.intersects(mask), dtype=bool)).astype(
        np.intp, copy=False
    )
    return _ClipPartitionOutput(
        geometry_values=values.take(keep_rows),
        local_rows=keep_rows,
    )


def _clip_homogeneous_point_candidates_native(
    source,
    mask,
    candidate_rows: np.ndarray,
    *,
    candidate_device_rows=None,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | None:
    """Clip point-only candidate rows without building a candidate frame."""
    if (
        clipping_by_rectangle
        or not isinstance(mask, Polygon | MultiPolygon)
        or candidate_rows.size == 0
        or not has_gpu_runtime()
    ):
        return None
    if not _clip_family_masks(source).all_point:
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None and hasattr(values, "to_owned"):
        owned = values.to_owned()
    if owned is None:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    rows = np.asarray(candidate_rows, dtype=np.intp)
    d_rows = (
        cp.asarray(candidate_device_rows, dtype=cp.int64)
        if candidate_device_rows is not None
        else cp.asarray(rows, dtype=cp.int64)
    )

    if rows.size == owned.row_count and np.array_equal(
        rows, np.arange(owned.row_count, dtype=rows.dtype)
    ):
        candidate_owned = owned
    else:
        candidate_owned = owned.device_take(d_rows)

    geometry_name = (
        source._geometry_column_name
        if hasattr(source, "_geometry_column_name")
        else getattr(source, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(source, geometry_name)
    if (
        source_state is not None
        and source_state.geometry_name == geometry_name
        and source_state.index_plan.kind
        in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        device_output = _clip_point_owned_with_polygon_mask_device(
            candidate_owned,
            mask,
            crs=geometry.crs,
        )
        if device_output is not None:
            source_rowset = _clip_source_rowset_for_positions(
                source,
                rows,
                device_row_positions=d_rows,
            )
            if source_rowset is None:
                return None
            capacity_result = _clip_native_tabular_result_from_rowset(
                source,
                geometry_name=geometry_name,
                geometry=device_output.geometry,
                rowset=source_rowset,
                keep_geom_type=keep_geom_type,
            )
            if capacity_result is not None and device_output.selection is not None:
                return NativeTabularSelection(
                    capacity_result=capacity_result,
                    selection=device_output.selection,
                )

    geometry_values = _clip_point_owned_with_polygon_mask(
        candidate_owned,
        mask,
        crs=geometry.crs,
    )
    if geometry_values is None:
        return None

    parts_tuple = (
        _build_clip_partition_result(
            source,
            rows,
            geometry_values,
            row_positions_device=d_rows,
        ),
    )
    ordered_rows = rows.astype(np.intp, copy=False)
    return _clip_constructive_parts_to_native_tabular_result(
        source=source,
        parts=parts_tuple,
        ordered_row_positions=ordered_rows,
        clipping_by_rectangle=False,
        has_non_point_candidates=False,
        keep_geom_type=keep_geom_type,
        spatial_materializer=lambda: ClipNativeResult(
            source=source,
            parts=parts_tuple,
            ordered_index=geometry.index.take(ordered_rows),
            ordered_row_positions=ordered_rows,
            clipping_by_rectangle=False,
            has_non_point_candidates=False,
            keep_geom_type=keep_geom_type,
        ).to_spatial(),
    )


def _clip_homogeneous_point_device_candidates_native(
    source,
    mask,
    candidate_device_rows,
    *,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | NativeTabularSelection | None:
    """Clip point-only device candidate rows without first exporting row ids."""
    if not isinstance(mask, Polygon | MultiPolygon) or not has_gpu_runtime():
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.api._native_rowset import NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if hasattr(source, "_geometry_column_name")
        else getattr(source, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(source, geometry_name)
    if (
        source_state is None
        or source_state.geometry_name != geometry_name
        or source_state.index_plan.kind
        not in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    candidate_selection = (
        candidate_device_rows
        if isinstance(candidate_device_rows, NativeDeviceSelection)
        else None
    )
    if candidate_selection is not None:
        if candidate_selection.source_row_count != source_state.row_count:
            return None
        d_rows = cp.asarray(
            candidate_selection.partition_capacity_positions(),
            dtype=cp.int64,
        )
        d_candidate_active = candidate_selection.active_capacity_mask()
    else:
        d_rows = cp.asarray(candidate_device_rows, dtype=cp.int64)
        d_candidate_active = cp.ones(int(d_rows.size), dtype=cp.bool_)
    if int(d_rows.size) == 0:
        from vibespatial.geometry.owned import build_null_owned_array

        empty_owned = build_null_owned_array(0, residency=owned.residency)
        empty_rowset = NativeRowSet.from_positions(
            cp.empty(0, dtype=cp.int64),
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=False,
        )
        return _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=GeometryNativeResult.from_owned(empty_owned, crs=geometry.crs),
            rowset=empty_rowset,
            keep_geom_type=keep_geom_type,
        )

    if candidate_selection is not None:
        from vibespatial.geometry.owned import device_take_owned_capacity_selection

        candidate_owned = device_take_owned_capacity_selection(
            owned,
            candidate_selection,
        )
    else:
        candidate_owned = owned.device_take(d_rows)
    from vibespatial.geometry.buffers import GeometryFamily

    if not _owned_active_family_subset(
        candidate_owned,
        {GeometryFamily.POINT},
    ):
        return None
    if clipping_by_rectangle:
        geometry_result = GeometryNativeResult.from_owned(
            candidate_owned,
            crs=geometry.crs,
        )
        source_rowset = NativeRowSet.from_positions(
            d_rows,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=False,
        )
        capacity_result = _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=geometry_result,
            rowset=source_rowset,
            keep_geom_type=keep_geom_type,
        )
        if capacity_result is None or candidate_selection is None:
            return capacity_result
        return NativeTabularSelection(
            capacity_result=capacity_result,
            selection=NativeDeviceSelection.from_mask(d_candidate_active),
        )

    expression = _clip_point_polygon_mask_expression(candidate_owned, mask)
    if expression is None:
        return None
    selection = expression.equal_to_selection(True)
    if not isinstance(selection, NativeDeviceSelection):
        return None
    d_output_active = d_candidate_active & selection.source_mask()
    source_rowset = NativeRowSet.from_positions(
        d_rows,
        source_token=source_state.lineage_token,
        source_row_count=source_state.row_count,
        ordered=True,
        unique=True,
        identity=False,
    )
    capacity_result = _clip_native_tabular_result_from_rowset(
        source,
        geometry_name=geometry_name,
        geometry=GeometryNativeResult.from_owned(candidate_owned, crs=geometry.crs),
        rowset=source_rowset,
        keep_geom_type=keep_geom_type,
    )
    if capacity_result is None:
        return None
    return NativeTabularSelection(
        capacity_result=capacity_result,
        selection=NativeDeviceSelection.from_mask(d_output_active),
    )


def _clip_homogeneous_line_rectangle_device_candidates_native(
    source,
    candidate_device_rows,
    *,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | NativeTabularSelection | None:
    """Clip line-only rectangle candidates through a capacity carrier.

    List-like rectangle clip follows ``clip_by_rect`` semantics and may drop a
    zero-length line contact. A Polygon mask is exact set intersection, so it
    must use line/polygon topology when lower-dimensional output is retained.
    """
    if rectangle_bounds is None or not has_gpu_runtime():
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.api._native_rowset import NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if hasattr(source, "_geometry_column_name")
        else getattr(source, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(source, geometry_name)
    if (
        source_state is None
        or source_state.geometry_name != geometry_name
        or source_state.index_plan.kind
        not in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    from vibespatial.constructive.clip_rect import clip_by_rect_owned
    from vibespatial.runtime.dispatch import record_dispatch_event

    candidate_selection = (
        candidate_device_rows
        if isinstance(candidate_device_rows, NativeDeviceSelection)
        else None
    )
    if candidate_selection is not None:
        if candidate_selection.source_row_count != source_state.row_count:
            return None
        d_rows = cp.asarray(
            candidate_selection.partition_capacity_positions(),
            dtype=cp.int64,
        )
        d_candidate_active = candidate_selection.active_capacity_mask()
    else:
        d_rows = cp.asarray(candidate_device_rows, dtype=cp.int64)
        d_candidate_active = cp.ones(int(d_rows.size), dtype=cp.bool_)
    if int(d_rows.size) == 0:
        return None

    if candidate_selection is not None:
        from vibespatial.geometry.owned import device_take_owned_capacity_selection

        candidate_owned = device_take_owned_capacity_selection(
            owned,
            candidate_selection,
        )
    else:
        candidate_owned = owned.device_take(d_rows)
    from vibespatial.geometry.buffers import GeometryFamily

    if not _owned_active_family_subset(
        candidate_owned,
        {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING},
    ):
        return None
    clip_result = clip_by_rect_owned(
        candidate_owned,
        *rectangle_bounds,
        dispatch_mode=ExecutionMode.GPU,
    )
    clipped_owned = clip_result.owned_result
    d_local_rows = clip_result.owned_result_rows_device
    if clipped_owned is None or d_local_rows is None:
        return None

    record_dispatch_event(
        surface="DeviceGeometryArray.clip_by_rect",
        operation="clip_by_rect",
        implementation="owned_clip_by_rect",
        reason=(
            "certified rectangle mask consumed lineal device candidates through "
            "the owned rectangle clip kernel"
        ),
        detail=f"candidate_rows={int(d_rows.size)}",
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )

    d_local_rows = cp.asarray(d_local_rows, dtype=cp.int64)
    if int(clipped_owned.row_count) != int(d_local_rows.size):
        return None

    if clip_result.owned_result_is_row_capacity:
        if int(clipped_owned.row_count) != int(d_rows.size):
            return None
        device_state = clipped_owned.device_state
        if device_state is None:
            return None
        d_fast_keep = cp.asarray(device_state.validity, dtype=cp.bool_)
        geometry_result = GeometryNativeResult.from_owned(
            clipped_owned,
            crs=geometry.crs,
        )
        d_output_keep = d_fast_keep & d_candidate_active
        if not clipping_by_rectangle and not keep_geom_type:
            from vibespatial.api._native_results import (
                NativeGeometryComposition,
                NativeGeometryCompositionPart,
            )
            from vibespatial.constructive.line_polygon_difference import (
                lineal_polygonal_constructive_topology_gpu,
            )
            from vibespatial.geometry.owned import tile_single_row

            d_unresolved = d_candidate_active & ~d_fast_keep
            unresolved_owned = candidate_owned._apply_row_activity(d_unresolved)
            rectangle_owned = _device_rectangle_owned_from_bounds(
                rectangle_bounds,
                residency=owned.residency,
            )
            if rectangle_owned is None:
                return None
            exact_geometry = lineal_polygonal_constructive_topology_gpu(
                unresolved_owned,
                tile_single_row(rectangle_owned, int(d_rows.size)),
                operation="intersection",
                dispatch_mode=ExecutionMode.GPU,
                crs=geometry.crs,
            )
            if exact_geometry is None:
                return None
            if not isinstance(exact_geometry, GeometryNativeResult):
                exact_geometry = GeometryNativeResult.from_owned(
                    exact_geometry,
                    crs=geometry.crs,
                )
            exact_geometry = exact_geometry.mask_capacity(d_unresolved)
            d_exact_keep = exact_geometry.valid_nonempty_mask_device()
            if d_exact_keep is None:
                return None
            d_output_keep = d_output_keep | (
                d_candidate_active & cp.asarray(d_exact_keep, dtype=cp.bool_)
            )
            d_capacity_rows = cp.arange(int(d_rows.size), dtype=cp.int64)
            composition_parts = [
                NativeGeometryCompositionPart(
                    geometry=geometry_result.mask_capacity(d_fast_keep),
                    output_rows=d_capacity_rows,
                )
            ]
            if exact_geometry.composition is None:
                composition_parts.append(
                    NativeGeometryCompositionPart(
                        geometry=exact_geometry,
                        output_rows=d_capacity_rows,
                    )
                )
            else:
                composition_parts.extend(exact_geometry.composition.parts)
            geometry_result = GeometryNativeResult.from_composition(
                NativeGeometryComposition(
                    parts=tuple(composition_parts),
                    row_count=int(d_rows.size),
                    crs=geometry.crs,
                    trusted_all_ogc_valid=True,
                ),
                crs=geometry.crs,
            )

        selection = NativeDeviceSelection.from_mask(d_output_keep)
        source_rowset = NativeRowSet.from_positions(
            d_rows,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=_clip_source_rows_identity_hint(d_rows, source_state.row_count),
        )
        capacity_result = _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=geometry_result,
            rowset=source_rowset,
            keep_geom_type=keep_geom_type,
        )
        if capacity_result is None:
            return None
        return NativeTabularSelection(
            capacity_result=capacity_result,
            selection=selection,
        )

    geometry_result = GeometryNativeResult.from_owned(clipped_owned, crs=geometry.crs)
    d_source_rows = d_rows[d_local_rows]
    source_rowset = NativeRowSet.from_positions(
        d_source_rows,
        source_token=source_state.lineage_token,
        source_row_count=source_state.row_count,
        ordered=True,
        unique=True,
        identity=_clip_source_rows_identity_hint(d_source_rows, source_state.row_count),
    )
    capacity_result = _clip_native_tabular_result_from_rowset(
        source,
        geometry_name=geometry_name,
        geometry=geometry_result,
        rowset=source_rowset,
        keep_geom_type=keep_geom_type,
    )
    if capacity_result is None or candidate_selection is None:
        return capacity_result
    return NativeTabularSelection(
        capacity_result=capacity_result,
        selection=NativeDeviceSelection.from_mask(
            d_candidate_active[d_local_rows],
        ),
    )


def _clip_homogeneous_polygon_candidates_native(
    source,
    mask,
    candidate_rows: np.ndarray,
    *,
    candidate_device_rows=None,
    mask_owned=None,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | None:
    """Clip polygon-only candidate rows without building a candidate frame."""
    if (
        clipping_by_rectangle
        or rectangle_bounds is not None
        or (mask_owned is None and not isinstance(mask, Polygon | MultiPolygon))
        or candidate_rows.size == 0
        or not has_gpu_runtime()
    ):
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None and hasattr(values, "to_owned"):
        owned = values.to_owned()
    if owned is None:
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    rows = np.asarray(candidate_rows, dtype=np.intp)
    d_rows = (
        cp.asarray(candidate_device_rows, dtype=cp.int64)
        if candidate_device_rows is not None
        else cp.asarray(rows, dtype=cp.int64)
    )
    if owned.residency is not Residency.DEVICE:
        promoted_source = _promote_geometry_backing_to_device(
            source,
            reason=(
                "clip polygon host-candidate rowset promoted source geometry "
                "for native device constructive assembly"
            ),
        )
        promoted_geometry = (
            promoted_source.geometry
            if isinstance(promoted_source, GeoDataFrame)
            else promoted_source
        )
        promoted_owned = getattr(promoted_geometry.values, "_owned", None)
        if promoted_owned is None or promoted_owned.residency is not Residency.DEVICE:
            return None
        return _clip_homogeneous_polygon_device_candidates_native(
            promoted_source,
            mask,
            d_rows,
            mask_owned=mask_owned,
            clipping_by_rectangle=clipping_by_rectangle,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=keep_geom_type,
        )

    return _clip_homogeneous_polygon_device_candidates_native(
        source,
        mask,
        d_rows,
        mask_owned=mask_owned,
        clipping_by_rectangle=clipping_by_rectangle,
        rectangle_bounds=rectangle_bounds,
        keep_geom_type=keep_geom_type,
    )


def _clip_homogeneous_polygon_device_candidates_native(
    source,
    mask,
    candidate_device_rows,
    *,
    mask_owned=None,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | NativeTabularSelection | None:
    """Clip polygon-only device candidate rows without first exporting row ids."""
    if (
        (clipping_by_rectangle and not (keep_geom_type and rectangle_bounds is not None))
        or (
            rectangle_bounds is None
            and mask_owned is None
            and not isinstance(mask, Polygon | MultiPolygon)
        )
        or not has_gpu_runtime()
    ):
        return None

    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if hasattr(source, "_geometry_column_name")
        else getattr(source, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(source, geometry_name)
    if (
        source_state is None
        or source_state.geometry_name != geometry_name
        or source_state.index_plan.kind
        not in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
        return None

    candidate_selection = (
        candidate_device_rows if isinstance(candidate_device_rows, NativeDeviceSelection) else None
    )
    if candidate_selection is not None:
        if candidate_selection.source_row_count != source_state.row_count:
            return None
        d_rows = cp.asarray(
            candidate_selection.partition_capacity_positions(),
            dtype=cp.int64,
        )
        d_candidate_active = candidate_selection.active_capacity_mask()
    else:
        d_rows = cp.asarray(candidate_device_rows, dtype=cp.int64)
        d_candidate_active = cp.ones(int(d_rows.size), dtype=cp.bool_)
    candidate_count = int(d_rows.size)
    if candidate_count == 0:
        return None
    candidate_owned = None

    def _ensure_candidate_owned():
        nonlocal candidate_owned
        if candidate_owned is None:
            if candidate_selection is not None:
                from vibespatial.geometry.owned import (
                    device_take_owned_capacity_selection,
                )

                candidate_owned = device_take_owned_capacity_selection(
                    owned,
                    candidate_selection,
                )
            else:
                candidate_owned = owned._device_indexed_take(
                    d_rows,
                    assume_unique_indices=True,
                )
        return candidate_owned

    from vibespatial.geometry.buffers import GeometryFamily

    if not _owned_active_family_subset(
        _ensure_candidate_owned(),
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON},
    ):
        return None

    if clipping_by_rectangle and keep_geom_type and rectangle_bounds is not None:
        native_attributes = getattr(source_state, "attributes", None)
        if getattr(native_attributes, "device_table", None) is not None:
            result_owned = _clip_polygonal_rectangle_keep_geom_type_device_owned(
                _ensure_candidate_owned(),
                rectangle_bounds,
            )
        else:
            candidate_partition = GeoSeries(
                _geometry_values_from_owned(
                    _ensure_candidate_owned(),
                    crs=geometry.crs,
                ),
                crs=geometry.crs,
            )
            clipped_values = _clip_polygon_partition_with_rectangle_mask(
                candidate_partition,
                rectangle_bounds,
                keep_geom_type_only=True,
            )
            result_owned = getattr(clipped_values, "_owned", None)
        if (
            result_owned is None
            or result_owned.residency is not Residency.DEVICE
            or result_owned.row_count != candidate_count
        ):
            return None
        d_keep = _owned_nonempty_polygon_device_mask(result_owned)
        if d_keep is None or int(d_keep.size) != candidate_count:
            return None
        d_keep = cp.asarray(d_keep, dtype=cp.bool_) & d_candidate_active
        source_rowset = NativeRowSet.from_positions(
            d_rows,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=_clip_source_rows_identity_hint(
                d_rows,
                source_state.row_count,
            ),
        )
        capacity_result = _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=GeometryNativeResult.from_owned(result_owned, crs=geometry.crs),
            rowset=source_rowset,
            keep_geom_type=True,
        )
        if capacity_result is None:
            return None
        from vibespatial.runtime.dispatch import record_dispatch_event

        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation="polygon_device_candidate_direct_rowset_assembly_gpu",
            reason=(
                "polygon device-candidate rectangle clip assembled final owned "
                "output through the canonical polygonal area carrier"
            ),
            detail=(
                f"candidate_capacity={candidate_count}; "
                "partition_counts=device-resident; "
                f"output_capacity={result_owned.row_count}; "
                "keep_geom_type=True"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        selected_result = NativeTabularSelection(
            capacity_result=capacity_result,
            selection=NativeDeviceSelection.from_mask(
                d_keep,
                source_token=(f"{source_state.lineage_token}:clip-rectangle-polygon-capacity"),
                source_row_count=candidate_count,
            ),
        )
        source_rows = getattr(capacity_result.provenance, "source_rows", None)
        return (
            selected_result
            if source_rows is None or candidate_count <= 1
            else selected_result.sort_selected_by_int64(source_rows)
        )

    from vibespatial.geometry.owned import (
        build_null_owned_array,
        device_take_owned_capacity_selection,
        from_shapely_geometries,
    )
    from vibespatial.predicates.binary import binary_predicate_expressions
    from vibespatial.runtime.dispatch import record_dispatch_event

    if mask_owned is None and rectangle_bounds is not None:
        mask_owned = _device_rectangle_owned_from_bounds(
            rectangle_bounds,
            residency=owned.residency,
        )
        if mask_owned is None:
            return None
    if mask_owned is not None and (
        mask_owned.residency is not Residency.DEVICE or int(mask_owned.row_count) != 1
    ):
        return None

    if mask_owned is None:
        mask_owned = from_shapely_geometries([mask], residency=owned.residency)
    source_token = f"clip-polygon-device-candidates:{id(owned)}"
    predicate_expressions = _clip_polygon_single_mask_candidate_predicates_device(
        owned,
        mask_owned,
        d_rows,
        d_candidate_active=d_candidate_active,
        source_token=source_token,
        operation_prefix="clip.polygon_mask.device_candidate_rowset",
        mask_bounds_are_exact=rectangle_bounds is not None,
    )
    if predicate_expressions is None:
        predicate_expressions = binary_predicate_expressions(
            ("intersects", "covered_by"),
            _ensure_candidate_owned(),
            mask_owned,
            source_token=source_token,
            operation_prefix="clip.polygon_mask.device_candidates",
        )

    if predicate_expressions is not None:
        if candidate_owned is None:
            source_device_state = owned._ensure_device_state(
                preserve_indexed_view=True,
            )
            d_source_nonmissing = _owned_valid_nonempty_device_mask(owned)
            if d_source_nonmissing is None:
                if source_device_state.trusted_all_valid is True:
                    d_nonmissing = cp.ones(candidate_count, dtype=cp.bool_)
                else:
                    d_nonmissing = cp.asarray(source_device_state.validity)[d_rows].astype(
                        cp.bool_, copy=True
                    )
            else:
                d_nonmissing = cp.asarray(d_source_nonmissing, dtype=cp.bool_)[d_rows]
        else:
            device_state = candidate_owned._ensure_device_state(
                preserve_indexed_view=True,
            )
            d_nonmissing = _owned_valid_nonempty_device_mask(candidate_owned)
        if d_nonmissing is None:
            if device_state.trusted_all_valid is True:
                d_nonmissing = cp.ones(candidate_owned.row_count, dtype=cp.bool_)
            else:
                d_nonmissing = cp.asarray(device_state.validity).astype(
                    cp.bool_,
                    copy=True,
                )

        predicate_row_count = (
            candidate_count if candidate_owned is None else int(candidate_owned.row_count)
        )
        d_intersects = cp.asarray(
            predicate_expressions["intersects"].values,
            dtype=cp.bool_,
        )
        if int(d_intersects.size) != predicate_row_count:
            return None
        d_hit = d_intersects & d_nonmissing & d_candidate_active
        if "covered_by" in predicate_expressions:
            d_covered_by = cp.asarray(
                predicate_expressions["covered_by"].values,
                dtype=cp.bool_,
            )
            if int(d_covered_by.size) != predicate_row_count:
                return None
            d_inside_mask = d_hit & d_covered_by
        else:
            d_inside_mask = cp.zeros(predicate_row_count, dtype=cp.bool_)
        d_exact_mask = d_hit & ~d_inside_mask

        output_owned_parts = []
        output_row_parts = []
        d_output_mask = d_inside_mask.copy()
        d_local_rows = cp.arange(candidate_count, dtype=cp.int64)
        inside_selection = NativeDeviceSelection.from_mask(d_inside_mask)
        inside_owned = device_take_owned_capacity_selection(
            _ensure_candidate_owned(),
            inside_selection,
        )
        output_owned_parts.append(inside_owned)
        output_row_parts.append(
            inside_selection.gather_capacity(d_local_rows, fill_value=0).astype(
                cp.int64,
                copy=False,
            )
        )

        exact_selection = NativeDeviceSelection.from_mask(d_exact_mask)
        d_exact_active = exact_selection.active_capacity_mask()
        d_exact_output_rows = exact_selection.gather_capacity(
            d_local_rows,
            fill_value=0,
        ).astype(cp.int64, copy=False)
        exact_candidate_owned = device_take_owned_capacity_selection(
            _ensure_candidate_owned(),
            exact_selection,
        )
        exact_area_owned = _clip_polygon_area_intersection_owned(
            exact_candidate_owned,
            mask_owned,
            preserve_lower_dimensional=not keep_geom_type,
        )
        d_positive_mask = _owned_nonempty_polygon_device_mask(
            exact_area_owned,
            keep_pointlike_zero_area=not keep_geom_type,
        )
        if d_positive_mask is None:
            return None
        d_positive_mask = cp.asarray(d_positive_mask, dtype=cp.bool_) & d_exact_active
        positive_selection = NativeDeviceSelection.from_mask(d_positive_mask)
        output_owned_parts.append(
            device_take_owned_capacity_selection(
                exact_area_owned,
                positive_selection,
            )
        )
        output_row_parts.append(
            positive_selection.gather_capacity(
                d_exact_output_rows,
                fill_value=0,
            ).astype(cp.int64, copy=False)
        )
        d_output_mask |= exact_selection.source_mask(
            active_mask=d_positive_mask,
        )

        if not keep_geom_type:
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )
            from vibespatial.geometry.owned import (
                device_mask_owned_capacity,
                device_physicalize_owned_row_selection_capacity,
                device_select_owned_capacity_partitions,
            )

            topology_parts = getattr(
                exact_area_owned,
                "_polygon_intersection_lower_dimensional_parts",
                None,
            )
            direct_topology_parts = topology_parts is not None
            if direct_topology_parts:
                boundary_capacity_parts = tuple(topology_parts)
                if any(
                    int(part.row_count) != candidate_count
                    for part in boundary_capacity_parts
                ):
                    return None
                boundary_masks = []
                for boundary_part in boundary_capacity_parts:
                    d_boundary_keep = _owned_valid_nonempty_device_mask(
                        boundary_part,
                    )
                    if d_boundary_keep is None:
                        return None
                    boundary_masks.append(
                        cp.asarray(d_boundary_keep, dtype=cp.bool_)
                        & d_exact_active
                    )
                physical_boundary_parts = boundary_capacity_parts
            else:
                d_boundary_active = d_exact_active
                topology_remnants = getattr(
                    exact_area_owned,
                    "_polygon_intersection_lower_dimensional_remnant",
                    None,
                )
                if topology_remnants is not None:
                    d_topology_remnants = cp.asarray(
                        topology_remnants,
                        dtype=cp.bool_,
                    )
                    if int(d_topology_remnants.size) == candidate_count:
                        d_boundary_active &= d_topology_remnants
                boundary_capacity_parts = (
                    _clip_polygon_boundary_intersection_device_capacity_parts(
                        exact_candidate_owned,
                        mask_owned,
                        d_local_rows,
                        d_candidate_active=d_boundary_active,
                        d_polygon_area_active=d_positive_mask,
                    )
                )
                if boundary_capacity_parts is None:
                    return None
                boundary_masks = []
                for boundary_part in boundary_capacity_parts:
                    d_boundary_keep = _owned_valid_nonempty_device_mask(
                        boundary_part,
                    )
                    if d_boundary_keep is None:
                        return None
                    boundary_masks.append(
                        cp.asarray(d_boundary_keep, dtype=cp.bool_)
                        & d_exact_active
                    )
                physical_boundary_parts = tuple(
                    device_physicalize_owned_row_selection_capacity(part, active)
                    for part, active in zip(
                        boundary_capacity_parts,
                        boundary_masks,
                        strict=True,
                    )
                )
            for boundary_part, d_boundary_active in zip(
                physical_boundary_parts,
                boundary_masks,
                strict=True,
            ):
                if boundary_part is None:
                    continue
                sliver_capacity = _clip_point_contact_sliver_polygons_device(
                    exact_candidate_owned,
                    mask_owned,
                    boundary_part,
                    d_local_rows,
                )
                if sliver_capacity is not None:
                    sliver_owned, d_sliver_mask = sliver_capacity
                    d_sliver_mask = (
                        cp.asarray(d_sliver_mask, dtype=cp.bool_)
                        & ~d_positive_mask
                    )
                    boundary_part = device_select_owned_capacity_partitions(
                        boundary_part,
                        [(sliver_owned, d_sliver_mask & d_exact_active)],
                    )
                if not direct_topology_parts:
                    boundary_remainder = binary_constructive_owned(
                        "difference",
                        boundary_part,
                        device_mask_owned_capacity(
                            exact_area_owned,
                            d_boundary_active,
                        ),
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    if (
                        boundary_remainder is None
                        or boundary_remainder.row_count != candidate_count
                    ):
                        return None
                    boundary_part = device_select_owned_capacity_partitions(
                        boundary_part,
                        [(boundary_remainder, d_positive_mask)],
                    )
                output_owned_parts.append(boundary_part)
                output_row_parts.append(d_exact_output_rows)
                d_boundary_keep = _owned_valid_nonempty_device_mask(
                    boundary_part,
                )
                if d_boundary_keep is None:
                    return None
                d_output_mask |= exact_selection.source_mask(
                    active_mask=(cp.asarray(d_boundary_keep, dtype=cp.bool_) & d_exact_active),
                )

        if output_owned_parts:
            geometry_result = _clip_geometry_composition_at_capacity(
                list(zip(output_owned_parts, output_row_parts, strict=True)),
                row_count=predicate_row_count,
                crs=source.crs,
                trusted_all_ogc_valid=True,
            )
            if geometry_result is None:
                return None
        else:
            result_owned = build_null_owned_array(
                predicate_row_count,
                residency=owned.residency,
            )
            geometry_result = GeometryNativeResult.from_owned(
                result_owned,
                crs=source.crs,
            )

        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation="polygon_device_candidate_direct_rowset_assembly_gpu",
            reason=(
                "polygon device-candidate clip assembled final owned output "
                "from inside, exact-area, and boundary rowsets without a "
                "hit-row scatter carrier"
            ),
            detail=(
                f"candidate_capacity={candidate_count}; "
                "partition_counts=device-resident; "
                f"output_capacity={geometry_result.row_count}; "
                f"keep_geom_type={keep_geom_type}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )

        source_rowset = _clip_source_rowset_for_positions(
            source,
            None,
            device_row_positions=d_rows,
        )
        if source_rowset is None:
            return None
        native_result = _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=geometry_result,
            rowset=source_rowset,
            keep_geom_type=keep_geom_type,
        )
        if native_result is not None:
            selected_result = NativeTabularSelection(
                capacity_result=native_result,
                selection=NativeDeviceSelection.from_mask(
                    d_output_mask,
                    source_token=(f"{source_state.lineage_token}:clip-polygon-capacity"),
                    source_row_count=predicate_row_count,
                ),
            )
            source_rows = getattr(native_result.provenance, "source_rows", None)
            return (
                selected_result
                if source_rows is None or predicate_row_count <= 1
                else selected_result.sort_selected_by_int64(source_rows)
            )

        return None

    # Fused predicate admission is an optimization, not an alternate semantic
    # implementation. Keep declined polygon pairs in the general aligned exact
    # constructive carrier instead of rebuilding a host-indexed partition.
    from vibespatial.constructive.binary_constructive import (
        broadcast_right_polygon_intersection_capacity_gpu,
    )

    candidate_owned = _ensure_candidate_owned()
    result_owned = broadcast_right_polygon_intersection_capacity_gpu(
        candidate_owned,
        mask_owned,
        right_row=0,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result_owned is None or result_owned.residency is not Residency.DEVICE:
        return None
    d_keep = (
        _owned_nonempty_polygon_device_mask(result_owned)
        if keep_geom_type
        else _owned_valid_nonempty_device_mask(result_owned)
    )
    if d_keep is None or int(d_keep.size) != int(result_owned.row_count):
        return None
    source_rowset = NativeRowSet.from_positions(
        d_rows,
        source_token=source_state.lineage_token,
        source_row_count=source_state.row_count,
        ordered=True,
        unique=True,
        identity=_clip_source_rows_identity_hint(
            d_rows,
            source_state.row_count,
        ),
    )
    capacity_result = _clip_native_tabular_result_from_rowset(
        source,
        geometry_name=geometry_name,
        geometry=GeometryNativeResult.from_owned(result_owned, crs=source.crs),
        rowset=source_rowset,
        keep_geom_type=keep_geom_type,
    )
    if capacity_result is not None:
        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation="polygon_device_candidate_aligned_exact_gpu",
            reason=(
                "fused polygon predicate admission declined and the existing "
                "device candidate relation was consumed by aligned exact topology"
            ),
            detail=(
                f"candidate_rows={candidate_count}; "
                "output_rows=device-resident; "
                f"keep_geom_type={keep_geom_type}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        selected_result = NativeTabularSelection(
            capacity_result=capacity_result,
            selection=NativeDeviceSelection.from_mask(
                d_keep,
                source_token=f"{source_state.lineage_token}:clip-exact-capacity",
                source_row_count=candidate_count,
            ),
        )
        source_rows = getattr(capacity_result.provenance, "source_rows", None)
        return (
            selected_result
            if source_rows is None or candidate_count <= 1
            else selected_result.sort_selected_by_int64(source_rows)
        )
    return None


def _clip_mixed_device_candidates_native(
    source,
    mask,
    candidate_device_rows,
    *,
    clipping_by_rectangle: bool,
    rectangle_bounds,
    keep_geom_type: bool,
) -> NativeTabularResult | None:
    """Clip mixed-family device candidates without exporting candidate rows."""
    if not has_gpu_runtime():
        return None
    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None
    family_masks = _clip_family_masks(source)
    if family_masks.all_polygonal and not clipping_by_rectangle:
        return None

    geometry_name = (
        source._geometry_column_name
        if hasattr(source, "_geometry_column_name")
        else getattr(source, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(source, geometry_name)
    if (
        source_state is None
        or source_state.geometry_name != geometry_name
        or source_state.index_plan.kind
        not in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    candidate_selection = (
        candidate_device_rows
        if isinstance(candidate_device_rows, NativeDeviceSelection)
        else None
    )
    if candidate_selection is not None:
        if candidate_selection.source_row_count != source_state.row_count:
            return None
        d_rows = cp.asarray(
            candidate_selection.partition_capacity_positions(),
            dtype=cp.int64,
        )
        d_candidate_active = candidate_selection.active_capacity_mask()
    else:
        d_rows = cp.asarray(candidate_device_rows, dtype=cp.int64)
        d_candidate_active = cp.ones(int(d_rows.size), dtype=cp.bool_)
    candidate_count = int(d_rows.size)
    if candidate_count == 0:
        from vibespatial.api._native_rowset import NativeRowSet
        from vibespatial.geometry.owned import build_null_owned_array

        empty_owned = build_null_owned_array(0, residency=owned.residency)
        empty_rowset = NativeRowSet.from_positions(
            cp.empty(0, dtype=cp.int64),
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=False,
        )
        return _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=GeometryNativeResult.from_owned(empty_owned, crs=source.crs),
            rowset=empty_rowset,
            keep_geom_type=keep_geom_type,
        )

    if candidate_selection is not None:
        from vibespatial.geometry.owned import device_take_owned_capacity_selection

        candidate_owned = device_take_owned_capacity_selection(
            owned,
            candidate_selection,
        )
    else:
        candidate_owned = owned.device_take(d_rows)
    if keep_geom_type and rectangle_bounds is not None and _clip_family_masks(source).all_polygonal:
        return None

    def _row_aligned_result_from_owned(
        result_geometry,
        d_local_output_rows,
        *,
        dynamic: bool = False,
    ):
        from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet
        from vibespatial.runtime.dispatch import record_dispatch_event

        d_local_output_rows = cp.asarray(d_local_output_rows, dtype=cp.int64)
        if int(d_local_output_rows.size) != int(result_geometry.row_count):
            return None
        d_source_rows = d_rows[d_local_output_rows]
        source_rowset = NativeRowSet.from_positions(
            d_source_rows,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=_clip_source_rows_identity_hint(
                d_source_rows,
                source_state.row_count,
            ),
        )
        native_result = _clip_native_tabular_result_from_rowset(
            source,
            geometry_name=geometry_name,
            geometry=(
                result_geometry.with_crs(source.crs)
                if isinstance(result_geometry, GeometryNativeResult)
                else GeometryNativeResult.from_owned(result_geometry, crs=source.crs)
            ),
            rowset=source_rowset,
            keep_geom_type=keep_geom_type,
        )
        if native_result is not None:
            record_dispatch_event(
                surface="geopandas.clip",
                operation="clip",
                implementation="mixed_device_candidate_native",
                reason=(
                    "mixed-family clip consumed device candidate rows through "
                    "native constructive carriers"
                ),
                detail=(
                    f"candidate_rows={candidate_count}, "
                    f"output_rows={int(result_geometry.row_count)}, "
                    "physical_shape=mixed-family-candidate-rowset"
                ),
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
            )
        if native_result is None or (not dynamic and candidate_selection is None):
            return native_result
        geometry_result = (
            result_geometry
            if isinstance(result_geometry, GeometryNativeResult)
            else GeometryNativeResult.from_owned(result_geometry, crs=source.crs)
        )
        d_keep = d_candidate_active[d_local_output_rows]
        if dynamic:
            d_geometry_keep = geometry_result.valid_nonempty_mask_device()
            if d_geometry_keep is None:
                return None
            d_keep &= cp.asarray(d_geometry_keep, dtype=cp.bool_)
        return NativeTabularSelection(
            capacity_result=native_result,
            selection=NativeDeviceSelection.from_mask(
                cp.asarray(d_keep, dtype=cp.bool_),
            ),
        )

    if clipping_by_rectangle:
        if rectangle_bounds is None:
            return None
        if keep_geom_type and family_masks.all_polygonal:
            return None
        from vibespatial.constructive.clip_rect import clip_by_rect_owned
        from vibespatial.geometry.buffers import GeometryFamily

        admitted = {
            GeometryFamily.POINT,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
        }
        if not _owned_active_family_subset(candidate_owned, admitted):
            return None

        clip_result = clip_by_rect_owned(
            candidate_owned,
            *rectangle_bounds,
            dispatch_mode=ExecutionMode.GPU,
        )
        result_owned = clip_result.owned_result
        if (
            result_owned is None
            or not clip_result.owned_result_is_row_capacity
            or result_owned.row_count != candidate_count
        ):
            return None
        return _row_aligned_result_from_owned(
            result_owned,
            cp.arange(candidate_count, dtype=cp.int64),
            dynamic=True,
        )

    if not isinstance(mask, Polygon | MultiPolygon):
        return None
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.geometry.owned import from_shapely_geometries, tile_single_row

    mask_owned = from_shapely_geometries([mask], residency=owned.residency)
    right_owned = tile_single_row(mask_owned, candidate_count)
    result_geometry = _binary_constructive_gpu(
        "intersection",
        candidate_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )
    if result_geometry is None or result_geometry.residency is not Residency.DEVICE:
        return None
    d_keep = (
        result_geometry.valid_nonempty_mask_device()
        if isinstance(result_geometry, GeometryNativeResult)
        else _owned_valid_nonempty_device_mask(result_geometry)
    )
    if d_keep is None:
        return None
    return _row_aligned_result_from_owned(
        result_geometry,
        cp.arange(candidate_count, dtype=cp.int64),
        dynamic=True,
    )


def _clip_concat_source_ordered_native_results(
    results,
    *,
    source_state,
    geometry_name: str,
    crs,
):
    """Concatenate source-row clip partitions and restore source position order."""
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    if any(
        not isinstance(result, NativeTabularSelection)
        and result.terminal_geodataframe_materializer is not None
        for result in results
    ):
        return None

    import cupy as cp

    from vibespatial.api._native_results import _concat_native_tabular_results

    if any(isinstance(result, NativeTabularSelection) for result in results):
        results = [
            replace(
                result,
                public_index_source_plan=None,
                public_index_source_rows=None,
            )
            if isinstance(result, NativeTabularSelection)
            and result.public_index_source_plan is not None
            else result
            for result in results
        ]
        combined_selection = _concat_native_tabular_selections(
            results,
            geometry_name=geometry_name,
            crs=crs,
            attrs=None,
            ignore_index=True,
        )
        provenance = combined_selection.capacity_result.provenance
        source_rows = getattr(provenance, "source_rows", None)
        if source_rows is None:
            return None
        return combined_selection.sort_selected_by_int64(
            source_rows,
        ).with_public_index_source(
            source_state.index_plan,
            source_rows,
        )

    source_row_parts = []
    for result in results:
        provenance = result.provenance
        source_rows = getattr(provenance, "source_rows", None)
        if source_rows is None:
            return None
        source_row_parts.append(cp.asarray(source_rows, dtype=cp.int64))
    d_source_rows = cp.concatenate(source_row_parts)
    d_order = cp.argsort(d_source_rows).astype(cp.int64, copy=False)
    d_sorted_source_rows = d_source_rows[d_order].astype(cp.int64, copy=False)

    combined = _concat_native_tabular_results(
        results,
        geometry_name=geometry_name,
        crs=crs,
        attrs=None,
        ignore_index=True,
    ).take(d_order, preserve_index=False)
    return replace(
        combined,
        index_plan=source_state.index_plan.take(
            d_sorted_source_rows,
            preserve_index=True,
            unique=True,
            strict_disallowed=False,
        ),
    )


def _clip_grouped_polygon_pair_capacity(
    pair_intersections,
    d_source_rows,
    d_pair_active,
    *,
    output_row_count: int,
    pair_intersections_axis_rectangles: bool = False,
):
    """Reduce polygon pair intersections into source-row output capacity."""
    import cupy as cp

    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        OwnedGeometryArray,
        build_empty_polygon_rows_device,
        device_scatter_owned_capacity_selection,
    )
    from vibespatial.kernels.constructive.segmented_union import (
        segmented_union_all_device_grouped,
    )
    from vibespatial.runtime.precision import (
        CompensationMode,
        PrecisionMode,
        PrecisionPlan,
        RefinementMode,
    )

    pair_capacity = int(pair_intersections.row_count)
    if output_row_count <= 0:
        return None
    d_source_rows = cp.asarray(d_source_rows, dtype=cp.int64)
    d_pair_active = cp.asarray(d_pair_active, dtype=cp.bool_)
    if int(d_source_rows.size) != pair_capacity or int(d_pair_active.size) != pair_capacity:
        raise ValueError("grouped clip pair capacity metadata must align")

    d_polygon = _owned_nonempty_polygon_device_mask(pair_intersections)
    if d_polygon is None:
        return None
    d_polygon_active = d_pair_active & d_polygon
    polygon_selection = NativeDeviceSelection.from_mask(d_polygon_active)
    grouped = NativeGroupedSelection(
        selection=polygon_selection,
        group_codes=d_source_rows.astype(cp.int32, copy=False),
        group_count=output_row_count,
    )
    d_group_counts = grouped.reduce_numeric(
        cp.ones(pair_capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int64, copy=False)
    d_group_counts += 1
    d_group_counts[0] += (
        cp.int64(pair_capacity) - cp.asarray(polygon_selection.logical_count, dtype=cp.int64)[0]
    )
    d_group_offsets = cp.empty(output_row_count + 1, dtype=cp.int64)
    d_group_offsets[0] = 0
    cp.cumsum(d_group_counts, out=d_group_offsets[1:])

    total_capacity = output_row_count + pair_capacity
    if total_capacity > np.iinfo(np.uint32).max or output_row_count > np.iinfo(np.uint32).max:
        raise OverflowError("grouped clip pair capacity exceeds radix lane width")
    d_pair_group_codes = polygon_selection.gather_capacity(
        d_source_rows,
        fill_value=0,
    ).astype(cp.int64, copy=False)
    d_all_group_codes = cp.concatenate(
        [
            cp.arange(output_row_count, dtype=cp.int64),
            d_pair_group_codes,
        ]
    )
    d_sort_keys = (d_all_group_codes.astype(cp.uint64, copy=False) << cp.uint64(32)) | cp.arange(
        total_capacity, dtype=cp.uint64
    )
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(total_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)

    selected_parts = pair_intersections._device_indexed_take(
        polygon_selection.safe_capacity_positions(),
    )
    pair_noops = device_scatter_owned_capacity_selection(
        build_empty_polygon_rows_device(pair_capacity),
        selected_parts,
        polygon_selection.as_capacity_prefix(),
        active_mask=polygon_selection.active_capacity_mask(),
    )
    pair_noops.device_state.trusted_all_valid = True
    pair_noops.device_state.trusted_polygonal_only = True
    all_parts = OwnedGeometryArray.concat(
        [
            build_empty_polygon_rows_device(output_row_count),
            pair_noops,
        ]
    )
    all_parts.device_state.trusted_all_valid = True
    all_parts.device_state.trusted_polygonal_only = True
    ordered_parts = all_parts._device_indexed_take(d_order)
    ordered_state = ordered_parts._ensure_device_state(preserve_indexed_view=True)
    ordered_state.trusted_all_valid = True
    ordered_state.trusted_polygonal_only = True
    ordered_state.trusted_family_domain = (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    )
    if pair_intersections_axis_rectangles:
        from vibespatial.geometry.owned import DeviceFixedGeometrySizeMetadata

        ordered_polygon_buffer = ordered_parts.device_state.families.get(
            GeometryFamily.POLYGON
        )
        if ordered_polygon_buffer is not None:
            ordered_polygon_buffer.axis_aligned_rectangles = True
            ordered_polygon_buffer.fixed_size = DeviceFixedGeometrySizeMetadata(
                max_first_level_count_per_row=1,
                max_coord_count_per_row=5,
            )
    empty_output = build_empty_polygon_rows_device(output_row_count)
    grouped_result = segmented_union_all_device_grouped(
        ordered_parts,
        d_group_offsets,
        cp.arange(output_row_count, dtype=cp.int64),
        output_row_count=output_row_count,
        precision_plan=PrecisionPlan(
            storage_precision=PrecisionMode.FP64,
            compute_precision=PrecisionMode.FP64,
            kernel_class=KernelClass.CONSTRUCTIVE,
            compensation=CompensationMode.NONE,
            refinement=RefinementMode.NONE,
            center_coordinates=False,
            reason="collective polygon clip uses grouped constructive fp64",
        ),
        empty_output=empty_output,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=pair_capacity + 1,
        nonempty_rows_positive_area=True,
        _capacity_all_valid_noops=True,
    )
    if grouped_result is None or grouped_result.row_count != output_row_count:
        raise RuntimeError("grouped clip topology did not preserve source capacity")
    grouped_state = grouped_result._ensure_device_state(preserve_indexed_view=True)
    grouped_state.trusted_polygonal_only = True
    grouped_state.trusted_family_domain = (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    )
    d_keep = _owned_valid_nonempty_device_mask(grouped_result)
    if d_keep is None:
        raise RuntimeError("grouped clip result has no device validity metadata")
    return grouped_result, d_keep


def _clip_gdf_with_lazy_grouped_union_mask_native(
    gdf,
    mask,
    lazy_mask_owned,
    *,
    keep_geom_type: bool,
    sort: bool,
) -> NativeTabularResult | NativeTabularSelection | None:
    """Clip polygon rows against a lazy grouped-union mask without union export.

    Physical shape: native spatial relation pairs from grouped mask members to
    source rows -> row-aligned exact intersections -> grouped constructive
    reduction by source row. The public mask is a single dissolved geometry,
    but the GPU work stays pair/group shaped until the explicit clip export.
    """
    if (
        sort
        or not has_gpu_runtime()
        or lazy_mask_owned is None
        or int(getattr(lazy_mask_owned, "row_count", -1)) != 1
    ):
        return None

    geometry = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None or owned.residency is not Residency.DEVICE:
        return None

    mask_source_owned = getattr(lazy_mask_owned, "_source_owned", None)
    if mask_source_owned is None or mask_source_owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.geometry.buffers import GeometryFamily

    polygonal = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    if not _owned_active_family_subset(owned, polygonal):
        return None
    if not _owned_active_family_subset(mask_source_owned, polygonal):
        return None

    geometry_name = (
        gdf._geometry_column_name
        if isinstance(gdf, GeoDataFrame)
        else getattr(gdf, "name", None) or "geometry"
    )
    source_state = _clip_native_state_for_source(gdf, geometry_name)
    if (
        source_state is None
        or source_state.geometry_name != geometry_name
        or source_state.index_plan.kind
        not in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }
    ):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    relation, _execution = gdf.sindex.query_relation(
        mask_source_owned,
        predicate="intersects",
        sort=False,
        source_token=source_state.lineage_token,
        query_row_count=mask_source_owned.row_count,
        return_device=True,
    )

    d_mask_rows = cp.asarray(relation.left_indices, dtype=cp.int64)
    d_source_rows = cp.asarray(relation.right_indices, dtype=cp.int64)
    if int(d_mask_rows.size) != int(d_source_rows.size):
        return None

    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.geometry.owned import build_null_owned_array
    from vibespatial.runtime.dispatch import record_dispatch_event

    pair_count = int(d_source_rows.size)
    if pair_count == 0:
        empty_owned = build_null_owned_array(0, residency=owned.residency)
        empty_rowset = NativeRowSet.from_positions(
            cp.empty(0, dtype=cp.int64),
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=False,
        )
        return _clip_native_tabular_result_from_rowset(
            gdf,
            geometry_name=geometry_name,
            geometry=GeometryNativeResult.from_owned(empty_owned, crs=geometry.crs),
            rowset=empty_rowset,
            keep_geom_type=keep_geom_type,
        )

    result_partitions = []
    materialized_mask_owned = None

    def _materialized_mask_owned_once():
        nonlocal materialized_mask_owned
        if materialized_mask_owned is not None:
            return materialized_mask_owned
        materialize_owned = getattr(lazy_mask_owned, "_materialize_owned", None)
        if not callable(materialize_owned):
            return None
        candidate = materialize_owned()
        if (
            candidate is None
            or candidate.residency is not Residency.DEVICE
            or int(getattr(candidate, "row_count", -1)) != 1
        ):
            return None
        materialized_mask_owned = candidate
        return materialized_mask_owned

    def _append_physicalized_source_rows(selection, *, implementation: str) -> bool:
        materialized = _materialized_mask_owned_once()
        if materialized is None:
            return False
        partition = _clip_homogeneous_polygon_device_candidates_native(
            gdf,
            mask,
            selection,
            mask_owned=materialized,
            clipping_by_rectangle=False,
            rectangle_bounds=None,
            keep_geom_type=keep_geom_type,
        )
        if partition is None:
            return False
        result_partitions.append(partition)
        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation=implementation,
            reason=(
                "lazy grouped-union mask was device-physicalized once and "
                "consumed only by the selected source rowset"
            ),
            detail=(
                f"source_rows={owned.row_count}, mask_rows={mask_source_owned.row_count}, "
                f"selected_capacity={selection.capacity}, "
                "selected_rows=device-resident"
            ),
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
        )
        return True

    # Shape admission must precede semantic pair refinement. Dense relations
    # can make exact pairwise coverage itself the dominant operation even when
    # the only feasible constructive carrier is one reduced mask. Build the
    # candidate source rowset directly from the resident relation and use
    # conservative host-known capacities for this first decision.
    d_candidate_source_mask = cp.zeros(owned.row_count, dtype=cp.bool_)
    d_candidate_source_mask[d_source_rows] = True
    candidate_source_selection = NativeDeviceSelection.from_mask(
        d_candidate_source_mask,
        source_token=source_state.lineage_token,
        source_row_count=source_state.row_count,
        geometry_family_domain=tuple(polygonal),
        trusted_all_valid_rows=None,
    )
    available_device_bytes = _clip_available_device_bytes()
    coarse_source_count = min(int(owned.row_count), pair_count)
    (
        coarse_relation_selected,
        coarse_relation_estimate,
        coarse_union_estimate,
    ) = _clip_collective_grouped_mask_prefers_relation(
        owned,
        mask_source_owned,
        relation_pair_count=pair_count,
        collective_source_count=coarse_source_count,
        available_device_bytes=available_device_bytes,
    )
    if not coarse_relation_selected:
        record_dispatch_event(
            surface="geopandas.clip",
            operation="clip",
            implementation="lazy_grouped_union_mask_union_plan_gpu",
            reason=(
                "grouped-mask clip selected union-first execution before exact "
                "pair refinement from resident relation shape and memory admission"
            ),
            detail=(
                f"source_rows<={coarse_source_count}, pair_rows={pair_count}, "
                f"relation_units={coarse_relation_estimate.dispatch_unit_count()}, "
                f"union_units={coarse_union_estimate.dispatch_unit_count()}, "
                f"relation_live_bytes={coarse_relation_estimate.live_device_byte_count()}, "
                f"available_device_bytes={available_device_bytes}, "
                "semantic_pair_refinement=skipped"
            ),
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
        )
        if not _append_physicalized_source_rows(
            candidate_source_selection,
            implementation="lazy_grouped_union_mask_union_physicalized_gpu",
        ):
            return None
        return _clip_concat_source_ordered_native_results(
            result_partitions,
            source_state=source_state,
            geometry_name=geometry_name,
            crs=geometry.crs,
        )

    coverage = _clip_lazy_grouped_union_coverage_rows_device(
        owned=owned,
        mask_source_owned=mask_source_owned,
        source_state=source_state,
        d_mask_rows=d_mask_rows,
        d_source_rows=d_source_rows,
        pair_count=pair_count,
    )
    if coverage is None:
        return None
    covered_selection, unresolved_selection = coverage
    covered_result = _native_state_passthrough_take(gdf, covered_selection)
    if covered_result is None:
        return None
    result_partitions.append(covered_result)
    d_unresolved_source_mask = unresolved_selection.source_mask()
    d_unresolved_pair_active = d_unresolved_source_mask[d_source_rows]
    relation_pair_selection = NativeDeviceSelection.from_mask(
        d_unresolved_pair_active,
    )
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_trusted_rectangle_bounds_matrix,
    )

    d_source_rectangle_bounds = device_trusted_rectangle_bounds_matrix(owned)
    d_mask_rectangle_bounds = device_trusted_rectangle_bounds_matrix(mask_source_owned)
    if d_source_rectangle_bounds is not None and d_mask_rectangle_bounds is not None:
        d_source_pair_bounds = cp.asarray(d_source_rectangle_bounds, dtype=cp.float64)[
            d_source_rows
        ]
        d_mask_pair_bounds = cp.asarray(d_mask_rectangle_bounds, dtype=cp.float64)[d_mask_rows]
        d_positive_area_pairs = d_unresolved_pair_active & (
            cp.minimum(d_source_pair_bounds[:, 2], d_mask_pair_bounds[:, 2])
            > cp.maximum(d_source_pair_bounds[:, 0], d_mask_pair_bounds[:, 0])
        ) & (
            cp.minimum(d_source_pair_bounds[:, 3], d_mask_pair_bounds[:, 3])
            > cp.maximum(d_source_pair_bounds[:, 1], d_mask_pair_bounds[:, 1])
        )
        d_positive_area_pair_count = cp.count_nonzero(d_positive_area_pairs).astype(cp.int64)

        d_mask_bounds = cp.asarray(d_mask_rectangle_bounds, dtype=cp.float64)
        d_scale = cp.maximum(cp.max(cp.abs(d_mask_bounds)), cp.float64(1.0))
        tolerance = d_scale * cp.float64(1.0e-12)
        d_same_y = cp.all(cp.abs(d_mask_bounds[:, 1] - d_mask_bounds[0, 1]) <= tolerance) & cp.all(
            cp.abs(d_mask_bounds[:, 3] - d_mask_bounds[0, 3]) <= tolerance
        )
        d_same_x = cp.all(cp.abs(d_mask_bounds[:, 0] - d_mask_bounds[0, 0]) <= tolerance) & cp.all(
            cp.abs(d_mask_bounds[:, 2] - d_mask_bounds[0, 2]) <= tolerance
        )
        d_lower = cp.where(d_same_y, d_mask_bounds[:, 0], d_mask_bounds[:, 1])
        d_upper = cp.where(d_same_y, d_mask_bounds[:, 2], d_mask_bounds[:, 3])
        d_interval_order = cp.argsort(d_lower)
        d_sorted_lower = d_lower[d_interval_order]
        d_sorted_upper = d_upper[d_interval_order]
        # Adjacent overlap is a conservative strip proof. Nested intervals
        # may decline this planner optimization and remain relation-shaped.
        d_connected = cp.all(d_sorted_lower[1:] < d_sorted_upper[:-1])
        d_mask_rectangle_strip = (d_same_y | d_same_x) & d_connected
    else:
        d_positive_area_pair_count = cp.asarray(
            relation_pair_selection.logical_count,
            dtype=cp.int64,
        )[0]
        d_mask_rectangle_strip = cp.asarray(False, dtype=cp.bool_)

    d_plan_counts = cp.concatenate(
        [
            cp.asarray(unresolved_selection.logical_count, dtype=cp.int64),
            cp.asarray(relation_pair_selection.logical_count, dtype=cp.int64),
            cp.asarray(d_positive_area_pair_count, dtype=cp.int64).reshape(1),
            cp.asarray(d_mask_rectangle_strip, dtype=cp.int64).reshape(1),
        ]
    )
    (
        unresolved_source_count,
        unresolved_pair_count,
        positive_area_pair_count,
        mask_rectangle_strip_admissible,
    ) = (
        int(value)
        for value in np.asarray(
            get_cuda_runtime().copy_device_to_host(
                d_plan_counts,
                reason="clip grouped-mask physical-plan aggregate admission counts",
            ),
            dtype=np.int64,
        )
    )
    if unresolved_source_count == 0:
        return _clip_concat_source_ordered_native_results(
            result_partitions,
            source_state=source_state,
            geometry_name=geometry_name,
            crs=geometry.crs,
        )
    (
        relation_selected,
        relation_estimate,
        union_estimate,
    ) = _clip_collective_grouped_mask_prefers_relation(
        owned,
        mask_source_owned,
        relation_pair_count=unresolved_pair_count,
        collective_source_count=unresolved_source_count,
        positive_area_pair_count=positive_area_pair_count,
        mask_rectangle_strip_admissible=bool(mask_rectangle_strip_admissible),
        available_device_bytes=available_device_bytes,
    )
    record_dispatch_event(
        surface="geopandas.clip",
        operation="clip",
        implementation=(
            "lazy_grouped_union_mask_relation_plan_gpu"
            if relation_selected
            else "lazy_grouped_union_mask_union_plan_gpu"
        ),
        reason=(
            "grouped-mask clip selected relation-first or union-first execution "
            "from resident pair and source capacities"
        ),
        detail=(
            f"source_rows={unresolved_source_count}, "
            f"pair_rows={unresolved_pair_count}, "
            f"positive_area_pairs={positive_area_pair_count}, "
            f"rectangle_strip={bool(mask_rectangle_strip_admissible)}, "
            f"relation_units={relation_estimate.dispatch_unit_count()}, "
            f"union_units={union_estimate.dispatch_unit_count()}, "
            f"relation_live_bytes={relation_estimate.live_device_byte_count()}, "
            f"available_device_bytes={available_device_bytes}, "
            "geometry_allocation=source-or-pair-capacity"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
    )
    if not relation_selected:
        if not _append_physicalized_source_rows(
            unresolved_selection,
            implementation="lazy_grouped_union_mask_union_physicalized_gpu",
        ):
            return None
        return _clip_concat_source_ordered_native_results(
            result_partitions,
            source_state=source_state,
            geometry_name=geometry_name,
            crs=geometry.crs,
        )

    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
    from vibespatial.overlay.gpu import _compact_bounded_device_work_spans

    d_unresolved_pair_positions = cp.flatnonzero(d_unresolved_pair_active).astype(
        cp.int64,
        copy=False,
    )
    compact_pair_count = int(d_unresolved_pair_positions.size)
    d_compact_source_rows = d_source_rows[d_unresolved_pair_positions]
    d_compact_mask_rows = d_mask_rows[d_unresolved_pair_positions]
    d_sort_keys = (
        d_compact_source_rows.astype(cp.uint64, copy=False) << cp.uint64(32)
    ) | cp.arange(compact_pair_count, dtype=cp.uint64)
    d_pair_order = sort_pairs(
        d_sort_keys,
        cp.arange(compact_pair_count, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)
    d_sorted_source_rows = d_compact_source_rows[d_pair_order]
    d_sorted_mask_rows = d_compact_mask_rows[d_pair_order]
    d_source_pair_counts = cp.bincount(
        d_sorted_source_rows,
        minlength=source_state.row_count,
    ).astype(cp.int64, copy=False)
    relation_live_bytes = max(int(relation_estimate.live_device_byte_count()), 1)
    estimated_bytes_per_pair = max(
        (relation_live_bytes + max(compact_pair_count, 1) - 1)
        // max(compact_pair_count, 1),
        256,
    )
    page_available_bytes = max(int(available_device_bytes or relation_live_bytes) // 5, 1)
    page_pair_budget = max(
        64 * 1024,
        page_available_bytes // estimated_bytes_per_pair,
    )
    source_spans = _compact_bounded_device_work_spans(
        d_source_pair_counts,
        live_event_budget=page_pair_budget,
    )
    d_pair_offsets = cp.empty(source_state.row_count + 1, dtype=cp.int64)
    d_pair_offsets[0] = 0
    d_pair_offsets[1:] = cp.cumsum(d_source_pair_counts, dtype=cp.int64)
    d_span_boundaries = cp.asarray(
        [source_spans[0][0], *(end for _start, end in source_spans)],
        dtype=cp.int64,
    )
    host_pair_boundaries = np.asarray(
        get_cuda_runtime().copy_device_to_host(
            d_pair_offsets[d_span_boundaries],
            reason="clip grouped-mask complete-source page-offset planning packet",
        ),
        dtype=np.int64,
    )

    for page_index, (source_start, source_end) in enumerate(source_spans):
        pair_start = int(host_pair_boundaries[page_index])
        pair_end = int(host_pair_boundaries[page_index + 1])
        if pair_end <= pair_start:
            continue
        page_source_count = source_end - source_start
        d_page_source_rows = d_sorted_source_rows[pair_start:pair_end]
        d_page_mask_rows = d_sorted_mask_rows[pair_start:pair_end]
        d_page_group_rows = (d_page_source_rows - np.int64(source_start)).astype(
            cp.int64,
            copy=False,
        )
        d_page_active = cp.ones(pair_end - pair_start, dtype=cp.bool_)
        source_pairs = owned.device_take(d_page_source_rows)
        mask_pairs = mask_source_owned.device_take(d_page_mask_rows)
        pair_intersections = _binary_constructive_gpu(
            "intersection",
            source_pairs,
            mask_pairs,
            dispatch_mode=ExecutionMode.GPU,
        )
        if pair_intersections is None or pair_intersections.residency is not Residency.DEVICE:
            raise RuntimeError(
                "grouped-mask relation page intersection declined after GPU plan admission"
            )
        d_pair_polygon_active = _owned_nonempty_polygon_device_mask(pair_intersections)
        if d_pair_polygon_active is None:
            raise RuntimeError("grouped-mask relation page lost polygon activity metadata")
        d_pair_polygon_active = cp.asarray(d_pair_polygon_active, dtype=cp.bool_)
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            device_trusted_rectangle_bounds_matrix,
        )

        pair_intersections_axis_rectangles = bool(
            device_trusted_rectangle_bounds_matrix(source_pairs) is not None
            and device_trusted_rectangle_bounds_matrix(mask_pairs) is not None
        )
        pair_boundary_parts = ()
        if not keep_geom_type:
            from vibespatial.constructive.grouped_mixed_union import (
                polygon_pair_boundary_capacity_parts_device,
            )

            pair_boundary_parts = polygon_pair_boundary_capacity_parts_device(
                source_pairs,
                mask_pairs,
                pair_active=(d_page_active & ~d_pair_polygon_active),
            )
            if pair_boundary_parts is None:
                raise RuntimeError("grouped-mask relation page boundary capacity declined")

        grouped_capacity = _clip_grouped_polygon_pair_capacity(
            pair_intersections,
            d_page_group_rows,
            d_page_active,
            output_row_count=page_source_count,
            pair_intersections_axis_rectangles=pair_intersections_axis_rectangles,
        )
        if grouped_capacity is None:
            return None
        polygon_capacity, d_polygon_keep = grouped_capacity
        if keep_geom_type:
            capacity_geometry = GeometryNativeResult.from_owned(
                polygon_capacity,
                crs=geometry.crs,
            )
            d_output_keep = d_polygon_keep
        else:
            from vibespatial.constructive.grouped_mixed_union import (
                grouped_mixed_union_capacity_device,
            )

            mixed_capacity = grouped_mixed_union_capacity_device(
                pair_intersections,
                d_page_group_rows,
                d_page_active,
                polygon_capacity,
                d_polygon_keep,
                output_row_count=page_source_count,
                crs=geometry.crs,
                pair_boundary_parts=pair_boundary_parts,
            )
            capacity_geometry = mixed_capacity.geometry
            d_output_keep = mixed_capacity.keep_mask
        source_rowset = NativeRowSet.from_positions(
            cp.arange(source_start, source_end, dtype=cp.int64),
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
            ordered=True,
            unique=True,
            identity=(source_start == 0 and source_end == source_state.row_count),
        )
        capacity_result = _clip_native_tabular_result_from_rowset(
            gdf,
            geometry_name=geometry_name,
            geometry=capacity_geometry,
            rowset=source_rowset,
            keep_geom_type=keep_geom_type,
        )
        if capacity_result is None:
            return None
        result_partitions.append(
            NativeTabularSelection(
                capacity_result=capacity_result,
                selection=NativeDeviceSelection.from_mask(d_output_keep),
            )
        )

    record_dispatch_event(
        surface="geopandas.clip",
        operation="clip",
        implementation="lazy_grouped_union_mask_relation_clip_gpu",
        reason=(
            "lazy grouped-union mask was consumed as device relation pairs and "
            "reduced by source row without materializing the dissolved mask"
        ),
        detail=(
            f"source_rows={owned.row_count}, mask_rows={mask_source_owned.row_count}, "
            f"pairs={compact_pair_count}, pages={len(source_spans)}, "
            f"pair_budget={page_pair_budget}, output_rows=device-resident"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
    )
    source_provenance = getattr(lazy_mask_owned, "_source_provenance", None)
    if (
        source_provenance is not None
        and getattr(source_provenance, "operation", None) == "buffer"
        and getattr(source_provenance, "source_geom_types", frozenset())
        <= frozenset({"linestring", "multilinestring"})
    ):
        record_dispatch_event(
            surface="geopandas.geodataframe.dissolve",
            operation="dissolve",
            implementation="buffered_line_grouped_union_gpu",
            reason=(
                "lazy buffered-line dissolve was reduced by the downstream "
                "native grouped constructive consumer"
            ),
            detail=(
                f"rows={mask_source_owned.row_count}, groups={lazy_mask_owned.row_count}, "
                "physical_shape=relation_pairs_to_grouped_union"
            ),
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
        )
    return _clip_concat_source_ordered_native_results(
        result_partitions,
        source_state=source_state,
        geometry_name=geometry_name,
        crs=geometry.crs,
    )


def _clip_lazy_grouped_union_coverage_rows_device(
    *,
    owned,
    mask_source_owned,
    source_state,
    d_mask_rows,
    d_source_rows,
    pair_count: int,
):
    """Partition grouped-mask relation candidates into covered and unresolved rows.

    Physical shape: relation pairs -> aligned pairwise `covered_by` predicates
    -> source-row boolean OR -> two source-capacity device selections.
    """
    if pair_count <= 0 or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    from vibespatial.predicates.binary import _binary_predicate_relation_pair_values_device

    expressions = _binary_predicate_relation_pair_values_device(
        ("covered_by",),
        owned,
        mask_source_owned,
        d_source_rows,
        d_mask_rows,
        operation_prefix="clip.lazy_grouped_union.coverage",
    )
    if expressions is None or "covered_by" not in expressions:
        return None

    d_pair_covered = cp.asarray(expressions["covered_by"], dtype=cp.bool_)
    if int(d_pair_covered.size) != pair_count:
        return None

    d_candidate_mask = cp.zeros(source_state.row_count, dtype=cp.bool_)
    d_candidate_mask[d_source_rows] = True
    d_pair_lanes = cp.arange(pair_count, dtype=cp.int64)
    d_covered_destinations = cp.where(
        d_pair_covered,
        d_source_rows,
        cp.int64(source_state.row_count) + d_pair_lanes,
    )
    d_covered_capacity = cp.zeros(
        source_state.row_count + pair_count,
        dtype=cp.bool_,
    )
    d_covered_capacity[d_covered_destinations] = True
    d_covered_mask = d_covered_capacity[: source_state.row_count]
    d_unresolved_mask = d_candidate_mask & ~d_covered_mask
    from vibespatial.api._native_rowset import NativeDeviceSelection

    return (
        NativeDeviceSelection.from_mask(
            d_covered_mask,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
        ),
        NativeDeviceSelection.from_mask(
            d_unresolved_mask,
            source_token=source_state.lineage_token,
            source_row_count=source_state.row_count,
        ),
    )


def _clip_gdf_with_mask_native(
    gdf,
    mask,
    sort=False,
    *,
    query_geometry=None,
    mask_owned=None,
    keep_geom_type: bool = False,
) -> NativeTabularResult | NativeTabularSelection:
    """Build a native clip result and defer GeoPandas assembly to explicit export."""
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    rectangle_bounds = _rectangle_bounds_from_mask(mask)
    if clipping_by_rectangle:
        if rectangle_bounds is None or not np.all(np.isfinite(rectangle_bounds)):
            return _clip_constructive_parts_to_native_tabular_result(
                source=gdf,
                parts=(),
                ordered_row_positions=np.empty(0, dtype=np.intp),
                clipping_by_rectangle=True,
                has_non_point_candidates=False,
                keep_geom_type=keep_geom_type,
            )
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask

    candidate_query_geometry = (
        query_geometry
        if isinstance(query_geometry, GeoDataFrame | GeoSeries | Polygon | MultiPolygon)
        else intersection_polygon
    )
    device_candidate_row_result = _bbox_device_candidate_rows_for_scalar_clip_mask_result(
        gdf,
        candidate_query_geometry,
        sort=sort,
    )
    if device_candidate_row_result is not None:
        direct_point_result = _clip_homogeneous_point_device_candidates_native(
            gdf,
            intersection_polygon,
            device_candidate_row_result.device_rows,
            clipping_by_rectangle=clipping_by_rectangle,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=keep_geom_type,
        )
        if direct_point_result is not None:
            return direct_point_result
        direct_line_result = _clip_homogeneous_line_rectangle_device_candidates_native(
            gdf,
            device_candidate_row_result.device_rows,
            clipping_by_rectangle=clipping_by_rectangle,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=keep_geom_type,
        )
        if direct_line_result is not None:
            return direct_line_result
        direct_polygon_result = _clip_homogeneous_polygon_device_candidates_native(
            gdf,
            intersection_polygon,
            device_candidate_row_result.device_rows,
            mask_owned=mask_owned,
            clipping_by_rectangle=clipping_by_rectangle,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=keep_geom_type,
        )
        if direct_polygon_result is not None:
            return direct_polygon_result
        mixed_result = _clip_mixed_device_candidates_native(
            gdf,
            intersection_polygon,
            device_candidate_row_result.device_rows,
            clipping_by_rectangle=clipping_by_rectangle,
            rectangle_bounds=rectangle_bounds,
            keep_geom_type=keep_geom_type,
        )
        if mixed_result is not None:
            return mixed_result

    relation_device_candidate_row_result = None
    if device_candidate_row_result is None and query_geometry is not None:
        relation_device_candidate_row_result = (
            _clip_device_candidate_rows_from_native_relation_result(
                gdf,
                query_geometry,
                sort=sort,
            )
        )
        if relation_device_candidate_row_result is not None:
            direct_point_result = _clip_homogeneous_point_device_candidates_native(
                gdf,
                intersection_polygon,
                relation_device_candidate_row_result.device_rows,
                clipping_by_rectangle=clipping_by_rectangle,
                rectangle_bounds=rectangle_bounds,
                keep_geom_type=keep_geom_type,
            )
            if direct_point_result is not None:
                return direct_point_result
            direct_polygon_result = _clip_homogeneous_polygon_device_candidates_native(
                gdf,
                intersection_polygon,
                relation_device_candidate_row_result.device_rows,
                mask_owned=mask_owned,
                clipping_by_rectangle=clipping_by_rectangle,
                rectangle_bounds=rectangle_bounds,
                keep_geom_type=keep_geom_type,
            )
            if direct_polygon_result is not None:
                return direct_polygon_result
            mixed_result = _clip_mixed_device_candidates_native(
                gdf,
                intersection_polygon,
                relation_device_candidate_row_result.device_rows,
                clipping_by_rectangle=clipping_by_rectangle,
                rectangle_bounds=rectangle_bounds,
                keep_geom_type=keep_geom_type,
            )
            if mixed_result is not None:
                return mixed_result

    candidate_row_result = _bbox_candidate_rows_for_scalar_clip_mask_result(
        gdf,
        candidate_query_geometry,
        sort=sort,
    )
    candidate_rows_spatially_ordered = bool(
        candidate_row_result is not None and candidate_row_result.spatially_ordered
    )
    candidate_device_rows = (
        None if candidate_row_result is None else candidate_row_result.device_rows
    )
    if candidate_row_result is None:
        query_input = query_geometry if query_geometry is not None else intersection_polygon
        if relation_device_candidate_row_result is not None:
            candidate_row_result = None
        else:
            candidate_row_result = _clip_candidate_rows_from_native_relation(
                gdf,
                query_input,
                sort=sort,
            )
        if candidate_row_result is None:
            candidate_rows = np.asarray(
                gdf.sindex.query(query_input, predicate="intersects", sort=sort),
                dtype=np.int32,
            )
        else:
            candidate_rows = candidate_row_result.rows
            candidate_device_rows = candidate_row_result.device_rows
            candidate_rows_spatially_ordered = candidate_row_result.spatially_ordered
    else:
        candidate_rows = candidate_row_result.rows
    if candidate_rows.ndim == 2:
        candidate_device_rows = None
        candidate_rows_spatially_ordered = False
        right_rows = candidate_rows[1]
        if sort:
            candidate_rows = np.unique(right_rows).astype(np.int32, copy=False)
        else:
            _unique_rows, first_hits = np.unique(right_rows, return_index=True)
            candidate_rows = right_rows[np.sort(first_hits)].astype(np.int32, copy=False)
    if not sort and candidate_rows.size > 1 and not candidate_rows_spatially_ordered:
        source_bounds = np.asarray(
            (gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf).bounds,
            dtype=np.float64,
        )
        candidate_bounds = source_bounds[candidate_rows]
        order = np.lexsort(
            (
                candidate_rows,
                candidate_bounds[:, 3],
                candidate_bounds[:, 2],
                candidate_bounds[:, 1],
                candidate_bounds[:, 0],
            )
        )
        candidate_rows = candidate_rows[order].astype(np.int32, copy=False)
        candidate_device_rows = None
    _raise_for_invalid_polygon_clip_candidates(
        gdf,
        intersection_polygon,
        candidate_rows,
        clipping_by_rectangle=clipping_by_rectangle,
    )
    ordered_row_positions = candidate_rows.astype(np.intp, copy=False)
    direct_point_result = _clip_homogeneous_point_candidates_native(
        gdf,
        intersection_polygon,
        ordered_row_positions,
        candidate_device_rows=candidate_device_rows,
        clipping_by_rectangle=clipping_by_rectangle,
        rectangle_bounds=rectangle_bounds,
        keep_geom_type=keep_geom_type,
    )
    if direct_point_result is not None:
        return direct_point_result
    direct_polygon_result = _clip_homogeneous_polygon_candidates_native(
        gdf,
        intersection_polygon,
        ordered_row_positions,
        candidate_device_rows=candidate_device_rows,
        mask_owned=mask_owned,
        clipping_by_rectangle=clipping_by_rectangle,
        rectangle_bounds=rectangle_bounds,
        keep_geom_type=keep_geom_type,
    )
    if direct_polygon_result is not None:
        return direct_polygon_result

    gdf_sub = gdf.iloc[candidate_rows]

    family_masks = _clip_family_masks(gdf_sub)
    point_mask = family_masks.point
    non_point_mask = family_masks.non_point
    multiline_mask = family_masks.multiline
    simple_line_mask = family_masks.simple_line
    polygon_mask = family_masks.polygon
    generic_mask = family_masks.generic
    rectangle_cleanup_safe = bool(
        rectangle_bounds is not None and (clipping_by_rectangle or family_masks.all_polygon)
    )

    def _clip_partition_values(partition, *, use_rect_fast_path=False):
        partition_families = _clip_family_masks(partition)
        if not clipping_by_rectangle and partition_families.all_point:
            return _clip_point_partition_with_polygon_mask(
                partition,
                mask,
            )
        if rectangle_bounds is not None and keep_geom_type and partition_families.all_polygonal:
            return _clip_polygon_partition_with_rectangle_mask(
                partition,
                rectangle_bounds,
                keep_geom_type_only=True,
            )
        if rectangle_bounds is not None and partition_families.all_polygon:
            return _clip_polygon_partition_with_rectangle_mask(partition, rectangle_bounds)
        if rectangle_bounds is not None and partition_families.all_polygonal:
            return _clip_complex_polygon_partition_with_rectangle_mask(
                partition,
                rectangle_bounds,
            )

        if isinstance(partition, GeoDataFrame):
            if not clipping_by_rectangle and partition_families.all_polygonal:
                return _clip_polygon_partition_with_polygon_mask(
                    partition,
                    mask,
                    keep_geom_type_only=keep_geom_type,
                )

            return (
                partition.geometry.values.clip_by_rect(*rectangle_bounds)
                if use_rect_fast_path
                else partition.geometry.values.intersection(mask)
            )

        if not clipping_by_rectangle and partition_families.all_polygonal:
            return _clip_polygon_partition_with_polygon_mask(
                partition,
                mask,
                keep_geom_type_only=keep_geom_type,
            )

        return (
            partition.values.clip_by_rect(*rectangle_bounds)
            if use_rect_fast_path
            else partition.values.intersection(mask)
        )

    parts: list[LeftConstructiveResult] = []

    def _append_part(selection_mask, *, use_rect_fast_path=False, passthrough=False):
        if not selection_mask.any():
            return
        local_mask = np.asarray(selection_mask, dtype=bool)
        if local_mask.size == len(gdf_sub) and bool(local_mask.all()):
            partition = gdf_sub
            row_positions = candidate_rows
            row_positions_device = candidate_device_rows
        else:
            partition = gdf_sub[local_mask]
            row_positions = candidate_rows[local_mask]
            row_positions_device = _clip_take_candidate_device_rows(
                candidate_device_rows,
                local_mask,
            )
        if (
            has_gpu_runtime()
            and _clip_partition_supports_device_promotion(partition)
            and (passthrough or use_rect_fast_path or not clipping_by_rectangle)
        ):
            partition = _promote_geometry_backing_to_device(
                partition,
                reason="clip selected candidate-limited GPU-native partition execution",
            )
        geometry_values = (
            (partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values)
            if passthrough
            else _clip_partition_values(
                partition,
                use_rect_fast_path=use_rect_fast_path,
            )
        )
        parts.append(
            _build_clip_partition_result(
                gdf,
                row_positions,
                geometry_values,
                row_positions_device=row_positions_device,
            )
        )

    _append_part(point_mask, passthrough=clipping_by_rectangle)
    _append_part(
        simple_line_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )
    _append_part(
        multiline_mask,
        use_rect_fast_path=rectangle_bounds is not None,
    )
    _append_part(
        polygon_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )
    _append_part(
        generic_mask,
        use_rect_fast_path=clipping_by_rectangle,
    )

    parts_tuple = tuple(parts)
    return _clip_constructive_parts_to_native_tabular_result(
        source=gdf,
        parts=parts_tuple,
        ordered_row_positions=ordered_row_positions,
        clipping_by_rectangle=rectangle_cleanup_safe,
        has_non_point_candidates=bool(non_point_mask.any()),
        keep_geom_type=keep_geom_type,
        spatial_materializer=lambda: ClipNativeResult(
            source=gdf,
            parts=parts_tuple,
            ordered_index=gdf_sub.index,
            ordered_row_positions=ordered_row_positions,
            clipping_by_rectangle=rectangle_cleanup_safe,
            has_non_point_candidates=bool(non_point_mask.any()),
            keep_geom_type=keep_geom_type,
        ).to_spatial(),
    )


def _clip_source_bounds_geometry(bounds):
    bounds = tuple(float(value) for value in bounds)
    if not np.all(np.isfinite(bounds)):
        return None
    xmin, ymin, xmax, ymax = bounds
    if abs(xmax - xmin) <= SPATIAL_EPSILON and abs(ymax - ymin) <= SPATIAL_EPSILON:
        return Point(xmin, ymin)
    if abs(xmax - xmin) <= SPATIAL_EPSILON:
        return LineString([(xmin, ymin), (xmax, ymax)])
    if abs(ymax - ymin) <= SPATIAL_EPSILON:
        return LineString([(xmin, ymin), (xmax, ymax)])
    return box(xmin, ymin, xmax, ymax)


def _clip_mask_covers_source_bounds(mask, source_bounds) -> bool:
    source_extent = _clip_source_bounds_geometry(source_bounds)
    if source_extent is None:
        return False

    if _mask_is_list_like_rectangle(mask):
        try:
            xmin, ymin, xmax, ymax = (float(value) for value in mask)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite((xmin, ymin, xmax, ymax))):
            return False
        sxmin, symin, sxmax, symax = (float(value) for value in source_bounds)
        return bool(
            xmin <= sxmin + SPATIAL_EPSILON
            and ymin <= symin + SPATIAL_EPSILON
            and xmax + SPATIAL_EPSILON >= sxmax
            and ymax + SPATIAL_EPSILON >= symax
        )

    if not isinstance(mask, Polygon | MultiPolygon) or mask.is_empty:
        return False
    return bool(shapely.covers(mask, source_extent))


def _clip_mask_covers_source_bounds_passthrough_native(
    source,
    mask,
    source_bounds,
) -> NativeTabularResult | NativeTabularSelection | None:
    if not _clip_mask_covers_source_bounds(mask, source_bounds):
        return None
    _raise_for_invalid_polygon_clip_candidates(
        source,
        mask,
        np.arange(len(source), dtype=np.intp),
        clipping_by_rectangle=_mask_is_list_like_rectangle(mask),
    )

    keep_rowset = _native_state_valid_nonempty_rowset(source)
    keep_mask = None
    if keep_rowset is not None:
        native_result = _native_state_passthrough_take(source, keep_rowset)
        kept_rows = (
            "device-resident"
            if isinstance(keep_rowset, NativeDeviceSelection)
            else len(keep_rowset)
        )
    else:
        geometry = source.geometry if isinstance(source, GeoDataFrame) else source
        keep_mask = ~(
            np.asarray(geometry.isna(), dtype=bool) | np.asarray(geometry.is_empty, dtype=bool)
        )
        native_result = _native_state_passthrough_take(source, keep_mask)
        kept_rows = int(np.count_nonzero(keep_mask))
    if native_result is None:
        if keep_mask is None:
            keep_rows = keep_rowset.to_host_positions(
                surface="vibespatial.api.tools.clip.mask_cover_passthrough_rows",
                strict_disallowed=False,
            )
            keep_mask = np.zeros(len(source), dtype=bool)
            keep_mask[np.asarray(keep_rows, dtype=np.intp)] = True
        passthrough = _take_spatial_rows(source, keep_mask)
        values = (
            passthrough.geometry.values
            if isinstance(passthrough, GeoDataFrame)
            else passthrough.values
        )
        owned = getattr(values, "_owned", None)
        native_result = _spatial_to_native_tabular_result(passthrough)
    else:
        exact_result = (
            native_result.capacity_result
            if isinstance(native_result, NativeTabularSelection)
            else native_result
        )
        owned = getattr(exact_result.geometry, "owned", None)
    selected = (
        ExecutionMode.GPU
        if owned is not None and owned.residency is Residency.DEVICE
        else ExecutionMode.CPU
    )

    from vibespatial.runtime.dispatch import record_dispatch_event

    record_dispatch_event(
        surface="geopandas.clip",
        operation="clip",
        implementation="mask_covers_source_bounds_passthrough",
        reason=(
            "mask clip physical shape reduced to source passthrough because the "
            "mask covers the source total-bounds extent"
        ),
        detail=(f"rows={len(source)}; kept_rows={kept_rows}; physical_shape=mask_clip"),
        requested=ExecutionMode.AUTO,
        selected=selected,
    )
    return native_result


def _native_state_valid_nonempty_rowset(source):
    from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if isinstance(source, GeoDataFrame)
        else getattr(source, "name", None) or "geometry"
    )
    state = _clip_native_state_for_source(source, geometry_name)
    if state is None:
        return None
    owned = getattr(state.geometry, "owned", None)
    if owned is None:
        return None

    if owned.residency is Residency.DEVICE and owned.device_state is not None:
        if not has_gpu_runtime():
            return None
        d_keep = _owned_valid_nonempty_device_mask(owned)
        if d_keep is None:
            return None
        return NativeDeviceSelection.from_mask(
            d_keep,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
        )

    keep = _native_state_valid_nonempty_keep_mask(source)
    if keep is None:
        return None
    positions = np.flatnonzero(keep).astype(np.int64, copy=False)
    return NativeRowSet.from_positions(
        positions,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
        identity=positions.size == state.row_count,
    )


def _native_state_valid_nonempty_keep_mask(source) -> np.ndarray | None:
    from vibespatial.geometry.owned import FAMILY_TAGS

    geometry_name = (
        source._geometry_column_name
        if isinstance(source, GeoDataFrame)
        else getattr(source, "name", None) or "geometry"
    )
    state = _clip_native_state_for_source(source, geometry_name)
    if state is None:
        return None
    owned = getattr(state.geometry, "owned", None)
    if owned is None:
        return None

    if owned.residency is Residency.DEVICE and owned.device_state is not None:
        if not has_gpu_runtime():
            return None
        d_keep = _owned_valid_nonempty_device_mask(owned)
        if d_keep is None:
            return None
        return _clip_device_to_host(
            d_keep,
            reason="clip mask-cover passthrough valid-nonempty row mask",
        ).astype(bool, copy=False)

    keep = np.asarray(owned.validity, dtype=bool).copy()
    tags = np.asarray(owned.tags, dtype=np.int8)
    family_row_offsets = np.asarray(owned.family_row_offsets, dtype=np.int64)
    for family, buffer in owned.families.items():
        family_rows = np.flatnonzero(keep & (tags == np.int8(FAMILY_TAGS[family])))
        if family_rows.size == 0:
            continue
        local_rows = family_row_offsets[family_rows]
        keep[family_rows] &= ~np.asarray(buffer.empty_mask, dtype=bool)[local_rows]
    return keep


def _native_state_passthrough_take(
    source,
    keep_mask,
) -> NativeTabularResult | NativeTabularSelection | None:
    from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet

    geometry_name = (
        source._geometry_column_name
        if isinstance(source, GeoDataFrame)
        else getattr(source, "name", None) or "geometry"
    )
    state = _clip_native_state_for_source(source, geometry_name)
    if state is None:
        return None
    if isinstance(keep_mask, NativeDeviceSelection):
        import cupy as cp

        from vibespatial.api._native_result_core import NativeGeometryProvenance

        selection = keep_mask
        if selection.source_token is None:
            selection = replace(
                selection,
                source_token=state.lineage_token,
                source_row_count=state.row_count,
            )
        if (
            selection.source_token != state.lineage_token
            or selection.source_row_count != state.row_count
        ):
            return None
        capacity_result = state.to_native_tabular_result()
        if capacity_result.provenance is None:
            capacity_result = replace(
                capacity_result,
                provenance=NativeGeometryProvenance(
                    operation="clip_passthrough",
                    row_count=state.row_count,
                    source_rows=cp.arange(state.row_count, dtype=cp.int64),
                    source_tokens=(state.lineage_token,),
                ),
            )
        return NativeTabularSelection(
            capacity_result=capacity_result,
            selection=selection,
        )
    if isinstance(keep_mask, NativeRowSet):
        rowset = keep_mask
        if rowset.source_token is None:
            rowset = NativeRowSet.from_positions(
                rowset.positions,
                source_token=state.lineage_token,
                source_row_count=state.row_count,
                ordered=rowset.ordered,
                unique=rowset.unique,
                identity=rowset.identity,
            )
        return state.take(rowset, preserve_index=True).to_native_tabular_result()
    keep_rows = np.flatnonzero(np.asarray(keep_mask, dtype=bool)).astype(
        np.int64,
        copy=False,
    )
    if keep_rows.size == state.row_count:
        return state.to_native_tabular_result()

    positions = keep_rows
    if (
        state.index_plan.kind in {"range", "device-labels", "host-labels", "host-labels-take"}
        and has_gpu_runtime()
    ):
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by GPU runtime
            cp = None
        if cp is not None:
            positions = cp.asarray(keep_rows, dtype=cp.int64)
    rowset = NativeRowSet.from_positions(
        positions,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
    )
    return state.take(rowset, preserve_index=True).to_native_tabular_result()


def _drop_clip_terminal_native_state(spatial):
    from vibespatial.api._native_state import drop_native_state

    drop_native_state(spatial)
    return spatial


def _apply_clip_keep_geom_type_terminal(source, clipped, *, keep_geom_type: bool):
    """Apply GeoPandas keep-geometry-type semantics at the public clip export."""
    if not keep_geom_type:
        return clipped

    from vibespatial.api.tools.overlay import _strip_non_polygon_collection_parts

    clipped = _drop_clip_terminal_native_state(clipped)
    source_geom_type = source.geom_type
    clipped_geom_type = clipped.geom_type
    geomcoll_concat = (clipped_geom_type == "GeometryCollection").any()
    geomcoll_orig = (source_geom_type == "GeometryCollection").any()
    new_collection = geomcoll_concat and not geomcoll_orig

    if geomcoll_orig:
        warnings.warn(
            "keep_geom_type can not be called on a GeoDataFrame with GeometryCollection.",
            stacklevel=3,
        )
        return clipped

    orig_types_total = sum(
        [
            source_geom_type.isin(POLYGON_GEOM_TYPES).any(),
            source_geom_type.isin(LINE_GEOM_TYPES).any(),
            source_geom_type.isin(POINT_GEOM_TYPES).any(),
        ]
    )
    clip_types_total = sum(
        [
            clipped_geom_type.isin(POLYGON_GEOM_TYPES).any(),
            clipped_geom_type.isin(LINE_GEOM_TYPES).any(),
            clipped_geom_type.isin(POINT_GEOM_TYPES).any(),
        ]
    )
    more_types = orig_types_total < clip_types_total

    if orig_types_total > 1:
        warnings.warn(
            "keep_geom_type can not be called on a mixed type GeoDataFrame.",
            stacklevel=3,
        )
        return clipped

    if not (new_collection or more_types):
        return clipped

    orig_type = source_geom_type.iloc[0]
    if orig_type in POLYGON_GEOM_TYPES:
        if new_collection:
            geometry = clipped.geometry if isinstance(clipped, GeoDataFrame) else clipped
            cleaned = _strip_non_polygon_collection_parts(np.asarray(geometry, dtype=object))
            keep = ~(shapely.is_missing(cleaned) | shapely.is_empty(cleaned))
            if isinstance(clipped, GeoDataFrame):
                clipped = _replace_geometry_column(
                    clipped.copy(deep=not PANDAS_GE_30),
                    _geometryarray_from_shapely(
                        cleaned,
                        crs=getattr(source, "crs", None),
                    ),
                )
            else:
                clipped = GeoSeries(
                    _geometryarray_from_shapely(
                        cleaned,
                        crs=getattr(source, "crs", None),
                    ),
                    index=clipped.index,
                    crs=getattr(source, "crs", None),
                    name=clipped.name,
                )
            clipped = clipped[keep]
        return clipped.loc[clipped.geom_type.isin(POLYGON_GEOM_TYPES)]

    if orig_type in LINE_GEOM_TYPES:
        if new_collection:
            clipped = clipped.explode(index_parts=False)
        return clipped.loc[clipped.geom_type.isin(LINE_GEOM_TYPES)]

    if orig_type in POINT_GEOM_TYPES:
        return clipped.loc[clipped.geom_type.isin(POINT_GEOM_TYPES)]

    return clipped


def _clip_native_tabular_to_spatial(
    result: NativeTabularResult | NativeTabularSelection,
    *,
    source: GeoDataFrame | GeoSeries,
    keep_geom_type: bool = False,
):
    if isinstance(source, GeoDataFrame):
        clipped = result.to_geodataframe(lazy_public_index=False)
        clipped = _apply_clip_keep_geom_type_terminal(
            source,
            clipped,
            keep_geom_type=keep_geom_type,
        )
        _maybe_seed_polygon_validity_cache(clipped)
        return clipped

    if isinstance(result, NativeTabularSelection):
        result = result.to_native_tabular_result(
            surface="vibespatial.api.tools.clip._clip_native_tabular_to_spatial",
            strict_disallowed=False,
        )

    export_attributes = result.attributes_for_export(
        surface="vibespatial.api.tools.clip._clip_native_tabular_to_spatial",
        include_index=True,
        strict_disallowed=False,
        lazy_public_index=False,
    )
    clipped = result.geometry.to_geoseries(
        index=export_attributes.index,
        name=getattr(source, "name", None) or result.geometry_name,
    )
    if result.attrs:
        clipped.attrs.update(result.attrs)
    clipped = _apply_clip_keep_geom_type_terminal(
        source,
        clipped,
        keep_geom_type=keep_geom_type,
    )
    _maybe_seed_polygon_validity_cache(clipped)
    return clipped


def _clip_take_owned_rows_native(owned, rows: np.ndarray, *, device_rows=None):
    """Take concrete owned rows while preserving device row indirection."""
    from vibespatial.runtime.residency import Residency

    rows = np.asarray(rows, dtype=np.intp)
    if rows.size == 0:
        return owned.take(rows)
    if rows.size == owned.row_count and np.array_equal(
        rows, np.arange(owned.row_count, dtype=rows.dtype)
    ):
        return owned
    if owned.residency is Residency.DEVICE and has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            d_rows = (
                cp.asarray(device_rows, dtype=cp.int64)
                if device_rows is not None
                else cp.asarray(rows, dtype=cp.int64)
            )
            return owned.device_take(
                d_rows,
                host_indices_for_sizing=np.asarray(rows, dtype=np.int64),
            )
    return owned.take(rows)


def _build_sparse_owned_clip_output(
    *,
    partition_crs,
    left_owned,
    inside_rows: np.ndarray,
    inside_rows_device=None,
    exact_area_owned,
    positive_local_rows: np.ndarray,
    positive_local_rows_device=None,
    positive_rows: np.ndarray,
    positive_rows_device=None,
    extra_owned_parts=(),
):
    from vibespatial.geometry.owned import OwnedGeometryArray, build_null_owned_array

    normalized_extra_parts = []
    for part in tuple(extra_owned_parts):
        if len(part) == 2:
            rows, owned = part
            device_rows = None
        elif len(part) == 3:
            rows, owned, device_rows = part
        else:
            raise ValueError("extra owned clip parts must be (rows, owned[, device_rows])")
        normalized_extra_parts.append((rows, owned, device_rows))
    extra_owned_parts = tuple(normalized_extra_parts)

    row_parts = [
        np.asarray(inside_rows, dtype=np.intp),
        np.asarray(positive_rows, dtype=np.intp),
        *[np.asarray(rows, dtype=np.intp) for rows, _owned, _device_rows in extra_owned_parts],
    ]
    kept_local_rows = np.concatenate(row_parts)
    if kept_local_rows.size == 0:
        return _ClipPartitionOutput(
            geometry_values=_geometry_values_from_owned(
                build_null_owned_array(0, residency=left_owned.residency),
                crs=partition_crs,
            ),
            local_rows=np.empty(0, dtype=np.intp),
        )

    kept_local_rows_device = None
    if has_gpu_runtime():
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            device_row_parts = []
            if inside_rows.size > 0:
                if inside_rows_device is None:
                    device_row_parts = []
                else:
                    device_row_parts.append(cp.asarray(inside_rows_device, dtype=cp.int64))
            if positive_rows.size > 0 and (device_row_parts or inside_rows.size == 0):
                if positive_rows_device is None:
                    device_row_parts = []
                else:
                    device_row_parts.append(cp.asarray(positive_rows_device, dtype=cp.int64))
            for rows, _owned, device_rows in extra_owned_parts:
                if np.asarray(rows, dtype=np.intp).size == 0:
                    continue
                if device_rows is None:
                    device_row_parts = []
                    break
                if device_row_parts or (inside_rows.size == 0 and positive_rows.size == 0):
                    device_row_parts.append(cp.asarray(device_rows, dtype=cp.int64))
                else:
                    device_row_parts = []
                    break
            if device_row_parts:
                kept_local_rows_device = cp.concatenate(device_row_parts)

    owned_parts = []
    if inside_rows.size > 0:
        owned_parts.append(
            _clip_take_owned_rows_native(
                left_owned,
                np.asarray(inside_rows, dtype=np.intp),
                device_rows=inside_rows_device,
            )
        )
    if positive_local_rows.size > 0:
        owned_parts.append(
            _clip_take_owned_rows_native(
                exact_area_owned,
                np.asarray(positive_local_rows, dtype=np.intp),
                device_rows=positive_local_rows_device,
            )
        )
    owned_parts.extend(owned for _rows, owned, _device_rows in extra_owned_parts)

    result_owned = OwnedGeometryArray.concat(owned_parts)
    ordered_local_rows = kept_local_rows
    if np.any(ordered_local_rows[1:] < ordered_local_rows[:-1]):
        reorder = np.argsort(ordered_local_rows, kind="stable").astype(np.intp, copy=False)
        reorder_device = None
        if kept_local_rows_device is not None and has_gpu_runtime():
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                cp = None
            if cp is not None:
                d_ordered_local_rows = cp.asarray(kept_local_rows_device, dtype=cp.int64)
                reorder_device = cp.argsort(d_ordered_local_rows).astype(
                    cp.int64,
                    copy=False,
                )
                kept_local_rows_device = d_ordered_local_rows[reorder_device]
        result_owned = _clip_take_owned_rows_native(
            result_owned,
            reorder,
            device_rows=reorder_device,
        )
        ordered_local_rows = ordered_local_rows[reorder]
        if reorder_device is None:
            kept_local_rows_device = None
    result_owned._clip_semantically_clean = True

    return _ClipPartitionOutput(
        geometry_values=_geometry_values_from_owned(result_owned, crs=partition_crs),
        local_rows=ordered_local_rows,
        local_rows_device=kept_local_rows_device,
    )


def _clip_gdf_with_mask(gdf, mask, sort=False, *, query_geometry=None):
    """
    Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    native_result = _clip_gdf_with_mask_native(
        gdf,
        mask,
        sort=sort,
        query_geometry=query_geometry,
        keep_geom_type=False,
    )
    return _clip_native_tabular_to_spatial(
        native_result,
        source=gdf,
        keep_geom_type=False,
    )


def _flatten_geometrycollection_clip_ingress(values):
    """Flatten GeometryCollection members at the public compatibility ingress."""
    objects = np.asarray(values, dtype=object)
    geometries: list[object] = []
    source_rows: list[int] = []

    def append_geometry(geometry, source_row: int) -> None:
        if geometry is None or bool(shapely.is_missing(geometry)):
            return
        if geometry.geom_type != "GeometryCollection":
            geometries.append(geometry)
            source_rows.append(source_row)
            return
        for member in geometry.geoms:
            append_geometry(member, source_row)

    for source_row, geometry in enumerate(objects):
        append_geometry(geometry, source_row)
    return (
        geometries,
        np.asarray(source_rows, dtype=np.int64),
    )


def _clip_geometrycollection_source_native(
    source,
    mask,
    *,
    sort: bool,
) -> NativeTabularResult | NativeTabularSelection | None:
    """Clip public GeometryCollection rows through native grouped set union."""
    if not has_gpu_runtime():
        return None
    geometry = source.geometry if isinstance(source, GeoDataFrame) else source
    values = geometry.values
    # DeviceGeometryArray cannot contain a GeometryCollection: owned geometry
    # stores its concrete families columnarly.  Reading ``_data`` here would
    # materialize every device-resident source through Shapely merely to prove
    # that no collection is present.
    if isinstance(values, DeviceGeometryArray):
        return None
    public_values = getattr(values, "_data", None)
    if public_values is None:
        return None
    type_ids = shapely.get_type_id(np.asarray(public_values, dtype=object))
    if not bool(np.any(type_ids == 7)):
        return None

    flat_geometries, flat_source_rows = (
        _flatten_geometrycollection_clip_ingress(public_values)
    )
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if not flat_geometries:
        return _clip_constructive_parts_to_native_tabular_result(
            source=source,
            parts=(),
            ordered_row_positions=np.empty(0, dtype=np.intp),
            clipping_by_rectangle=clipping_by_rectangle,
            has_non_point_candidates=False,
            keep_geom_type=False,
        )

    from vibespatial.geometry.owned import from_shapely_geometries

    flat_owned = from_shapely_geometries(
        flat_geometries,
        residency=Residency.DEVICE if has_gpu_runtime() else Residency.HOST,
    )
    flat_values = (
        DeviceGeometryArray._from_owned(flat_owned, crs=source.crs)
        if flat_owned.residency is Residency.DEVICE
        else GeometryArray.from_owned(flat_owned, crs=source.crs)
    )
    flat_index = pd.RangeIndex(len(flat_geometries))
    if isinstance(source, GeoDataFrame):
        flat_source = source.iloc[flat_source_rows].copy(deep=False)
        flat_source.index = flat_index
        flat_source = _replace_geometry_column(flat_source, flat_values)
    else:
        flat_source = _geometry_series_from_values(
            flat_values,
            index=flat_index,
            crs=source.crs,
            name=getattr(source, "name", None),
        )

    flat_result = evaluate_geopandas_clip_native(
        flat_source,
        mask,
        keep_geom_type=False,
        sort=sort,
    )
    capacity_result = (
        flat_result.capacity_result
        if isinstance(flat_result, NativeTabularSelection)
        else flat_result
    )

    import cupy as cp

    capacity = capacity_result.geometry.row_count
    provenance_rows = getattr(capacity_result.provenance, "source_rows", None)
    d_capacity_source_rows = (
        cp.arange(capacity, dtype=cp.int64)
        if provenance_rows is None
        else cp.asarray(provenance_rows, dtype=cp.int64)
    )
    if isinstance(flat_result, NativeTabularSelection):
        d_active_capacity = flat_result.selection.active_capacity_mask()
        d_capacity_positions = flat_result.selection.safe_capacity_positions()
        selected_geometry = capacity_result.geometry.take(d_capacity_positions).mask_capacity(
            d_active_capacity
        )
        d_selected_flat_rows = d_capacity_source_rows[d_capacity_positions]
    else:
        d_active_capacity = cp.ones(capacity, dtype=cp.bool_)
        selected_geometry = capacity_result.geometry
        d_selected_flat_rows = d_capacity_source_rows

    from vibespatial.api._native_result_core import (
        NativeGeometryCompositionPart,
        NativeGeometryProvenance,
    )

    if selected_geometry.composition is None:
        concrete_parts = (
            NativeGeometryCompositionPart(
                geometry=selected_geometry,
                output_rows=cp.arange(capacity, dtype=cp.int64),
                collection_position=0,
            ),
        )
    else:
        concrete_parts = selected_geometry.composition.parts

    d_flat_source_rows = cp.asarray(flat_source_rows, dtype=cp.int64)
    source_row_count = len(source)
    concrete_owned = []
    concrete_group_rows = []
    concrete_active = []
    for child_part in concrete_parts:
        owned = child_part.geometry.owned
        if owned is None:
            raise StrictNativeFallbackError(
                "GeometryCollection clip reduction lost concrete device storage"
            )
        d_child_rows = cp.asarray(child_part.output_rows, dtype=cp.int64)
        d_child_flat_rows = d_selected_flat_rows[d_child_rows]
        d_child_valid = child_part.geometry.valid_nonempty_mask_device()
        if d_child_valid is None:
            raise StrictNativeFallbackError(
                "GeometryCollection clip reduction lost device validity metadata"
            )
        concrete_owned.append(owned)
        concrete_group_rows.append(d_flat_source_rows[d_child_flat_rows])
        concrete_active.append(cp.asarray(d_child_valid, dtype=cp.bool_))

    from vibespatial.geometry.owned import OwnedGeometryArray

    grouped_input = OwnedGeometryArray.concat(concrete_owned)
    if grouped_input.row_count == 0:
        return _clip_constructive_parts_to_native_tabular_result(
            source=source,
            parts=(),
            ordered_row_positions=np.empty(0, dtype=np.intp),
            clipping_by_rectangle=clipping_by_rectangle,
            has_non_point_candidates=False,
            keep_geom_type=False,
        )
    d_group_rows = cp.concatenate(concrete_group_rows)
    d_group_active = cp.concatenate(concrete_active)
    grouped_polygon = _clip_grouped_polygon_pair_capacity(
        grouped_input,
        d_group_rows,
        d_group_active,
        output_row_count=source_row_count,
    )
    if grouped_polygon is None:
        raise StrictNativeFallbackError(
            "GeometryCollection clip polygon coverage reduction declined"
        )
    polygon_capacity, d_polygon_keep = grouped_polygon
    from vibespatial.constructive.grouped_mixed_union import (
        grouped_mixed_union_capacity_device,
    )

    mixed_capacity = grouped_mixed_union_capacity_device(
        grouped_input,
        d_group_rows,
        d_group_active,
        polygon_capacity,
        d_polygon_keep,
        output_row_count=source_row_count,
        crs=source.crs,
    )
    d_source_active = cp.asarray(mixed_capacity.keep_mask, dtype=cp.bool_)

    geometry_name = (
        source._geometry_column_name
        if isinstance(source, GeoDataFrame)
        else getattr(source, "name", None) or "geometry"
    )
    attributes = (
        source.drop(columns=[geometry_name]).copy(deep=False)
        if isinstance(source, GeoDataFrame)
        else pd.DataFrame(index=source.index)
    )
    result = NativeTabularResult(
        attributes=attributes,
        geometry=mixed_capacity.geometry,
        geometry_name=geometry_name,
        column_order=(
            tuple(source.columns)
            if isinstance(source, GeoDataFrame)
            else (geometry_name,)
        ),
        attrs=source.attrs.copy() or None,
        provenance=NativeGeometryProvenance(
            operation="clip_geometrycollection_grouped_union",
            row_count=source_row_count,
            source_rows=cp.arange(source_row_count, dtype=cp.int64),
        ),
    )
    from vibespatial.runtime.dispatch import record_dispatch_event

    record_dispatch_event(
        surface="clip",
        operation="clip",
        implementation="geometrycollection_grouped_union_gpu",
        reason=(
            "GeometryCollection members clipped and reduced through native grouped "
            "mixed union semantics"
        ),
        detail=f"rows={source_row_count},parts={len(flat_geometries)}",
        selected=ExecutionMode.GPU,
    )
    selected_result = NativeTabularSelection(
        capacity_result=result,
        selection=NativeDeviceSelection.from_mask(d_source_active),
    )
    if not sort or source_row_count <= 1:
        return selected_result

    source_order = np.asarray(
        pd.Series(
            np.arange(source_row_count, dtype=np.int64),
            index=source.index,
        )
        .sort_index(kind="stable")
        .to_numpy(copy=False),
        dtype=np.int64,
    )
    source_sort_rank = np.empty(source_row_count, dtype=np.int64)
    source_sort_rank[source_order] = np.arange(source_row_count, dtype=np.int64)
    return selected_result.sort_selected_by_int64(cp.asarray(source_sort_rank))


def evaluate_geopandas_clip_native(
    gdf,
    mask,
    *,
    keep_geom_type: bool = False,
    sort: bool = False,
) -> NativeTabularResult | NativeTabularSelection:
    """Build a native clip result and defer GeoPandas export to the boundary."""
    original = gdf

    if not isinstance(gdf, GeoDataFrame | GeoSeries):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if (
        not isinstance(mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon)
        and not clipping_by_rectangle
    ):
        raise TypeError(
            "'mask' should be GeoDataFrame, GeoSeries,"
            f"(Multi)Polygon or list-like, got {type(mask)}"
        )

    if clipping_by_rectangle and len(mask) != 4:
        raise TypeError("If 'mask' is list-like, it must have four values (minx, miny, maxx, maxy)")

    if isinstance(mask, GeoDataFrame | GeoSeries) and not _check_crs(gdf, mask):
        _crs_mismatch_warn(gdf, mask, stacklevel=3)

    geometrycollection_result = _clip_geometrycollection_source_native(
        gdf,
        mask,
        sort=sort,
    )
    if geometrycollection_result is not None:
        return geometrycollection_result

    lazy_grouped_mask_owned = (
        _lazy_grouped_union_mask_owned_private(mask)
        if isinstance(mask, GeoDataFrame | GeoSeries)
        else None
    )
    host_source_bounds = (
        None if _spatial_prefers_device_bounds_private(gdf) else _spatial_total_bounds_private(gdf)
    )

    if clipping_by_rectangle:
        polygon_mask_bounds = None
    elif lazy_grouped_mask_owned is not None:
        polygon_mask_bounds = None
    elif isinstance(mask, GeoDataFrame | GeoSeries) and len(mask) == 1:
        polygon_mask_bounds = None
        if _single_row_polygon_mask_owned_private(mask) is None:
            polygon_mask_bounds = _rectangle_bounds_from_mask(mask.geometry.iloc[0])
    else:
        polygon_mask_bounds = _rectangle_bounds_from_mask(mask)

    promote_polygon_mask_to_device = (
        has_gpu_runtime()
        and not clipping_by_rectangle
        and (
            polygon_mask_bounds is None
            or (strict_native_mode_enabled() and _spatial_all_polygonal_private(gdf))
        )
    )

    if promote_polygon_mask_to_device:
        gdf = _promote_geometry_backing_to_device(
            gdf,
            reason="clip(): GPU boundary selection for source geometry",
        )
        if isinstance(mask, GeoDataFrame | GeoSeries):
            mask = _promote_geometry_backing_to_device(
                mask,
                reason="clip(): GPU boundary selection for mask geometry",
            )

    device_bounds_routing = _spatial_prefers_device_bounds_private(gdf) or (
        isinstance(mask, GeoDataFrame | GeoSeries) and _spatial_prefers_device_bounds_private(mask)
    )
    box_gdf = host_source_bounds
    if not device_bounds_routing:
        if isinstance(mask, GeoDataFrame | GeoSeries):
            box_mask = _spatial_total_bounds_private(mask)
        elif clipping_by_rectangle:
            box_mask = mask
        else:
            box_mask = mask.bounds if not mask.is_empty else (np.nan,) * 4
        if box_gdf is None:
            box_gdf = _spatial_total_bounds_private(gdf)
        if not (
            ((box_mask[0] <= box_gdf[2]) and (box_gdf[0] <= box_mask[2]))
            and ((box_mask[1] <= box_gdf[3]) and (box_gdf[1] <= box_mask[3]))
        ):
            return _clip_constructive_parts_to_native_tabular_result(
                source=original,
                parts=(),
                ordered_row_positions=np.empty(0, dtype=np.intp),
                clipping_by_rectangle=clipping_by_rectangle,
                has_non_point_candidates=False,
                keep_geom_type=keep_geom_type,
            )

    if lazy_grouped_mask_owned is not None and not clipping_by_rectangle:
        lazy_clip_result = _clip_gdf_with_lazy_grouped_union_mask_native(
            gdf,
            mask,
            lazy_grouped_mask_owned,
            keep_geom_type=keep_geom_type,
            sort=sort,
        )
        if lazy_clip_result is not None:
            return lazy_clip_result

    single_row_mask_owned = None
    mask_query_geometry = None
    if isinstance(mask, GeoDataFrame | GeoSeries):
        mask_query_geometry = mask.geometry if isinstance(mask, GeoDataFrame) else mask
        if (
            len(mask) == 1
            and device_bounds_routing
            and not keep_geom_type
            and _spatial_all_polygonal_private(gdf)
        ):
            single_row_mask_owned = _single_row_polygon_mask_owned_private(mask)
        if len(mask) == 1 and single_row_mask_owned is not None:
            combined_mask = None
        elif len(mask) == 1:
            combined_mask = mask.geometry.iloc[0]
        else:
            combined_mask = mask.geometry.union_all()
    else:
        combined_mask = mask

    if box_gdf is not None:
        passthrough_result = _clip_mask_covers_source_bounds_passthrough_native(
            gdf,
            combined_mask,
            box_gdf,
        )
        if passthrough_result is not None:
            return passthrough_result

    return _clip_gdf_with_mask_native(
        gdf,
        combined_mask,
        sort=sort,
        query_geometry=mask_query_geometry,
        mask_owned=single_row_mask_owned,
        keep_geom_type=keep_geom_type,
    )


def clip(gdf, mask, keep_geom_type=False, sort=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The ``gdf`` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from ``gdf`` will be
    clipped to the total boundary of all polygons in mask.

    If the ``mask`` is list-like with four elements ``(minx, miny, maxx, maxy)``, a
    faster rectangle clipping algorithm will be used. Note that this can lead to
    slightly different results in edge cases, e.g. if a line would be reduced to a
    point, this point might not be returned.
    The geometry is clipped in a fast but possibly dirty way. The output is not
    guaranteed to be valid. No exceptions will be raised for topological errors.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.
        The mask's geometry is dissolved into one geometric feature
        and intersected with ``gdf``.
        If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
        ``clip`` will use a faster rectangle clipping (:meth:`~GeoSeries.clip_by_rect`),
        possibly leading to slightly different results.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).
    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from ``gdf`` clipped to
         polygon boundary from mask.

    See Also
    --------
    GeoDataFrame.clip : equivalent GeoDataFrame method
    GeoSeries.clip : equivalent GeoSeries method

    Examples
    --------
    Clip points (grocery stores) with polygons (the Near West Side community):

    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)
    >>> groceries.shape
    (148, 8)

    >>> nws_groceries = geopandas.clip(groceries, near_west_side)
    >>> nws_groceries.shape
    (7, 8)
    """
    native_result = evaluate_geopandas_clip_native(
        gdf,
        mask,
        keep_geom_type=keep_geom_type,
        sort=sort,
    )
    return _clip_native_tabular_to_spatial(
        native_result,
        source=gdf,
        keep_geom_type=keep_geom_type,
    )
