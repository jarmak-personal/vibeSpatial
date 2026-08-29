"""Prepared device carrier for bounded polygon-against-one-mask classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from vibespatial.constructive.representative_point import representative_point_owned
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
from vibespatial.kernels.core.spatial_query_kernels import _morton_range_kernels
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.precision import (
    CompensationMode,
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    RefinementMode,
)
from vibespatial.spatial.indexing import build_flat_spatial_index, generate_bounds_pairs
from vibespatial.spatial.segment_primitives import (
    DeviceRingLocalSegmentRelation,
    _extract_segments_gpu,
    device_segment_table_as_linestrings,
)
from vibespatial.spatial.spatial_index_device import (
    _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
    _morton_reduction_span_schedule,
    _prepare_morton_range_query,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - GPU-only carrier
    cp = None


# Charge each capacity lane for its compact relation and scan footprint.
_CANDIDATE_BYTES_ESTIMATE = 32
_LIVE_MEMORY_FRACTION = 8
_LIVE_MEMORY_RESERVE_BYTES = 256 << 20
_MAX_CANDIDATES_PER_QUERY_TILE = 8 * 1024

request_warmup(["exclusive_scan_i64"])


def prepared_polygon_mask_fp64_plan() -> PrecisionPlan:
    """Return the exact predicate plan implemented by the prepared mask kernels."""
    return PrecisionPlan(
        storage_precision=PrecisionMode.FP64,
        compute_precision=PrecisionMode.FP64,
        kernel_class=KernelClass.PREDICATE,
        compensation=CompensationMode.NONE,
        refinement=RefinementMode.NONE,
        center_coordinates=False,
        reason=(
            "prepared polygon mask classification uses authoritative fp64 "
            "orientation with exact expansion refinement"
        ),
    )


def _require_prepared_polygon_mask_precision_plan(
    precision_plan: PrecisionPlan,
) -> PrecisionPlan:
    if not isinstance(precision_plan, PrecisionPlan):
        raise TypeError("prepared polygon mask classification requires a PrecisionPlan")
    if precision_plan.kernel_class is not KernelClass.PREDICATE:
        raise ValueError("prepared polygon mask classification requires a PREDICATE PrecisionPlan")
    if (
        precision_plan.storage_precision is not PrecisionMode.FP64
        or precision_plan.compute_precision is not PrecisionMode.FP64
    ):
        raise NotImplementedError(
            "prepared polygon mask classification currently requires authoritative fp64"
        )
    if (
        precision_plan.compensation is not CompensationMode.NONE
        or precision_plan.refinement is not RefinementMode.NONE
        or precision_plan.center_coordinates
    ):
        raise ValueError(
            "prepared polygon mask fp64 plans must be uncentered and require "
            "neither compensation nor staged refinement"
        )
    return precision_plan


_MASK_CLASSIFICATION_KERNEL_SOURCE = ORIENT2D_DEVICE + r"""
__device__ inline void vs_mask_two_diff(
    double a,
    double b,
    double &difference,
    double &tail
) {
    difference = a - b;
    const double b_virtual = a - difference;
    const double a_virtual = difference + b_virtual;
    const double b_roundoff = b_virtual - b;
    const double a_roundoff = a - a_virtual;
    tail = a_roundoff + b_roundoff;
}

__device__ inline int vs_mask_grow_expansion(
    const int expansion_length,
    const double* expansion,
    const double value,
    double* output
) {
    double accumulator = value;
    int output_length = 0;
    for (int i = 0; i < expansion_length; ++i) {
        double sum;
        double roundoff;
        vs_two_sum(accumulator, expansion[i], sum, roundoff);
        if (roundoff != 0.0) output[output_length++] = roundoff;
        accumulator = sum;
    }
    if (accumulator != 0.0 || output_length == 0) {
        output[output_length++] = accumulator;
    }
    return output_length;
}

__device__ inline int vs_mask_add_scalar(
    double* expansion,
    const int expansion_length,
    const double value
) {
    double scratch[24];
    const int output_length = vs_mask_grow_expansion(
        expansion_length,
        expansion,
        value,
        scratch
    );
    for (int i = 0; i < output_length; ++i) expansion[i] = scratch[i];
    return output_length;
}

__device__ inline int vs_mask_add_product(
    double* expansion,
    int expansion_length,
    const double left,
    const double right,
    const double sign
) {
    double product;
    double roundoff;
    vs_two_product(left, right, product, roundoff);
    expansion_length = vs_mask_add_scalar(
        expansion,
        expansion_length,
        sign * roundoff
    );
    return vs_mask_add_scalar(
        expansion,
        expansion_length,
        sign * product
    );
}

__device__ inline int vs_mask_orient2d_exact(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double px,
    const double py
) {
    double abx;
    double abx_tail;
    double aby;
    double aby_tail;
    double apx;
    double apx_tail;
    double apy;
    double apy_tail;
    vs_mask_two_diff(bx, ax, abx, abx_tail);
    vs_mask_two_diff(by, ay, aby, aby_tail);
    vs_mask_two_diff(px, ax, apx, apx_tail);
    vs_mask_two_diff(py, ay, apy, apy_tail);

    double expansion[24];
    expansion[0] = 0.0;
    int expansion_length = 1;
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, abx, apy, 1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, abx, apy_tail, 1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, abx_tail, apy, 1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, abx_tail, apy_tail, 1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, aby, apx, -1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, aby, apx_tail, -1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, aby_tail, apx, -1.0
    );
    expansion_length = vs_mask_add_product(
        expansion, expansion_length, aby_tail, apx_tail, -1.0
    );

    const double determinant = expansion[expansion_length - 1];
    return determinant > 0.0 ? 1 : (determinant < 0.0 ? -1 : 0);
}

__device__ inline int vs_mask_orient2d(
    const double ax,
    const double ay,
    const double bx,
    const double by,
    const double px,
    const double py
) {
    const double det_left = (bx - ax) * (py - ay);
    const double det_right = (by - ay) * (px - ax);
    const double determinant = det_left - det_right;
    double determinant_sum;
    if (det_left > 0.0) {
        if (det_right <= 0.0) return determinant > 0.0 ? 1 : -1;
        determinant_sum = det_left + det_right;
    } else if (det_left < 0.0) {
        if (det_right >= 0.0) return determinant > 0.0 ? 1 : -1;
        determinant_sum = -det_left - det_right;
    } else {
        return det_right == 0.0
            ? vs_mask_orient2d_exact(ax, ay, bx, by, px, py)
            : (determinant > 0.0 ? 1 : -1);
    }
    const double error_bound =
        3.3306690738754716e-16 * determinant_sum;
    if (determinant >= error_bound) return 1;
    if (-determinant >= error_bound) return -1;
    return vs_mask_orient2d_exact(ax, ay, bx, by, px, py);
}

extern "C" __global__ void __launch_bounds__(256, 4)
classify_mask_candidate_relation(
    const int* __restrict__ pair_query,
    const int* __restrict__ pair_segment,
    const long long* __restrict__ logical_count,
    const double* __restrict__ point_x,
    const double* __restrict__ point_y,
    const double* __restrict__ segment_x,
    const double* __restrict__ segment_y,
    unsigned int* __restrict__ row_flags,
    const int row_count,
    const int pair_capacity
) {
    const int stride = blockDim.x * gridDim.x;
    const long long active_count = logical_count[0];
    for (int candidate = blockIdx.x * blockDim.x + threadIdx.x;
         candidate < pair_capacity;
         candidate += stride) {
        if ((long long)candidate >= active_count) continue;

        const int query = pair_query[candidate];
        const int segment = pair_segment[candidate];
        if (query < row_count) {
            /* The indexed relation already proved exact fp64 MBR overlap. */
            atomicOr(&row_flags[query], 1u);
            continue;
        }

        const int row = query - row_count;
        if (row < 0 || row >= row_count) continue;
        const double2 sx =
            reinterpret_cast<const double2*>(segment_x)[segment];
        const double2 sy =
            reinterpret_cast<const double2*>(segment_y)[segment];
        const double px = point_x[row];
        const double py = point_y[row];
        const bool straddles = (sy.x > py) != (sy.y > py);
        if (!straddles) continue;

        /*
         * For the leftward ray, x_intersection < px exactly when orient2d's
         * sign is opposite the directed segment's dy.  This avoids division
         * and makes the topology decision through exact-sign refinement.
         */
        const int orientation = vs_mask_orient2d(
            sx.x,
            sy.x,
            sx.y,
            sy.y,
            px,
            py
        );
        const bool crosses = sy.y > sy.x ? orientation < 0 : orientation > 0;
        if (crosses) atomicXor(&row_flags[row], 2u);
    }
}
"""
_MASK_CLASSIFICATION_KERNEL_NAMES = ("classify_mask_candidate_relation",)

if cp is not None:
    request_nvrtc_warmup(
        [
            (
                "prepared-polygon-mask-indexed-relation-fp64",
                _MASK_CLASSIFICATION_KERNEL_SOURCE,
                _MASK_CLASSIFICATION_KERNEL_NAMES,
            )
        ]
    )


@dataclass(frozen=True)
class PreparedPolygonMaskPhysicalShape:
    """Indexed candidate-refine shape for one-mask classification."""

    row_count: int
    segment_count: int
    indexed_query_count: int
    dense_candidate_work: int
    candidate_work: int
    scheduled_index_lane_bound: int
    exact_lane_bound: int
    total_scheduled_lane_bound: int
    free_device_bytes: int
    output_bytes: int
    candidate_tile_capacity: int
    row_tile_size: int
    segment_tile_size: int
    tile_count: int


def _plan_mask_classification_shape(
    row_count: int,
    segment_count: int,
    free_device_bytes: int,
    *,
    span_bucket_counts=None,
) -> PreparedPolygonMaskPhysicalShape:
    """Plan structural Morton tiles and bound all scheduled pair lanes."""
    row_count = max(0, int(row_count))
    segment_count = max(0, int(segment_count))
    free_device_bytes = max(0, int(free_device_bytes))
    indexed_query_count = row_count * 2
    dense_candidate_work = row_count * segment_count
    if span_bucket_counts is None:
        candidate_work = indexed_query_count * segment_count
        bucket_counts = None
    else:
        bucket_counts = tuple(max(0, int(value)) for value in span_bucket_counts)
        if len(bucket_counts) != len(_MORTON_SPAN_BUCKET_UPPER_BOUNDS):
            raise ValueError("Morton span bucket counts must cover every span bucket")
        if sum(bucket_counts) != indexed_query_count:
            raise ValueError("Morton span bucket counts must cover every indexed query")
        candidate_work = sum(
            count * min(int(span), segment_count)
            for count, span in zip(
                bucket_counts,
                _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
                strict=True,
            )
        )

    scheduled_index_lane_bound = candidate_work * 2
    exact_lane_bound = candidate_work
    total_scheduled_lane_bound = scheduled_index_lane_bound + exact_lane_bound
    output_bytes = row_count * 4
    reserve_bytes = min(_LIVE_MEMORY_RESERVE_BYTES, free_device_bytes // 4)
    usable_bytes = max(
        0,
        (free_device_bytes - reserve_bytes - output_bytes) // _LIVE_MEMORY_FRACTION,
    )
    candidate_tile_capacity = min(
        candidate_work,
        usable_bytes // _CANDIDATE_BYTES_ESTIMATE,
    )
    if candidate_work and candidate_tile_capacity == 0:
        candidate_tile_capacity = 1

    segment_tile_size = min(
        segment_count,
        _MAX_CANDIDATES_PER_QUERY_TILE,
        candidate_tile_capacity,
    )
    row_tile_size = min(
        indexed_query_count,
        candidate_tile_capacity // max(1, segment_tile_size),
    )
    if candidate_work:
        row_tile_size = max(1, row_tile_size)
        segment_tile_size = max(1, segment_tile_size)
        if bucket_counts is None:
            row_tiles = (
                indexed_query_count + row_tile_size - 1
            ) // row_tile_size
            segment_tiles = (
                segment_count + segment_tile_size - 1
            ) // segment_tile_size
            tile_count = row_tiles * segment_tiles
        else:
            tile_count = sum(
                ((count + row_tile_size - 1) // row_tile_size)
                * (
                    (min(int(span), segment_count) + segment_tile_size - 1)
                    // segment_tile_size
                )
                for count, span in zip(
                    bucket_counts,
                    _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
                    strict=True,
                )
                if count and span
            )
    else:
        row_tile_size = 0
        segment_tile_size = 0
        tile_count = 0

    return PreparedPolygonMaskPhysicalShape(
        row_count=row_count,
        segment_count=segment_count,
        indexed_query_count=indexed_query_count,
        dense_candidate_work=dense_candidate_work,
        candidate_work=candidate_work,
        scheduled_index_lane_bound=scheduled_index_lane_bound,
        exact_lane_bound=exact_lane_bound,
        total_scheduled_lane_bound=total_scheduled_lane_bound,
        free_device_bytes=free_device_bytes,
        output_bytes=output_bytes,
        candidate_tile_capacity=candidate_tile_capacity,
        row_tile_size=row_tile_size,
        segment_tile_size=segment_tile_size,
        tile_count=tile_count,
    )


def _live_device_free_bytes() -> int:
    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    return int(free_bytes)


def _mask_classification_kernels(precision_plan: PrecisionPlan):
    precision_plan = _require_prepared_polygon_mask_precision_plan(precision_plan)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        (
            "prepared-polygon-mask-indexed-relation-"
            f"{precision_plan.compute_precision.value}"
        ),
        _MASK_CLASSIFICATION_KERNEL_SOURCE,
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_MASK_CLASSIFICATION_KERNEL_SOURCE,
        kernel_names=_MASK_CLASSIFICATION_KERNEL_NAMES,
    )


def _classify_mask_candidate_relation_tile_device(
    d_pair_query,
    d_pair_segment,
    d_logical_count,
    d_point_x,
    d_point_y,
    d_line_x,
    d_line_y,
    row_count: int,
    *,
    precision_plan: PrecisionPlan,
    d_row_flags=None,
):
    """Refine one capacity-backed candidate prefix without reading its count."""
    pair_capacity = int(d_pair_query.size)
    if d_row_flags is None:
        d_row_flags = cp.zeros(row_count, dtype=cp.uint32)
    if pair_capacity == 0:
        return d_row_flags, 0

    runtime = get_cuda_runtime()
    precision_plan = _require_prepared_polygon_mask_precision_plan(precision_plan)
    kernel = _mask_classification_kernels(precision_plan)[
        "classify_mask_candidate_relation"
    ]
    params = (
        (
            runtime.pointer(d_pair_query),
            runtime.pointer(d_pair_segment),
            runtime.pointer(d_logical_count),
            runtime.pointer(d_point_x),
            runtime.pointer(d_point_y),
            runtime.pointer(d_line_x),
            runtime.pointer(d_line_y),
            runtime.pointer(d_row_flags),
            row_count,
            pair_capacity,
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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, pair_capacity)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=cp.cuda.get_current_stream(),
    )
    return d_row_flags, pair_capacity


def _classify_mask_indexed_relation_device(
    segment_index,
    d_query_bounds,
    d_point_x,
    d_point_y,
    d_line_x,
    d_line_y,
    row_count: int,
    free_device_bytes: int,
    *,
    precision_plan: PrecisionPlan,
):
    """Count/scan/scatter indexed candidates through fixed-capacity tiles."""
    precision_plan = _require_prepared_polygon_mask_precision_plan(precision_plan)
    flat_index = segment_index.to_flat_index()
    state = _prepare_morton_range_query(
        flat_index,
        d_query_bounds,
        d_query_bounds,
    )
    if state is None:
        raise RuntimeError("prepared mask segment index produced no Morton query state")

    runtime = get_cuda_runtime()
    d_row_flags = cp.zeros(row_count, dtype=cp.uint32)
    d_query_order = None
    try:
        d_query_order, bucket_counts = _morton_reduction_span_schedule(
            state.d_starts,
            state.d_ends,
        )
        shape = _plan_mask_classification_shape(
            row_count,
            int(segment_index.row_count),
            free_device_bytes,
            span_bucket_counts=bucket_counts,
        )
        if not shape.candidate_work:
            return d_row_flags, shape

        kernels = _morton_range_kernels()
        count_kernel = kernels["morton_range_tile_count"]
        scatter_kernel = kernels["morton_range_tile_scatter"]
        ptr = runtime.pointer
        bucket_start = 0
        for bucket_index, bucket_count_raw in enumerate(bucket_counts):
            bucket_count = int(bucket_count_raw)
            bucket_stop = bucket_start + bucket_count
            bucket_span = min(
                int(_MORTON_SPAN_BUCKET_UPPER_BOUNDS[bucket_index]),
                shape.segment_count,
            )
            if bucket_span == 0 or bucket_count == 0:
                bucket_start = bucket_stop
                continue
            for query_order_start in range(
                bucket_start,
                bucket_stop,
                shape.row_tile_size,
            ):
                query_batch_count = min(
                    shape.row_tile_size,
                    bucket_stop - query_order_start,
                )
                for position_start in range(
                    0,
                    bucket_span,
                    shape.segment_tile_size,
                ):
                    current_tile_width = min(
                        shape.segment_tile_size,
                        bucket_span - position_start,
                    )
                    pair_capacity = query_batch_count * current_tile_width
                    d_counts = cp.zeros(query_batch_count, dtype=cp.int32)
                    d_counts_i64 = None
                    d_offsets = None
                    d_logical_count = None
                    d_pair_query = None
                    d_pair_segment = None
                    try:
                        count_grid, count_block = runtime.launch_config(
                            count_kernel,
                            query_batch_count,
                        )
                        runtime.launch(
                            count_kernel,
                            grid=count_grid,
                            block=count_block,
                            params=(
                                (
                                    ptr(state.d_starts),
                                    ptr(state.d_ends),
                                    ptr(d_query_order),
                                    ptr(state.d_sorted_tree_bounds),
                                    ptr(state.d_query_bounds),
                                    ptr(d_counts),
                                    query_order_start,
                                    query_batch_count,
                                    position_start,
                                    current_tile_width,
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
                                    KERNEL_PARAM_I64,
                                    KERNEL_PARAM_I32,
                                ),
                            ),
                        )
                        d_counts_i64 = d_counts.astype(cp.int64, copy=False)
                        d_offsets = exclusive_sum(d_counts_i64, synchronize=False)
                        d_logical_count = d_offsets[-1:] + d_counts_i64[-1:]
                        d_pair_query = cp.empty(pair_capacity, dtype=cp.int32)
                        d_pair_segment = cp.empty(pair_capacity, dtype=cp.int32)
                        scatter_grid, scatter_block = runtime.launch_config(
                            scatter_kernel,
                            query_batch_count,
                        )
                        runtime.launch(
                            scatter_kernel,
                            grid=scatter_grid,
                            block=scatter_block,
                            params=(
                                (
                                    ptr(state.d_starts),
                                    ptr(state.d_ends),
                                    ptr(d_query_order),
                                    ptr(state.d_order),
                                    ptr(state.d_sorted_tree_bounds),
                                    ptr(state.d_query_bounds),
                                    ptr(d_offsets),
                                    ptr(d_pair_query),
                                    ptr(d_pair_segment),
                                    query_order_start,
                                    query_batch_count,
                                    position_start,
                                    current_tile_width,
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
                                    KERNEL_PARAM_I32,
                                    KERNEL_PARAM_I32,
                                    KERNEL_PARAM_I64,
                                    KERNEL_PARAM_I32,
                                ),
                            ),
                        )
                        _classify_mask_candidate_relation_tile_device(
                            d_pair_query,
                            d_pair_segment,
                            d_logical_count,
                            d_point_x,
                            d_point_y,
                            d_line_x,
                            d_line_y,
                            row_count,
                            precision_plan=precision_plan,
                            d_row_flags=d_row_flags,
                        )
                    finally:
                        runtime.free(d_counts)
                        runtime.free(d_counts_i64)
                        runtime.free(d_offsets)
                        runtime.free(d_logical_count)
                        runtime.free(d_pair_query)
                        runtime.free(d_pair_segment)
            bucket_start = bucket_stop
        return d_row_flags, shape
    finally:
        runtime.free(d_query_order)
        state.close()


@dataclass(frozen=True)
class PreparedPolygonMaskClassification:
    """Device row classification relative to one polygonal mask boundary."""

    valid: object
    covered_by: object
    exterior: object
    boundary_unresolved: object


@dataclass
class PreparedPolygonMask:
    """One polygonal mask prepared as an indexed physical boundary table."""

    mask: OwnedGeometryArray
    boundary_lines: OwnedGeometryArray
    segment_index: object
    source_segments: object
    segment_ring_ids: object
    ring_starts: object
    ring_ends: object
    ancestor_shell_ring_ids: object
    precision_plan: PrecisionPlan

    @classmethod
    def from_owned(
        cls,
        mask: OwnedGeometryArray,
        *,
        precision_plan: PrecisionPlan,
    ) -> PreparedPolygonMask | None:
        precision_plan = _require_prepared_polygon_mask_precision_plan(precision_plan)
        if cp is None or int(mask.row_count) != 1:
            return None
        state = mask._ensure_device_state(preserve_indexed_view=True)
        polygonal = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
        if not set(state.families).intersection(polygonal):
            return None
        segments = _extract_segments_gpu(mask)
        boundary_lines = device_segment_table_as_linestrings(segments)
        d_parts = cp.asarray(segments.part_indices, dtype=cp.int32)
        d_rings = cp.asarray(segments.ring_indices, dtype=cp.int32)
        if int(segments.count) == 0:
            d_segment_ring_ids = cp.empty(0, dtype=cp.int32)
            d_ring_starts = cp.empty(0, dtype=cp.int64)
            d_ring_ends = cp.empty(0, dtype=cp.int64)
            d_shell_ring_ids = cp.empty(0, dtype=cp.int32)
        else:
            d_ring_start_mask = cp.empty(segments.count, dtype=cp.bool_)
            d_ring_start_mask[0] = True
            d_ring_start_mask[1:] = (d_parts[1:] != d_parts[:-1]) | (
                d_rings[1:] != d_rings[:-1]
            )
            d_ring_starts = cp.flatnonzero(d_ring_start_mask).astype(
                cp.int64,
                copy=False,
            )
            d_ring_ends = cp.concatenate(
                (d_ring_starts[1:], cp.asarray([segments.count], dtype=cp.int64))
            )
            d_segment_ring_ids = cp.cumsum(
                d_ring_start_mask.astype(cp.int32, copy=False),
                dtype=cp.int32,
            ) - np.int32(1)
            d_ring_parts = d_parts[d_ring_starts]
            d_ring_locals = d_rings[d_ring_starts]
            d_ring_keys = (
                d_ring_parts.astype(cp.uint64, copy=False) << cp.uint64(32)
            ) | d_ring_locals.astype(cp.uint32, copy=False).astype(
                cp.uint64,
                copy=False,
            )
            d_shell_keys = d_ring_parts.astype(cp.uint64, copy=False) << cp.uint64(32)
            d_shell_ring_ids = cp.searchsorted(d_ring_keys, d_shell_keys).astype(
                cp.int32,
                copy=False,
            )
        flat_index = build_flat_spatial_index(
            boundary_lines,
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
                reason="prepared polygon mask requires a device segment index",
            ),
        )
        segment_index = flat_index.to_native_spatial_index(
            source_token="prepared-polygon-mask-boundary-segments",
        )
        device_index_arrays = (
            segment_index.order,
            segment_index.morton_keys,
            None if segment_index.metadata is None else segment_index.metadata.bounds,
        )
        if not all(
            hasattr(value, "__cuda_array_interface__")
            for value in device_index_arrays
        ):
            raise RuntimeError(
                "prepared polygon mask requires a fully device-resident Morton index"
            )
        return cls(
            mask=mask,
            boundary_lines=boundary_lines,
            segment_index=segment_index,
            source_segments=segments,
            segment_ring_ids=d_segment_ring_ids,
            ring_starts=d_ring_starts,
            ring_ends=d_ring_ends,
            ancestor_shell_ring_ids=d_shell_ring_ids,
            precision_plan=precision_plan,
        )

    def close(self) -> None:
        """Release the retained singular source segment table."""
        self.source_segments.free()

    def complete_ring_relation(
        self,
        rows: OwnedGeometryArray,
    ) -> DeviceRingLocalSegmentRelation:
        """Relate rows to intersecting complete rings plus winding shells.

        The Morton relation is segment-local, but topology cannot consume a
        clipped fragment of a closed ring.  Candidate segment ids are lowered
        to unique ``(row, ring)`` pairs and every hole pair adds its component
        shell.  That ancestor shell supplies the exact face-walk winding
        baseline without copying unrelated holes or components.
        """
        from vibespatial.api._native_relation import NativeRelation

        ring_count = int(self.ring_starts.size)
        pairs = generate_bounds_pairs(rows, self.boundary_lines)
        d_pair_rows = cp.asarray(pairs.device_left_indices, dtype=cp.int64)
        d_pair_segments = cp.asarray(pairs.device_right_indices, dtype=cp.int64)
        if int(d_pair_rows.size) == 0 or ring_count == 0:
            d_rows = cp.empty(0, dtype=cp.int32)
            d_ring_ids = cp.empty(0, dtype=cp.int32)
        else:
            d_candidate_ring_ids = cp.asarray(
                self.segment_ring_ids,
                dtype=cp.int32,
            )[d_pair_segments]
            d_shell_ring_ids = cp.asarray(
                self.ancestor_shell_ring_ids,
                dtype=cp.int32,
            )[d_candidate_ring_ids]
            d_candidate_keys = (
                d_pair_rows.astype(cp.int64, copy=False) * np.int64(ring_count)
                + d_candidate_ring_ids.astype(cp.int64, copy=False)
            )
            d_shell_keys = (
                d_pair_rows.astype(cp.int64, copy=False) * np.int64(ring_count)
                + d_shell_ring_ids.astype(cp.int64, copy=False)
            )
            d_complete_keys = cp.unique(
                cp.concatenate((d_candidate_keys, d_shell_keys))
            )
            d_rows = (d_complete_keys // np.int64(ring_count)).astype(
                cp.int32,
                copy=False,
            )
            d_ring_ids = (d_complete_keys % np.int64(ring_count)).astype(
                cp.int32,
                copy=False,
            )
        relation = NativeRelation(
            left_indices=d_rows,
            right_indices=d_ring_ids,
            left_token="prepared-mask-unresolved-rows",
            right_token="prepared-mask-complete-rings",
            predicate="bounds-intersects-plus-ancestor-shell",
            left_row_count=int(rows.row_count),
            right_row_count=ring_count,
            sorted_by_left=True,
            duplicate_policy="drop",
        )
        return DeviceRingLocalSegmentRelation(
            source_segments=self.source_segments,
            ring_relation=relation,
            ring_starts=self.ring_starts,
            ring_ends=self.ring_ends,
            ancestor_shell_ring_ids=self.ancestor_shell_ring_ids,
        )

    def _representative_points(self, rows: OwnedGeometryArray):
        representatives = representative_point_owned(
            rows,
            dispatch_mode=ExecutionMode.GPU,
        )
        state = representatives._ensure_device_state(preserve_indexed_view=True)
        points = state.families.get(GeometryFamily.POINT)
        if points is None:
            return None
        d_x = cp.asarray(points.x, dtype=cp.float64)
        d_y = cp.asarray(points.y, dtype=cp.float64)
        if int(d_x.size) != int(rows.row_count) or int(d_y.size) != int(rows.row_count):
            return None
        return representatives, d_x, d_y

    def classify_polygon_rows(
        self,
        rows: OwnedGeometryArray,
    ) -> PreparedPolygonMaskClassification | None:
        """Classify rows through an indexed, capacity-backed candidate relation.

        Boundary MBR queries and representative-point ray queries share one
        reusable segment index. Count/scan/scatter emits only refined candidates
        into structural tile capacity; exact-sign ray parity then classifies rows
        without a candidate-count host fence.
        """
        if cp is None or int(rows.row_count) == 0:
            return None
        precision_plan = _require_prepared_polygon_mask_precision_plan(
            self.precision_plan
        )
        row_count = int(rows.row_count)
        row_state = rows._ensure_device_state(preserve_indexed_view=True)
        mask_state = self.mask._ensure_device_state(preserve_indexed_view=True)
        d_valid = cp.asarray(row_state.validity, dtype=cp.bool_) & cp.asarray(
            mask_state.validity,
            dtype=cp.bool_,
        ).reshape(-1)[0]

        representative_points = self._representative_points(rows)
        if representative_points is None:
            return None
        representatives, d_point_x, d_point_y = representative_points
        d_row_bounds = cp.ascontiguousarray(
            compute_geometry_bounds_device(
                rows,
                preserve_indexed_view=True,
            ).reshape(row_count, 4),
            dtype=cp.float64,
        )
        line_state = self.boundary_lines._ensure_device_state(
            preserve_indexed_view=True,
        )
        lines = line_state.families[GeometryFamily.LINESTRING]
        d_line_x = cp.ascontiguousarray(lines.x, dtype=cp.float64)
        d_line_y = cp.ascontiguousarray(lines.y, dtype=cp.float64)
        segment_count = int(self.boundary_lines.row_count)
        total_bounds = self.segment_index.total_bounds
        if total_bounds is None or not all(isfinite(value) for value in total_bounds):
            raise RuntimeError("prepared polygon mask segment index has invalid bounds")
        d_ray_min_x = cp.minimum(d_point_x, cp.float64(total_bounds[0]))
        d_ray_bounds = cp.stack(
            (d_ray_min_x, d_point_y, d_point_x, d_point_y),
            axis=1,
        )
        d_query_bounds = cp.ascontiguousarray(
            cp.concatenate((d_row_bounds, d_ray_bounds), axis=0),
            dtype=cp.float64,
        )
        d_row_flags, shape = _classify_mask_indexed_relation_device(
            self.segment_index,
            d_query_bounds,
            d_point_x,
            d_point_y,
            d_line_x,
            d_line_y,
            row_count,
            _live_device_free_bytes(),
            precision_plan=precision_plan,
        )

        d_boundary = (d_row_flags & cp.uint32(1)) != 0
        d_inside = (d_row_flags & cp.uint32(2)) != 0

        representative_state = representatives._ensure_device_state(
            preserve_indexed_view=True,
        )
        d_representative_valid = cp.asarray(
            representative_state.validity,
            dtype=cp.bool_,
        )
        d_inside = (
            d_valid
            & d_representative_valid
            & d_inside
        )
        d_covered = d_inside & ~d_boundary
        d_exterior = d_valid & ~d_inside & ~d_boundary
        d_unresolved = d_valid & d_boundary
        record_dispatch_event(
            surface="vibespatial.spatial.prepared_polygon_mask",
            operation="classify_polygon_rows",
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            implementation="prepared_single_mask_indexed_relation_exact_ray_gpu",
            reason=(
                "one physical polygon boundary was classified through a reusable "
                "segment index, capacity-backed candidate tiles, and exact rays"
            ),
            detail=(
                f"rows={row_count}; physical_mask_segments={segment_count}; "
                f"indexed_queries={shape.indexed_query_count}; "
                f"dense_candidate_work={shape.dense_candidate_work}; "
                f"candidate_work={shape.candidate_work}; "
                f"scheduled_index_lane_bound={shape.scheduled_index_lane_bound}; "
                f"exact_lane_bound={shape.exact_lane_bound}; "
                f"total_scheduled_lane_bound={shape.total_scheduled_lane_bound}; "
                f"candidate_tile_capacity={shape.candidate_tile_capacity}; "
                f"row_tile_size={shape.row_tile_size}; "
                f"segment_tile_size={shape.segment_tile_size}; "
                f"tiles={shape.tile_count}; free_device_bytes={shape.free_device_bytes}; "
                f"precision={precision_plan.compute_precision.value}; "
                f"precision_reason={precision_plan.reason}"
            ),
        )
        return PreparedPolygonMaskClassification(
            valid=d_valid,
            covered_by=d_covered,
            exterior=d_exterior,
            boundary_unresolved=d_unresolved,
        )
