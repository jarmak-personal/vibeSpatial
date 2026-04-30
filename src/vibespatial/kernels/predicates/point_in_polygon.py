from __future__ import annotations

import logging
import weakref
from time import perf_counter

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import compact_indices
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.kernels.core.geometry_analysis import _launch_family_bounds_kernel
from vibespatial.kernels.predicates.point_in_polygon_source import (
    _PIP_BLOCK_PER_PAIR_KERNEL_NAMES,
    _POINT_IN_POLYGON_KERNEL_NAMES,
    _format_block_per_pair_source,
    _format_pip_kernel_source,
)
from vibespatial.predicates.point_in_polygon_cpu import (
    point_in_polygon_cpu_variant as _point_in_polygon_cpu_variant,
)
from vibespatial.predicates.point_in_polygon_host import (
    candidate_rows_by_family as _candidate_rows_by_family_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    candidate_rows_from_coarse,
    cpu_return_device_fallback,
    dense_true_mask,
    empty_bool_mask_like,
    empty_gpu_bounds_placeholder,
    initialize_coarse_result,
    work_cv,
)
from vibespatial.predicates.point_in_polygon_host import (
    compute_pip_center as _compute_pip_center_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    compute_work_estimates_for_candidates as _compute_work_estimates_for_candidates_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    estimate_pip_work_multipolygon as _estimate_pip_work_multipolygon_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    estimate_pip_work_polygon as _estimate_pip_work_polygon_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    select_gpu_strategy as _select_gpu_strategy_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    should_bin_dispatch as _should_bin_dispatch_host,
)
from vibespatial.predicates.point_in_polygon_host import (
    to_python_result as _to_python_result_host,
)
from vibespatial.predicates.support import (
    PointSequence,
    coerce_geometry_array,
    resolve_predicate_context,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

from .point_within_bounds import (
    NormalizedBoundsInput,
    _evaluate_point_within_bounds,
    _normalize_right_input,
)

if cp is not None:
    request_warmup(["select_i32", "select_i64"])

_POINT_IN_POLYGON_KERNEL_SOURCE = _format_pip_kernel_source("double")


def _current_cuda_stream():
    if cp is None:
        return None
    return cp.cuda.get_current_stream()



# ---------------------------------------------------------------------------
# Block-per-pair kernel: for complex polygons (>1024 vertices), one thread
# block cooperatively processes a single (point, polygon) pair.  Threads
# split ring iteration and use shared-memory XOR reduction for the even-odd
# containment result.
# ---------------------------------------------------------------------------


def _pip_block_per_pair_kernels(compute_type: str = "double"):
    source = _format_block_per_pair_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        f"pip-block-per-pair-{compute_type}", source
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_PIP_BLOCK_PER_PAIR_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Work-size binning infrastructure
# ---------------------------------------------------------------------------

# Complexity bins: simple (<64 verts), medium (64-1024), complex (>1024)
_PIP_WORK_BINS = [64, 1024]


def _estimate_pip_work_polygon(
    polygon_geometry_offsets,
    polygon_ring_offsets,
    candidate_right_family_rows,
):
    return _estimate_pip_work_polygon_host(
        polygon_geometry_offsets,
        polygon_ring_offsets,
        candidate_right_family_rows,
    )


def _estimate_pip_work_multipolygon(
    multipolygon_geometry_offsets,
    multipolygon_part_offsets,
    multipolygon_ring_offsets,
    candidate_right_family_rows,
):
    return _estimate_pip_work_multipolygon_host(
        multipolygon_geometry_offsets,
        multipolygon_part_offsets,
        multipolygon_ring_offsets,
        candidate_right_family_rows,
    )


def _should_bin_dispatch(work_estimates) -> bool:
    return _should_bin_dispatch_host(work_estimates)


def _scatter_bin_results(
    bin_indices_device,
    bin_results_device,
    full_output_device,
    bin_count: int,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    """Scatter bin results back into the full output buffer at original positions."""
    runtime = get_cuda_runtime()
    kernel = _point_in_polygon_kernels(compute_type)["scatter_compacted_hits"]
    params = (
        (
            runtime.pointer(bin_indices_device),
            runtime.pointer(bin_results_device),
            runtime.pointer(full_output_device),
            bin_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, bin_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=_current_cuda_stream(),
    )


def _free_device_buffers(runtime, buffers) -> None:
    for buffer in buffers:
        runtime.free(buffer)


_DEVICE_RESULT_KEEPALIVES: dict[int, tuple[object, ...]] = {}


def _release_device_result_keepalive(key: int) -> None:
    _DEVICE_RESULT_KEEPALIVES.pop(key, None)


def _wrap_device_result_with_keepalive(device_array, *buffers):
    keepalive = tuple(buffer for buffer in buffers if buffer is not None)
    if not keepalive:
        return device_array
    key = id(device_array)
    _DEVICE_RESULT_KEEPALIVES[key] = keepalive
    weakref.finalize(device_array, _release_device_result_keepalive, key)
    return device_array


def _binned_polygon_dispatch(
    candidate_rows_device,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    work_estimates,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Dispatch polygon PIP kernel in work-balanced bins.

    For simple/medium bins uses the thread-per-pair kernel.  For the complex
    bin (>1024 vertices) uses the block-per-pair kernel where one thread block
    cooperatively processes a single candidate pair.

    All binning is performed on-device using CuPy to avoid D->H/H->D
    round-trips for candidate rows and bin indices.
    """
    import cupy as _cp

    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]

    device_out = runtime.allocate((candidate_count,), cp.uint8)

    # Move work estimates to device once for all bins.
    d_work = _cp.asarray(work_estimates)
    bin_edges = [0] + _PIP_WORK_BINS + [int(work_estimates.max()) + 1]
    _log.debug(
        "binned_polygon_dispatch: %d candidates, bin_edges=%s, "
        "work min=%d max=%d mean=%.1f std=%.1f",
        candidate_count, bin_edges,
        work_estimates.min(), work_estimates.max(),
        work_estimates.mean(), work_estimates.std(),
    )

    pending_free = []
    try:
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (d_work >= lo) & (d_work < hi)
            bin_count = int(mask.sum())
            if bin_count == 0:
                continue

            # Keep per-bin arrays alive until the caller hits a real
            # synchronization boundary. Scatter reads these buffers on the
            # same stream, so freeing them eagerly forces avoidable syncs.
            device_bin_indices = _cp.flatnonzero(mask).astype(_cp.int32)
            device_bin_candidates = candidate_rows_device[device_bin_indices].astype(_cp.int32)
            device_bin_out = runtime.allocate((bin_count,), cp.uint8)
            pending_free.extend(
                (device_bin_candidates, device_bin_indices, device_bin_out)
            )

            use_block_per_pair = lo >= _PIP_WORK_BINS[-1]

            if use_block_per_pair:
                kernel = _pip_block_per_pair_kernels(compute_type)["pip_block_per_pair_polygon"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(polygon_buffer.empty_mask),
                        runtime.pointer(polygon_buffer.geometry_offsets),
                        runtime.pointer(polygon_buffer.ring_offsets),
                        runtime.pointer(polygon_buffer.x),
                        runtime.pointer(polygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.POLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(
                    kernel,
                    grid=grid,
                    block=block,
                    params=params,
                    stream=_current_cuda_stream(),
                )
            else:
                kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_compacted_tagged"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(polygon_buffer.empty_mask),
                        runtime.pointer(polygon_buffer.geometry_offsets),
                        runtime.pointer(polygon_buffer.ring_offsets),
                        runtime.pointer(polygon_buffer.x),
                        runtime.pointer(polygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.POLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(
                    kernel,
                    grid=grid,
                    block=block,
                    params=params,
                    stream=_current_cuda_stream(),
                )

            _scatter_bin_results(
                device_bin_indices, device_bin_out, device_out, bin_count,
                compute_type=compute_type, center_x=center_x, center_y=center_y,
            )
        return device_out, pending_free
    except Exception:
        runtime.synchronize()
        _free_device_buffers(runtime, pending_free)
        runtime.free(device_out)
        raise


def _binned_multipolygon_dispatch(
    candidate_rows_device,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    work_estimates,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Dispatch multipolygon PIP kernel in work-balanced bins.

    All binning is performed on-device using CuPy to avoid D->H/H->D
    round-trips for candidate rows and bin indices.
    """
    import cupy as _cp

    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]

    device_out = runtime.allocate((candidate_count,), cp.uint8)

    # Move work estimates to device once for all bins.
    d_work = _cp.asarray(work_estimates)
    bin_edges = [0] + _PIP_WORK_BINS + [int(work_estimates.max()) + 1]
    _log.debug(
        "binned_multipolygon_dispatch: %d candidates, work min=%d max=%d",
        candidate_count, work_estimates.min(), work_estimates.max(),
    )

    pending_free = []
    try:
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (d_work >= lo) & (d_work < hi)
            bin_count = int(mask.sum())
            if bin_count == 0:
                continue

            # Keep per-bin arrays alive until the caller hits a real
            # synchronization boundary. Scatter reads these buffers on the
            # same stream, so freeing them eagerly forces avoidable syncs.
            device_bin_indices = _cp.flatnonzero(mask).astype(_cp.int32)
            device_bin_candidates = candidate_rows_device[device_bin_indices].astype(_cp.int32)
            device_bin_out = runtime.allocate((bin_count,), cp.uint8)
            pending_free.extend(
                (device_bin_candidates, device_bin_indices, device_bin_out)
            )

            use_block_per_pair = lo >= _PIP_WORK_BINS[-1]

            if use_block_per_pair:
                kernel = _pip_block_per_pair_kernels(compute_type)["pip_block_per_pair_multipolygon"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(multipolygon_buffer.empty_mask),
                        runtime.pointer(multipolygon_buffer.geometry_offsets),
                        runtime.pointer(multipolygon_buffer.part_offsets),
                        runtime.pointer(multipolygon_buffer.ring_offsets),
                        runtime.pointer(multipolygon_buffer.x),
                        runtime.pointer(multipolygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(
                    kernel,
                    grid=grid,
                    block=block,
                    params=params,
                    stream=_current_cuda_stream(),
                )
            else:
                kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_compacted_tagged"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(multipolygon_buffer.empty_mask),
                        runtime.pointer(multipolygon_buffer.geometry_offsets),
                        runtime.pointer(multipolygon_buffer.part_offsets),
                        runtime.pointer(multipolygon_buffer.ring_offsets),
                        runtime.pointer(multipolygon_buffer.x),
                        runtime.pointer(multipolygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(
                    kernel,
                    grid=grid,
                    block=block,
                    params=params,
                    stream=_current_cuda_stream(),
                )

            _scatter_bin_results(
                device_bin_indices, device_bin_out, device_out, bin_count,
                compute_type=compute_type, center_x=center_x, center_y=center_y,
            )
        return device_out, pending_free
    except Exception:
        runtime.synchronize()
        _free_device_buffers(runtime, pending_free)
        runtime.free(device_out)
        raise


def _compute_work_estimates_for_candidates(
    candidate_rows_host,
    right_array: OwnedGeometryArray,
):
    return _compute_work_estimates_for_candidates_host(candidate_rows_host, right_array)


def _compute_work_estimates_for_candidates_device(
    candidate_rows_device,
    right_array: OwnedGeometryArray,
):
    """Compute per-candidate polygon work estimates fully on device."""
    right_state = right_array._ensure_device_state()
    d_candidate_rows = cp.asarray(candidate_rows_device, dtype=cp.int32)
    d_tags = cp.asarray(right_state.tags)[d_candidate_rows]
    d_family_row_offsets = cp.asarray(right_state.family_row_offsets)[d_candidate_rows]
    d_work = cp.zeros(d_candidate_rows.size, dtype=cp.int32)

    if GeometryFamily.POLYGON in right_state.families:
        poly_indices = cp.flatnonzero(
            d_tags == FAMILY_TAGS[GeometryFamily.POLYGON]
        ).astype(cp.int32, copy=False)
        if poly_indices.size > 0:
            poly_family_rows = d_family_row_offsets[poly_indices]
            poly_valid = poly_family_rows >= 0
            poly_valid_indices = poly_indices[poly_valid]
            poly_family_rows = poly_family_rows[poly_valid].astype(cp.int32, copy=False)
            if poly_family_rows.size > 0:
                poly_buf = right_state.families[GeometryFamily.POLYGON]
                d_geometry_offsets = cp.asarray(poly_buf.geometry_offsets)
                d_ring_offsets = cp.asarray(poly_buf.ring_offsets)
                ring_start = d_geometry_offsets[poly_family_rows]
                ring_end = d_geometry_offsets[poly_family_rows + 1]
                coord_start = d_ring_offsets[ring_start]
                coord_end = d_ring_offsets[ring_end]
                d_work[poly_valid_indices] = (coord_end - coord_start).astype(
                    cp.int32, copy=False,
                )

    if GeometryFamily.MULTIPOLYGON in right_state.families:
        mp_indices = cp.flatnonzero(
            d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
        ).astype(cp.int32, copy=False)
        if mp_indices.size > 0:
            mp_family_rows = d_family_row_offsets[mp_indices]
            mp_valid = mp_family_rows >= 0
            mp_valid_indices = mp_indices[mp_valid]
            mp_family_rows = mp_family_rows[mp_valid].astype(cp.int32, copy=False)
            if mp_family_rows.size > 0:
                mp_buf = right_state.families[GeometryFamily.MULTIPOLYGON]
                d_geometry_offsets = cp.asarray(mp_buf.geometry_offsets)
                d_part_offsets = cp.asarray(mp_buf.part_offsets)
                d_ring_offsets = cp.asarray(mp_buf.ring_offsets)
                poly_start = d_geometry_offsets[mp_family_rows]
                poly_end = d_geometry_offsets[mp_family_rows + 1]
                ring_start = d_part_offsets[poly_start]
                ring_end = d_part_offsets[poly_end]
                coord_start = d_ring_offsets[ring_start]
                coord_end = d_ring_offsets[ring_end]
                d_work[mp_valid_indices] = (coord_end - coord_start).astype(
                    cp.int32, copy=False,
                )

    return d_work





if cp is not None:
    request_nvrtc_warmup([
        ("point-in-polygon", _POINT_IN_POLYGON_KERNEL_SOURCE, _POINT_IN_POLYGON_KERNEL_NAMES),
    ])


def _point_in_polygon_kernels(compute_type: str = "double"):
    source = _format_pip_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"point-in-polygon-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_POINT_IN_POLYGON_KERNEL_NAMES,
    )


def _to_python_result(values) -> list[bool | None]:
    return _to_python_result_host(values)


def _candidate_rows_by_family(
    right: OwnedGeometryArray,
    candidate_rows,
):
    return _candidate_rows_by_family_host(right, candidate_rows)


def _select_gpu_strategy(
    row_count: int,
    *,
    strategy: str,
    right_array: OwnedGeometryArray | None = None,
) -> str:
    return _select_gpu_strategy_host(row_count, strategy=strategy, right_array=right_array)


def _launch_polygon_dense(
    candidate_indices,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    device_out,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> tuple[object, object]:
    """Launch dense polygon PIP kernel and return temporary buffers to keep alive."""
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]
    n = left.row_count
    stream = _current_cuda_stream()
    device_mask = runtime.allocate((n,), cp.uint8, zero=True)
    device_indices = runtime.from_host(candidate_indices.astype("int32", copy=False))
    device_mask[device_indices] = cp.uint8(1)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_dense"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(device_mask),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(polygon_buffer.empty_mask),
                runtime.pointer(polygon_buffer.geometry_offsets),
                runtime.pointer(polygon_buffer.ring_offsets),
                runtime.pointer(polygon_buffer.x),
                runtime.pointer(polygon_buffer.y),
                runtime.pointer(device_out),
                left.row_count,
                center_x,
                center_y,
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
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, left.row_count)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=params,
            stream=stream,
        )
        return device_indices, device_mask
    except Exception:
        runtime.synchronize()
        runtime.free(device_indices)
        runtime.free(device_mask)
        raise


def _launch_polygon_compacted(
    candidate_rows,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]
    device_out = runtime.allocate((candidate_count,), cp.uint8)
    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_compacted_tagged"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            runtime.pointer(polygon_buffer.empty_mask),
            runtime.pointer(polygon_buffer.geometry_offsets),
            runtime.pointer(polygon_buffer.ring_offsets),
            runtime.pointer(polygon_buffer.x),
            runtime.pointer(polygon_buffer.y),
            FAMILY_TAGS[GeometryFamily.POLYGON],
            runtime.pointer(device_out),
            candidate_count,
            center_x,
            center_y,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=_current_cuda_stream(),
    )
    return device_out


def _launch_multipolygon_dense(
    candidate_indices,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    device_out,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> tuple[object, object]:
    """Launch dense multipolygon PIP kernel and return temporary buffers to keep alive."""
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]
    n = left.row_count
    stream = _current_cuda_stream()
    device_mask = runtime.allocate((n,), cp.uint8, zero=True)
    device_indices = runtime.from_host(candidate_indices.astype("int32", copy=False))
    device_mask[device_indices] = cp.uint8(1)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_dense"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(device_mask),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(multipolygon_buffer.empty_mask),
                runtime.pointer(multipolygon_buffer.geometry_offsets),
                runtime.pointer(multipolygon_buffer.part_offsets),
                runtime.pointer(multipolygon_buffer.ring_offsets),
                runtime.pointer(multipolygon_buffer.x),
                runtime.pointer(multipolygon_buffer.y),
                runtime.pointer(device_out),
                left.row_count,
                center_x,
                center_y,
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
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, left.row_count)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=params,
            stream=stream,
        )
        return device_indices, device_mask
    except Exception:
        runtime.synchronize()
        runtime.free(device_indices)
        runtime.free(device_mask)
        raise


def _launch_multipolygon_compacted(
    candidate_rows,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]
    device_out = runtime.allocate((candidate_count,), cp.uint8)
    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_compacted_tagged"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            runtime.pointer(multipolygon_buffer.empty_mask),
            runtime.pointer(multipolygon_buffer.geometry_offsets),
            runtime.pointer(multipolygon_buffer.part_offsets),
            runtime.pointer(multipolygon_buffer.ring_offsets),
            runtime.pointer(multipolygon_buffer.x),
            runtime.pointer(multipolygon_buffer.y),
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            runtime.pointer(device_out),
            candidate_count,
            center_x,
            center_y,
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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=_current_cuda_stream(),
    )
    return device_out


def _scatter_compacted_hits(
    candidate_rows,
    compacted_hits,
    dense_out,
    candidate_count: int,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    runtime = get_cuda_runtime()
    kernel = _point_in_polygon_kernels(compute_type)["scatter_compacted_hits"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(compacted_hits),
            runtime.pointer(dense_out),
            candidate_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=_current_cuda_stream(),
    )


def _launch_fused(
    points: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Single-kernel fused bounds check + PIP test for all rows."""
    runtime = get_cuda_runtime()
    left_state = points._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]

    has_polygon = GeometryFamily.POLYGON in right_state.families
    has_multipolygon = GeometryFamily.MULTIPOLYGON in right_state.families
    polygon_buffer = right_state.families[GeometryFamily.POLYGON] if has_polygon else None
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON] if has_multipolygon else None

    # Ensure per-family bounds exist on device.  When the fused path skips
    # CPU compute_geometry_bounds, the device buffers have bounds=None.
    # Compute them directly on-device — this is much cheaper than the CPU
    # path that was the original bottleneck we're eliminating.
    for family, device_buffer in ((GeometryFamily.POLYGON, polygon_buffer), (GeometryFamily.MULTIPOLYGON, multipolygon_buffer)):
        if device_buffer is not None and device_buffer.bounds is None:
            row_count = right.families[family].row_count
            device_buffer.bounds = runtime.allocate((row_count, 4), cp.float64)
            _launch_family_bounds_kernel(family, device_buffer, row_count=row_count)

    device_out = runtime.allocate((points.row_count,), cp.uint8)

    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_fused"]
    params = (
        (
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.empty_mask),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            # polygon family (null if absent)
            runtime.pointer(polygon_buffer.bounds if has_polygon else None),
            runtime.pointer(polygon_buffer.empty_mask if has_polygon else None),
            runtime.pointer(polygon_buffer.geometry_offsets if has_polygon else None),
            runtime.pointer(polygon_buffer.ring_offsets if has_polygon else None),
            runtime.pointer(polygon_buffer.x if has_polygon else None),
            runtime.pointer(polygon_buffer.y if has_polygon else None),
            # multipolygon family (null if absent)
            runtime.pointer(multipolygon_buffer.bounds if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.empty_mask if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.geometry_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.part_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.ring_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.x if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.y if has_multipolygon else None),
            # tags
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            # output
            runtime.pointer(device_out),
            points.row_count,
            center_x,
            center_y,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, points.row_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=params,
        stream=_current_cuda_stream(),
    )
    return device_out


def _launch_bounds_candidate_rows(
    points: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = points._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_bounds = (
        None
        if GeometryFamily.POLYGON not in right_state.families
        else right_state.families[GeometryFamily.POLYGON].bounds
    )
    multipolygon_bounds = (
        None
        if GeometryFamily.MULTIPOLYGON not in right_state.families
        else right_state.families[GeometryFamily.MULTIPOLYGON].bounds
    )
    device_mask = runtime.allocate((points.row_count,), cp.uint8)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_bounds_mask"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.empty_mask),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(right_state.tags),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(polygon_bounds),
                runtime.pointer(multipolygon_bounds),
                FAMILY_TAGS[GeometryFamily.POLYGON],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                runtime.pointer(device_mask),
                points.row_count,
                center_x,
                center_y,
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, points.row_count)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=params,
            stream=_current_cuda_stream(),
        )
        return compact_indices(device_mask)
    finally:
        runtime.free(device_mask)


def launch_point_region_candidate_rows(
    points: OwnedGeometryArray,
    regions: OwnedGeometryArray,
):
    """Return device-resident candidate rows for aligned point/region bounds hits."""
    return _launch_bounds_candidate_rows(points, regions)

_log = logging.getLogger("vibespatial.kernels.point_in_polygon")

_last_gpu_substage_timings: dict[str, float] | None = None


def get_last_gpu_substage_timings() -> dict[str, float] | None:
    """Return sub-stage timing breakdown from the most recent GPU point-in-polygon call."""
    return _last_gpu_substage_timings


def _compute_pip_center(
    points: OwnedGeometryArray,
    right_array: OwnedGeometryArray,
) -> tuple[float, float]:
    return _compute_pip_center_host(points, right_array)


def _initialize_host_coarse_result(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
):
    """Build the public object-dtype coarse result at an explicit host boundary."""
    return initialize_coarse_result(points, ~points.validity | right.null_mask)


def _empty_device_bool_result(row_count: int):
    runtime = get_cuda_runtime()
    return runtime.allocate((int(row_count),), cp.bool_, zero=True)


def _evaluate_point_in_polygon_gpu(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
    *,
    strategy: str = "auto",
    return_device: bool = False,
):
    global _last_gpu_substage_timings
    timings: dict[str, float] = {}

    right_array = right.geometry_array
    assert right_array is not None

    # Determine compute precision from device profile.
    from vibespatial.runtime.adaptive import get_cached_snapshot
    snapshot = get_cached_snapshot()
    use_fp32 = not snapshot.device_profile.favors_native_fp64
    compute_type = "float" if use_fp32 else "double"

    # Compute center for coordinate centering.
    center_x, center_y = _compute_pip_center(points, right_array)

    t0 = perf_counter()
    points._ensure_device_state()
    timings["point_upload_s"] = perf_counter() - t0

    t0 = perf_counter()
    right_array._ensure_device_state()
    timings["polygon_upload_s"] = perf_counter() - t0

    selected_strategy = _select_gpu_strategy(
        points.row_count, strategy=strategy, right_array=right_array,
    )
    timings["requested_strategy"] = selected_strategy
    if return_device and selected_strategy == "dense":
        # Dense builds a full-row mask from a host candidate list, which is
        # the least GPU-native variant in this stack. For device-resident
        # consumers, keep execution on the compacted path instead.
        selected_strategy = "compacted"
        timings["strategy_rewrite"] = "dense->compacted(return_device)"
    timings["strategy"] = selected_strategy

    if selected_strategy == "dense":
        t0 = perf_counter()
        coarse = _evaluate_point_within_bounds(points, right)
        timings["coarse_filter_s"] = perf_counter() - t0

        t0 = perf_counter()
        candidate_rows = candidate_rows_from_coarse(coarse)
        timings["candidate_mask_s"] = perf_counter() - t0
        timings["candidate_count"] = int(candidate_rows.size)
        timings["total_rows"] = int(points.row_count)

        if candidate_rows.size == 0:
            _last_gpu_substage_timings = timings
            return coarse

        t0 = perf_counter()
        rows_by_family = _candidate_rows_by_family(right_array, candidate_rows)
        timings["family_split_s"] = perf_counter() - t0
        if not rows_by_family:
            _last_gpu_substage_timings = timings
            return coarse

        t0 = perf_counter()
        runtime = get_cuda_runtime()
        device_dense_out = runtime.allocate((points.row_count,), cp.uint8, zero=True)
        _returned = False
        pending_free = []
        cleanup_requires_sync = False
        try:
            if GeometryFamily.POLYGON in rows_by_family:
                pending_free.extend(
                    _launch_polygon_dense(
                        rows_by_family[GeometryFamily.POLYGON],
                        points,
                        right_array,
                        device_dense_out,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                )
                cleanup_requires_sync = True
            if GeometryFamily.MULTIPOLYGON in rows_by_family:
                pending_free.extend(
                    _launch_multipolygon_dense(
                        rows_by_family[GeometryFamily.MULTIPOLYGON],
                        points,
                        right_array,
                        device_dense_out,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                )
                cleanup_requires_sync = True
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                cleanup_requires_sync = False
                keepalive_out = _wrap_device_result_with_keepalive(
                    device_dense_out,
                    *pending_free,
                )
                pending_free.clear()
                _last_gpu_substage_timings = timings
                _returned = True
                return keepalive_out
            dense_out = runtime.copy_device_to_host(
                device_dense_out,
                reason="point-in-polygon dense strategy result host export",
            )
            cleanup_requires_sync = False
        finally:
            if cleanup_requires_sync:
                runtime.synchronize()
            _free_device_buffers(runtime, pending_free)
            if not _returned:
                runtime.free(device_dense_out)
        coarse[candidate_rows] = dense_out[candidate_rows].astype(bool, copy=False)
    elif selected_strategy == "compacted":
        # Ensure per-family bounds exist on device (same as fused/binned paths).
        runtime = get_cuda_runtime()
        right_state = right_array._ensure_device_state()
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if family in right_state.families:
                device_buffer = right_state.families[family]
                if device_buffer.bounds is None:
                    fam_row_count = right_array.families[family].row_count
                    device_buffer.bounds = runtime.allocate(
                        (fam_row_count, 4), cp.float64,
                    )
                    _launch_family_bounds_kernel(
                        family, device_buffer, row_count=fam_row_count,
                    )
        t0 = perf_counter()
        candidate_rows = _launch_bounds_candidate_rows(points, right_array, compute_type=compute_type, center_x=center_x, center_y=center_y)
        timings["coarse_filter_s"] = perf_counter() - t0
        timings["candidate_mask_s"] = 0.0
        timings["candidate_count"] = int(candidate_rows.count)
        timings["total_rows"] = int(points.row_count)
        timings["family_split_s"] = 0.0

        if candidate_rows.count == 0:
            _last_gpu_substage_timings = timings
            if return_device:
                return _empty_device_bool_result(points.row_count)
            coarse = _initialize_host_coarse_result(points, right)
            return coarse

        runtime = get_cuda_runtime()
        device_dense_out = runtime.allocate((points.row_count,), cp.uint8)
        device_dense_out[...] = 0
        _returned = False
        pending_free = []
        cleanup_requires_sync = False
        try:
            t0 = perf_counter()
            if GeometryFamily.POLYGON in right_array.families:
                polygon_hits = _launch_polygon_compacted(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                pending_free.append(polygon_hits)
                cleanup_requires_sync = True
                _scatter_compacted_hits(
                    candidate_rows.values,
                    polygon_hits,
                    device_dense_out,
                    candidate_rows.count,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                multipolygon_hits = _launch_multipolygon_compacted(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                pending_free.append(multipolygon_hits)
                cleanup_requires_sync = True
                _scatter_compacted_hits(
                    candidate_rows.values,
                    multipolygon_hits,
                    device_dense_out,
                    candidate_rows.count,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                cleanup_requires_sync = False
                keepalive_out = _wrap_device_result_with_keepalive(
                    device_dense_out,
                    candidate_rows.values,
                    *pending_free,
                )
                pending_free.clear()
                _last_gpu_substage_timings = timings
                _returned = True
                return keepalive_out
            dense_out = runtime.copy_device_to_host(
                device_dense_out,
                reason="point-in-polygon compacted strategy result host export",
            )
            cleanup_requires_sync = False
        finally:
            if cleanup_requires_sync:
                runtime.synchronize()
            _free_device_buffers(runtime, pending_free)
            if not _returned:
                runtime.free(device_dense_out)
                runtime.free(candidate_rows.values)
        coarse = _initialize_host_coarse_result(points, right)
        coarse[dense_true_mask(dense_out)] = True
    elif selected_strategy == "binned":
        # Binned: bounds filter first, then dispatch PIP in work-balanced bins
        # with a block-per-pair kernel for the complex (>1024 vertex) bin.
        runtime = get_cuda_runtime()
        # Ensure per-family bounds exist on device (same as fused path).
        right_state = right_array._ensure_device_state()
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if family in right_state.families:
                device_buffer = right_state.families[family]
                if device_buffer.bounds is None:
                    fam_row_count = right_array.families[family].row_count
                    device_buffer.bounds = runtime.allocate(
                        (fam_row_count, 4), cp.float64,
                    )
                    _launch_family_bounds_kernel(
                        family, device_buffer, row_count=fam_row_count,
                    )
        t0 = perf_counter()
        candidate_rows = _launch_bounds_candidate_rows(
            points, right_array, compute_type=compute_type,
            center_x=center_x, center_y=center_y,
        )
        timings["coarse_filter_s"] = perf_counter() - t0
        timings["candidate_mask_s"] = 0.0
        timings["candidate_count"] = int(candidate_rows.count)
        timings["total_rows"] = int(points.row_count)
        timings["family_split_s"] = 0.0

        if candidate_rows.count == 0:
            _last_gpu_substage_timings = timings
            if return_device:
                return _empty_device_bool_result(points.row_count)
            coarse = _initialize_host_coarse_result(points, right)
            return coarse

        # Compute per-candidate work estimates on device from the cached
        # family routing metadata and offset buffers. This avoids copying
        # the candidate row vector back to host just to bin work.
        t0 = perf_counter()
        work_estimates = _compute_work_estimates_for_candidates_device(
            candidate_rows.values,
            right_array,
        )
        timings["work_estimation_s"] = perf_counter() - t0
        timings["work_cv"] = work_cv(work_estimates)

        device_dense_out = runtime.allocate((points.row_count,), cp.uint8)
        device_dense_out[...] = 0
        _returned = False
        pending_free = []
        cleanup_requires_sync = False
        try:
            t0 = perf_counter()
            if GeometryFamily.POLYGON in right_array.families:
                polygon_hits, polygon_temps = _binned_polygon_dispatch(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    work_estimates,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                pending_free.extend(polygon_temps)
                pending_free.append(polygon_hits)
                cleanup_requires_sync = True
                _scatter_compacted_hits(
                    candidate_rows.values,
                    polygon_hits,
                    device_dense_out,
                    candidate_rows.count,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                multipolygon_hits, multipolygon_temps = _binned_multipolygon_dispatch(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    work_estimates,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                pending_free.extend(multipolygon_temps)
                pending_free.append(multipolygon_hits)
                cleanup_requires_sync = True
                _scatter_compacted_hits(
                    candidate_rows.values,
                    multipolygon_hits,
                    device_dense_out,
                    candidate_rows.count,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                cleanup_requires_sync = False
                keepalive_out = _wrap_device_result_with_keepalive(
                    device_dense_out,
                    candidate_rows.values,
                    *pending_free,
                )
                pending_free.clear()
                _last_gpu_substage_timings = timings
                _returned = True
                return keepalive_out
            dense_out = runtime.copy_device_to_host(
                device_dense_out,
                reason="point-in-polygon binned strategy result host export",
            )
            cleanup_requires_sync = False
        finally:
            if cleanup_requires_sync:
                runtime.synchronize()
            _free_device_buffers(runtime, pending_free)
            if not _returned:
                runtime.free(device_dense_out)
                runtime.free(candidate_rows.values)
        coarse = _initialize_host_coarse_result(points, right)
        coarse[dense_true_mask(dense_out)] = True
    else:
        # Fused: single kernel does bounds check + PIP in one launch.
        runtime = get_cuda_runtime()
        t0 = perf_counter()
        device_out = _launch_fused(points, right_array, compute_type=compute_type, center_x=center_x, center_y=center_y)
        timings["fused_kernel_s"] = perf_counter() - t0
        timings["coarse_filter_s"] = 0.0
        timings["candidate_mask_s"] = 0.0
        timings["family_split_s"] = 0.0
        timings["total_rows"] = int(points.row_count)
        if return_device:
            # Keep result on GPU — caller is responsible for freeing.
            _last_gpu_substage_timings = timings
            return device_out
        try:
            dense_out = runtime.copy_device_to_host(
                device_out,
                reason="point-in-polygon fused strategy result host export",
            )
        finally:
            runtime.free(device_out)
        coarse = _initialize_host_coarse_result(points, right)
        null_mask = ~points.validity | right.null_mask
        # Normalize to object-dtype ndarray matching dense/compacted paths.
        coarse[dense_true_mask(dense_out)] = True
        coarse[null_mask] = None

    _last_gpu_substage_timings = timings
    _log.info(
        "point_in_polygon GPU substages: %s",
        " | ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in timings.items()
        ),
    )
    return coarse


@register_kernel_variant(
    "point_in_polygon",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    preferred_residency=Residency.DEVICE,
    tags=("coarse-filter", "refine", "cuda-python", "dense-or-compacted"),
)
def _point_in_polygon_gpu_variant(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
    *,
    strategy: str = "auto",
):
    return _evaluate_point_in_polygon_gpu(points, right, strategy=strategy)


def point_in_polygon(
    points: PointSequence,
    polygons: PointSequence,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _return_device: bool = False,
) -> list[bool | None]:
    global _last_gpu_substage_timings

    t0 = perf_counter()
    left = coerce_geometry_array(
        points,
        arg_name="points",
        expected_families=(GeometryFamily.POINT,),
    )
    coerce_left_s = perf_counter() - t0

    # Coerce the right side to an OwnedGeometryArray cheaply (no bounds).
    # Full _normalize_right_input (including compute_geometry_bounds) is
    # deferred to the CPU path — the fused GPU kernel reads device-resident
    # bounds directly so the CPU bounds computation is redundant there.
    t0 = perf_counter()
    right_array = coerce_geometry_array(
        polygons,
        arg_name="polygons",
        expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
    )
    coerce_right_s = perf_counter() - t0

    if right_array.row_count != left.row_count:
        raise ValueError(
            f"point_in_polygon requires aligned inputs; "
            f"got {left.row_count} points and {right_array.row_count} polygon rows"
        )

    context = resolve_predicate_context(
        kernel_name="point_in_polygon",
        left=left,
        right=right_array,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    if context.runtime_selection.selected is ExecutionMode.GPU:
        # Build lightweight NormalizedBoundsInput without CPU bounds.
        if _return_device:
            # Device-return consumers use the fused/compacted result directly;
            # host null/empty masks are only needed when exporting public
            # object-dtype predicate results.
            null_mask = empty_bool_mask_like(empty_gpu_bounds_placeholder())
            empty_mask = null_mask
        else:
            null_mask = ~right_array.validity
            empty_mask = empty_bool_mask_like(right_array.validity)
        right = NormalizedBoundsInput(
            bounds=empty_gpu_bounds_placeholder(),
            null_mask=null_mask,
            empty_mask=empty_mask,
            geometry_array=right_array,
        )
        t0 = perf_counter()
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="point_in_polygon selected GPU execution",
        )
        right_array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="point_in_polygon selected GPU execution",
        )
        move_to_device_s = perf_counter() - t0

        gpu_out = _evaluate_point_in_polygon_gpu(
            left, right, return_device=_return_device,
        )

        # Merge outer timing into substage report
        if _last_gpu_substage_timings is not None:
            _last_gpu_substage_timings["coerce_left_s"] = coerce_left_s
            _last_gpu_substage_timings["coerce_right_s"] = coerce_right_s
            _last_gpu_substage_timings["move_to_device_s"] = move_to_device_s

        # _return_device: keep result on GPU as CuPy bool array for
        # zero-copy pipelines (feeds into device_take).
        if _return_device:
            import cupy as _cp
            bool_out = _cp.asarray(gpu_out, dtype=_cp.bool_)
            if bool_out is gpu_out:
                return bool_out
            return _wrap_device_result_with_keepalive(bool_out, gpu_out)

        # All strategies now return a normalized object-dtype ndarray.
        return _to_python_result(gpu_out)

    # CPU path: ensure host-side buffers are materialized (device-resident
    # arrays from the GPU byte-classify parser may have empty host buffers).
    left._ensure_host_state()
    right_array._ensure_host_state()

    # CPU path: full normalize with bounds (needed by CPU coarse filter).
    t0 = perf_counter()
    right = _normalize_right_input(polygons, expected_len=left.row_count)
    normalize_right_s = perf_counter() - t0
    if _last_gpu_substage_timings is not None:
        _last_gpu_substage_timings["normalize_right_s"] = normalize_right_s
    cpu_out = _point_in_polygon_cpu_variant(left, right)
    if _return_device:
        # CPU fallback for _return_device: return numpy bool array
        return cpu_return_device_fallback(cpu_out)
    return _to_python_result(cpu_out)


def point_in_polygon_expression(
    points: PointSequence,
    polygons: PointSequence,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    source_token: str | None = None,
    operation: str = "point_in_polygon",
):
    """Return private point-in-polygon predicate results as a NativeExpression.

    Physical shape: row-aligned point-vs-polygon predicate flow. The existing
    staged PIP pipeline produces a dense bool vector; sanctioned native
    consumers can compare it into a `NativeRowSet` instead of exporting a
    public bool Series.
    """
    left = coerce_geometry_array(
        points,
        arg_name="points",
        expected_families=(GeometryFamily.POINT,),
    )
    right = coerce_geometry_array(
        polygons,
        arg_name="polygons",
        expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
    )
    values = point_in_polygon(
        left,
        right,
        dispatch_mode=dispatch_mode,
        precision=precision,
        _return_device=True,
    )
    selected = (
        ExecutionMode.GPU
        if hasattr(values, "__cuda_array_interface__")
        else ExecutionMode.CPU
    )
    record_dispatch_event(
        surface="vibespatial.kernels.predicates.point_in_polygon_expression",
        operation=operation,
        requested=ExecutionMode.GPU,
        selected=selected,
        implementation=(
            "native_point_in_polygon_expression_gpu"
            if selected is ExecutionMode.GPU
            else "native_point_in_polygon_expression_host"
        ),
        reason="point-in-polygon predicate result returned as NativeExpression",
        detail=(
            f"row_count={left.row_count}; "
            "workload_shape=aligned_pairwise_point_region; carrier=NativeExpression"
        ),
    )
    from vibespatial.api._native_expression import NativeExpression

    return NativeExpression(
        operation=operation,
        values=values,
        source_token=source_token,
        source_row_count=left.row_count,
        dtype=str(getattr(values, "dtype", "bool")),
        precision="predicate",
    )
