"""Reusable dense-cell index for conservative point-tree candidates.

The carrier stores each point exactly once. Predicate queries may consume its
cell-aligned bbox superset because an exact predicate immediately refines the
relation. Bbox-only queries continue to use the exact Morton path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log2, sqrt

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import PairSortStrategy, exclusive_sum, sort_pairs
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event

from .point_grid_index_kernels import (
    _POINT_GRID_INDEX_SOURCE,
    POINT_GRID_INDEX_KERNEL_NAMES,
)
from .query_types import _DeviceCandidates, require_device_candidate_pair_capacity

_MIN_POINT_GRID_ROWS = 100_000
_TARGET_POINTS_PER_CELL = 8
_MIN_GRID_SIZE = 64
_MAX_GRID_SIZE = 2_048
_SCATTER_THREADS = 128

request_nvrtc_warmup(
    [("point-grid-index", _POINT_GRID_INDEX_SOURCE, POINT_GRID_INDEX_KERNEL_NAMES)]
)


@dataclass(frozen=True)
class PreparedPointGridIndex:
    """Device-resident point rows grouped into a fixed square grid."""

    grid_size: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    sorted_tree_rows: object
    cell_counts: object
    cell_offsets: object
    integral_counts: object

    @property
    def device_bytes(self) -> int:
        return sum(
            int(getattr(value, "nbytes", 0))
            for value in (
                self.sorted_tree_rows,
                self.cell_counts,
                self.cell_offsets,
                self.integral_counts,
            )
        )


def point_grid_index_kernels():
    return compile_kernel_group(
        "point-grid-index",
        _POINT_GRID_INDEX_SOURCE,
        POINT_GRID_INDEX_KERNEL_NAMES,
    )


def _grid_size_for_rows(row_count: int) -> int:
    target = max(sqrt(max(row_count, 1) / _TARGET_POINTS_PER_CELL), 1.0)
    power_of_two = 1 << int(ceil(log2(target)))
    return min(max(power_of_two, _MIN_GRID_SIZE), _MAX_GRID_SIZE)


def _point_grid_required_bytes(
    row_count: int,
    cell_count: int,
    *,
    needs_device_bounds: bool,
) -> int:
    """Bound simultaneously-live point-grid rows, radix scratch, and cells."""
    # Row temporaries include optional fp64 bounds, finite masks, cell x/y/id,
    # source rows, radix key/value outputs, and conservative radix workspace.
    row_bytes = 192 if needs_device_bounds else 160
    # Cell storage includes counts, offsets, integral image, both cumsum
    # workspaces, and library allocation slack.
    return int(row_count) * row_bytes + int(cell_count) * 96 + (1 << 20)


def prepare_point_grid_index(flat_index) -> PreparedPointGridIndex | None:
    """Build or return the cached point grid for a homogeneous point tree."""
    cached = getattr(flat_index, "point_grid", None)
    if cached is not None:
        return cached
    owned = flat_index.geometry_array
    if (
        owned.row_count < _MIN_POINT_GRID_ROWS
        or set(owned.families) != {GeometryFamily.POINT}
    ):
        return None
    xmin, ymin, xmax, ymax = map(float, flat_index.total_bounds)
    if not all(isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        return None
    if not xmax > xmin or not ymax > ymin:
        return None

    import cupy as cp

    grid_size = _grid_size_for_rows(owned.row_count)
    cell_count = grid_size * grid_size
    runtime = get_cuda_runtime()
    required_bytes = _point_grid_required_bytes(
        owned.row_count,
        cell_count,
        needs_device_bounds=flat_index.device_bounds is None,
    )
    admission = runtime.admit_device_memory(
        stage="spatial.point_grid_index",
        required_bytes=required_bytes,
        requested_units=owned.row_count,
    )
    if not admission.admitted:
        return None

    device_bounds = flat_index.device_bounds
    if device_bounds is None:
        device_bounds = runtime.from_host(
            np.ascontiguousarray(flat_index.bounds, dtype=np.float64)
        )
        object.__setattr__(flat_index, "device_bounds", device_bounds)
    bounds = cp.asarray(device_bounds, dtype=cp.float64).reshape(-1, 4)
    finite = cp.isfinite(bounds).all(axis=1)
    cell_x = cp.floor(
        (bounds[:, 0] - xmin) * (grid_size / (xmax - xmin))
    ).astype(cp.int32, copy=False)
    cell_y = cp.floor(
        (bounds[:, 1] - ymin) * (grid_size / (ymax - ymin))
    ).astype(cp.int32, copy=False)
    cell_x = cp.clip(cell_x, 0, grid_size - 1)
    cell_y = cp.clip(cell_y, 0, grid_size - 1)
    invalid_cell = np.uint64(cell_count)
    cell_ids = cp.where(
        finite,
        cell_y.astype(cp.uint64) * np.uint64(grid_size) + cell_x.astype(cp.uint64),
        invalid_cell,
    )
    rows = cp.arange(owned.row_count, dtype=cp.int32)
    sorted_pairs = sort_pairs(
        cell_ids,
        rows,
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    sorted_tree_rows = cp.asarray(sorted_pairs.values, dtype=cp.int32)
    all_counts = cp.bincount(cell_ids, minlength=cell_count + 1).astype(
        cp.int32,
        copy=False,
    )
    cell_counts = cp.ascontiguousarray(all_counts[:cell_count])
    cell_offsets = exclusive_sum(cell_counts.astype(cp.int64), synchronize=False)
    integral_counts = cp.zeros((grid_size + 1, grid_size + 1), dtype=cp.int64)
    interior = integral_counts[1:, 1:]
    interior[...] = cell_counts.reshape(grid_size, grid_size)
    cp.cumsum(interior, axis=0, out=interior)
    cp.cumsum(interior, axis=1, out=interior)
    prepared = PreparedPointGridIndex(
        grid_size=grid_size,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        sorted_tree_rows=sorted_tree_rows,
        cell_counts=cell_counts,
        cell_offsets=cell_offsets,
        integral_counts=integral_counts,
    )
    object.__setattr__(flat_index, "point_grid", prepared)
    record_dispatch_event(
        surface="vibespatial.spatial.point_grid_index",
        operation="prepare",
        implementation="dense_point_cell_index_gpu",
        reason=(
            f"grouped {owned.row_count} point rows once into "
            f"{grid_size}x{grid_size} cells"
        ),
        selected=ExecutionMode.GPU,
    )
    return prepared


def point_grid_superset_query(flat_index, query_bounds) -> _DeviceCandidates | None:
    """Return cell-conservative pairs for immediate exact refinement."""
    prepared = prepare_point_grid_index(flat_index)
    if prepared is None:
        return None

    import cupy as cp

    runtime = get_cuda_runtime()
    bounds = cp.ascontiguousarray(cp.asarray(query_bounds, dtype=cp.float64)).reshape(-1, 4)
    query_count = int(bounds.shape[0])
    if query_count == 0:
        return _DeviceCandidates(
            d_left=cp.empty(0, dtype=cp.int32),
            d_right=cp.empty(0, dtype=cp.int32),
            total_pairs=0,
        )
    query_counts = cp.empty(query_count, dtype=cp.int64)
    query_offsets = None
    query_cursors = None
    out_left = None
    out_right = None
    kernels = point_grid_index_kernels()
    ptr = runtime.pointer
    try:
        count_kernel = kernels["point_grid_query_counts"]
        grid, block = runtime.launch_config(count_kernel, query_count)
        runtime.launch(
            count_kernel,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(bounds),
                    prepared.xmin,
                    prepared.ymin,
                    prepared.xmax,
                    prepared.ymax,
                    prepared.grid_size,
                    ptr(prepared.integral_counts),
                    ptr(query_counts),
                    query_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        query_offsets = exclusive_sum(query_counts, synchronize=False)
        total_pairs = count_scatter_total(
            runtime,
            query_counts,
            query_offsets,
            reason="point-grid conservative candidate allocation fence",
        )
        if total_pairs == 0:
            return _DeviceCandidates(
                d_left=cp.empty(0, dtype=cp.int32),
                d_right=cp.empty(0, dtype=cp.int32),
                total_pairs=0,
            )
        require_device_candidate_pair_capacity(
            total_pairs,
            relation_name="device point-grid conservative candidate relation",
        )
        out_left = cp.empty(total_pairs, dtype=cp.int32)
        out_right = cp.empty(total_pairs, dtype=cp.int32)
        query_cursors = query_offsets.astype(cp.uint64, copy=True)
        scatter_kernel = kernels["point_grid_query_scatter"]
        runtime.launch(
            scatter_kernel,
            grid=(query_count, 1, 1),
            block=(_SCATTER_THREADS, 1, 1),
            params=(
                (
                    ptr(bounds),
                    prepared.xmin,
                    prepared.ymin,
                    prepared.xmax,
                    prepared.ymax,
                    prepared.grid_size,
                    ptr(prepared.cell_counts),
                    ptr(prepared.cell_offsets),
                    ptr(prepared.sorted_tree_rows),
                    ptr(query_offsets),
                    ptr(query_cursors),
                    ptr(out_left),
                    ptr(out_right),
                    query_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        result = _DeviceCandidates(
            d_left=out_left,
            d_right=out_right,
            total_pairs=total_pairs,
        )
        out_left = None
        out_right = None
        return result
    finally:
        runtime.free(query_counts)
        runtime.free(query_offsets)
        runtime.free(query_cursors)
        runtime.free(out_left)
        runtime.free(out_right)
