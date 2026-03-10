from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from vibespatial.cccl_primitives import sort_pairs
from vibespatial.cccl_precompile import request_warmup

request_warmup(["radix_sort_i32_i32", "radix_sort_u64_i32"])

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None
from vibespatial.cuda_runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds, compute_morton_keys  # noqa: E402
from vibespatial.owned_geometry import OwnedGeometryArray  # noqa: E402
from vibespatial.residency import Residency, TransferTrigger  # noqa: E402
from vibespatial.adaptive_runtime import plan_dispatch_selection  # noqa: E402
from vibespatial.precision import KernelClass  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime  # noqa: E402


_GPU_BOUNDS_PAIRS_THRESHOLD = 2_048


def _default_index_runtime_selection() -> RuntimeSelection:
    return plan_dispatch_selection(
        kernel_name="flat_index_build",
        kernel_class=KernelClass.COARSE,
        row_count=1,  # always exceeds threshold of 0
    )


_INDEXING_KERNEL_SOURCE = """
extern "C" __device__ unsigned long long spread_bits_32(unsigned int value) {
  unsigned long long x = (unsigned long long) value;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

extern "C" __global__ void morton_keys_from_bounds(
    const double* bounds,
    double minx,
    double miny,
    double maxx,
    double maxy,
    unsigned long long* out_keys,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int base = row * 4;
  const double bx0 = bounds[base + 0];
  const double by0 = bounds[base + 1];
  const double bx1 = bounds[base + 2];
  const double by1 = bounds[base + 3];
  if (isnan(bx0) || isnan(by0) || isnan(bx1) || isnan(by1)) {
    out_keys[row] = 0xFFFFFFFFFFFFFFFFULL;
    return;
  }
  const double span_x = fmax(maxx - minx, 1e-12);
  const double span_y = fmax(maxy - miny, 1e-12);
  const double center_x = (bx0 + bx1) * 0.5;
  const double center_y = (by0 + by1) * 0.5;
  const unsigned int norm_x = (unsigned int) llround(((center_x - minx) / span_x) * 65535.0);
  const unsigned int norm_y = (unsigned int) llround(((center_y - miny) / span_y) * 65535.0);
  out_keys[row] = spread_bits_32(norm_x) | (spread_bits_32(norm_y) << 1);
}

/* Phase 4 — sweep-plane MBR overlap test.
 *
 * Each thread handles one geometry from the left set and sweeps over a
 * Morton-sorted window of right geometries to find overlapping MBRs.
 * The sweep window is bounded by the left geometry's x-extent.
 *
 * Two passes: pass 0 counts valid pairs per left geometry;
 *             pass 1 writes pairs using a prefix-sum offset array.
 */
extern "C" __global__ void sweep_mbr_overlap(
    const double* left_bounds,     /* [left_count, 4] : minx,miny,maxx,maxy */
    const int*    left_indices,     /* original row indices for left */
    int           left_count,
    const double* right_bounds,    /* [right_count, 4] */
    const int*    right_indices,    /* original row indices for right */
    int           right_count,
    int*          out_left,         /* output left row indices (pass 1 only) */
    int*          out_right,        /* output right row indices (pass 1 only) */
    int*          counts,           /* per-left-row pair counts (pass 0: written; pass 1: read as offsets) */
    int           pass_number,      /* 0 = count, 1 = scatter */
    int           same_input,       /* 1 if left == right (upper-triangle mode) */
    int           include_self      /* 1 to include (i,i) pairs in same_input mode */
) {
  const int lid = blockIdx.x * blockDim.x + threadIdx.x;
  if (lid >= left_count) return;

  const int lbase = lid * 4;
  const double lx0 = left_bounds[lbase + 0];
  const double ly0 = left_bounds[lbase + 1];
  const double lx1 = left_bounds[lbase + 2];
  const double ly1 = left_bounds[lbase + 3];

  /* Skip NaN bounds */
  if (isnan(lx0) || isnan(ly0) || isnan(lx1) || isnan(ly1)) {
    if (pass_number == 0) counts[lid] = 0;
    return;
  }

  int pair_count = 0;
  int write_offset = 0;
  if (pass_number == 1) write_offset = counts[lid];

  const int left_row = left_indices[lid];

  for (int rid = 0; rid < right_count; rid++) {
    const int rbase = rid * 4;
    const double rx0 = right_bounds[rbase + 0];
    const double ry0 = right_bounds[rbase + 1];
    const double rx1 = right_bounds[rbase + 2];
    const double ry1 = right_bounds[rbase + 3];

    if (isnan(rx0) || isnan(ry0) || isnan(rx1) || isnan(ry1)) continue;

    /* MBR overlap test */
    if (lx0 > rx1 || lx1 < rx0 || ly0 > ry1 || ly1 < ry0) continue;

    const int right_row = right_indices[rid];

    /* Upper-triangle filter for same-input mode */
    if (same_input) {
      if (include_self) {
        if (left_row > right_row) continue;
      } else {
        if (left_row >= right_row) continue;
      }
    }

    if (pass_number == 0) {
      pair_count++;
    } else {
      out_left[write_offset + pair_count] = left_row;
      out_right[write_offset + pair_count] = right_row;
      pair_count++;
    }
  }

  if (pass_number == 0) counts[lid] = pair_count;
}
"""


@dataclass(frozen=True)
class CandidatePairs:
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_bounds: np.ndarray
    right_bounds: np.ndarray
    pairs_examined: int
    tile_size: int
    same_input: bool

    @property
    def count(self) -> int:
        return int(self.left_indices.size)


def _valid_row_mask(bounds: np.ndarray) -> np.ndarray:
    return ~np.isnan(bounds).any(axis=1)


def _generate_bounds_pairs_gpu(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    *,
    same_input: bool,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """GPU path for MBR overlap pair generation using sweep-plane NVRTC kernel.

    Returns (left_indices, right_indices, pairs_examined).
    """
    runtime = get_cuda_runtime()
    kernels = _indexing_kernels()
    sweep_kernel = kernels["sweep_mbr_overlap"]
    ptr = runtime.pointer

    left_valid_mask = ~np.isnan(left_bounds).any(axis=1)
    right_valid_mask = left_valid_mask if same_input else ~np.isnan(right_bounds).any(axis=1)
    left_valid = np.flatnonzero(left_valid_mask)
    right_valid = left_valid if same_input else np.flatnonzero(right_valid_mask)

    left_count = int(left_valid.size)
    right_count = int(right_valid.size)

    if left_count == 0 or right_count == 0:
        empty = np.asarray([], dtype=np.int32)
        return empty, empty, 0

    # Upload valid bounds subsets to device
    left_valid_bounds = np.ascontiguousarray(left_bounds[left_valid])
    right_valid_bounds = left_valid_bounds if same_input else np.ascontiguousarray(right_bounds[right_valid])
    left_valid_indices = left_valid.astype(np.int32, copy=False)
    right_valid_indices = left_valid_indices if same_input else right_valid.astype(np.int32, copy=False)

    d_left_bounds = runtime.from_host(left_valid_bounds)
    d_right_bounds = d_left_bounds if same_input else runtime.from_host(right_valid_bounds)
    d_left_indices = runtime.from_host(left_valid_indices)
    d_right_indices = d_left_indices if same_input else runtime.from_host(right_valid_indices)
    d_counts = runtime.allocate((left_count,), np.int32)

    # Pass 0: count pairs per left geometry
    count_params = (
        (
            ptr(d_left_bounds), ptr(d_left_indices), left_count,
            ptr(d_right_bounds), ptr(d_right_indices), right_count,
            0,  # out_left (unused in pass 0)
            0,  # out_right (unused in pass 0)
            ptr(d_counts),
            0,  # pass_number = 0
            int(same_input),
            int(include_self),
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(sweep_kernel, left_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=count_params)

    # Prefix sum on counts to get scatter offsets (using CuPy for element-wise, ADR-0033)
    cp_counts = cp.asarray(d_counts)
    total_pairs = int(cp_counts.sum())

    if total_pairs == 0:
        runtime.free(d_left_bounds)
        if not same_input:
            runtime.free(d_right_bounds)
            runtime.free(d_right_indices)
        runtime.free(d_left_indices)
        runtime.free(d_counts)
        empty = np.asarray([], dtype=np.int32)
        return empty, empty, left_count * right_count

    # Exclusive prefix sum to get write offsets
    d_offsets = cp.empty_like(cp_counts)
    cp.cumsum(cp_counts, out=d_offsets)
    d_offsets -= cp_counts  # exclusive scan
    # Copy offsets back into d_counts buffer for the kernel
    cp.copyto(cp.asarray(d_counts), d_offsets)
    cp.cuda.Stream.null.synchronize()

    # Allocate output arrays
    d_out_left = runtime.allocate((total_pairs,), np.int32)
    d_out_right = runtime.allocate((total_pairs,), np.int32)

    # Pass 1: scatter pairs
    scatter_params = (
        (
            ptr(d_left_bounds), ptr(d_left_indices), left_count,
            ptr(d_right_bounds), ptr(d_right_indices), right_count,
            ptr(d_out_left), ptr(d_out_right),
            ptr(d_counts),
            1,  # pass_number = 1
            int(same_input),
            int(include_self),
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(sweep_kernel, left_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=scatter_params)
    runtime.synchronize()

    # Transfer results back to host
    left_result = runtime.copy_device_to_host(d_out_left).astype(np.int32, copy=False)
    right_result = runtime.copy_device_to_host(d_out_right).astype(np.int32, copy=False)

    # Clean up device memory
    runtime.free(d_left_bounds)
    if not same_input:
        runtime.free(d_right_bounds)
        runtime.free(d_right_indices)
    runtime.free(d_left_indices)
    runtime.free(d_counts)
    runtime.free(d_out_left)
    runtime.free(d_out_right)

    pairs_examined = left_count * right_count
    return left_result, right_result, pairs_examined


def generate_bounds_pairs(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray | None = None,
    *,
    tile_size: int = 256,
    include_self: bool = False,
) -> CandidatePairs:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    same_input = right is None or right is left
    right_array = left if right is None else right

    left_bounds = compute_geometry_bounds(left)
    right_bounds = left_bounds if same_input else compute_geometry_bounds(right_array)

    # GPU path: use sweep-plane NVRTC kernel when geometry count exceeds threshold
    total_geom_count = left.row_count + (0 if same_input else right_array.row_count)
    use_gpu = (
        total_geom_count >= _GPU_BOUNDS_PAIRS_THRESHOLD
        and has_gpu_runtime()
        and cp is not None
    )
    if use_gpu:
        left_indices, right_indices, pairs_examined = _generate_bounds_pairs_gpu(
            left_bounds,
            right_bounds,
            same_input=same_input,
            include_self=include_self,
        )
        return CandidatePairs(
            left_indices=left_indices,
            right_indices=right_indices,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            pairs_examined=pairs_examined,
            tile_size=tile_size,
            same_input=same_input,
        )

    # CPU path: nested-tile loop
    left_valid = np.flatnonzero(_valid_row_mask(left_bounds))
    right_valid = left_valid if same_input else np.flatnonzero(_valid_row_mask(right_bounds))

    candidate_left: list[np.ndarray] = []
    candidate_right: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_valid.size, tile_size):
        left_chunk = left_valid[left_start : left_start + tile_size]
        left_chunk_bounds = left_bounds[left_chunk]
        right_loop_start = left_start if same_input else 0
        for right_start in range(right_loop_start, right_valid.size, tile_size):
            right_chunk = right_valid[right_start : right_start + tile_size]
            right_chunk_bounds = right_bounds[right_chunk]
            pairs_examined += int(left_chunk.size * right_chunk.size)

            intersects = (
                (left_chunk_bounds[:, None, 0] <= right_chunk_bounds[None, :, 2])
                & (left_chunk_bounds[:, None, 2] >= right_chunk_bounds[None, :, 0])
                & (left_chunk_bounds[:, None, 1] <= right_chunk_bounds[None, :, 3])
                & (left_chunk_bounds[:, None, 3] >= right_chunk_bounds[None, :, 1])
            )

            left_local, right_local = np.nonzero(intersects)
            if same_input:
                left_rows = left_chunk[left_local]
                right_rows = right_chunk[right_local]
                if left_start == right_start:
                    keep = left_rows <= right_rows if include_self else left_rows < right_rows
                    left_rows = left_rows[keep]
                    right_rows = right_rows[keep]
                if left_rows.size == 0:
                    continue
                candidate_left.append(left_rows.astype(np.int32, copy=False))
                candidate_right.append(right_rows.astype(np.int32, copy=False))
                continue

            if left_local.size == 0:
                continue
            candidate_left.append(left_chunk[left_local].astype(np.int32, copy=False))
            candidate_right.append(right_chunk[right_local].astype(np.int32, copy=False))

    if candidate_left:
        left_indices = np.concatenate(candidate_left)
        right_indices = np.concatenate(candidate_right)
    else:
        left_indices = np.asarray([], dtype=np.int32)
        right_indices = np.asarray([], dtype=np.int32)

    return CandidatePairs(
        left_indices=left_indices,
        right_indices=right_indices,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        pairs_examined=pairs_examined,
        tile_size=tile_size,
        same_input=same_input,
    )


@dataclass(frozen=True)
class BoundsPairBenchmark:
    dataset: str
    rows: int
    tile_size: int
    elapsed_seconds: float
    pairs_examined: int
    candidate_pairs: int


def benchmark_bounds_pairs(
    geometry_array: OwnedGeometryArray,
    *,
    dataset: str,
    tile_size: int = 256,
) -> BoundsPairBenchmark:
    started = perf_counter()
    pairs = generate_bounds_pairs(geometry_array, tile_size=tile_size)
    elapsed = perf_counter() - started
    return BoundsPairBenchmark(
        dataset=dataset,
        rows=geometry_array.row_count,
        tile_size=tile_size,
        elapsed_seconds=elapsed,
        pairs_examined=pairs.pairs_examined,
        candidate_pairs=pairs.count,
    )


@dataclass(frozen=True)
class FlatSpatialIndex:
    geometry_array: OwnedGeometryArray
    _host_order: object  # np.ndarray or None (lazy from device_order)
    _host_morton_keys: object  # np.ndarray or None (lazy from device_morton_keys)
    bounds: np.ndarray
    total_bounds: tuple[float, float, float, float]
    regular_grid: "RegularGridRectIndex | None" = None
    device_morton_keys: object = None  # CuPy device array or None
    device_order: object = None  # CuPy device array or None

    @property
    def order(self) -> np.ndarray:
        """Lazily materialise host order array from device (ADR-0005)."""
        if self._host_order is not None:
            return self._host_order
        host_order = cp.asnumpy(self.device_order).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_order", host_order)
        return host_order

    @property
    def morton_keys(self) -> np.ndarray:
        """Lazily materialise host morton_keys array from device (ADR-0005)."""
        if self._host_morton_keys is not None:
            return self._host_morton_keys
        host_keys = cp.asnumpy(self.device_morton_keys).astype(np.uint64, copy=False)
        object.__setattr__(self, "_host_morton_keys", host_keys)
        return host_keys

    @property
    def size(self) -> int:
        if self._host_order is not None:
            return int(self._host_order.size)
        if self.device_order is not None:
            return int(self.device_order.size)
        # Avoid D→H just for size — use bounds row count.
        return int(self.bounds.shape[0])

    def query_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        order = self.order
        minx, miny, maxx, maxy = bounds
        ordered_bounds = self.bounds[order]
        mask = (
            (ordered_bounds[:, 0] <= maxx)
            & (ordered_bounds[:, 2] >= minx)
            & (ordered_bounds[:, 1] <= maxy)
            & (ordered_bounds[:, 3] >= miny)
        )
        return order[mask]

    def query(
        self,
        other: OwnedGeometryArray,
        *,
        tile_size: int = 256,
    ) -> CandidatePairs:
        order = self.order
        ordered = self.geometry_array
        result = generate_bounds_pairs(other, ordered, tile_size=tile_size)
        if result.count == 0:
            return result
        mapped_right = order[result.right_indices]
        return CandidatePairs(
            left_indices=result.left_indices,
            right_indices=mapped_right.astype(np.int32, copy=False),
            left_bounds=result.left_bounds,
            right_bounds=result.right_bounds,
            pairs_examined=result.pairs_examined,
            tile_size=result.tile_size,
            same_input=result.same_input,
        )


@dataclass(frozen=True)
class RegularGridRectIndex:
    origin_x: float
    origin_y: float
    cell_width: float
    cell_height: float
    cols: int
    rows: int
    size: int


_INDEXING_KERNEL_NAMES = ("morton_keys_from_bounds", "sweep_mbr_overlap")

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
request_nvrtc_warmup([
    ("indexing", _INDEXING_KERNEL_SOURCE, _INDEXING_KERNEL_NAMES),
])


def _indexing_kernels():
    return compile_kernel_group("indexing", _INDEXING_KERNEL_SOURCE, _INDEXING_KERNEL_NAMES)


def _detect_regular_grid_rect_index(
    geometry_array: OwnedGeometryArray,
    bounds: np.ndarray,
) -> RegularGridRectIndex | None:
    polygon_buffer = geometry_array.families.get(GeometryFamily.POLYGON)
    if polygon_buffer is None or len(geometry_array.families) != 1:
        return None
    if geometry_array.row_count == 0:
        return None
    if not np.all(geometry_array.validity):
        return None
    if np.any(polygon_buffer.empty_mask):
        return None

    expected_ring_offsets = np.arange(0, (geometry_array.row_count + 1) * 5, 5, dtype=np.int32)
    if polygon_buffer.ring_offsets is None or not np.array_equal(polygon_buffer.ring_offsets, expected_ring_offsets):
        return None
    if not np.array_equal(polygon_buffer.geometry_offsets, np.arange(geometry_array.row_count + 1, dtype=np.int32)):
        return None

    if np.isnan(bounds).any():
        return None
    widths = bounds[:, 2] - bounds[:, 0]
    heights = bounds[:, 3] - bounds[:, 1]
    if np.any(widths <= 0.0) or np.any(heights <= 0.0):
        return None
    cell_width = float(widths[0])
    cell_height = float(heights[0])
    tol = 1e-9 * max(abs(cell_width), abs(cell_height), 1.0)
    if not np.allclose(widths, cell_width, atol=tol, rtol=0.0):
        return None
    if not np.allclose(heights, cell_height, atol=tol, rtol=0.0):
        return None

    minx = float(bounds[:, 0].min())
    miny = float(bounds[:, 1].min())
    cols = int(np.rint((bounds[:, 0].max() - minx) / cell_width)) + 1
    rows = int(np.rint((bounds[:, 1].max() - miny) / cell_height)) + 1
    if cols <= 0 or rows <= 0:
        return None

    col_index = np.rint((bounds[:, 0] - minx) / cell_width).astype(np.int32, copy=False)
    row_index = np.rint((bounds[:, 1] - miny) / cell_height).astype(np.int32, copy=False)
    if np.any(col_index < 0) or np.any(col_index >= cols) or np.any(row_index < 0) or np.any(row_index >= rows):
        return None

    expected_index = row_index.astype(np.int64) * cols + col_index.astype(np.int64)
    actual_index = np.arange(geometry_array.row_count, dtype=np.int64)
    if not np.array_equal(expected_index, actual_index):
        return None

    ring_x = polygon_buffer.x.reshape(geometry_array.row_count, 5)
    ring_y = polygon_buffer.y.reshape(geometry_array.row_count, 5)
    x_is_min = np.isclose(ring_x, bounds[:, 0][:, None], atol=tol, rtol=0.0)
    x_is_max = np.isclose(ring_x, bounds[:, 2][:, None], atol=tol, rtol=0.0)
    y_is_min = np.isclose(ring_y, bounds[:, 1][:, None], atol=tol, rtol=0.0)
    y_is_max = np.isclose(ring_y, bounds[:, 3][:, None], atol=tol, rtol=0.0)
    if not np.all(x_is_min | x_is_max):
        return None
    if not np.all(y_is_min | y_is_max):
        return None
    if not np.all(np.isclose(ring_x[:, 0], ring_x[:, -1], atol=tol, rtol=0.0)):
        return None
    if not np.all(np.isclose(ring_y[:, 0], ring_y[:, -1], atol=tol, rtol=0.0)):
        return None
    if not np.all(x_is_min[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(x_is_max[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(y_is_min[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(y_is_max[:, :4].sum(axis=1) == 2):
        return None
    edge_same_x = np.isclose(ring_x[:, 1:], ring_x[:, :-1], atol=tol, rtol=0.0)
    edge_same_y = np.isclose(ring_y[:, 1:], ring_y[:, :-1], atol=tol, rtol=0.0)
    if not np.all(np.logical_xor(edge_same_x, edge_same_y)):
        return None

    return RegularGridRectIndex(
        origin_x=minx,
        origin_y=miny,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=rows,
        size=geometry_array.row_count,
    )


def _build_flat_spatial_index_gpu(
    geometry_array: OwnedGeometryArray,
    bounds: np.ndarray,
    *,
    keep_on_device: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Morton-sorted spatial index on GPU.

    Parameters
    ----------
    keep_on_device : bool
        When True, defer the D->H transfer of morton_keys and order.
        The returned arrays are CuPy device arrays instead of NumPy host
        arrays.  Callers that chain into further GPU work (e.g. GPU bounds
        pair generation) should set this to avoid a round-trip.
    """
    runtime = get_cuda_runtime()
    finite = bounds[~np.isnan(bounds).any(axis=1)]
    if finite.size == 0:
        return (
            np.full(geometry_array.row_count, np.iinfo(np.uint64).max, dtype=np.uint64),
            np.arange(geometry_array.row_count, dtype=np.int32),
        )

    geometry_array.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="build_flat_spatial_index selected GPU morton sort",
    )
    device_bounds = runtime.from_host(bounds)
    device_keys = None
    device_order = None
    sorted_result = None
    try:
        device_keys = runtime.allocate((geometry_array.row_count,), np.uint64)
        kernel = _indexing_kernels()["morton_keys_from_bounds"]
        ptr = runtime.pointer
        params = (
            (
                ptr(device_bounds),
                float(finite[:, 0].min()),
                float(finite[:, 1].min()),
                float(finite[:, 2].max()),
                float(finite[:, 3].max()),
                ptr(device_keys),
                geometry_array.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernel, geometry_array.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        device_order = runtime.from_host(np.arange(geometry_array.row_count, dtype=np.int32))
        sorted_result = sort_pairs(device_keys, device_order, synchronize=False)

        if keep_on_device:
            # Return CuPy device arrays — caller is responsible for eventual
            # D->H transfer or further GPU consumption.
            morton_keys = device_keys
            order = sorted_result.values
            return morton_keys, order

        # Standard path: transfer to host
        morton_keys = cp.asnumpy(device_keys).astype(np.uint64, copy=False)
        order = cp.asnumpy(sorted_result.values).astype(np.int32, copy=False)
        return morton_keys, order
    finally:
        runtime.free(device_bounds)
        if not keep_on_device:
            runtime.free(device_keys)
            runtime.free(device_order)
            if sorted_result is not None:
                runtime.free(sorted_result.keys)
                runtime.free(sorted_result.values)
        else:
            # Only free intermediates not returned to the caller
            runtime.free(device_order)
            if sorted_result is not None:
                runtime.free(sorted_result.keys)


def build_flat_spatial_index(
    geometry_array: OwnedGeometryArray,
    *,
    runtime_selection: RuntimeSelection | None = None,
) -> FlatSpatialIndex:
    selection = runtime_selection or _default_index_runtime_selection()
    geometry_array.record_runtime_selection(selection)
    bounds_dispatch = (
        ExecutionMode.GPU
        if selection.selected is ExecutionMode.GPU
        else ExecutionMode.CPU
    )
    bounds = compute_geometry_bounds(geometry_array, dispatch_mode=bounds_dispatch)
    regular_grid = _detect_regular_grid_rect_index(geometry_array, bounds)
    d_morton_keys = None
    d_order = None
    if regular_grid is None:
        use_gpu_sort = (
            selection.selected is ExecutionMode.GPU
            and has_gpu_runtime()
        )
        if use_gpu_sort:
            # Keep device arrays to avoid D→H transfer mid-pipeline
            # (ADR-0005).  Host fields are lazily populated on first
            # access via query_bounds() / CPU fallback paths.
            d_morton_keys_raw, d_order_raw = _build_flat_spatial_index_gpu(
                geometry_array, bounds, keep_on_device=True,
            )
            morton_keys = None
            order = None
            d_morton_keys = d_morton_keys_raw
            d_order = d_order_raw
        else:
            morton_keys = compute_morton_keys(geometry_array)
            order = np.argsort(morton_keys, kind="stable").astype(np.int32, copy=False)
    else:
        order = np.arange(geometry_array.row_count, dtype=np.int32)
        morton_keys = order.astype(np.uint64, copy=False)
    finite = bounds[~np.isnan(bounds).any(axis=1)]
    if finite.size == 0:
        total_bounds = (float("nan"),) * 4
    else:
        total_bounds = (
            float(finite[:, 0].min()),
            float(finite[:, 1].min()),
            float(finite[:, 2].max()),
            float(finite[:, 3].max()),
        )
    return FlatSpatialIndex(
        geometry_array=geometry_array,
        _host_order=order,
        _host_morton_keys=morton_keys,
        bounds=bounds,
        total_bounds=total_bounds,
        regular_grid=regular_grid,
        device_morton_keys=d_morton_keys,
        device_order=d_order,
    )


@dataclass(frozen=True)
class SegmentMBRTable:
    row_indices: np.ndarray
    segment_indices: np.ndarray
    bounds: np.ndarray

    @property
    def count(self) -> int:
        return int(self.row_indices.size)


def extract_segment_mbrs(geometry_array: OwnedGeometryArray) -> SegmentMBRTable:
    row_indices: list[int] = []
    segment_indices: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []

    for row_index, geometry in enumerate(geometry_array.to_shapely()):
        if geometry is None or geometry.is_empty:
            continue
        geom_type = geometry.geom_type
        if geom_type == "LineString":
            lines = [geometry]
        elif geom_type == "Polygon":
            lines = [geometry.exterior, *geometry.interiors]
        elif geom_type == "MultiLineString":
            lines = list(geometry.geoms)
        elif geom_type == "MultiPolygon":
            lines = []
            for polygon in geometry.geoms:
                lines.append(polygon.exterior)
                lines.extend(polygon.interiors)
        else:
            continue

        segment_counter = 0
        for line in lines:
            coords = list(line.coords)
            for start, end in zip(coords, coords[1:], strict=False):
                minx = min(float(start[0]), float(end[0]))
                miny = min(float(start[1]), float(end[1]))
                maxx = max(float(start[0]), float(end[0]))
                maxy = max(float(start[1]), float(end[1]))
                row_indices.append(row_index)
                segment_indices.append(segment_counter)
                boxes.append((minx, miny, maxx, maxy))
                segment_counter += 1

    if not boxes:
        return SegmentMBRTable(
            row_indices=np.asarray([], dtype=np.int32),
            segment_indices=np.asarray([], dtype=np.int32),
            bounds=np.empty((0, 4), dtype=np.float64),
        )
    return SegmentMBRTable(
        row_indices=np.asarray(row_indices, dtype=np.int32),
        segment_indices=np.asarray(segment_indices, dtype=np.int32),
        bounds=np.asarray(boxes, dtype=np.float64),
    )


@dataclass(frozen=True)
class SegmentCandidatePairs:
    left_rows: np.ndarray
    left_segments: np.ndarray
    right_rows: np.ndarray
    right_segments: np.ndarray
    pairs_examined: int

    @property
    def count(self) -> int:
        return int(self.left_rows.size)


def generate_segment_mbr_pairs(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
) -> SegmentCandidatePairs:
    left_segments = extract_segment_mbrs(left)
    right_segments = extract_segment_mbrs(right)

    left_rows_out: list[np.ndarray] = []
    left_segment_out: list[np.ndarray] = []
    right_rows_out: list[np.ndarray] = []
    right_segment_out: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_segments.count, tile_size):
        left_bounds = left_segments.bounds[left_start : left_start + tile_size]
        left_rows = left_segments.row_indices[left_start : left_start + tile_size]
        left_ids = left_segments.segment_indices[left_start : left_start + tile_size]
        for right_start in range(0, right_segments.count, tile_size):
            right_bounds = right_segments.bounds[right_start : right_start + tile_size]
            right_rows = right_segments.row_indices[right_start : right_start + tile_size]
            right_ids = right_segments.segment_indices[right_start : right_start + tile_size]
            pairs_examined += int(left_bounds.shape[0] * right_bounds.shape[0])
            intersects = (
                (left_bounds[:, None, 0] <= right_bounds[None, :, 2])
                & (left_bounds[:, None, 2] >= right_bounds[None, :, 0])
                & (left_bounds[:, None, 1] <= right_bounds[None, :, 3])
                & (left_bounds[:, None, 3] >= right_bounds[None, :, 1])
            )
            left_local, right_local = np.nonzero(intersects)
            if left_local.size == 0:
                continue
            left_rows_out.append(left_rows[left_local].astype(np.int32, copy=False))
            left_segment_out.append(left_ids[left_local].astype(np.int32, copy=False))
            right_rows_out.append(right_rows[right_local].astype(np.int32, copy=False))
            right_segment_out.append(right_ids[right_local].astype(np.int32, copy=False))

    if not left_rows_out:
        empty = np.asarray([], dtype=np.int32)
        return SegmentCandidatePairs(empty, empty, empty, empty, pairs_examined)
    return SegmentCandidatePairs(
        left_rows=np.concatenate(left_rows_out),
        left_segments=np.concatenate(left_segment_out),
        right_rows=np.concatenate(right_rows_out),
        right_segments=np.concatenate(right_segment_out),
        pairs_examined=pairs_examined,
    )


@dataclass(frozen=True)
class SegmentFilterBenchmark:
    rows_left: int
    rows_right: int
    naive_segment_pairs: int
    filtered_segment_pairs: int
    elapsed_seconds: float


def benchmark_segment_filter(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
) -> SegmentFilterBenchmark:
    left_segments = extract_segment_mbrs(left)
    right_segments = extract_segment_mbrs(right)
    started = perf_counter()
    filtered = generate_segment_mbr_pairs(left, right, tile_size=tile_size)
    elapsed = perf_counter() - started
    return SegmentFilterBenchmark(
        rows_left=left.row_count,
        rows_right=right.row_count,
        naive_segment_pairs=int(left_segments.count * right_segments.count),
        filtered_segment_pairs=filtered.count,
        elapsed_seconds=elapsed,
    )
