from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs

request_warmup(["radix_sort_i32_i32", "radix_sort_u64_i32", "exclusive_scan_i32"])


try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import (  # noqa: E402
    compute_geometry_bounds,
    compute_morton_keys,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.fallbacks import record_fallback_event  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import Residency, TransferTrigger  # noqa: E402

_GPU_BOUNDS_PAIRS_THRESHOLD = 2_048

logger = logging.getLogger(__name__)


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

/* Sort-and-sweep MBR overlap test.
 *
 * Operates on a concatenated array of geometries, sorted by minx.
 * Each thread handles one element and sweeps forward through the sorted
 * array.  The sweep terminates when sorted_minx[j] > current_maxx,
 * pruning the search space from O(n) to O(k) where k is the average
 * x-overlap count.
 *
 * For same_input mode: thread i sweeps forward and emits upper-triangle
 * pairs (based on original row index) to avoid duplicates.
 *
 * For two-input mode: thread i sweeps forward and emits a pair whenever
 * the two elements come from different sides (left vs right).  Because
 * the sweep is directional (i < j in sorted order), each cross-side
 * pair is discovered exactly once -- by whichever element appears first.
 * The emitted pair is always (left_orig, right_orig) regardless of which
 * side the sweeping element belongs to.
 *
 * Two passes: pass 0 counts valid pairs per sorted element;
 *             pass 1 writes pairs using a prefix-sum offset array.
 */
extern "C" __global__ void sweep_sorted_mbr_overlap(
    const double* __restrict__ sorted_bounds,
    const int*    __restrict__ sorted_orig,
    const int*    __restrict__ sorted_side,
    int           total_count,
    int*          out_left,
    int*          out_right,
    int*          counts,
    int           pass_number,
    int           same_input,
    int           include_self
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_count) return;

  const int ibase = i * 4;
  const double ix0 = sorted_bounds[ibase + 0];
  const double iy0 = sorted_bounds[ibase + 1];
  const double ix1 = sorted_bounds[ibase + 2];
  const double iy1 = sorted_bounds[ibase + 3];
  const int i_orig = sorted_orig[i];
  const int i_side = sorted_side[i];

  int pair_count = 0;
  int write_offset = 0;
  if (pass_number == 1) write_offset = counts[i];

  /* Sweep forward: elements are sorted by minx, so once sorted_minx[j] > ix1
   * no further element can overlap in x with element i. */
  for (int j = i + 1; j < total_count; j++) {
    const int jbase = j * 4;
    const double jx0 = sorted_bounds[jbase + 0];
    /* Early exit: sorted by minx, so jx0 > ix1 means no more x-overlap */
    if (jx0 > ix1) break;

    const double jy0 = sorted_bounds[jbase + 1];
    const double jx1 = sorted_bounds[jbase + 2];
    const double jy1 = sorted_bounds[jbase + 3];

    /* Full y-axis overlap test (x-overlap guaranteed by sweep condition) */
    if (iy0 > jy1 || iy1 < jy0) continue;

    const int j_orig = sorted_orig[j];
    const int j_side = sorted_side[j];

    if (same_input) {
      /* Same-input mode: each pair (i,j) where i < j in sorted order is
       * discovered exactly once by the thread at position i.  Emit as
       * (min_orig, max_orig) for upper-triangle.  Skip self-pairs
       * (same original row index) unless include_self is set. */
      if (i_orig == j_orig && !include_self) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Emit as (smaller_orig, larger_orig) */
        if (i_orig <= j_orig) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    } else {
      /* Two-input mode: emit only cross-side pairs */
      if (i_side == j_side) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Always emit as (left_orig, right_orig) */
        if (i_side == 0) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    }
  }

  if (pass_number == 0) counts[i] = pair_count;
}
"""


@dataclass(frozen=True)
class CandidatePairs:
    """MBR candidate pair result with optional device-resident arrays.

    When produced by the GPU path, ``_device_left_indices`` and
    ``_device_right_indices`` hold CuPy device arrays.  The public
    ``left_indices`` and ``right_indices`` properties lazily materialise
    host (NumPy) arrays on first access, following the same pattern as
    :class:`FlatSpatialIndex`.
    """

    _host_left_indices: object  # np.ndarray or None (lazy from device)
    _host_right_indices: object  # np.ndarray or None (lazy from device)
    left_bounds: np.ndarray
    right_bounds: np.ndarray
    pairs_examined: int
    tile_size: int
    same_input: bool
    _device_left_indices: object = None  # CuPy device array or None
    _device_right_indices: object = None  # CuPy device array or None

    @property
    def left_indices(self) -> np.ndarray:
        """Lazily materialise host left_indices from device (ADR-0005)."""
        if self._host_left_indices is not None:
            return self._host_left_indices
        host = cp.asnumpy(self._device_left_indices).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_left_indices", host)
        return host

    @property
    def right_indices(self) -> np.ndarray:
        """Lazily materialise host right_indices from device (ADR-0005)."""
        if self._host_right_indices is not None:
            return self._host_right_indices
        host = cp.asnumpy(self._device_right_indices).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_right_indices", host)
        return host

    @property
    def device_left_indices(self):
        """CuPy device array of left indices, or None if CPU-produced."""
        return self._device_left_indices

    @property
    def device_right_indices(self):
        """CuPy device array of right indices, or None if CPU-produced."""
        return self._device_right_indices

    @property
    def count(self) -> int:
        if self._host_left_indices is not None:
            return int(self._host_left_indices.size)
        if self._device_left_indices is not None:
            return int(self._device_left_indices.size)
        return 0


def _valid_row_mask(bounds: np.ndarray) -> np.ndarray:
    return ~np.isnan(bounds).any(axis=1)


def _generate_bounds_pairs_gpu(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    *,
    same_input: bool,
    include_self: bool,
) -> tuple[object, object, int]:
    """GPU sort-and-sweep MBR overlap pair generation.

    Returns ``(d_left_indices, d_right_indices, pairs_examined)`` where the
    index arrays are CuPy device arrays (zero-copy -- no D->H transfer).
    Returns numpy arrays for the empty-result edge case.
    """
    runtime = get_cuda_runtime()
    kernels = _indexing_kernels()
    sweep_kernel = kernels["sweep_sorted_mbr_overlap"]
    ptr = runtime.pointer

    # --- Filter NaN bounds on device (Tier 2: CuPy element-wise) ---
    d_left_bounds_full = cp.asarray(left_bounds)
    d_left_valid_mask = ~cp.isnan(d_left_bounds_full).any(axis=1)
    d_left_valid = cp.flatnonzero(d_left_valid_mask).astype(cp.int32, copy=False)
    left_count = int(d_left_valid.size)

    if same_input:
        d_right_bounds_full = d_left_bounds_full
        d_right_valid = d_left_valid
        right_count = left_count
    else:
        d_right_bounds_full = cp.asarray(right_bounds)
        d_right_valid_mask = ~cp.isnan(d_right_bounds_full).any(axis=1)
        d_right_valid = cp.flatnonzero(d_right_valid_mask).astype(cp.int32, copy=False)
        right_count = int(d_right_valid.size)

    if left_count == 0 or right_count == 0:
        empty = np.asarray([], dtype=np.int32)
        return empty, empty, 0

    # --- Extract valid bounds subsets on device ---
    d_left_valid_bounds = d_left_bounds_full[d_left_valid]  # [left_count, 4]
    d_left_orig = d_left_valid  # original row indices (not mutated, no copy needed)

    if same_input:
        d_right_valid_bounds = d_left_valid_bounds
        d_right_orig = d_left_orig
    else:
        d_right_valid_bounds = d_right_bounds_full[d_right_valid]
        d_right_orig = d_right_valid  # not mutated, no copy needed

    # --- Build concatenated sorted array for sweep ---
    if same_input:
        # Same-input mode: single array, all elements are both left and right
        total_count = left_count
        # Sort by minx (column 0 of bounds) on device
        sort_order = cp.argsort(d_left_valid_bounds[:, 0]).astype(cp.int32, copy=False)
        d_sorted_bounds = d_left_valid_bounds[sort_order]
        d_sorted_bounds_flat = d_sorted_bounds.ravel().astype(cp.float64, copy=False)
        d_sorted_orig = d_left_orig[sort_order]
        d_sorted_side = cp.zeros(total_count, dtype=cp.int32)
    else:
        # Two-input mode: concatenate left (side=0) + right (side=1)
        total_count = left_count + right_count
        d_all_bounds = cp.concatenate([d_left_valid_bounds, d_right_valid_bounds], axis=0)
        d_all_orig = cp.concatenate([d_left_orig, d_right_orig])
        d_all_side = cp.concatenate([
            cp.zeros(left_count, dtype=cp.int32),
            cp.ones(right_count, dtype=cp.int32),
        ])
        # Sort by minx (column 0) on device
        sort_order = cp.argsort(d_all_bounds[:, 0]).astype(cp.int32, copy=False)
        d_sorted_bounds = d_all_bounds[sort_order]
        d_sorted_bounds_flat = d_sorted_bounds.ravel().astype(cp.float64, copy=False)
        d_sorted_orig = d_all_orig[sort_order]
        d_sorted_side = d_all_side[sort_order]

    # --- Pass 0: count pairs per sorted element ---
    d_counts = runtime.allocate((total_count,), np.int32, zero=True)
    _param_types = (
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
    )
    count_params = (
        (
            ptr(d_sorted_bounds_flat), ptr(d_sorted_orig), ptr(d_sorted_side),
            total_count,
            0,  # out_left (unused in pass 0)
            0,  # out_right (unused in pass 0)
            ptr(d_counts),
            0,  # pass_number = 0
            int(same_input),
            int(include_self),
        ),
        _param_types,
    )
    grid, block = runtime.launch_config(sweep_kernel, total_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=count_params)

    # Prefix sum for scatter offsets (CCCL exclusive_sum, ADR-0033)
    cp_counts = cp.asarray(d_counts)
    d_offsets = exclusive_sum(cp_counts, synchronize=False)
    total_pairs = count_scatter_total(runtime, cp_counts, d_offsets)

    if total_pairs == 0:
        empty = np.asarray([], dtype=np.int32)
        return empty, empty, left_count * right_count

    if total_pairs > 2_147_483_647:
        raise OverflowError(
            f"sweep_sorted_mbr_overlap: {total_pairs} candidate pairs exceed "
            f"int32 offset capacity (2^31-1). Consider spatial partitioning."
        )

    # Copy offsets into counts buffer for the scatter kernel to read
    cp.copyto(cp_counts, d_offsets)

    # Allocate output arrays on device
    d_out_left = runtime.allocate((total_pairs,), np.int32)
    d_out_right = runtime.allocate((total_pairs,), np.int32)

    # --- Pass 1: scatter pairs ---
    scatter_params = (
        (
            ptr(d_sorted_bounds_flat), ptr(d_sorted_orig), ptr(d_sorted_side),
            total_count,
            ptr(d_out_left), ptr(d_out_right),
            ptr(d_counts),
            1,  # pass_number = 1
            int(same_input),
            int(include_self),
        ),
        _param_types,
    )
    grid, block = runtime.launch_config(sweep_kernel, total_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=scatter_params)
    # No sync needed -- CuPy arrays stay on device.

    pairs_examined = left_count * right_count
    return cp.asarray(d_out_left), cp.asarray(d_out_right), pairs_examined


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
        d_left, d_right, pairs_examined = _generate_bounds_pairs_gpu(
            left_bounds,
            right_bounds,
            same_input=same_input,
            include_self=include_self,
        )
        # Determine if GPU returned device arrays or host arrays (empty case)
        is_device = cp is not None and isinstance(d_left, cp.ndarray)
        return CandidatePairs(
            _host_left_indices=None if is_device else d_left,
            _host_right_indices=None if is_device else d_right,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            pairs_examined=pairs_examined,
            tile_size=tile_size,
            same_input=same_input,
            _device_left_indices=d_left if is_device else None,
            _device_right_indices=d_right if is_device else None,
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
        _host_left_indices=left_indices,
        _host_right_indices=right_indices,
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
    regular_grid: RegularGridRectIndex | None = None
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
        ordered = self.geometry_array
        result = generate_bounds_pairs(other, ordered, tile_size=tile_size)
        if result.count == 0:
            return result
        # When both order and result are device-resident, remap on device
        # to avoid a D->H round-trip (ADR-0005).
        if (
            cp is not None
            and self.device_order is not None
            and result.device_right_indices is not None
        ):
            d_order = cp.asarray(self.device_order)
            d_mapped_right = d_order[result.device_right_indices]
            return CandidatePairs(
                _host_left_indices=None,
                _host_right_indices=None,
                left_bounds=result.left_bounds,
                right_bounds=result.right_bounds,
                pairs_examined=result.pairs_examined,
                tile_size=result.tile_size,
                same_input=result.same_input,
                _device_left_indices=result.device_left_indices,
                _device_right_indices=d_mapped_right.astype(cp.int32, copy=False),
            )
        # CPU fallback: materialise to host and remap
        order = self.order
        mapped_right = order[result.right_indices]
        return CandidatePairs(
            _host_left_indices=result.left_indices,
            _host_right_indices=mapped_right.astype(np.int32, copy=False),
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


_INDEXING_KERNEL_NAMES = ("morton_keys_from_bounds", "sweep_sorted_mbr_overlap")

# ---------------------------------------------------------------------------
# Segment MBR extraction kernels (Tier 1: geometry-specific inner loops)
# ---------------------------------------------------------------------------
# One kernel per geometry family.  Two-pass count/scatter:
#   Pass 0 (pass_number==0): each thread counts segments for one geometry row.
#   Pass 1 (pass_number==1): each thread scatters (row, seg_idx, minx,miny,maxx,maxy).
#
# The kernels receive coordinate buffers in SoA layout (separate x[], y[])
# plus the offset hierarchy for the family.
# ---------------------------------------------------------------------------

_SEGMENT_MBR_KERNEL_SOURCE = """
/* ---- LineString segment MBR ---- */
extern "C" __global__ void segment_mbr_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int c0 = geom_offsets[g];
    const int c1 = geom_offsets[g + 1];
    const int n_seg = c1 - c0 - 1;
    if (n_seg <= 0) {
      if (pass_number == 0) counts[g] = 0;
      continue;
    }
    if (pass_number == 0) {
      counts[g] = n_seg;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      for (int s = 0; s < n_seg; s++) {
        const int ci = c0 + s;
        const double x0 = x[ci], y0 = y[ci];
        const double x1 = x[ci + 1], y1 = y[ci + 1];
        const int idx = base + s;
        row_out[idx] = grow;
        seg_out[idx] = s;
        bounds_out[idx * 4 + 0] = fmin(x0, x1);
        bounds_out[idx * 4 + 1] = fmin(y0, y1);
        bounds_out[idx * 4 + 2] = fmax(x0, x1);
        bounds_out[idx * 4 + 3] = fmax(y0, y1);
      }
    }
  }
}

/* ---- Polygon segment MBR ---- */
extern "C" __global__ void segment_mbr_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int r0 = geom_offsets[g];
    const int r1 = geom_offsets[g + 1];
    int total = 0;
    for (int r = r0; r < r1; r++) {
      const int rc0 = ring_offsets[r];
      const int rc1 = ring_offsets[r + 1];
      total += rc1 - rc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        const int n_seg = rc1 - rc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = rc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiLineString segment MBR ---- */
extern "C" __global__ void segment_mbr_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int pc0 = part_offsets[p];
      const int pc1 = part_offsets[p + 1];
      total += pc1 - pc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int pc0 = part_offsets[p];
        const int pc1 = part_offsets[p + 1];
        const int n_seg = pc1 - pc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = pc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiPolygon segment MBR ---- */
extern "C" __global__ void segment_mbr_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int r0 = part_offsets[p];
      const int r1 = part_offsets[p + 1];
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        total += rc1 - rc0 - 1;
      }
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int r0 = part_offsets[p];
        const int r1 = part_offsets[p + 1];
        for (int r = r0; r < r1; r++) {
          const int rc0 = ring_offsets[r];
          const int rc1 = ring_offsets[r + 1];
          const int n_seg = rc1 - rc0 - 1;
          for (int s = 0; s < n_seg; s++) {
            const int ci = rc0 + s;
            const double x0 = x[ci], y0 = y[ci];
            const double x1 = x[ci + 1], y1 = y[ci + 1];
            const int idx = base + seg;
            row_out[idx] = grow;
            seg_out[idx] = seg;
            bounds_out[idx * 4 + 0] = fmin(x0, x1);
            bounds_out[idx * 4 + 1] = fmin(y0, y1);
            bounds_out[idx * 4 + 2] = fmax(x0, x1);
            bounds_out[idx * 4 + 3] = fmax(y0, y1);
            seg++;
          }
        }
      }
    }
  }
}
"""

_SEGMENT_MBR_KERNEL_NAMES = (
    "segment_mbr_linestring",
    "segment_mbr_polygon",
    "segment_mbr_multilinestring",
    "segment_mbr_multipolygon",
)

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("indexing", _INDEXING_KERNEL_SOURCE, _INDEXING_KERNEL_NAMES),
    ("segment-mbr", _SEGMENT_MBR_KERNEL_SOURCE, _SEGMENT_MBR_KERNEL_NAMES),
])


def _indexing_kernels():
    return compile_kernel_group("indexing", _INDEXING_KERNEL_SOURCE, _INDEXING_KERNEL_NAMES)


def _segment_mbr_kernels():
    return compile_kernel_group("segment-mbr", _SEGMENT_MBR_KERNEL_SOURCE, _SEGMENT_MBR_KERNEL_NAMES)


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

    # Structural checks above use only offsets/bounds (available on host
    # even for device-resident OGAs).  The rectangle vertex verification
    # below needs coordinate data.  Lazily materialise host state here --
    # _ensure_host_state only transfers x/y since offsets are already
    # populated.  At 10K polygons this is ~800KB.
    geometry_array._ensure_host_state()
    polygon_buffer = geometry_array.families[GeometryFamily.POLYGON]
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
    # Use GPU for bounds when geometry is device-resident to avoid pulling
    # the full coordinate arrays to host just for min/max.  Structural
    # metadata (offsets, masks) is host-available even on device-resident
    # OGAs (populated at GPU read time), so regular grid detection and
    # morton key computation still work.
    bounds_dispatch = (
        ExecutionMode.GPU
        if (
            selection.selected is ExecutionMode.GPU
            or (geometry_array.residency is Residency.DEVICE and has_gpu_runtime())
        )
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
    """Segment MBR table with optional device-resident arrays.

    When produced by the GPU path, arrays are CuPy device arrays and
    ``residency`` is ``Residency.DEVICE``.  The public properties
    ``row_indices``, ``segment_indices``, and ``bounds`` return the
    underlying arrays as-is (device or host).  Use ``to_host()`` to get
    a copy with NumPy arrays on the host side.
    """

    row_indices: object  # np.ndarray or CuPy device array
    segment_indices: object  # np.ndarray or CuPy device array
    bounds: object  # np.ndarray (N,4) or CuPy device array (N,4)
    residency: Residency = Residency.HOST

    @property
    def count(self) -> int:
        if self.row_indices is None:
            return 0
        return int(self.row_indices.shape[0]) if hasattr(self.row_indices, "shape") else int(self.row_indices.size)

    def to_host(self) -> SegmentMBRTable:
        """Return a host-resident copy (NumPy arrays).

        If already host-resident, returns self.
        """
        if self.residency is Residency.HOST:
            return self
        return SegmentMBRTable(
            row_indices=cp.asnumpy(self.row_indices),
            segment_indices=cp.asnumpy(self.segment_indices),
            bounds=cp.asnumpy(self.bounds),
            residency=Residency.HOST,
        )


# ---------------------------------------------------------------------------
# Kernel-family dispatch table for segment MBR extraction
# ---------------------------------------------------------------------------

_SEGMENT_FAMILY_KERNEL_MAP = {
    GeometryFamily.LINESTRING: "segment_mbr_linestring",
    GeometryFamily.POLYGON: "segment_mbr_polygon",
    GeometryFamily.MULTILINESTRING: "segment_mbr_multilinestring",
    GeometryFamily.MULTIPOLYGON: "segment_mbr_multipolygon",
}

# Families that produce segments (Points/MultiPoints do not have segments)
_SEGMENT_FAMILIES = frozenset(_SEGMENT_FAMILY_KERNEL_MAP.keys())


def _extract_segment_mbrs_cpu(geometry_array: OwnedGeometryArray) -> SegmentMBRTable:
    """CPU fallback: extract segment MBRs via Shapely (triple-nested loop)."""
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
            residency=Residency.HOST,
        )
    return SegmentMBRTable(
        row_indices=np.asarray(row_indices, dtype=np.int32),
        segment_indices=np.asarray(segment_indices, dtype=np.int32),
        bounds=np.asarray(boxes, dtype=np.float64),
        residency=Residency.HOST,
    )


def _launch_segment_mbr_family(
    runtime,
    kernels: dict,
    family: GeometryFamily,
    d_buf,
    d_global_rows,
    geom_count: int,
) -> tuple[object, object, object] | None:
    """Run the two-pass count/scatter for one geometry family on GPU.

    Returns ``(d_row_out, d_seg_out, d_bounds_out)`` as CuPy arrays, or
    ``None`` if no segments were produced.
    """
    kernel_name = _SEGMENT_FAMILY_KERNEL_MAP[family]
    kernel = kernels[kernel_name]
    ptr = runtime.pointer

    # --- Build parameter list based on family ---
    # All families share: x, y, geom_offsets, global_row_indices, geom_count,
    #                      counts, row_out, seg_out, bounds_out, pass_number
    # Polygon adds: ring_offsets
    # MultiLineString adds: part_offsets
    # MultiPolygon adds: part_offsets, ring_offsets

    d_x = cp.asarray(d_buf.x)
    d_y = cp.asarray(d_buf.y)
    d_geom_offsets = cp.asarray(d_buf.geometry_offsets)

    # Pass 0: count segments per geometry
    d_counts = cp.zeros(geom_count, dtype=cp.int32)

    if family == GeometryFamily.LINESTRING:
        param_values_base = (
            ptr(d_x), ptr(d_y), ptr(d_geom_offsets),
            ptr(d_global_rows), geom_count,
            ptr(d_counts),
        )
        param_types_base = (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        )
    elif family == GeometryFamily.POLYGON:
        d_ring_offsets = cp.asarray(d_buf.ring_offsets)
        param_values_base = (
            ptr(d_x), ptr(d_y), ptr(d_geom_offsets),
            ptr(d_ring_offsets), ptr(d_global_rows), geom_count,
            ptr(d_counts),
        )
        param_types_base = (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        )
    elif family == GeometryFamily.MULTILINESTRING:
        d_part_offsets = cp.asarray(d_buf.part_offsets)
        param_values_base = (
            ptr(d_x), ptr(d_y), ptr(d_geom_offsets),
            ptr(d_part_offsets), ptr(d_global_rows), geom_count,
            ptr(d_counts),
        )
        param_types_base = (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        )
    elif family == GeometryFamily.MULTIPOLYGON:
        d_part_offsets = cp.asarray(d_buf.part_offsets)
        d_ring_offsets = cp.asarray(d_buf.ring_offsets)
        param_values_base = (
            ptr(d_x), ptr(d_y), ptr(d_geom_offsets),
            ptr(d_part_offsets), ptr(d_ring_offsets),
            ptr(d_global_rows), geom_count,
            ptr(d_counts),
        )
        param_types_base = (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        )
    else:
        return None

    # Count pass: row_out, seg_out, bounds_out are unused (null pointers)
    count_params = (
        (*param_values_base, 0, 0, 0, 0),  # row_out, seg_out, bounds_out, pass=0
        (*param_types_base, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, geom_count)
    runtime.launch(kernel, grid=grid, block=block, params=count_params)

    # Prefix sum for scatter offsets (CCCL, ADR-0033 Tier 3a)
    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total = count_scatter_total(runtime, d_counts, d_offsets)

    if total == 0:
        return None

    # Allocate output arrays on device
    d_row_out = cp.empty(total, dtype=cp.int32)
    d_seg_out = cp.empty(total, dtype=cp.int32)
    d_bounds_out = cp.empty(total * 4, dtype=cp.float64)

    # Copy offsets into counts buffer for scatter pass to read
    cp.copyto(d_counts, d_offsets)

    # Scatter pass
    scatter_params = (
        (*param_values_base, ptr(d_row_out), ptr(d_seg_out), ptr(d_bounds_out), 1),
        (*param_types_base, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, geom_count)
    runtime.launch(kernel, grid=grid, block=block, params=scatter_params)

    # Reshape bounds to (N, 4) -- no sync needed, stays on device
    d_bounds_out = d_bounds_out.reshape(total, 4)
    return d_row_out, d_seg_out, d_bounds_out


def _extract_segment_mbrs_gpu(geometry_array: OwnedGeometryArray) -> SegmentMBRTable:
    """GPU path: extract segment MBRs via NVRTC kernels.

    Returns device-resident SegmentMBRTable with CuPy arrays.
    No D->H transfer -- zero-copy end-to-end.
    """
    runtime = get_cuda_runtime()
    kernels = _segment_mbr_kernels()
    d_state = geometry_array._ensure_device_state()

    # Hoist tag array to device once -- used for all family lookups
    d_tags = cp.asarray(d_state.tags)

    all_rows = []
    all_segs = []
    all_bounds = []

    for family in _SEGMENT_FAMILIES:
        if family not in d_state.families:
            continue
        d_buf = d_state.families[family]
        geom_count = int(d_buf.geometry_offsets.shape[0]) - 1
        if geom_count == 0:
            continue

        # Global row indices for this family
        family_tag = FAMILY_TAGS[family]
        d_global_rows = cp.flatnonzero(d_tags == family_tag).astype(cp.int32, copy=False)

        result = _launch_segment_mbr_family(
            runtime, kernels, family, d_buf, d_global_rows, geom_count,
        )
        if result is not None:
            d_row_out, d_seg_out, d_bounds_out = result
            all_rows.append(d_row_out)
            all_segs.append(d_seg_out)
            all_bounds.append(d_bounds_out)

    if not all_rows:
        return SegmentMBRTable(
            row_indices=cp.empty(0, dtype=cp.int32),
            segment_indices=cp.empty(0, dtype=cp.int32),
            bounds=cp.empty((0, 4), dtype=cp.float64),
            residency=Residency.DEVICE,
        )

    # Concatenate results from all families (Tier 2: CuPy)
    d_all_rows = cp.concatenate(all_rows) if len(all_rows) > 1 else all_rows[0]
    d_all_segs = cp.concatenate(all_segs) if len(all_segs) > 1 else all_segs[0]
    d_all_bounds = cp.concatenate(all_bounds) if len(all_bounds) > 1 else all_bounds[0]

    return SegmentMBRTable(
        row_indices=d_all_rows,
        segment_indices=d_all_segs,
        bounds=d_all_bounds,
        residency=Residency.DEVICE,
    )


def extract_segment_mbrs(geometry_array: OwnedGeometryArray) -> SegmentMBRTable:
    """Extract per-segment MBRs from all line/polygon geometries.

    Dispatches to GPU when available, falling back to CPU otherwise.
    The GPU path returns device-resident CuPy arrays (no D->H transfer).
    """
    use_gpu = has_gpu_runtime() and cp is not None
    if use_gpu:
        try:
            return _extract_segment_mbrs_gpu(geometry_array)
        except Exception:
            logger.debug("GPU segment MBR extraction failed, falling back to CPU", exc_info=True)
            record_fallback_event(
                surface="extract_segment_mbrs",
                reason="GPU kernel failed, falling back to CPU",
            )
    return _extract_segment_mbrs_cpu(geometry_array)


@dataclass(frozen=True)
class SegmentCandidatePairs:
    """Segment candidate pairs with lazy device-to-host materialization.

    When produced by the GPU path, ``_device_*`` fields hold CuPy device
    arrays and ``_host_*`` fields are ``None``.  The public properties
    lazily call ``cp.asnumpy()`` on first host access, following the
    ``CandidatePairs`` pattern (ADR-0005).
    """

    _host_left_rows: object  # np.ndarray or None (lazy from device)
    _host_left_segments: object  # np.ndarray or None (lazy from device)
    _host_right_rows: object  # np.ndarray or None (lazy from device)
    _host_right_segments: object  # np.ndarray or None (lazy from device)
    pairs_examined: int
    _device_left_rows: object = None  # CuPy device array or None
    _device_left_segments: object = None  # CuPy device array or None
    _device_right_rows: object = None  # CuPy device array or None
    _device_right_segments: object = None  # CuPy device array or None

    @property
    def left_rows(self) -> np.ndarray:
        """Lazily materialise host left_rows from device (ADR-0005)."""
        if self._host_left_rows is not None:
            return self._host_left_rows
        host = cp.asnumpy(self._device_left_rows).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_left_rows", host)
        return host

    @property
    def left_segments(self) -> np.ndarray:
        """Lazily materialise host left_segments from device (ADR-0005)."""
        if self._host_left_segments is not None:
            return self._host_left_segments
        host = cp.asnumpy(self._device_left_segments).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_left_segments", host)
        return host

    @property
    def right_rows(self) -> np.ndarray:
        """Lazily materialise host right_rows from device (ADR-0005)."""
        if self._host_right_rows is not None:
            return self._host_right_rows
        host = cp.asnumpy(self._device_right_rows).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_right_rows", host)
        return host

    @property
    def right_segments(self) -> np.ndarray:
        """Lazily materialise host right_segments from device (ADR-0005)."""
        if self._host_right_segments is not None:
            return self._host_right_segments
        host = cp.asnumpy(self._device_right_segments).astype(np.int32, copy=False)
        object.__setattr__(self, "_host_right_segments", host)
        return host

    @property
    def device_left_rows(self):
        """CuPy device array of left row indices, or None if CPU-produced."""
        return self._device_left_rows

    @property
    def device_left_segments(self):
        """CuPy device array of left segment indices, or None if CPU-produced."""
        return self._device_left_segments

    @property
    def device_right_rows(self):
        """CuPy device array of right row indices, or None if CPU-produced."""
        return self._device_right_rows

    @property
    def device_right_segments(self):
        """CuPy device array of right segment indices, or None if CPU-produced."""
        return self._device_right_segments

    @property
    def count(self) -> int:
        if self._host_left_rows is not None:
            return int(self._host_left_rows.size)
        if self._device_left_rows is not None:
            return int(self._device_left_rows.size)
        return 0


def _generate_segment_mbr_pairs_gpu(
    left_segments: SegmentMBRTable,
    right_segments: SegmentMBRTable,
) -> SegmentCandidatePairs:
    """GPU segment pair generation using sweep-sort overlap kernel.

    Operates directly on device-resident segment bounds -- no D->H->D
    ping-pong.  Uses the same ``sweep_sorted_mbr_overlap`` kernel as
    geometry-level pair generation, but on segment-level MBRs.

    The sweep kernel emits index pairs into the concatenated segment
    array.  Device-side gather then maps these back to (row, segment).
    """
    runtime = get_cuda_runtime()
    kernels = _indexing_kernels()
    sweep_kernel = kernels["sweep_sorted_mbr_overlap"]
    ptr = runtime.pointer

    d_left_bounds_full = cp.asarray(left_segments.bounds)    # (N, 4) device
    d_right_bounds_full = cp.asarray(right_segments.bounds)  # (M, 4) device
    d_left_rows_full = cp.asarray(left_segments.row_indices)
    d_left_segs_full = cp.asarray(left_segments.segment_indices)
    d_right_rows_full = cp.asarray(right_segments.row_indices)
    d_right_segs_full = cp.asarray(right_segments.segment_indices)

    # --- Filter NaN bounds on device (Tier 2: CuPy element-wise) ---
    d_left_valid = cp.flatnonzero(~cp.isnan(d_left_bounds_full).any(axis=1))
    d_right_valid = cp.flatnonzero(~cp.isnan(d_right_bounds_full).any(axis=1))
    left_count = int(d_left_valid.size)
    right_count = int(d_right_valid.size)
    pairs_examined = left_count * right_count

    if left_count == 0 or right_count == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return SegmentCandidatePairs(
            _host_left_rows=None,
            _host_left_segments=None,
            _host_right_rows=None,
            _host_right_segments=None,
            pairs_examined=pairs_examined,
            _device_left_rows=empty,
            _device_left_segments=empty,
            _device_right_rows=empty,
            _device_right_segments=empty,
        )

    # --- Extract valid subsets on device ---
    d_left_bounds = d_left_bounds_full[d_left_valid]
    d_right_bounds = d_right_bounds_full[d_right_valid]
    d_left_rows = d_left_rows_full[d_left_valid]
    d_left_segs = d_left_segs_full[d_left_valid]
    d_right_rows = d_right_rows_full[d_right_valid]
    d_right_segs = d_right_segs_full[d_right_valid]

    # --- Build concatenated sorted array on device (no host round-trip) ---
    total_count = left_count + right_count
    d_all_bounds = cp.concatenate([d_left_bounds, d_right_bounds], axis=0)
    d_all_orig = cp.concatenate([
        cp.arange(left_count, dtype=cp.int32),
        cp.arange(right_count, dtype=cp.int32),
    ])
    d_all_side = cp.concatenate([
        cp.zeros(left_count, dtype=cp.int32),
        cp.ones(right_count, dtype=cp.int32),
    ])

    # Sort by minx (column 0) on device
    sort_order = cp.argsort(d_all_bounds[:, 0]).astype(cp.int32, copy=False)
    d_sorted_bounds = d_all_bounds[sort_order]
    d_sorted_bounds_flat = d_sorted_bounds.ravel().astype(cp.float64, copy=False)
    d_sorted_orig = d_all_orig[sort_order]
    d_sorted_side = d_all_side[sort_order]

    # --- Pass 0: count pairs per sorted element ---
    d_counts = runtime.allocate((total_count,), np.int32, zero=True)
    _param_types = (
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_I32,
    )
    count_params = (
        (
            ptr(d_sorted_bounds_flat), ptr(d_sorted_orig), ptr(d_sorted_side),
            total_count,
            0, 0,  # out_left, out_right (unused in pass 0)
            ptr(d_counts),
            0,  # pass_number = 0
            0,  # same_input = False
            0,  # include_self = False
        ),
        _param_types,
    )
    grid, block = runtime.launch_config(sweep_kernel, total_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=count_params)

    # Prefix sum for scatter offsets (CCCL, ADR-0033 Tier 3a)
    cp_counts = cp.asarray(d_counts)
    d_offsets = exclusive_sum(cp_counts, synchronize=False)
    total_pairs = count_scatter_total(runtime, cp_counts, d_offsets)

    if total_pairs > 2_147_483_647:
        raise OverflowError(
            f"sweep_segment_mbr_overlap: {total_pairs} segment pairs exceed "
            f"int32 offset capacity (2^31-1). Consider spatial partitioning."
        )

    if total_pairs == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return SegmentCandidatePairs(
            _host_left_rows=None,
            _host_left_segments=None,
            _host_right_rows=None,
            _host_right_segments=None,
            pairs_examined=pairs_examined,
            _device_left_rows=empty,
            _device_left_segments=empty,
            _device_right_rows=empty,
            _device_right_segments=empty,
        )

    # Copy offsets into counts buffer for scatter pass
    cp.copyto(cp_counts, d_offsets)

    # Allocate output index arrays
    d_out_left = runtime.allocate((total_pairs,), np.int32)
    d_out_right = runtime.allocate((total_pairs,), np.int32)

    # --- Pass 1: scatter pairs ---
    scatter_params = (
        (
            ptr(d_sorted_bounds_flat), ptr(d_sorted_orig), ptr(d_sorted_side),
            total_count,
            ptr(d_out_left), ptr(d_out_right),
            ptr(d_counts),
            1,  # pass_number = 1
            0,  # same_input = False
            0,  # include_self = False
        ),
        _param_types,
    )
    grid, block = runtime.launch_config(sweep_kernel, total_count)
    runtime.launch(sweep_kernel, grid=grid, block=block, params=scatter_params)

    # Map index pairs back to (row, segment) via device-side gather
    d_left_pair_idx = cp.asarray(d_out_left)
    d_right_pair_idx = cp.asarray(d_out_right)

    return SegmentCandidatePairs(
        _host_left_rows=None,
        _host_left_segments=None,
        _host_right_rows=None,
        _host_right_segments=None,
        pairs_examined=pairs_examined,
        _device_left_rows=d_left_rows[d_left_pair_idx],
        _device_left_segments=d_left_segs[d_left_pair_idx],
        _device_right_rows=d_right_rows[d_right_pair_idx],
        _device_right_segments=d_right_segs[d_right_pair_idx],
    )


def _generate_segment_mbr_pairs_cpu(
    left_segments: SegmentMBRTable,
    right_segments: SegmentMBRTable,
    *,
    tile_size: int = 512,
) -> SegmentCandidatePairs:
    """CPU fallback: tiled numpy MBR overlap on host."""
    # Ensure host arrays
    left_seg = left_segments.to_host() if left_segments.residency is Residency.DEVICE else left_segments
    right_seg = right_segments.to_host() if right_segments.residency is Residency.DEVICE else right_segments

    left_rows_out: list[np.ndarray] = []
    left_segment_out: list[np.ndarray] = []
    right_rows_out: list[np.ndarray] = []
    right_segment_out: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_seg.count, tile_size):
        left_bounds = left_seg.bounds[left_start : left_start + tile_size]
        left_rows = left_seg.row_indices[left_start : left_start + tile_size]
        left_ids = left_seg.segment_indices[left_start : left_start + tile_size]
        for right_start in range(0, right_seg.count, tile_size):
            right_bounds = right_seg.bounds[right_start : right_start + tile_size]
            right_rows = right_seg.row_indices[right_start : right_start + tile_size]
            right_ids = right_seg.segment_indices[right_start : right_start + tile_size]
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
        return SegmentCandidatePairs(
            _host_left_rows=empty,
            _host_left_segments=empty,
            _host_right_rows=empty,
            _host_right_segments=empty,
            pairs_examined=pairs_examined,
        )
    return SegmentCandidatePairs(
        _host_left_rows=np.concatenate(left_rows_out),
        _host_left_segments=np.concatenate(left_segment_out),
        _host_right_rows=np.concatenate(right_rows_out),
        _host_right_segments=np.concatenate(right_segment_out),
        pairs_examined=pairs_examined,
    )


def generate_segment_mbr_pairs(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
) -> SegmentCandidatePairs:
    """Generate candidate segment pairs by MBR overlap filtering.

    Dispatches to GPU when available.  The GPU path uses the existing
    sweep-sort overlap kernel (``_generate_bounds_pairs_gpu``) on segment
    bounds, returning device-resident CuPy arrays (no eager D->H transfer).
    """
    left_segments = extract_segment_mbrs(left)
    right_segments = extract_segment_mbrs(right)

    use_gpu = (
        has_gpu_runtime()
        and cp is not None
        and left_segments.residency is Residency.DEVICE
        and right_segments.residency is Residency.DEVICE
    )
    if use_gpu:
        try:
            return _generate_segment_mbr_pairs_gpu(left_segments, right_segments)
        except Exception:
            logger.debug("GPU segment pair generation failed, falling back to CPU", exc_info=True)
            record_fallback_event(
                surface="generate_segment_mbr_pairs",
                reason="GPU kernel failed, falling back to CPU",
            )
    return _generate_segment_mbr_pairs_cpu(
        left_segments, right_segments, tile_size=tile_size,
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
