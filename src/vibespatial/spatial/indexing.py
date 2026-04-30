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
    compute_geometry_bounds_device,
    compute_morton_keys,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.config import COARSE_BOUNDS_TILE_SIZE, SEGMENT_TILE_SIZE  # noqa: E402
from vibespatial.runtime.fallbacks import record_fallback_event  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import (  # noqa: E402
    Residency,
    TransferTrigger,
    combined_residency,
)
from vibespatial.spatial.indexing_cpu import iter_geometry_parts  # noqa: E402

logger = logging.getLogger(__name__)

_REGULAR_GRID_SINGLE_BLOCK_CERTIFY_LIMIT = 256
_REGULAR_GRID_MAX_CERTIFY_BLOCKS = 65535


def _runtime_device_to_host(device_array: object, dtype, *, reason: str) -> np.ndarray:
    host = get_cuda_runtime().copy_device_to_host(device_array, reason=reason)
    return host.astype(dtype, copy=False)


def _default_index_runtime_selection() -> RuntimeSelection:
    return plan_dispatch_selection(
        kernel_name="flat_index_build",
        kernel_class=KernelClass.COARSE,
        row_count=1,  # always exceeds threshold of 0
        current_residency=Residency.HOST,
    )


from vibespatial.spatial.indexing_kernels import (  # noqa: E402
    _INDEXING_KERNEL_NAMES,
    _INDEXING_KERNEL_SOURCE,
    _SEGMENT_MBR_KERNEL_NAMES,
    _SEGMENT_MBR_KERNEL_SOURCE,
)


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
        host = _runtime_device_to_host(
            self._device_left_indices,
            np.int32,
            reason="spatial index candidate left-index host export",
        )
        object.__setattr__(self, "_host_left_indices", host)
        return host

    @property
    def right_indices(self) -> np.ndarray:
        """Lazily materialise host right_indices from device (ADR-0005)."""
        if self._host_right_indices is not None:
            return self._host_right_indices
        host = _runtime_device_to_host(
            self._device_right_indices,
            np.int32,
            reason="spatial index candidate right-index host export",
        )
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
        empty = cp.empty(0, dtype=cp.int32)
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
    total_pairs = count_scatter_total(
        runtime,
        cp_counts,
        d_offsets,
        reason="spatial index sweep candidate-pair allocation fence",
    )

    if total_pairs == 0:
        empty = cp.empty(0, dtype=cp.int32)
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
    tile_size: int = COARSE_BOUNDS_TILE_SIZE,
    include_self: bool = False,
) -> CandidatePairs:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    same_input = right is None or right is left
    right_array = left if right is None else right

    left_bounds = compute_geometry_bounds(left)
    right_bounds = left_bounds if same_input else compute_geometry_bounds(right_array)

    total_geom_count = left.row_count + (0 if same_input else right_array.row_count)
    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=total_geom_count,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=has_gpu_runtime(),
        current_residency=left.residency if same_input else combined_residency(left, right_array),
    )
    use_gpu = selection.selected is ExecutionMode.GPU and cp is not None
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
    tile_size: int = COARSE_BOUNDS_TILE_SIZE,
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
    _host_bounds: object  # np.ndarray or None (lazy from device_bounds)
    total_bounds: tuple[float, float, float, float]
    regular_grid: RegularGridRectIndex | None = None
    device_morton_keys: object = None  # CuPy device array or None
    device_order: object = None  # CuPy device array or None
    device_bounds: object = None  # CuPy device array or None

    @property
    def bounds(self) -> np.ndarray:
        """Lazily materialise host bounds for CPU/public compatibility paths."""
        if self._host_bounds is not None:
            return self._host_bounds
        if self.device_bounds is None:
            raise ValueError("FlatSpatialIndex has neither host nor device bounds")
        host_bounds = _runtime_device_to_host(
            self.device_bounds,
            np.float64,
            reason="flat spatial index device bounds host export",
        )
        object.__setattr__(self, "_host_bounds", host_bounds)
        return host_bounds

    @property
    def order(self) -> np.ndarray:
        """Lazily materialise host order array from device (ADR-0005)."""
        if self._host_order is not None:
            return self._host_order
        host_order = _runtime_device_to_host(
            self.device_order,
            np.int32,
            reason="flat spatial index device order host export",
        )
        object.__setattr__(self, "_host_order", host_order)
        return host_order

    @property
    def morton_keys(self) -> np.ndarray:
        """Lazily materialise host morton_keys array from device (ADR-0005)."""
        if self._host_morton_keys is not None:
            return self._host_morton_keys
        host_keys = _runtime_device_to_host(
            self.device_morton_keys,
            np.uint64,
            reason="flat spatial index device morton-key host export",
        )
        object.__setattr__(self, "_host_morton_keys", host_keys)
        return host_keys

    @property
    def size(self) -> int:
        if self._host_order is not None:
            return int(self._host_order.size)
        if self.device_order is not None:
            return int(self.device_order.size)
        if self._host_bounds is not None:
            return int(self._host_bounds.shape[0])
        if self.device_bounds is not None:
            return int(self.device_bounds.shape[0])
        # Avoid D→H just for size — use bounds row count.
        return int(self.bounds.shape[0])

    def geometry_metadata(self, *, source_token: str | None = None):
        """Wrap index bounds/metadata as a private native carrier."""
        from vibespatial.api._native_metadata import NativeGeometryMetadata

        return NativeGeometryMetadata.from_spatial_index(
            self,
            source_token=source_token,
        )

    def to_native_spatial_index(self, *, source_token: str | None = None):
        """Wrap this index as reusable private native execution state."""
        from vibespatial.api._native_metadata import NativeSpatialIndex

        return NativeSpatialIndex.from_flat_index(
            self,
            source_token=source_token,
        )

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
        tile_size: int = COARSE_BOUNDS_TILE_SIZE,
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


def _coords_form_axis_aligned_box(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if xs.size != 5 or ys.size != 5:
        return None
    minx = float(xs.min())
    miny = float(ys.min())
    maxx = float(xs.max())
    maxy = float(ys.max())
    if minx >= maxx or miny >= maxy:
        return None
    tol = 1e-9 * max(abs(maxx - minx), abs(maxy - miny), 1.0)
    if abs(xs[0] - xs[-1]) > tol or abs(ys[0] - ys[-1]) > tol:
        return None
    x_matches = (np.abs(xs - minx) <= tol) | (np.abs(xs - maxx) <= tol)
    y_matches = (np.abs(ys - miny) <= tol) | (np.abs(ys - maxy) <= tol)
    if not bool(np.all(x_matches) and np.all(y_matches)):
        return None
    edge_same_x = np.abs(xs[1:] - xs[:-1]) <= tol
    edge_same_y = np.abs(ys[1:] - ys[:-1]) <= tol
    if not bool(np.all(np.logical_xor(edge_same_x, edge_same_y))):
        return None
    return minx, miny, maxx, maxy


def _detect_single_row_device_rect_index(
    geometry_array: OwnedGeometryArray,
) -> tuple[RegularGridRectIndex, np.ndarray] | None:
    """Validate a one-row device polygon rectangle with minimal D2H."""
    if (
        cp is None
        or geometry_array.row_count != 1
        or GeometryFamily.POLYGON not in geometry_array.families
        or len(geometry_array.families) != 1
        or geometry_array.residency is not Residency.DEVICE
    ):
        return None
    state = geometry_array._ensure_device_state()
    device_buffer = state.families.get(GeometryFamily.POLYGON)
    if device_buffer is None:
        return None
    if int(getattr(device_buffer.x, "size", 0)) != 5 or int(
        getattr(device_buffer.y, "size", 0)
    ) != 5:
        return None

    runtime = get_cuda_runtime()
    xs = runtime.copy_device_to_host(
        device_buffer.x,
        reason="spatial index single-rectangle x-coordinate validation fence",
    ).astype(np.float64, copy=False)
    ys = runtime.copy_device_to_host(
        device_buffer.y,
        reason="spatial index single-rectangle y-coordinate validation fence",
    ).astype(np.float64, copy=False)
    bounds_tuple = _coords_form_axis_aligned_box(xs, ys)
    if bounds_tuple is None:
        return None

    minx, miny, maxx, maxy = bounds_tuple
    bounds = np.asarray([[minx, miny, maxx, maxy]], dtype=np.float64)
    return (
        RegularGridRectIndex(
            origin_x=minx,
            origin_y=miny,
            cell_width=maxx - minx,
            cell_height=maxy - miny,
            cols=1,
            rows=1,
            size=1,
        ),
        bounds,
    )


def _detect_regular_grid_rect_index_device(
    geometry_array: OwnedGeometryArray,
) -> tuple[RegularGridRectIndex, object, tuple[float, float, float, float]] | None:
    """Detect a device-resident regular rectangle grid without host metadata."""
    if (
        cp is None
        or geometry_array.row_count <= 1
        or GeometryFamily.POLYGON not in geometry_array.families
        or len(geometry_array.families) != 1
        or geometry_array.residency is not Residency.DEVICE
    ):
        return None

    state = geometry_array._ensure_device_state()
    polygon_buffer = state.families.get(GeometryFamily.POLYGON)
    if polygon_buffer is None or polygon_buffer.ring_offsets is None:
        return None
    host_polygon_buffer = geometry_array.families.get(GeometryFamily.POLYGON)
    if (
        host_polygon_buffer is not None
        and host_polygon_buffer.host_materialized
        and geometry_array._validity is not None
        and geometry_array._tags is not None
        and geometry_array._family_row_offsets is not None
    ):
        return None

    row_count = int(geometry_array.row_count)
    if (
        int(getattr(polygon_buffer.geometry_offsets, "size", 0)) != row_count + 1
        or int(getattr(polygon_buffer.ring_offsets, "size", 0)) != row_count + 1
    ):
        return None

    x_count = int(getattr(polygon_buffer.x, "size", 0))
    y_count = int(getattr(polygon_buffer.y, "size", 0))
    if x_count < 5 or y_count != x_count:
        return None

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    d_bounds = runtime.allocate((row_count, 4), cp.float64)
    d_summary = runtime.allocate((8,), cp.float64)
    d_block_summaries = None
    d_valid_flag = None
    try:
        kernels = _indexing_kernels()
        certify_kernel = kernels["regular_rect_grid_certify"]
        block_size = min(runtime.optimal_block_size(certify_kernel), 256)
        block_size = max(32, 1 << (block_size.bit_length() - 1))
        certify_values = (
            ptr(polygon_buffer.x),
            ptr(polygon_buffer.y),
            ptr(polygon_buffer.geometry_offsets),
            ptr(polygon_buffer.ring_offsets),
            ptr(polygon_buffer.empty_mask),
            ptr(state.validity),
            ptr(state.tags),
            row_count,
            x_count,
            int(FAMILY_TAGS[GeometryFamily.POLYGON]),
            ptr(d_bounds),
        )
        certify_types = (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        )
        if row_count <= _REGULAR_GRID_SINGLE_BLOCK_CERTIFY_LIMIT:
            runtime.launch(
                certify_kernel,
                grid=(1, 1, 1),
                block=(block_size, 1, 1),
                params=(
                    certify_values + (ptr(d_summary),),
                    certify_types + (KERNEL_PARAM_PTR,),
                ),
            )
        else:
            # Row-shaped certification: each CTA validates a strided row chunk,
            # then one tiny finalize pass reduces summaries for the host scalar.
            block_count = min(
                _REGULAR_GRID_MAX_CERTIFY_BLOCKS,
                max(1, (row_count + block_size - 1) // block_size),
            )
            d_block_summaries = runtime.allocate((block_count, 8), cp.float64)
            d_valid_flag = runtime.allocate((1,), cp.int32, zero=True)
            d_valid_flag[...] = 1
            runtime.launch(
                kernels["regular_rect_grid_certify_blocks"],
                grid=(block_count, 1, 1),
                block=(block_size, 1, 1),
                params=(
                    certify_values + (ptr(d_block_summaries),),
                    certify_types + (KERNEL_PARAM_PTR,),
                ),
            )
            runtime.launch(
                kernels["regular_rect_grid_finalize"],
                grid=(1, 1, 1),
                block=(block_size, 1, 1),
                params=(
                    (
                        ptr(d_bounds),
                        ptr(d_block_summaries),
                        block_count,
                        row_count,
                        ptr(d_summary),
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                    ),
                ),
            )
            runtime.launch(
                kernels["regular_rect_grid_validate_positions"],
                grid=(block_count, 1, 1),
                block=(block_size, 1, 1),
                params=(
                    (ptr(d_bounds), row_count, ptr(d_summary), ptr(d_valid_flag)),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                    ),
                ),
            )
            runtime.launch(
                kernels["regular_rect_grid_apply_valid"],
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=(
                    (ptr(d_valid_flag), ptr(d_summary)),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR),
                ),
            )
        host_summary = runtime.copy_device_to_host(
            d_summary,
            reason="spatial index regular-grid summary scalar fence",
        )
    except Exception:
        runtime.free(d_bounds)
        raise
    finally:
        runtime.free(d_block_summaries)
        runtime.free(d_valid_flag)
        runtime.free(d_summary)
    (
        h_minx,
        h_miny,
        h_maxx,
        h_maxy,
        h_cell_width,
        h_cell_height,
        h_cols,
        h_rows,
    ) = host_summary.tolist()
    if h_cols <= 0.0 or h_rows <= 0.0:
        runtime.free(d_bounds)
        return None
    state.row_bounds = d_bounds
    return (
        RegularGridRectIndex(
            origin_x=float(h_minx),
            origin_y=float(h_miny),
            cell_width=float(h_cell_width),
            cell_height=float(h_cell_height),
            cols=int(h_cols),
            rows=int(h_rows),
            size=row_count,
        ),
        d_bounds,
        (float(h_minx), float(h_miny), float(h_maxx), float(h_maxy)),
    )


def _sample_regular_grid_polygon_vertices(
    geometry_array: OwnedGeometryArray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only the polygon vertices needed for regular-grid verification."""
    if sample_indices.size == 0:
        empty = np.empty((0, 5), dtype=np.float64)
        return empty, empty

    polygon_buffer = geometry_array.families[GeometryFamily.POLYGON]
    if polygon_buffer.host_materialized:
        ring_x_full = polygon_buffer.x.reshape(geometry_array.row_count, 5)
        ring_y_full = polygon_buffer.y.reshape(geometry_array.row_count, 5)
        return ring_x_full[sample_indices], ring_y_full[sample_indices]

    if (
        cp is None
        or geometry_array.device_state is None
        or GeometryFamily.POLYGON not in geometry_array.device_state.families
    ):
        geometry_array._ensure_host_state()
        polygon_buffer = geometry_array.families[GeometryFamily.POLYGON]
        ring_x_full = polygon_buffer.x.reshape(geometry_array.row_count, 5)
        ring_y_full = polygon_buffer.y.reshape(geometry_array.row_count, 5)
        return ring_x_full[sample_indices], ring_y_full[sample_indices]

    runtime = get_cuda_runtime()
    device_buffer = geometry_array.device_state.families[GeometryFamily.POLYGON]
    sample_coord_indices = (
        sample_indices.astype(np.int64, copy=False)[:, None] * 5
        + np.arange(5, dtype=np.int64)[None, :]
    ).reshape(-1)
    d_sample_coord_indices = cp.asarray(sample_coord_indices)
    ring_x = runtime.copy_device_to_host(
        device_buffer.x[d_sample_coord_indices],
        reason="spatial index regular-grid sampled x-coordinate validation export",
    ).reshape(sample_indices.size, 5)
    ring_y = runtime.copy_device_to_host(
        device_buffer.y[d_sample_coord_indices],
        reason="spatial index regular-grid sampled y-coordinate validation export",
    ).reshape(sample_indices.size, 5)
    return (
        np.ascontiguousarray(ring_x, dtype=np.float64),
        np.ascontiguousarray(ring_y, dtype=np.float64),
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
    # Device-resident decode paths keep metadata and offsets lazy on host.
    # Hydrate only the routing/structure arrays needed for these checks;
    # coordinate payload stays on device until the sampled verification below.
    if geometry_array._validity is None:
        geometry_array._ensure_host_metadata()
    if (
        not polygon_buffer.host_materialized
        and (
            polygon_buffer.geometry_offsets.size == 0
            or polygon_buffer.empty_mask.size == 0
            or polygon_buffer.ring_offsets is None
        )
    ):
        geometry_array._ensure_host_family_structure(GeometryFamily.POLYGON)
        polygon_buffer = geometry_array.families[GeometryFamily.POLYGON]
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
    # Use direct absolute-difference instead of np.allclose for speed:
    # np.allclose has significant per-call overhead (~0.3ms) that dominates
    # for the simple uniform-check we need here.
    if np.any(np.abs(widths - cell_width) > tol):
        return None
    if np.any(np.abs(heights - cell_height) > tol):
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

    # Structural checks above use only metadata/offsets. The rectangle
    # verification below still needs coordinates, but only for a small sample.
    # Pull those sampled vertices directly from device instead of
    # materializing the full polygon batch on host.
    n = geometry_array.row_count
    _SAMPLE_THRESHOLD = 256
    _SAMPLE_SIZE = 128
    if n > _SAMPLE_THRESHOLD:
        # Deterministic evenly-spaced sampling: covers first, last, and
        # interior rows.  Avoids np.random.RandomState overhead (~1ms).
        step = max(1, n // _SAMPLE_SIZE)
        sample_indices = np.arange(0, n, step, dtype=np.intp)[:_SAMPLE_SIZE]
    else:
        sample_indices = np.arange(n, dtype=np.intp)

    ring_x, ring_y = _sample_regular_grid_polygon_vertices(
        geometry_array,
        sample_indices,
    )
    sample_bounds = bounds[sample_indices]

    # Use direct absolute-difference comparisons instead of np.isclose
    # (which has high per-call overhead) since rtol=0.0 makes them identical.
    x_is_min = np.abs(ring_x - sample_bounds[:, 0:1]) <= tol
    x_is_max = np.abs(ring_x - sample_bounds[:, 2:3]) <= tol
    y_is_min = np.abs(ring_y - sample_bounds[:, 1:2]) <= tol
    y_is_max = np.abs(ring_y - sample_bounds[:, 3:4]) <= tol
    if not np.all(x_is_min | x_is_max):
        return None
    if not np.all(y_is_min | y_is_max):
        return None
    if not np.all(np.abs(ring_x[:, 0] - ring_x[:, -1]) <= tol):
        return None
    if not np.all(np.abs(ring_y[:, 0] - ring_y[:, -1]) <= tol):
        return None
    if not np.all(x_is_min[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(x_is_max[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(y_is_min[:, :4].sum(axis=1) == 2):
        return None
    if not np.all(y_is_max[:, :4].sum(axis=1) == 2):
        return None
    edge_same_x = np.abs(ring_x[:, 1:] - ring_x[:, :-1]) <= tol
    edge_same_y = np.abs(ring_y[:, 1:] - ring_y[:, :-1]) <= tol
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
        if keep_on_device and cp is not None:
            return (
                cp.full(
                    geometry_array.row_count,
                    np.iinfo(np.uint64).max,
                    dtype=cp.uint64,
                ),
                cp.arange(geometry_array.row_count, dtype=cp.int32),
            )
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
    total_bounds = (
        float(finite[:, 0].min()),
        float(finite[:, 1].min()),
        float(finite[:, 2].max()),
        float(finite[:, 3].max()),
    )
    try:
        return _build_flat_spatial_index_gpu_from_device_bounds(
            geometry_array,
            device_bounds,
            total_bounds=total_bounds,
            keep_on_device=keep_on_device,
        )
    finally:
        runtime.free(device_bounds)


def _device_total_bounds(d_bounds) -> tuple[float, float, float, float]:
    """Summarize device row bounds with one named scalar fence."""
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is required for device total bounds")
    row_count = int(getattr(d_bounds, "shape", (0,))[0])
    if row_count == 0:
        return (float("nan"),) * 4

    finite = cp.isfinite(d_bounds).all(axis=1)
    summary = cp.empty(5, dtype=cp.float64)
    summary[0] = cp.count_nonzero(finite).astype(cp.float64)
    summary[1] = cp.min(cp.where(finite, d_bounds[:, 0], cp.inf))
    summary[2] = cp.min(cp.where(finite, d_bounds[:, 1], cp.inf))
    summary[3] = cp.max(cp.where(finite, d_bounds[:, 2], -cp.inf))
    summary[4] = cp.max(cp.where(finite, d_bounds[:, 3], -cp.inf))
    host = get_cuda_runtime().copy_device_to_host(
        summary,
        reason="flat spatial index device total-bounds scalar fence",
    )
    if int(host[0]) == 0:
        return (float("nan"),) * 4
    return (float(host[1]), float(host[2]), float(host[3]), float(host[4]))


def _build_flat_spatial_index_gpu_from_device_bounds(
    geometry_array: OwnedGeometryArray,
    device_bounds,
    *,
    total_bounds: tuple[float, float, float, float],
    keep_on_device: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Morton order from already device-resident bounds."""
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is required for device spatial indexing")

    runtime = get_cuda_runtime()
    if not np.isfinite(np.asarray(total_bounds, dtype=np.float64)).all():
        if keep_on_device:
            return (
                cp.full(
                    geometry_array.row_count,
                    np.iinfo(np.uint64).max,
                    dtype=cp.uint64,
                ),
                cp.arange(geometry_array.row_count, dtype=cp.int32),
            )
        return (
            np.full(geometry_array.row_count, np.iinfo(np.uint64).max, dtype=np.uint64),
            np.arange(geometry_array.row_count, dtype=np.int32),
        )

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
                float(total_bounds[0]),
                float(total_bounds[1]),
                float(total_bounds[2]),
                float(total_bounds[3]),
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
        device_order = cp.arange(geometry_array.row_count, dtype=cp.int32)
        sorted_result = sort_pairs(device_keys, device_order, synchronize=False)

        if keep_on_device:
            # Return CuPy device arrays — caller is responsible for eventual
            # D->H transfer or further GPU consumption.
            morton_keys = device_keys
            order = sorted_result.values
            return morton_keys, order

        # Standard path: explicit host export for CPU/public compatibility.
        morton_keys = _runtime_device_to_host(
            device_keys,
            np.uint64,
            reason="flat spatial index GPU morton-key build host export",
        )
        order = _runtime_device_to_host(
            sorted_result.values,
            np.int32,
            reason="flat spatial index GPU order build host export",
        )
        return morton_keys, order
    finally:
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
    # Use GPU for bounds only when geometry is already device-resident.
    # Host-resident arrays should not be uploaded purely to compute bounds;
    # the GPU sort path can still take over later if selected.
    bounds_dispatch = (
        ExecutionMode.GPU
        if geometry_array.residency is Residency.DEVICE and has_gpu_runtime()
        else ExecutionMode.CPU
    )

    if (
        selection.selected is ExecutionMode.GPU
        and geometry_array.residency is Residency.DEVICE
        and geometry_array.row_count <= 1
        and has_gpu_runtime()
    ):
        if geometry_array.row_count == 0:
            bounds = np.empty((0, 4), dtype=np.float64)
            device_bounds = None
            regular_grid = None
        else:
            single_rect = _detect_single_row_device_rect_index(geometry_array)
            if single_rect is None:
                bounds = None
                device_bounds = compute_geometry_bounds_device(geometry_array)
                regular_grid = None
            else:
                regular_grid, bounds = single_rect
                device_bounds = None
        order = np.arange(geometry_array.row_count, dtype=np.int32)
        morton_keys = order.astype(np.uint64, copy=False)
        if bounds is not None and bounds.shape[0] > 0:
            total_bounds = (
                float(bounds[:, 0].min()),
                float(bounds[:, 1].min()),
                float(bounds[:, 2].max()),
                float(bounds[:, 3].max()),
            )
        else:
            total_bounds = (float("nan"),) * 4
        return FlatSpatialIndex(
            geometry_array=geometry_array,
            _host_order=order,
            _host_morton_keys=morton_keys,
            _host_bounds=bounds,
            total_bounds=total_bounds,
            regular_grid=regular_grid,
            device_bounds=device_bounds,
        )

    device_regular_grid = _detect_regular_grid_rect_index_device(geometry_array)
    if device_regular_grid is not None:
        regular_grid, device_bounds, total_bounds = device_regular_grid
        order = np.arange(geometry_array.row_count, dtype=np.int32)
        morton_keys = order.astype(np.uint64, copy=False)
        return FlatSpatialIndex(
            geometry_array=geometry_array,
            _host_order=order,
            _host_morton_keys=morton_keys,
            _host_bounds=None,
            total_bounds=total_bounds,
            regular_grid=regular_grid,
            device_bounds=device_bounds,
        )

    device_bounds = None
    total_bounds = None
    if (
        selection.selected is ExecutionMode.GPU
        and geometry_array.residency is Residency.DEVICE
        and has_gpu_runtime()
    ):
        device_bounds = compute_geometry_bounds_device(geometry_array)
        regular_grid = None
    else:
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
            if device_bounds is not None:
                total_bounds = _device_total_bounds(device_bounds)
                d_morton_keys_raw, d_order_raw = (
                    _build_flat_spatial_index_gpu_from_device_bounds(
                        geometry_array,
                        device_bounds,
                        total_bounds=total_bounds,
                        keep_on_device=True,
                    )
                )
                bounds = None
            else:
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
    # Compute total bounds.  When a regular grid was detected, all bounds
    # are finite (validated during detection), so skip NaN filtering.
    # Otherwise use np.nanmin/np.nanmax which is faster than building a
    # boolean mask + fancy-indexing the finite rows.
    if total_bounds is not None:
        pass
    elif regular_grid is not None:
        total_bounds = (
            float(bounds[:, 0].min()),
            float(bounds[:, 1].min()),
            float(bounds[:, 2].max()),
            float(bounds[:, 3].max()),
        )
    elif bounds.shape[0] == 0 or np.all(np.isnan(bounds[:, 0])):
        total_bounds = (float("nan"),) * 4
    else:
        total_bounds = (
            float(np.nanmin(bounds[:, 0])),
            float(np.nanmin(bounds[:, 1])),
            float(np.nanmax(bounds[:, 2])),
            float(np.nanmax(bounds[:, 3])),
        )
    return FlatSpatialIndex(
        geometry_array=geometry_array,
        _host_order=order,
        _host_morton_keys=morton_keys,
        _host_bounds=bounds,
        total_bounds=total_bounds,
        regular_grid=regular_grid,
        device_morton_keys=d_morton_keys,
        device_order=d_order,
        device_bounds=device_bounds,
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
        runtime = get_cuda_runtime()
        return SegmentMBRTable(
            row_indices=runtime.copy_device_to_host(
                self.row_indices,
                reason="segment MBR row-index host export",
            ),
            segment_indices=runtime.copy_device_to_host(
                self.segment_indices,
                reason="segment MBR segment-index host export",
            ),
            bounds=runtime.copy_device_to_host(
                self.bounds,
                reason="segment MBR bounds host export",
            ),
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
            lines = iter_geometry_parts(geometry)
        elif geom_type == "MultiPolygon":
            lines = []
            for polygon in iter_geometry_parts(geometry):
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
    total = count_scatter_total(
        runtime,
        d_counts,
        d_offsets,
        reason="spatial index segment-bounds allocation fence",
    )

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
    selection = plan_dispatch_selection(
        kernel_name="segment_mbr_extract",
        kernel_class=KernelClass.COARSE,
        row_count=geometry_array.row_count,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=has_gpu_runtime() and cp is not None,
        current_residency=geometry_array.residency,
    )
    if selection.selected is ExecutionMode.GPU and cp is not None:
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
        host = _runtime_device_to_host(
            self._device_left_rows,
            np.int32,
            reason="segment candidate left-row host export",
        )
        object.__setattr__(self, "_host_left_rows", host)
        return host

    @property
    def left_segments(self) -> np.ndarray:
        """Lazily materialise host left_segments from device (ADR-0005)."""
        if self._host_left_segments is not None:
            return self._host_left_segments
        host = _runtime_device_to_host(
            self._device_left_segments,
            np.int32,
            reason="segment candidate left-segment host export",
        )
        object.__setattr__(self, "_host_left_segments", host)
        return host

    @property
    def right_rows(self) -> np.ndarray:
        """Lazily materialise host right_rows from device (ADR-0005)."""
        if self._host_right_rows is not None:
            return self._host_right_rows
        host = _runtime_device_to_host(
            self._device_right_rows,
            np.int32,
            reason="segment candidate right-row host export",
        )
        object.__setattr__(self, "_host_right_rows", host)
        return host

    @property
    def right_segments(self) -> np.ndarray:
        """Lazily materialise host right_segments from device (ADR-0005)."""
        if self._host_right_segments is not None:
            return self._host_right_segments
        host = _runtime_device_to_host(
            self._device_right_segments,
            np.int32,
            reason="segment candidate right-segment host export",
        )
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
    total_pairs = count_scatter_total(
        runtime,
        cp_counts,
        d_offsets,
        reason="spatial index segment sweep-pair allocation fence",
    )

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
    tile_size: int = SEGMENT_TILE_SIZE,
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
    tile_size: int = SEGMENT_TILE_SIZE,
) -> SegmentCandidatePairs:
    """Generate candidate segment pairs by MBR overlap filtering.

    Dispatches to GPU when available.  The GPU path uses the existing
    sweep-sort overlap kernel (``_generate_bounds_pairs_gpu``) on segment
    bounds, returning device-resident CuPy arrays (no eager D->H transfer).
    """
    left_segments = extract_segment_mbrs(left)
    right_segments = extract_segment_mbrs(right)

    selection = plan_dispatch_selection(
        kernel_name="segment_mbr_pairs",
        kernel_class=KernelClass.COARSE,
        row_count=left_segments.count + right_segments.count,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=(
            has_gpu_runtime()
            and cp is not None
            and left_segments.residency is Residency.DEVICE
            and right_segments.residency is Residency.DEVICE
        ),
        current_residency=combined_residency(left, right),
    )
    if selection.selected is ExecutionMode.GPU and cp is not None:
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
    tile_size: int = SEGMENT_TILE_SIZE,
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
