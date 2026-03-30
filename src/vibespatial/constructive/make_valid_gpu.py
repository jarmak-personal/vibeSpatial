"""GPU-resident make_valid batch repair pipeline (ADR-0019 + ADR-0033).

Phase 16: Batch repolygonization. All invalid polygons are collected into one
contiguous batch and processed through the GPU repair pipeline together,
eliminating the per-polygon Python loop. shapely.polygonize and
shapely.make_valid are no longer used in the primary GPU path.

Pipeline stages (batched across all invalid polygons):
  Phase A: GPU self-intersection detection (extract ring segments, classify pairs)
  Phase B: Simple repair kernels (close rings, remove duplicate vertices, fix orientation)
  Phase C: Self-intersection splitting (count/scatter split events, sort, dedup, rebuild)
  Phase D: Re-polygonization (half-edges, face walk, face containment, assembly)
  Phase E: Output assembly (build OwnedGeometryArray, convert to Shapely)

All kernels use fp64 compute (CONSTRUCTIVE class per ADR-0002).
Tier 1: NVRTC for geometry-specific work.
Tier 3a: CCCL for scan/sort/compact.
Tier 2: CuPy for element-wise ops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibespatial.constructive.make_valid_gpu_kernels import (
    _REPAIR_KERNEL_NAMES,
    _REPAIR_KERNEL_SOURCE,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    segmented_reduce_sum,
    sort_pairs,
    unique_sorted_pairs,
)

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "unique_by_key_i32_i32", "unique_by_key_u64_i32",
    "segmented_reduce_sum_f64",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.overlay.gpu import (  # noqa: E402
    _build_polygon_output_from_faces_gpu,
    _gpu_face_walk,
    build_gpu_half_edge_graph,
)
from vibespatial.overlay.types import (  # noqa: E402
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    OverlayFaceDeviceState,
    OverlayFaceTable,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.residency import Residency  # noqa: E402

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES),
])


def _compile_repair_kernels():
    return compile_kernel_group("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Phase B: Simple repair operations
# ---------------------------------------------------------------------------

def _gpu_close_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Close unclosed rings by appending first vertex. Returns new x, y, ring_offsets.

    Closure check via check_ring_closure NVRTC kernel (Tier 1, 1 thread/ring).
    Copy+append via close_rings NVRTC kernel.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- Step 1: NVRTC kernel checks which rings need closure (Tier 1) ---
    d_needs_close = cp.empty(ring_count, dtype=cp.int32)
    grid, block = runtime.launch_config(kernels["check_ring_closure"], ring_count)
    runtime.launch(
        kernels["check_ring_closure"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_close), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # Early exit if no rings need closure (single scalar D2H)
    if int(cp.sum(d_needs_close)) == 0:
        return d_x, d_y, d_ring_offsets

    # --- Step 2: Compute new ring sizes and offsets on device (Tier 2: CuPy) ---
    d_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_new_sizes = d_sizes + d_needs_close
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)
    total_new = int(d_new_offsets[-1])

    # --- Step 3: Copy vertices + append closure vertex where needed (Tier 1) ---
    d_out_x = cp.empty(total_new, dtype=cp.float64)
    d_out_y = cp.empty(total_new, dtype=cp.float64)

    grid, block = runtime.launch_config(kernels["close_rings"], ring_count)
    runtime.launch(
        kernels["close_rings"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_close),
             ptr(d_new_offsets), ptr(d_out_x), ptr(d_out_y), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    return d_out_x, d_out_y, d_new_offsets


def _gpu_remove_duplicate_vertices(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Remove consecutive duplicate vertices within each ring.

    Tier 1: flag_duplicate_vertices NVRTC kernel (1 thread/vertex).
    Tier 2: CuPy searchsorted for vertex-to-ring mapping, fancy indexing for compaction.
    Tier 3a: CCCL compact_indices for keep-mask compaction, segmented_reduce_sum
    for per-ring kept counts.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    vertex_count = int(d_x.shape[0])
    if vertex_count == 0:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident vertex-to-ring mapping (Tier 2: CuPy searchsorted) ---
    d_vertex_ids = cp.arange(vertex_count, dtype=cp.int32)
    d_vertex_ring_ids = cp.searchsorted(
        d_ring_offsets[1:], d_vertex_ids, side="right"
    ).astype(cp.int32)

    # --- Flag duplicates via NVRTC kernel (Tier 1) ---
    d_keep = cp.empty(vertex_count, dtype=cp.uint8)
    grid, block = runtime.launch_config(kernels["flag_duplicate_vertices"], vertex_count)
    runtime.launch(
        kernels["flag_duplicate_vertices"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_vertex_ring_ids),
             ptr(d_keep), vertex_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    # No explicit sync needed -- cp.sum below triggers implicit sync on same stream.

    # Early exit if no duplicates found (single scalar D2H via cp.sum)
    kept_count = int(cp.sum(d_keep))
    if kept_count == vertex_count:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident compaction (Tier 3a: CCCL compact_indices) ---
    compact_result = compact_indices(d_keep)
    d_kept_indices = compact_result.values.astype(cp.int64)

    new_x = d_x[d_kept_indices]
    new_y = d_y[d_kept_indices]

    # --- GPU-resident ring offset rebuild ---
    # Per-ring kept count via segmented reduce (Tier 3a: CCCL)
    d_keep_i32 = d_keep.astype(cp.int32)
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_keep_i32, d_starts, d_ends, num_segments=ring_count,
    )
    d_new_sizes = seg_result.values.astype(cp.int32)
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)

    return new_x, new_y, d_new_offsets


def _gpu_fix_ring_orientation(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Fix ring orientation: exterior rings CCW (positive area), holes CW (negative area).

    Tier 1: compute_ring_shoelace NVRTC kernel for per-vertex cross products.
    Tier 3a: segmented_reduce_sum (CCCL) for per-ring signed area.
    Tier 2: CuPy element-wise for exterior/hole classification and reversal mask.
    Tier 1: reverse_ring_coords NVRTC kernel for coordinate reversal.
    All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    vertex_count = int(d_x.shape[0])
    if vertex_count < 3 or ring_count == 0:
        return d_x, d_y

    # --- Step 1: Compute per-vertex shoelace cross products (Tier 1: NVRTC) ---
    # compute_ring_shoelace writes x[v]*y[v+1] - x[v+1]*y[v] for each vertex.
    # Launch for vertex_count-1 to avoid out-of-bounds read on the last vertex.
    # Zero-init so last-vertex-per-ring contributions are automatically 0.
    d_cross = cp.zeros(vertex_count, dtype=cp.float64)
    safe_count = vertex_count - 1
    if safe_count > 0:
        grid, block = runtime.launch_config(kernels["compute_ring_shoelace"], safe_count)
        runtime.launch(
            kernels["compute_ring_shoelace"],
            grid=grid, block=block,
            params=(
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_cross),
                 ring_count, safe_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
            ),
        )

    # Zero out cross products at ring boundaries to prevent cross-ring contamination
    d_last_verts = d_ring_offsets[1:] - 1
    d_cross[d_last_verts] = 0.0

    # --- Step 2: Per-ring signed area via segmented reduce (Tier 3a: CCCL) ---
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_cross, d_starts, d_ends, num_segments=ring_count,
    )
    d_ring_areas = seg_result.values * 0.5

    # --- Step 3: Classify exterior vs hole and build reversal mask (Tier 2: CuPy) ---
    d_ring_ids = cp.arange(ring_count, dtype=cp.int32)
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_ring_ids, side="right"
    ).astype(cp.int32)
    d_first_ring_of_poly = d_geom_offsets[d_poly_of_ring]
    d_is_exterior = (d_ring_ids == d_first_ring_of_poly)

    # Exterior should be CCW (positive area); hole should be CW (negative area)
    d_needs_reverse = (
        (d_is_exterior & (d_ring_areas < 0))
        | (~d_is_exterior & (d_ring_areas > 0))
    ).astype(cp.uint8)

    # Early exit if no rings need reversal (single scalar D2H via cp.sum)
    if int(cp.sum(d_needs_reverse)) == 0:
        return d_x, d_y

    # --- Step 4: Reverse coordinates for wrong-orientation rings (Tier 1: NVRTC) ---
    grid, block = runtime.launch_config(kernels["reverse_ring_coords"], ring_count)
    runtime.launch(
        kernels["reverse_ring_coords"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_reverse), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    # No explicit sync needed -- caller will either launch more same-stream
    # work or trigger implicit sync via CuPy/D2H read.
    return d_x, d_y


# ---------------------------------------------------------------------------
# Phase A + C: Self-intersection detection and splitting
# ---------------------------------------------------------------------------

def _extract_ring_segments_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Extract consecutive vertex pairs as flat segment table from ring coords.

    GPU-resident: seg_counts via CuPy element-wise (Tier 2), seg_offsets via
    CuPy cumsum (Tier 4), then one NVRTC kernel thread per segment (Tier 1).

    Returns (seg_x0, seg_y0, seg_x1, seg_y1, seg_ring_ids) on device.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- Compute seg_counts and seg_offsets on device (Tier 2: CuPy) ---
    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_seg_counts)
    total_segments = int(d_seg_offsets[-1])

    if total_segments == 0:
        empty = cp.empty(0, dtype=cp.float64)
        return empty, empty, empty, empty, cp.empty(0, dtype=cp.int32)

    d_seg_x0 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_y0 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_x1 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_y1 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_ring_ids = cp.empty(total_segments, dtype=cp.int32)

    kernel = kernels["extract_ring_segments"]
    grid, block = runtime.launch_config(kernel, total_segments)
    runtime.launch(
        kernel,
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_seg_offsets),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_seg_ring_ids), ring_count, total_segments),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    return d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids


def _detect_intra_ring_intersections(
    d_seg_x0: cp.ndarray,
    d_seg_y0: cp.ndarray,
    d_seg_x1: cp.ndarray,
    d_seg_y1: cp.ndarray,
    d_seg_ring_ids: cp.ndarray,
    total_segments: int,
    ring_count: int,
    d_ring_offsets: cp.ndarray,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Detect self-intersections within each ring using segment pair classification.

    GPU-resident: per-ring segment counts via cp.bincount (Tier 2), pair counts
    via CuPy element-wise (Tier 2), pair offsets via CuPy cumsum (Tier 4),
    pair generation via NVRTC kernel (Tier 1), classification via
    classify_segment_pairs_v2 (Tier 1).

    Returns (seg_a_ids, seg_b_ids, kinds, point_x, point_y) as device
    (CuPy) arrays.  Only returns pairs with kind != 0 (actual intersections).
    """
    if total_segments < 2:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- GPU-resident segment offset table per ring (Tier 2: CuPy) ---
    d_ring_seg_counts = cp.bincount(d_seg_ring_ids, minlength=ring_count).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_ring_seg_counts)

    # --- Compute pair counts per ring on device (Tier 2: CuPy) ---
    # For ring with k segments: pairs = k*(k-1)/2 - k (all pairs minus adjacent minus wrap)
    d_k = d_ring_seg_counts.astype(cp.int64)
    d_pair_counts = cp.maximum(d_k * (d_k - 1) // 2 - d_k, 0).astype(cp.int32)

    d_pair_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_pair_offsets[1:] = cp.cumsum(d_pair_counts)
    total_pairs = int(d_pair_offsets[-1])

    if total_pairs == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    # --- Generate pairs on GPU (Tier 1: NVRTC, 1 thread per pair) ---
    d_left_lookup = cp.empty(total_pairs, dtype=cp.int32)
    d_right_lookup = cp.empty(total_pairs, dtype=cp.int32)

    pair_kernel = kernels["generate_intra_ring_pairs"]
    grid_gen, block_gen = runtime.launch_config(pair_kernel, total_pairs)
    runtime.launch(
        pair_kernel,
        grid=grid_gen, block=block_gen,
        params=(
            (ptr(d_pair_offsets), ptr(d_seg_offsets),
             ptr(d_ring_seg_counts), ptr(d_left_lookup),
             ptr(d_right_lookup), ring_count, total_pairs),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    # --- Classify segment pairs (Tier 1: NVRTC) ---
    # Uses on-GPU Shewchuk adaptive refinement (no ambiguous output)
    from vibespatial.spatial.segment_primitives import _classify_kernels

    pair_count = total_pairs
    d_out_kind = cp.zeros(pair_count, dtype=cp.int8)
    d_out_px = cp.zeros(pair_count, dtype=cp.float64)
    d_out_py = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox1 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy1 = cp.zeros(pair_count, dtype=cp.float64)

    seg_kernels = _classify_kernels("double")
    classify_kernel = seg_kernels["classify_segment_pairs_v2"]
    grid, block = runtime.launch_config(classify_kernel, pair_count)
    runtime.launch(
        classify_kernel,
        grid=grid, block=block,
        params=(
            (ptr(d_left_lookup), ptr(d_right_lookup),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_out_kind), ptr(d_out_px), ptr(d_out_py),
             ptr(d_out_ox0), ptr(d_out_oy0), ptr(d_out_ox1), ptr(d_out_oy1),
             pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        ),
    )
    # No sync needed -- CuPy ops below run on same null stream.

    # Filter hits on device (Tier 2: CuPy boolean mask + fancy indexing)
    d_hit_mask = d_out_kind != 0
    d_hit_indices = cp.flatnonzero(d_hit_mask)
    if d_hit_indices.size == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    return (
        d_left_lookup[d_hit_indices],
        d_right_lookup[d_hit_indices],
        d_out_kind[d_hit_indices],
        d_out_px[d_hit_indices],
        d_out_py[d_hit_indices],
    )


def _split_self_intersections_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, bool]:
    """Detect and split self-intersections. Returns new x, y, ring_offsets, had_splits.

    GPU-resident: segment extraction via NVRTC (Tier 1), pair generation via
    NVRTC (Tier 1), dedup via CuPy mask (Tier 2), split counts via
    cp.bincount (Tier 2), ring sizes via segmented_reduce_sum (Tier 3a).
    """
    # Phase A: Extract segments and detect intersections (GPU-resident)
    d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids = \
        _extract_ring_segments_gpu(d_x, d_y, d_ring_offsets, ring_count, kernels)
    total_segments = int(d_seg_x0.size)

    d_seg_a, d_seg_b, d_kinds, d_px, d_py = _detect_intra_ring_intersections(
        d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids,
        total_segments, ring_count, d_ring_offsets, kernels,
    )

    if d_seg_a.size == 0:
        return d_x, d_y, d_ring_offsets, False

    # Phase C: Count and scatter split events
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    pair_count = int(d_seg_a.size)
    d_event_counts = cp.zeros(pair_count, dtype=cp.int32)

    count_kernel = kernels["count_self_split_events"]
    grid_count, block_count = runtime.launch_config(count_kernel, pair_count)
    runtime.launch(
        count_kernel,
        grid=grid_count, block=block_count,
        params=(
            (ptr(d_kinds), ptr(d_event_counts), pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    d_event_offsets = exclusive_sum(d_event_counts)
    total_events = count_scatter_total(runtime, d_event_counts, d_event_offsets)
    if total_events == 0:
        return d_x, d_y, d_ring_offsets, False

    d_out_seg_ids = cp.empty(total_events, dtype=cp.int32)
    d_out_t = cp.empty(total_events, dtype=cp.float64)
    d_out_x = cp.empty(total_events, dtype=cp.float64)
    d_out_y = cp.empty(total_events, dtype=cp.float64)
    d_out_key = cp.empty(total_events, dtype=cp.uint64)

    scatter_kernel = kernels["scatter_self_split_events"]
    grid_scatter, block_scatter = runtime.launch_config(scatter_kernel, pair_count)
    runtime.launch(
        scatter_kernel,
        grid=grid_scatter, block=block_scatter,
        params=(
            (ptr(d_seg_a), ptr(d_seg_b), ptr(d_kinds), ptr(d_px), ptr(d_py),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_event_offsets),
             ptr(d_out_seg_ids), ptr(d_out_t), ptr(d_out_x), ptr(d_out_y),
             ptr(d_out_key), pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # Sort events by (segment_id, t) via packed key
    sorted_result = sort_pairs(d_out_key, d_out_seg_ids)
    d_sorted_keys = sorted_result.keys

    # Gather sorted t, x, y using the sorted order
    d_indices = cp.arange(total_events, dtype=cp.int32)
    idx_sort = sort_pairs(d_out_key, d_indices)
    d_perm = idx_sort.values
    d_perm_i64 = d_perm.astype(cp.int64)

    d_sorted_seg_ids = d_out_seg_ids[d_perm_i64]
    d_sorted_t = d_out_t[d_perm_i64]
    d_sorted_x = d_out_x[d_perm_i64]
    d_sorted_y = d_out_y[d_perm_i64]

    # --- Phase 9: GPU-resident dedup (CuPy mask instead of host loop) ---
    if total_events > 1:
        unique_result = unique_sorted_pairs(d_sorted_keys, d_sorted_seg_ids)
        unique_count = unique_result.count
        if unique_count < total_events:
            # CuPy device-side mask: keep first occurrence of each key
            d_mask = cp.ones(total_events, dtype=cp.bool_)
            d_mask[1:] = d_sorted_keys[1:] != d_sorted_keys[:-1]
            keep_idx = cp.flatnonzero(d_mask).astype(cp.int64)
            d_sorted_seg_ids = d_sorted_seg_ids[keep_idx]
            d_sorted_t = d_sorted_t[keep_idx]
            d_sorted_x = d_sorted_x[keep_idx]
            d_sorted_y = d_sorted_y[keep_idx]
            total_events = int(keep_idx.size)

    if total_events == 0:
        return d_x, d_y, d_ring_offsets, False

    # --- Phase 9: GPU-resident per-segment split counts (cp.bincount) ---
    d_seg_split_counts = cp.bincount(
        d_sorted_seg_ids, minlength=total_segments
    ).astype(cp.int32)
    d_seg_split_offsets = exclusive_sum(d_seg_split_counts)

    # --- Phase 9: GPU-resident ring size computation ---
    # ring_seg_offsets and ring_seg_counts via device ops
    d_ring_seg_counts = cp.bincount(d_seg_ring_ids, minlength=ring_count).astype(cp.int32)
    d_ring_seg_offsets = exclusive_sum(d_ring_seg_counts)

    # Per-ring extra splits = sum of seg_split_counts for segments in that ring
    # Use segmented_reduce_sum on d_seg_split_counts grouped by ring
    d_seg_starts = d_ring_seg_offsets.copy()
    d_seg_ends = d_seg_starts + d_ring_seg_counts
    seg_extra_result = segmented_reduce_sum(
        d_seg_split_counts.astype(cp.float64),
        d_seg_starts, d_seg_ends,
        num_segments=ring_count,
    )
    d_extra_per_ring = seg_extra_result.values.astype(cp.int32)

    # New ring size: n_segs original start vertices + extra splits + 1 closing vertex
    d_new_ring_sizes = d_ring_seg_counts + d_extra_per_ring + 1
    d_new_ring_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_ring_offsets[1:] = cp.cumsum(d_new_ring_sizes)
    total_new = int(d_new_ring_offsets[-1])

    # Launch rebuild kernel (all arrays already device-resident)
    d_new_x = cp.empty(total_new, dtype=cp.float64)
    d_new_y = cp.empty(total_new, dtype=cp.float64)

    rebuild_kernel = kernels["rebuild_ring_coords"]
    ring_grid, ring_block = runtime.launch_config(rebuild_kernel, ring_count)
    runtime.launch(
        rebuild_kernel,
        grid=ring_grid, block=ring_block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets),
             ptr(d_sorted_seg_ids), ptr(d_sorted_x), ptr(d_sorted_y), ptr(d_sorted_t),
             ptr(d_seg_split_offsets), ptr(d_seg_split_counts),
             ptr(d_new_ring_offsets), ptr(d_ring_seg_offsets),
             ptr(d_new_x), ptr(d_new_y),
             ring_count, total_segments),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )
    # No sync needed -- caller uses device arrays on same null stream.
    return d_new_x, d_new_y, d_new_ring_offsets, True


# ---------------------------------------------------------------------------
# Phase D + E: Re-polygonization and output assembly
# ---------------------------------------------------------------------------

def _atomic_edges_from_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
) -> AtomicEdgeTable:
    """Build an AtomicEdgeTable from split ring coordinates (Part A).

    Bridges make_valid's split-ring output to overlay's reconstruction input.
    Each ring segment (consecutive vertex pair) becomes a directed edge pair
    (forward + reverse), matching the overlay pipeline's atomic edge format.

    All computation uses CuPy vectorised operations (Tier 2, fp64).
    """
    # Compute segments per ring: for a ring with n vertices, n-1 segments
    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_seg_counts)
    total_segments = int(d_seg_offsets[-1])

    if total_segments == 0:
        empty_d_i32 = cp.empty(0, dtype=cp.int32)
        empty_d_i8 = cp.empty(0, dtype=cp.int8)
        empty_d_f64 = cp.empty(0, dtype=cp.float64)
        return AtomicEdgeTable(
            left_segment_count=0, right_segment_count=0,
            runtime_selection=runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=empty_d_i32, direction=empty_d_i8,
                src_x=empty_d_f64, src_y=empty_d_f64,
                dst_x=empty_d_f64, dst_y=empty_d_f64,
                row_indices=empty_d_i32, part_indices=empty_d_i32,
                ring_indices=empty_d_i32, source_side=empty_d_i8,
            ),
            _count=0,
        )

    # Expand segments: for each segment, find its ring and local index
    d_seg_ids = cp.arange(total_segments, dtype=cp.int32)
    d_ring_of_seg = cp.searchsorted(
        d_seg_offsets[1:], d_seg_ids, side="right"
    ).astype(cp.int32)
    d_local_seg = d_seg_ids - d_seg_offsets[d_ring_of_seg]
    d_v = d_ring_offsets[d_ring_of_seg] + d_local_seg

    # Forward edges: src = v[i], dst = v[i+1]
    d_fwd_src_x = d_x[d_v]
    d_fwd_src_y = d_y[d_v]
    d_fwd_dst_x = d_x[d_v + 1]
    d_fwd_dst_y = d_y[d_v + 1]

    # Reverse edges: src = v[i+1], dst = v[i]
    d_rev_src_x = d_fwd_dst_x
    d_rev_src_y = d_fwd_dst_y
    d_rev_dst_x = d_fwd_src_x
    d_rev_dst_y = d_fwd_src_y

    # Interleave forward and reverse: edge 2k = forward, edge 2k+1 = reverse
    total_atomic = total_segments * 2
    d_src_x = cp.empty(total_atomic, dtype=cp.float64)
    d_src_y = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_x = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_y = cp.empty(total_atomic, dtype=cp.float64)
    d_src_x[0::2] = d_fwd_src_x
    d_src_x[1::2] = d_rev_src_x
    d_src_y[0::2] = d_fwd_src_y
    d_src_y[1::2] = d_rev_src_y
    d_dst_x[0::2] = d_fwd_dst_x
    d_dst_x[1::2] = d_rev_dst_x
    d_dst_y[0::2] = d_fwd_dst_y
    d_dst_y[1::2] = d_rev_dst_y

    # Source segment IDs: forward and reverse share the same segment ID
    d_source_seg_ids = cp.empty(total_atomic, dtype=cp.int32)
    d_source_seg_ids[0::2] = d_seg_ids
    d_source_seg_ids[1::2] = d_seg_ids

    # Direction: +1 for forward, -1 for reverse
    d_direction = cp.empty(total_atomic, dtype=cp.int8)
    d_direction[0::2] = 1
    d_direction[1::2] = -1

    # Map segments to polygon indices via geom_offsets
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_ring_of_seg, side="right"
    ).astype(cp.int32)
    d_row_indices = cp.empty(total_atomic, dtype=cp.int32)
    d_row_indices[0::2] = d_poly_of_ring
    d_row_indices[1::2] = d_poly_of_ring

    # Ring indices
    d_ring_indices = cp.empty(total_atomic, dtype=cp.int32)
    d_ring_indices[0::2] = d_ring_of_seg
    d_ring_indices[1::2] = d_ring_of_seg

    # Source side: all left (1) since we have one geometry
    d_source_side = cp.ones(total_atomic, dtype=cp.int8)

    # Part indices: same as row indices for simple polygons
    d_part_indices = d_row_indices.copy()

    # Build AtomicEdgeTable with device-primary storage.
    # build_gpu_half_edge_graph reads device_state metadata directly,
    # so no D->H transfers needed here.  Host arrays lazily materialized.
    return AtomicEdgeTable(
        left_segment_count=total_segments,
        right_segment_count=0,
        runtime_selection=runtime_selection,
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=d_source_seg_ids,
            direction=d_direction,
            src_x=d_src_x, src_y=d_src_y,
            dst_x=d_dst_x, dst_y=d_dst_y,
            row_indices=d_row_indices,
            part_indices=d_part_indices,
            ring_indices=d_ring_indices,
            source_side=d_source_side,
        ),
        _count=total_atomic,
    )


def _repolygonize_from_split_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection | None = None,
) -> OwnedGeometryArray | None:
    """Re-polygonize from split ring coordinates using GPU overlay pipeline (Part B).

    Replaces the prior Shapely CPU polygonize with the fully GPU-parallel
    half-edge graph / face walk / face assembly from overlay_gpu.py.

    Returns an OwnedGeometryArray (device-resident per ADR-0005), or None
    if the GPU pipeline produces no valid faces.
    """
    if runtime_selection is None:
        runtime_selection = RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="make_valid GPU repolygonize",
        )

    # Step 1: Build AtomicEdgeTable from split ring coordinates
    atomic_edges = _atomic_edges_from_rings(
        d_x, d_y, d_ring_offsets, d_geom_offsets,
        ring_count, polygon_count, runtime_selection,
    )
    if atomic_edges.count == 0:
        return None

    # Step 2: Build half-edge graph
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    if half_edge_graph.edge_count == 0:
        return None

    # Step 3: Face walk to extract faces
    (face_offsets, face_edge_ids, bounded_mask, signed_area,
     centroid_x, centroid_y, label_x, label_y, face_count) = _gpu_face_walk(half_edge_graph)

    if face_count == 0:
        return None

    # Step 4: Select bounded positive-area faces (no coverage labeling needed)
    # For make_valid, we want all bounded faces with positive area — these
    # are the valid polygon faces extracted from the self-intersection split.
    d_select_mask = (bounded_mask != 0) & (signed_area > 0)
    selected_face_indices = cp.asnumpy(
        cp.flatnonzero(d_select_mask).astype(cp.int32)
    )

    if selected_face_indices.size == 0:
        return None

    # Build OverlayFaceTable for the assembly function
    faces = OverlayFaceTable(
        runtime_selection=runtime_selection,
        device_state=OverlayFaceDeviceState(
            face_offsets=face_offsets,
            face_edge_ids=face_edge_ids,
            bounded_mask=bounded_mask,
            signed_area=signed_area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            left_covered=cp.ones(face_count, dtype=cp.int8),
            right_covered=cp.zeros(face_count, dtype=cp.int8),
        ),
        _face_count=face_count,
    )

    # Step 5: Assemble output polygons from selected faces
    result = _build_polygon_output_from_faces_gpu(
        half_edge_graph, faces, selected_face_indices,
    )
    return result


# ---------------------------------------------------------------------------
# Main GPU repair entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPURepairResult:
    """Result of GPU make_valid repair."""
    repaired_owned: OwnedGeometryArray | None  # device-resident merged result
    repaired_count: int
    gpu_phases_used: tuple[str, ...]
    still_invalid_rows: np.ndarray  # global row indices GPU couldn't fix


def _extract_batch_coords_device(
    d_buffer,
    invalid_family_rows: np.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray] | None:
    """Device-side extraction of invalid polygon coordinates into contiguous batch.

    Like _extract_batch_coords but operates entirely on device arrays via
    _device_take_family_buffer. No D->H transfers except two tiny scalars
    (total_rings, total_coords) for allocation sizing.
    """
    from vibespatial.geometry.owned import _device_take_family_buffer

    if invalid_family_rows.size == 0:
        return None

    d_rows = cp.asarray(invalid_family_rows.astype(np.int32))
    taken = _device_take_family_buffer(d_buffer, GeometryFamily.POLYGON, d_rows)

    if taken.x.size == 0:
        return None

    return taken.x, taken.y, taken.ring_offsets, taken.geometry_offsets


def _build_batch_repaired_device(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray | None:
    """Build a device-resident OwnedGeometryArray from batch device coordinates.

    Device-resident replacement for _build_batch_repaired_owned. Filters
    degenerate rings (< 4 vertices) on device and produces a device-resident
    result via build_device_resident_owned. No D->H transfer for coordinate data.
    """
    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    if d_x.size == 0 or polygon_count == 0:
        return None

    # Filter degenerate rings on device (< 4 vertices)
    d_ring_lens = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_valid_rings = d_ring_lens >= 4

    if not bool(cp.any(d_valid_rings)):
        return None

    d_valid_ring_indices = cp.flatnonzero(d_valid_rings).astype(cp.int32)
    valid_ring_count = int(d_valid_ring_indices.size)

    if valid_ring_count == 0:
        return None

    # Gather valid ring coordinate ranges and build new ring offsets
    from vibespatial.geometry.owned import _device_gather_offset_slices

    coords_2d = (
        cp.column_stack([d_x, d_y])
        if d_x.size
        else cp.empty((0, 2), dtype=cp.float64)
    )
    gathered_coords, new_ring_offsets = _device_gather_offset_slices(
        coords_2d, d_ring_offsets, d_valid_ring_indices,
    )
    new_x = gathered_coords[:, 0].copy() if gathered_coords.size else cp.empty(0, dtype=cp.float64)
    new_y = gathered_coords[:, 1].copy() if gathered_coords.size else cp.empty(0, dtype=cp.float64)

    # Build new geom offsets: count valid rings per polygon
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_valid_ring_indices, side="right",
    ).astype(cp.int32)
    d_rings_per_poly = cp.bincount(d_poly_of_ring, minlength=polygon_count).astype(cp.int32)
    new_geom_offsets = cp.zeros(polygon_count + 1, dtype=cp.int32)
    cp.cumsum(d_rings_per_poly, out=new_geom_offsets[1:])

    # Validity: polygons with zero valid rings are invalid
    d_poly_valid = d_rings_per_poly > 0
    h_validity = cp.asnumpy(d_poly_valid)  # tiny: polygon_count bools
    h_tags = np.full(polygon_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    h_family_row_offsets = np.arange(polygon_count, dtype=np.int32)

    device_families = {
        GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=~d_poly_valid,
            ring_offsets=new_ring_offsets,
            bounds=None,
        ),
    }
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=polygon_count,
        tags=h_tags,
        validity=h_validity,
        family_row_offsets=h_family_row_offsets,
    )
    result.runtime_history.append(runtime_selection)
    return result


def _device_scatter_repaired(
    original_owned: OwnedGeometryArray,
    repaired_batch: OwnedGeometryArray,
    family_name: str,
    invalid_family_rows: np.ndarray,
    fam_to_global: dict[int, int],
) -> OwnedGeometryArray:
    """Merge repaired polygon family buffer back into original on device.

    Takes the original device-resident OwnedGeometryArray and the batch
    repair result, then produces a new device-resident OwnedGeometryArray
    where repaired rows are scattered back into the correct positions.
    Uses existing _device_take_family_buffer and
    _concat_device_family_buffers from owned.py for all buffer work.
    No D->H coordinate transfers.
    """
    from vibespatial.geometry.owned import (
        _concat_device_family_buffers,
        _device_take_family_buffer,
        build_device_resident_owned,
    )

    ds = original_owned.device_state
    family = GeometryFamily(family_name)
    family_tag = FAMILY_TAGS[family_name]
    orig_d_buf = ds.families[family]
    orig_polygon_count = int(orig_d_buf.geometry_offsets.size) - 1

    # Build the set of valid (non-repaired) family rows
    invalid_set = set(invalid_family_rows.tolist())
    valid_family_rows = np.array(
        [i for i in range(orig_polygon_count) if i not in invalid_set],
        dtype=np.int32,
    )

    # Take valid rows from original, repaired rows from batch
    bufs_to_concat = []
    valid_count = 0

    if valid_family_rows.size > 0:
        d_valid_rows = cp.asarray(valid_family_rows)
        valid_buf = _device_take_family_buffer(orig_d_buf, family, d_valid_rows)
        bufs_to_concat.append(valid_buf)
        valid_count = int(valid_buf.geometry_offsets.size) - 1

    # Get the repaired batch's device family buffers.
    # Repair can change family: a self-intersecting Polygon becomes a
    # MultiPolygon after repolygonization. Collect buffers from ALL
    # families present in the repaired batch.
    repair_ds = repaired_batch.device_state if repaired_batch.device_state is not None else None
    if repair_ds is None and hasattr(repaired_batch, '_ensure_device_state'):
        repaired_batch._ensure_device_state()
        repair_ds = repaired_batch.device_state

    # Collect per-family repair buffers and their row counts
    repair_families: dict[GeometryFamily, object] = {}
    if repair_ds is not None:
        for rfam, rbuf in repair_ds.families.items():
            rfam_count = int(rbuf.geometry_offsets.size) - 1 if rbuf.geometry_offsets is not None else 0
            if rfam_count > 0:
                repair_families[rfam] = rbuf

    # Same-family repair buffer (polygon -> polygon)
    same_family_buf = repair_families.get(family)
    if same_family_buf is not None:
        bufs_to_concat.append(same_family_buf)

    if not bufs_to_concat and not repair_families:
        return original_owned

    # Build new device families
    new_device_families = {}
    for fam, d_buf in ds.families.items():
        if fam == family:
            # Merge valid original rows + same-family repair rows
            if bufs_to_concat:
                new_device_families[fam] = _concat_device_family_buffers(family, bufs_to_concat)
            else:
                # All original rows were invalid and none repaired to same family
                # Only add if valid_count > 0
                if valid_count > 0:
                    new_device_families[fam] = bufs_to_concat[0] if bufs_to_concat else d_buf
        else:
            new_device_families[fam] = d_buf

    # Cross-family repair: if repair produced geometries in a different family
    # (e.g., Polygon input -> MultiPolygon output), add those to the
    # appropriate family buffer.
    for rfam, rbuf in repair_families.items():
        if rfam == family:
            continue  # already handled above
        if rfam in new_device_families:
            # Append to existing family buffer
            new_device_families[rfam] = _concat_device_family_buffers(
                rfam, [new_device_families[rfam], rbuf],
            )
        else:
            new_device_families[rfam] = rbuf

    # Rebuild metadata: tags and validity need remapping.
    h_tags = original_owned.tags.copy()
    h_validity = original_owned.validity.copy()
    h_fro = original_owned.family_row_offsets.copy()

    # Count same-family repair rows
    same_family_repair_count = (
        int(same_family_buf.geometry_offsets.size) - 1
        if same_family_buf is not None else 0
    )

    # Build old->new position mapping for the original family
    old_to_new = np.empty(orig_polygon_count, dtype=np.int32)
    for i, vr in enumerate(valid_family_rows):
        old_to_new[vr] = i

    # Track which repaired rows went to a different family
    # Build a map: batch_repair_index -> (target_family, family_row_offset)
    cross_family_map: dict[int, tuple[GeometryFamily, int]] = {}
    for rfam, rbuf in repair_families.items():
        if rfam == family:
            # Same-family: indices 0..same_family_repair_count-1
            continue
        rfam_count = int(rbuf.geometry_offsets.size) - 1
        # Cross-family rows: their positions in the target family buffer
        # are appended after whatever was already there.
        existing_count = 0
        if rfam in ds.families:
            existing_count = int(ds.families[rfam].geometry_offsets.size) - 1
        for i in range(rfam_count):
            cross_family_map[i] = (rfam, existing_count + i)

    # Remap family_row_offsets and tags for repaired rows
    poly_global_mask = h_tags == family_tag
    poly_globals = np.flatnonzero(poly_global_mask)

    if cross_family_map:
        # The repaired batch has rows in a different family.
        # For each invalid family row, determine if it went to same-family
        # or cross-family. The repaired batch order matches invalid_family_rows
        # order; rows in the batch are assigned to families by the batch output.
        #
        # Simple heuristic: if same_family_repair_count == 0 and there is
        # exactly one cross-family, ALL repaired rows went to that family.
        if same_family_repair_count == 0 and len(repair_families) == 1:
            target_fam = next(iter(repair_families))
            target_tag = FAMILY_TAGS[target_fam]
            target_count = int(repair_families[target_fam].geometry_offsets.size) - 1
            existing_in_target = 0
            if target_fam in ds.families:
                existing_in_target = int(ds.families[target_fam].geometry_offsets.size) - 1

            # Remap valid original rows
            if poly_globals.size > 0:
                old_offsets = h_fro[poly_globals]
                valid_idx = (old_offsets >= 0) & (old_offsets < orig_polygon_count)
                h_fro[poly_globals[valid_idx]] = old_to_new[old_offsets[valid_idx]]

            # Remap repaired rows: change tag and family_row_offset
            for i, rr in enumerate(invalid_family_rows):
                g = fam_to_global.get(int(rr))
                if g is not None:
                    h_tags[g] = target_tag
                    h_fro[g] = existing_in_target + i
                    if i < target_count:
                        h_validity[g] = True
        else:
            # Mixed output: some same-family, some cross-family.
            # Fall through to same-family-only mapping for now.
            for i, rr in enumerate(invalid_family_rows):
                old_to_new[rr] = valid_count + i
            if poly_globals.size > 0:
                old_offsets = h_fro[poly_globals]
                valid_idx = (old_offsets >= 0) & (old_offsets < orig_polygon_count)
                h_fro[poly_globals[valid_idx]] = old_to_new[old_offsets[valid_idx]]
    else:
        # All repairs stayed in the same family
        for i, rr in enumerate(invalid_family_rows):
            old_to_new[rr] = valid_count + i
        if poly_globals.size > 0:
            old_offsets = h_fro[poly_globals]
            valid_idx = (old_offsets >= 0) & (old_offsets < orig_polygon_count)
            h_fro[poly_globals[valid_idx]] = old_to_new[old_offsets[valid_idx]]

        # Update validity for repaired rows that produced empty/invalid output
        merged_family_buf = new_device_families.get(family)
        if same_family_buf is not None and merged_family_buf is not None and hasattr(merged_family_buf, "empty_mask"):
            d_merged_empty = merged_family_buf.empty_mask
            h_merged_empty = cp.asnumpy(d_merged_empty)  # tiny: polygon_count bools
            for gi, fro_val in zip(
                np.flatnonzero(poly_global_mask), h_fro[poly_globals],
            ):
                if 0 <= fro_val < len(h_merged_empty) and h_merged_empty[fro_val]:
                    h_validity[gi] = False

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=original_owned.row_count,
        tags=h_tags,
        validity=h_validity,
        family_row_offsets=h_fro,
    )


def _build_batch_repaired_owned(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray | None:
    """Build an OwnedGeometryArray from batch device coordinates.

    Used when Phase B repairs (close/dedup/orient) were sufficient and no
    self-intersection splitting was needed. Transfers coordinates to host
    in bulk (single D2H per array) and builds GeoArrow polygon buffers.
    """
    h_x = cp.asnumpy(d_x).astype(np.float64, copy=False)
    h_y = cp.asnumpy(d_y).astype(np.float64, copy=False)
    h_ro = cp.asnumpy(d_ring_offsets).astype(np.int32, copy=False)
    h_go = cp.asnumpy(d_geom_offsets).astype(np.int32, copy=False)

    if h_x.size == 0 or polygon_count == 0:
        return None

    # Filter degenerate rings (< 4 vertices) on host in bulk
    ring_lens = np.diff(h_ro)
    valid_rings = ring_lens >= 4

    if not np.any(valid_rings):
        return None

    # Build per-polygon ring counts and validity
    validity = np.ones(polygon_count, dtype=bool)
    tags = np.full(polygon_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(polygon_count, dtype=np.int32)

    # Rebuild ring/geom offsets filtering invalid rings
    new_ring_offsets_list = []
    new_geom_offsets = [0]
    new_x_chunks = []
    new_y_chunks = []
    coord_cursor = 0

    for p in range(polygon_count):
        r_start = int(h_go[p])
        r_end = int(h_go[p + 1])
        poly_ring_count = 0
        for r in range(r_start, r_end):
            if valid_rings[r]:
                c_start = int(h_ro[r])
                c_end = int(h_ro[r + 1])
                new_ring_offsets_list.append(coord_cursor)
                new_x_chunks.append(h_x[c_start:c_end])
                new_y_chunks.append(h_y[c_start:c_end])
                coord_cursor += c_end - c_start
                poly_ring_count += 1
        if poly_ring_count == 0:
            validity[p] = False
        new_geom_offsets.append(len(new_ring_offsets_list))

    new_ring_offsets_list.append(coord_cursor)

    if coord_cursor == 0:
        return None

    out_x = np.concatenate(new_x_chunks) if new_x_chunks else np.empty(0, dtype=np.float64)
    out_y = np.concatenate(new_y_chunks) if new_y_chunks else np.empty(0, dtype=np.float64)
    out_ring_offsets = np.asarray(new_ring_offsets_list, dtype=np.int32)
    out_geom_offsets = np.asarray(new_geom_offsets, dtype=np.int32)

    # Compute per-polygon bounds via vectorised operations
    bounds = np.empty((polygon_count, 4), dtype=np.float64)
    for p in range(polygon_count):
        gr_start = int(out_geom_offsets[p])
        gr_end = int(out_geom_offsets[p + 1])
        if gr_start >= gr_end or not validity[p]:
            bounds[p] = (0.0, 0.0, 0.0, 0.0)
            continue
        c_start = int(out_ring_offsets[gr_start])
        c_end = int(out_ring_offsets[gr_end])
        bounds[p] = (
            out_x[c_start:c_end].min(),
            out_y[c_start:c_end].min(),
            out_x[c_start:c_end].max(),
            out_y[c_start:c_end].max(),
        )

    families = {
        GeometryFamily.POLYGON: FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=polygon_count,
            x=out_x,
            y=out_y,
            geometry_offsets=out_geom_offsets,
            empty_mask=~validity,
            ring_offsets=out_ring_offsets,
            bounds=bounds,
        ),
    }
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
        runtime_history=[runtime_selection],
    )


def _extract_batch_coords(
    buffer: FamilyGeometryBuffer,
    geom_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    invalid_family_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract coordinates for all invalid polygons into contiguous batch arrays.

    Returns (batch_x, batch_y, batch_ring_offsets, batch_geom_offsets) as
    host NumPy arrays ready for device upload, or None if no data.
    """
    if invalid_family_rows.size == 0:
        return None

    # Vectorised ring/coord range extraction for all invalid polygons
    r_starts = geom_offsets[invalid_family_rows]      # first ring index per polygon
    r_ends = geom_offsets[invalid_family_rows + 1]    # one-past-last ring index
    ring_counts_per_poly = r_ends - r_starts

    # Skip polygons with zero rings
    has_rings = ring_counts_per_poly > 0
    if not np.any(has_rings):
        return None

    # Build batch ring offsets: for each polygon, remap its ring offsets to
    # be relative to the batch coordinate array.
    batch_ring_list = []
    batch_geom_list = [0]
    coord_cursor = 0

    for i in range(invalid_family_rows.size):
        rs = int(r_starts[i])
        re = int(r_ends[i])
        if rs >= re:
            batch_geom_list.append(len(batch_ring_list))
            continue
        local_ring_offs = ring_offsets[rs:re + 1]
        coord_start = int(local_ring_offs[0])
        local_ring_offs_rebased = local_ring_offs - coord_start + coord_cursor
        batch_ring_list.extend(local_ring_offs_rebased[:-1].tolist())
        coord_cursor = int(local_ring_offs_rebased[-1])
        batch_geom_list.append(len(batch_ring_list))

    # Sentinel for ring_offsets
    batch_ring_list.append(coord_cursor)
    batch_ring_offsets = np.asarray(batch_ring_list, dtype=np.int32)
    batch_geom_offsets = np.asarray(batch_geom_list, dtype=np.int32)
    total_coords = coord_cursor

    if total_coords == 0:
        return None

    # Gather coordinates: extract slices for each invalid polygon
    batch_x = np.empty(total_coords, dtype=np.float64)
    batch_y = np.empty(total_coords, dtype=np.float64)
    buf_x = np.asarray(buffer.x, dtype=np.float64)
    buf_y = np.asarray(buffer.y, dtype=np.float64)

    dst = 0
    for i in range(invalid_family_rows.size):
        rs = int(r_starts[i])
        re = int(r_ends[i])
        if rs >= re:
            continue
        c_start = int(ring_offsets[rs])
        c_end = int(ring_offsets[re])
        n = c_end - c_start
        batch_x[dst:dst + n] = buf_x[c_start:c_end]
        batch_y[dst:dst + n] = buf_y[c_start:c_end]
        dst += n

    return batch_x, batch_y, batch_ring_offsets, batch_geom_offsets


def gpu_repair_invalid_polygons(
    owned: OwnedGeometryArray,
    invalid_rows: np.ndarray,
    geometries: np.ndarray | None = None,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
) -> GPURepairResult | None:
    """GPU-resident batch repair of invalid polygon geometries (Phase 16).

    Implements the full make_valid pipeline on GPU with batch processing:
    1. Collect all invalid polygon coordinates into one contiguous batch
    2. Phase B: Close rings, remove duplicates, fix orientation (batched)
    3. Phase A+C: Detect and split self-intersections (batched)
    4. Phase D: Re-polygonize via overlay half-edge/face-walk pipeline (batched)
    5. Merge repaired rows back into original owned array on device

    When ``owned.device_state`` is available, the entire pipeline stays
    device-resident — no D->H coordinate transfers.  The result carries
    ``repaired_owned`` so callers can stay on device (ADR-0005).

    Returns None if GPU repair is not applicable (no GPU, no polygon families,
    or CuPy not available).

    Parameters
    ----------
    owned : OwnedGeometryArray with device_state
    invalid_rows : indices of invalid rows to repair
    geometries : optional shapely geometry array (unused in device path)
    method : repair method (only "linework" supported on GPU)
    keep_collapsed : whether to keep collapsed geometries
    """
    if cp is None:
        return None

    from vibespatial.runtime import has_gpu_runtime
    if not has_gpu_runtime():
        return None

    if invalid_rows.size == 0:
        return GPURepairResult(
            repaired_owned=owned,
            repaired_count=0,
            gpu_phases_used=(),
            still_invalid_rows=np.asarray([], dtype=np.int32),
        )

    # Only repair polygon family rows on GPU; non-polygon invalids are
    # collected into still_invalid_rows for the caller to handle.
    polygon_families = set()
    for family_name in ("polygon", "multipolygon"):
        if family_name in owned.families:
            polygon_families.add(family_name)

    if not polygon_families:
        return None

    # Compile repair kernels
    try:
        kernels = _compile_repair_kernels()
    except Exception:
        return None

    # Determine whether we have a device-resident path
    use_device_path = (
        owned.device_state is not None
        and any(
            GeometryFamily(fn) in owned.device_state.families
            for fn in polygon_families
        )
    )

    runtime_sel = RuntimeSelection(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        reason="make_valid GPU batch repair pipeline (Phase 16)",
    )

    phases_used: list[str] = []
    gpu_repaired_count = 0
    repaired_global_rows: set[int] = set()
    # Start with the original owned; each family merge updates it
    merged_owned = owned

    for family_name in polygon_families:
        buffer = owned.families[family_name]
        if not hasattr(buffer, "ring_offsets") or buffer.ring_offsets is None:
            continue

        # Map invalid rows to family rows (vectorised, no Python loop)
        family_tag = FAMILY_TAGS[family_name]
        tag_match = owned.tags == family_tag
        invalid_mask = np.zeros(len(owned.tags), dtype=bool)
        invalid_mask[invalid_rows] = True
        global_invalid = np.flatnonzero(tag_match & invalid_mask)
        if global_invalid.size == 0:
            continue

        fam_row_offsets = owned.family_row_offsets[global_invalid]
        polygon_count = int(buffer.geometry_offsets.size) - 1 if hasattr(buffer, "geometry_offsets") else 0
        if polygon_count <= 0:
            continue
        valid_fro = (fam_row_offsets >= 0) & (fam_row_offsets < polygon_count)
        global_invalid = global_invalid[valid_fro]
        fam_row_offsets = fam_row_offsets[valid_fro]
        if global_invalid.size == 0:
            continue

        # Unique family rows and their global row mapping
        invalid_family_rows = np.unique(fam_row_offsets)
        fam_to_global = {}
        for gi, fro in zip(global_invalid, fam_row_offsets):
            fam_to_global.setdefault(int(fro), int(gi))

        try:
            if use_device_path:
                # --- Device-resident path: no D->H transfers ---
                ds = owned.device_state
                family = GeometryFamily(family_name)
                d_buf = ds.families[family]

                batch = _extract_batch_coords_device(d_buf, invalid_family_rows)
                if batch is None:
                    continue
                d_x, d_y, d_ring_offsets, d_geom_offsets = batch
            else:
                # --- Host fallback path (no device_state) ---
                ring_offsets = np.asarray(buffer.ring_offsets, dtype=np.int32)
                geom_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int32)
                batch = _extract_batch_coords(
                    buffer, geom_offsets, ring_offsets, invalid_family_rows,
                )
                if batch is None:
                    continue
                batch_x, batch_y, batch_ring_offsets, batch_geom_offsets = batch
                d_x = cp.asarray(np.ascontiguousarray(batch_x))
                d_y = cp.asarray(np.ascontiguousarray(batch_y))
                d_ring_offsets = cp.asarray(batch_ring_offsets)
                d_geom_offsets = cp.asarray(batch_geom_offsets)

            batch_ring_count = int(d_ring_offsets.size) - 1
            batch_poly_count = int(d_geom_offsets.size) - 1

            if batch_ring_count == 0 or batch_poly_count == 0:
                continue

            # --- Step 2: Phase B — batched simple repair ---
            d_x, d_y, d_ring_offsets = _gpu_close_rings(
                d_x, d_y, d_ring_offsets,
                batch_ring_count, kernels,
            )
            phases_used.append("close_rings")

            d_x, d_y, d_ring_offsets = _gpu_remove_duplicate_vertices(
                d_x, d_y, d_ring_offsets,
                batch_ring_count, kernels,
            )
            phases_used.append("remove_duplicates")

            d_x, d_y = _gpu_fix_ring_orientation(
                d_x, d_y, d_ring_offsets,
                d_geom_offsets, batch_ring_count, batch_poly_count, kernels,
            )
            phases_used.append("fix_orientation")

            # --- Step 3: Phase A+C — batched intersection detection + splitting ---
            d_x, d_y, d_ring_offsets, had_splits = \
                _split_self_intersections_gpu(
                    d_x, d_y, d_ring_offsets,
                    batch_ring_count, kernels,
                )
            if had_splits:
                phases_used.append("split_intersections")

            # --- Step 4: Phase D — batched repolygonization ---
            if had_splits:
                batch_ring_count = int(d_ring_offsets.size) - 1
                batch_result = _repolygonize_from_split_rings(
                    d_x, d_y, d_ring_offsets,
                    d_geom_offsets, batch_ring_count, batch_poly_count,
                    runtime_selection=runtime_sel,
                )
                phases_used.append("repolygonize")
            else:
                # No self-intersections — simple repairs were sufficient.
                batch_result = _build_batch_repaired_device(
                    d_x, d_y, d_ring_offsets, d_geom_offsets,
                    batch_ring_count, batch_poly_count, runtime_sel,
                )

            # --- Step 5: Merge repaired batch back into owned on device ---
            if batch_result is not None:
                merged_owned = _device_scatter_repaired(
                    merged_owned, batch_result, family_name,
                    invalid_family_rows, fam_to_global,
                )
                gpu_repaired_count += batch_poly_count
                for fro in invalid_family_rows:
                    g = fam_to_global.get(int(fro))
                    if g is not None:
                        repaired_global_rows.add(g)

        except Exception:
            # GPU batch repair failed for this family; rows stay in still_invalid
            pass

    # Collect rows that GPU couldn't fix (non-polygon families, GPU exceptions)
    still_invalid = np.asarray(
        [int(r) for r in invalid_rows if r not in repaired_global_rows],
        dtype=np.int32,
    )

    return GPURepairResult(
        repaired_owned=merged_owned,
        repaired_count=gpu_repaired_count,
        gpu_phases_used=tuple(set(phases_used)),
        still_invalid_rows=still_invalid,
    )

