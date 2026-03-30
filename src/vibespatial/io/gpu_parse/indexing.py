"""Fused ingest + spatial index: build packed Hilbert R-tree during parsing.

Provides ``build_spatial_index`` which takes flat coordinate arrays and
geometry offsets (the output of any GPU reader stage) and produces a
packed Hilbert R-tree entirely on the device.  No separate indexing pass
is required.

ADR-0002: COARSE kernel class — bounding boxes stay fp64 because they are
memory-bound (not compute-bound) and fp32 rounding shrinks bounds, causing
false negatives in spatial filtering.

ADR-0033 tier classification:
  - Tier 1 (NVRTC): compute_feature_bounds, compute_hilbert_codes
  - Tier 2 (CuPy): R-tree node construction (element-wise gather/scatter)
  - Tier 3a (CCCL): sort_pairs for Hilbert-order sorting
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import sort_pairs
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------
# Hilbert curve encoding kernel — implements the threadlocalmutex/rawrunprotected
# algorithm in CUDA.  Produces 32-bit Hilbert codes from normalized integer
# coordinates on a 2^16 grid.
#
# The algorithm is a bitwise prefix-scan approach that produces the same
# codes as the CPU _encode() in hilbert_curve.py.  Reference:
# http://threadlocalmutex.com/ (public domain).
from vibespatial.io.gpu_parse.indexing_kernels import (
    _BOUNDS_KERNEL_NAMES,
    _BOUNDS_KERNEL_SOURCE,
    _HILBERT_KERNEL_NAMES,
    _HILBERT_KERNEL_SOURCE,
)

# ---------------------------------------------------------------------------
# Warmup registration (ADR-0034)
# ---------------------------------------------------------------------------

request_nvrtc_warmup([
    ("fused-index-bounds", _BOUNDS_KERNEL_SOURCE, _BOUNDS_KERNEL_NAMES),
    ("fused-index-hilbert", _HILBERT_KERNEL_SOURCE, _HILBERT_KERNEL_NAMES),
])

# CCCL warmup for radix sort used in Hilbert ordering
request_warmup(["radix_sort_u32_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _bounds_kernels():
    """Compile (cached) the feature bounds kernel."""
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("fused-index-bounds", _BOUNDS_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_BOUNDS_KERNEL_SOURCE,
        kernel_names=_BOUNDS_KERNEL_NAMES,
    )


def _hilbert_kernels():
    """Compile (cached) the Hilbert code kernel."""
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("fused-index-hilbert", _HILBERT_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_HILBERT_KERNEL_SOURCE,
        kernel_names=_HILBERT_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuSpatialIndex:
    """Packed Hilbert R-tree built entirely on the GPU.

    All arrays are CuPy device arrays.  No host materialization occurs
    during construction.

    Attributes
    ----------
    d_sorted_indices : cp.ndarray (int32, n_features)
        Feature indices in Hilbert curve order.
    d_node_bounds : cp.ndarray (float64, (n_nodes, 4))
        Per-node bounding boxes: min_x, min_y, max_x, max_y.
    d_node_children : cp.ndarray (int32, (n_internal_nodes, node_capacity))
        Child pointers per internal node.  Leaf nodes are implicit (groups
        of ``node_capacity`` features in sorted order).
    d_feature_bounds : cp.ndarray (float64, (n_features, 4))
        Per-feature bounding boxes (in original feature order).
    d_hilbert_codes : cp.ndarray (uint32, n_features)
        Hilbert codes per feature (in original feature order).
    n_features : int
        Number of features.
    n_nodes : int
        Total number of nodes (leaves + internal).
    n_leaf_nodes : int
        Number of leaf nodes.
    node_capacity : int
        Maximum children per node (fan-out).
    """

    d_sorted_indices: object   # cp.ndarray int32
    d_node_bounds: object      # cp.ndarray float64 (n_nodes, 4)
    d_node_children: object    # cp.ndarray int32 (n_internal_nodes, node_capacity)
    d_feature_bounds: object   # cp.ndarray float64 (n_features, 4)
    d_hilbert_codes: object    # cp.ndarray uint32
    n_features: int
    n_nodes: int
    n_leaf_nodes: int
    node_capacity: int


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _compute_bounds_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    geometry_offsets: cp.ndarray,
    n_features: int,
) -> cp.ndarray:
    """Compute per-feature bounding boxes on the GPU.

    Returns device array of shape (n_features, 4) with columns
    [min_x, min_y, max_x, max_y], dtype float64.
    """
    runtime = get_cuda_runtime()
    kernels = _bounds_kernels()
    kernel = kernels["compute_feature_bounds"]
    ptr = runtime.pointer

    d_bounds = cp.empty(n_features * 4, dtype=cp.float64)

    params = (
        (ptr(d_x), ptr(d_y), ptr(geometry_offsets), ptr(d_bounds), n_features),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, n_features)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    return d_bounds.reshape(n_features, 4)


def _compute_hilbert_codes_gpu(
    d_bounds: cp.ndarray,
    n_features: int,
) -> cp.ndarray:
    """Compute 32-bit Hilbert codes from feature bounding boxes on the GPU.

    Parameters
    ----------
    d_bounds : cp.ndarray, shape (n_features, 4), float64
        Per-feature bounding boxes.
    n_features : int
        Number of features.

    Returns
    -------
    cp.ndarray, shape (n_features,), uint32
        Hilbert codes.
    """
    runtime = get_cuda_runtime()
    kernels = _hilbert_kernels()
    kernel = kernels["compute_hilbert_codes"]
    ptr = runtime.pointer

    # Compute total extent from bounds.  The 4 scalar D->H transfers below
    # are unavoidable: extent values are passed as kernel parameters (not
    # pointers) to avoid an extra indirection in the hot loop.  This is the
    # same pattern used by the existing morton_keys_from_bounds kernel.
    d_bounds_flat = d_bounds.reshape(-1, 4)
    extent_minx = float(cp.nanmin(d_bounds_flat[:, 0]))
    extent_miny = float(cp.nanmin(d_bounds_flat[:, 1]))
    extent_maxx = float(cp.nanmax(d_bounds_flat[:, 2]))
    extent_maxy = float(cp.nanmax(d_bounds_flat[:, 3]))

    d_hilbert_codes = cp.empty(n_features, dtype=cp.uint32)

    params = (
        (
            ptr(d_bounds),
            extent_minx,
            extent_miny,
            extent_maxx,
            extent_maxy,
            ptr(d_hilbert_codes),
            n_features,
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
    grid, block = runtime.launch_config(kernel, n_features)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    return d_hilbert_codes


def _build_packed_rtree(
    d_sorted_indices: cp.ndarray,
    d_feature_bounds: cp.ndarray,
    n_features: int,
    node_capacity: int = 16,
) -> tuple[cp.ndarray, cp.ndarray, int, int]:
    """Build a packed R-tree from Hilbert-sorted features using CuPy.

    The tree is built bottom-up:
      1. Leaf level: consecutive groups of ``node_capacity`` features
         (in Hilbert-sorted order).
      2. Each subsequent level groups ``node_capacity`` children.
      3. Repeat until a single root node remains.

    Parameters
    ----------
    d_sorted_indices : cp.ndarray int32 (n_features,)
        Feature indices in Hilbert order.
    d_feature_bounds : cp.ndarray float64 (n_features, 4)
        Per-feature bounding boxes.
    n_features : int
    node_capacity : int
        Fan-out per node.

    Returns
    -------
    (d_node_bounds, d_node_children, n_nodes, n_leaf_nodes) : tuple
        - d_node_bounds: float64 (n_nodes, 4)
        - d_node_children: int32 (n_internal_nodes, node_capacity), -1 for unused slots
        - n_nodes: total node count
        - n_leaf_nodes: leaf node count
    """
    B = node_capacity

    # -- Leaf level: group sorted features into leaves --
    n_leaves = int((n_features + B - 1) // B)

    # Gather bounds in Hilbert order
    d_sorted_bounds = d_feature_bounds[d_sorted_indices]  # (n_features, 4)

    # Compute leaf bounding boxes by reducing groups of B features.
    # Pad to a multiple of B for uniform reshaping.
    padded_n = n_leaves * B
    if padded_n > n_features:
        nan_pad = cp.full((padded_n - n_features, 4), cp.nan, dtype=cp.float64)
        d_padded_bounds = cp.concatenate([d_sorted_bounds, nan_pad], axis=0)
    else:
        d_padded_bounds = d_sorted_bounds

    d_grouped = d_padded_bounds.reshape(n_leaves, B, 4)
    # Leaf bounds: nanmin of columns 0,1 and nanmax of columns 2,3
    d_leaf_bounds = cp.empty((n_leaves, 4), dtype=cp.float64)
    d_leaf_bounds[:, 0] = cp.nanmin(d_grouped[:, :, 0], axis=1)
    d_leaf_bounds[:, 1] = cp.nanmin(d_grouped[:, :, 1], axis=1)
    d_leaf_bounds[:, 2] = cp.nanmax(d_grouped[:, :, 2], axis=1)
    d_leaf_bounds[:, 3] = cp.nanmax(d_grouped[:, :, 3], axis=1)

    # Build internal levels bottom-up
    all_level_bounds = [d_leaf_bounds]
    all_level_children = []  # Each entry: int32 (n_nodes_at_level, B)

    current_bounds = d_leaf_bounds
    current_count = n_leaves
    # Offset into the flat node array where this level starts
    # Leaves are at the end, internal nodes at the beginning (reverse order).
    # We will concatenate levels at the end.

    while current_count > 1:
        n_parents = int((current_count + B - 1) // B)
        padded_count = n_parents * B

        # Pad current level bounds for uniform reshape
        if padded_count > current_count:
            nan_pad = cp.full((padded_count - current_count, 4), cp.nan, dtype=cp.float64)
            padded_bounds = cp.concatenate([current_bounds, nan_pad], axis=0)
        else:
            padded_bounds = current_bounds

        grouped = padded_bounds.reshape(n_parents, B, 4)
        parent_bounds = cp.empty((n_parents, 4), dtype=cp.float64)
        parent_bounds[:, 0] = cp.nanmin(grouped[:, :, 0], axis=1)
        parent_bounds[:, 1] = cp.nanmin(grouped[:, :, 1], axis=1)
        parent_bounds[:, 2] = cp.nanmax(grouped[:, :, 2], axis=1)
        parent_bounds[:, 3] = cp.nanmax(grouped[:, :, 3], axis=1)

        # Build child pointer array: each parent points to B children in the
        # previous level.  We compute global node indices after assembly.
        # For now, store local indices into the child level.
        # Vectorized: build (n_parents, B) grid of child indices in one shot.
        parent_ids = cp.arange(n_parents, dtype=cp.int32)
        slot_ids = cp.arange(B, dtype=cp.int32)
        # (n_parents, B) grid: parent_ids[:, None] * B + slot_ids[None, :]
        child_grid = parent_ids[:, None] * B + slot_ids[None, :]
        child_ptrs = cp.where(child_grid < current_count, child_grid, -1).astype(cp.int32)

        all_level_bounds.append(parent_bounds)
        all_level_children.append(child_ptrs)
        current_bounds = parent_bounds
        current_count = n_parents

    # Assemble the node arrays.
    # Layout: root at index 0, then level 1, level 2, ..., then leaves.
    # all_level_bounds is [leaves, level_1_parents, level_2_parents, ..., root]
    # We reverse to get [root, ..., level_1_parents, leaves].
    all_level_bounds.reverse()
    all_level_children.reverse()

    # Compute level offsets in the flat node array
    level_sizes = [int(lb.shape[0]) for lb in all_level_bounds]
    level_offsets = []
    offset = 0
    for s in level_sizes:
        level_offsets.append(offset)
        offset += s

    n_nodes = offset
    n_leaf_nodes = n_leaves

    # Concatenate all bounds
    d_node_bounds = cp.concatenate(all_level_bounds, axis=0)  # (n_nodes, 4)

    # Fix up child pointers to use global node indices.
    # Internal nodes are in levels 0..len(all_level_children)-1.
    # Level i's children are in level i+1.
    fixed_children = []
    for level_idx, child_ptrs in enumerate(all_level_children):
        child_level_offset = level_offsets[level_idx + 1]
        # Add the child level offset to valid entries
        valid = child_ptrs >= 0
        fixed = cp.where(valid, child_ptrs + child_level_offset, -1)
        fixed_children.append(fixed)

    if fixed_children:
        d_node_children = cp.concatenate(fixed_children, axis=0)
    else:
        # Single leaf node, no internal structure needed
        d_node_children = cp.empty((0, B), dtype=cp.int32)

    return d_node_bounds, d_node_children, n_nodes, n_leaf_nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_spatial_index(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    geometry_offsets: cp.ndarray,
    *,
    node_capacity: int = 16,
) -> GpuSpatialIndex:
    """Build a packed Hilbert R-tree from coordinate arrays.

    All computation stays on the GPU.  The returned ``GpuSpatialIndex``
    contains device-resident arrays only — no host round-trips occur
    during construction.

    Parameters
    ----------
    d_x : cp.ndarray, float64
        Flat X coordinate array on the device.
    d_y : cp.ndarray, float64
        Flat Y coordinate array on the device.
    geometry_offsets : cp.ndarray, int32
        Offset array of length ``n_features + 1`` delimiting each
        feature's coordinate span.  ``feature[i]`` owns coordinates
        ``d_x[geometry_offsets[i]:geometry_offsets[i+1]]``.
    node_capacity : int
        Fan-out per R-tree node (default 16).

    Returns
    -------
    GpuSpatialIndex
        Packed Hilbert R-tree with all arrays on the device.

    ADR-0002
    --------
    COARSE kernel class.  Bounds computed in fp64 (memory-bound; fp32
    rounding shrinks bounds causing false negatives).  Hilbert codes use
    integer arithmetic (uint32) — no precision concern.
    """
    n_features = int(geometry_offsets.shape[0]) - 1
    if n_features <= 0:
        # Degenerate: no features — return an empty index
        return GpuSpatialIndex(
            d_sorted_indices=cp.empty(0, dtype=cp.int32),
            d_node_bounds=cp.empty((0, 4), dtype=cp.float64),
            d_node_children=cp.empty((0, node_capacity), dtype=cp.int32),
            d_feature_bounds=cp.empty((0, 4), dtype=cp.float64),
            d_hilbert_codes=cp.empty(0, dtype=cp.uint32),
            n_features=0,
            n_nodes=0,
            n_leaf_nodes=0,
            node_capacity=node_capacity,
        )

    # Step 1: Compute per-feature bounding boxes (Tier 1 NVRTC)
    d_feature_bounds = _compute_bounds_gpu(d_x, d_y, geometry_offsets, n_features)

    # Step 2: Compute Hilbert codes (Tier 1 NVRTC)
    d_hilbert_codes = _compute_hilbert_codes_gpu(d_feature_bounds, n_features)

    # Step 3: Sort features by Hilbert code (Tier 3a CCCL radix sort)
    d_indices = cp.arange(n_features, dtype=cp.int32)
    sorted_result = sort_pairs(d_hilbert_codes, d_indices, synchronize=False)
    d_sorted_indices = sorted_result.values

    # Step 4: Build packed R-tree from sorted order (Tier 2 CuPy)
    d_node_bounds, d_node_children, n_nodes, n_leaf_nodes = _build_packed_rtree(
        d_sorted_indices, d_feature_bounds, n_features, node_capacity=node_capacity,
    )

    return GpuSpatialIndex(
        d_sorted_indices=d_sorted_indices,
        d_node_bounds=d_node_bounds,
        d_node_children=d_node_children,
        d_feature_bounds=d_feature_bounds,
        d_hilbert_codes=d_hilbert_codes,
        n_features=n_features,
        n_nodes=n_nodes,
        n_leaf_nodes=n_leaf_nodes,
        node_capacity=node_capacity,
    )


def build_index_from_reader(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    geometry_offsets: cp.ndarray,
    *,
    node_capacity: int = 16,
) -> GpuSpatialIndex:
    """Build spatial index from reader output — fused ingest step.

    This is the intended entry point when building a spatial index as a
    side effect of file parsing.  Functionally identical to
    ``build_spatial_index`` but documents the intended use as a composable
    stage in a reader pipeline.

    Parameters
    ----------
    d_x, d_y, geometry_offsets
        Same as ``build_spatial_index``.
    node_capacity : int
        Fan-out per R-tree node (default 16).

    Returns
    -------
    GpuSpatialIndex
    """
    return build_spatial_index(
        d_x, d_y, geometry_offsets, node_capacity=node_capacity,
    )
