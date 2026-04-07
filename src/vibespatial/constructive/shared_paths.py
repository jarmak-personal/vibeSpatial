"""GPU-accelerated shared_paths (binary constructive).

Detects collinear overlapping segments shared between two linear geometries
and classifies them as forward (same direction) or backward (opposite
direction).  Returns a numpy array of Shapely GeometryCollection objects,
each containing two MultiLineStrings: [forward_paths, backward_paths].

Architecture (ADR-0033): Tier 1 NVRTC -- geometry-specific inner loops
iterating all segment pairs across two geometries to detect collinearity
and overlap.

Precision (ADR-0002): CONSTRUCTIVE class -- stays fp64 on all devices per
policy.  PrecisionPlan wired through dispatch for observability.

Zero D2H transfers in the hot path.  Geometry data stays device-resident;
only the final shared segment coordinates (small output) cross the bus for
GeometryCollection assembly on host.
"""

from __future__ import annotations

import logging

import numpy as np

from vibespatial.constructive.shared_paths_cpu import (
    empty_shared_paths_result,
    init_shared_paths_result_array,
    merge_shared_paths_segments,
    shared_paths_cpu,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    OwnedGeometryArray,
    tile_single_row,
    unique_tag_pairs,
)
from vibespatial.kernels.constructive.shared_paths import (
    _SHARED_PATHS_KERNEL_SOURCE,
    SHARED_PATHS_KERNEL_NAMES,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import WorkloadShape, detect_workload_shape
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034)
# ---------------------------------------------------------------------------

request_nvrtc_warmup([
    ("shared-paths", _SHARED_PATHS_KERNEL_SOURCE, SHARED_PATHS_KERNEL_NAMES),
])


def _shared_paths_kernels():
    return compile_kernel_group(
        "shared-paths", _SHARED_PATHS_KERNEL_SOURCE, SHARED_PATHS_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------

_LS = GeometryFamily.LINESTRING
_MLS = GeometryFamily.MULTILINESTRING

# Supported kernel pairs: (left_family, right_family) -> (count_kernel, scatter_kernel)
_KERNEL_PAIRS: dict[tuple[GeometryFamily, GeometryFamily], tuple[str, str]] = {
    (_LS, _LS): ("shared_paths_ls_ls_count", "shared_paths_ls_ls_scatter"),
    (_MLS, _LS): ("shared_paths_mls_ls_count", "shared_paths_mls_ls_scatter"),
    (_LS, _MLS): ("shared_paths_mls_ls_count", "shared_paths_mls_ls_scatter"),
    (_MLS, _MLS): ("shared_paths_mls_mls_count", "shared_paths_mls_mls_scatter"),
}


# ---------------------------------------------------------------------------
# Family args builder
# ---------------------------------------------------------------------------

def _family_args(state, family, runtime):
    """Build (args, arg_types) for one side of the kernel."""
    ptr = runtime.pointer
    P = KERNEL_PARAM_PTR

    buf = state.families[family]

    args = [ptr(state.validity), ptr(state.tags), ptr(state.family_row_offsets)]
    types = [P, P, P]

    args.append(ptr(buf.geometry_offsets))
    types.append(P)

    if family == _MLS:
        args.append(ptr(buf.part_offsets))
        types.append(P)

    args.extend([ptr(buf.empty_mask), ptr(buf.x), ptr(buf.y)])
    types.extend([P, P, P])

    args.append(FAMILY_TAGS[family])
    types.append(KERNEL_PARAM_I32)

    return args, types


# ---------------------------------------------------------------------------
# GPU kernel dispatch
# ---------------------------------------------------------------------------

def _launch_shared_paths_subgroup(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    d_left_idx,
    d_right_idx,
    sub_count: int,
    left_family: GeometryFamily,
    right_family: GeometryFamily,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Launch shared_paths count+scatter kernels for a family pair.

    Returns (counts, offsets, seg_x1, seg_y1, seg_x2, seg_y2, seg_dir) on host,
    or None if the family pair is not supported.
    """
    # Determine the kernel pair, handling left/right swap for LS x MLS
    canonical_key = (left_family, right_family)
    if canonical_key not in _KERNEL_PAIRS:
        return None

    # For LS x MLS, swap to MLS x LS (the kernel is MLS-left, LS-right)
    if canonical_key == (_LS, _MLS):
        eff_left_owned, eff_right_owned = right_owned, left_owned
        eff_left_idx, eff_right_idx = d_right_idx, d_left_idx
        eff_left_family, eff_right_family = _MLS, _LS
    else:
        eff_left_owned, eff_right_owned = left_owned, right_owned
        eff_left_idx, eff_right_idx = d_left_idx, d_right_idx
        eff_left_family, eff_right_family = left_family, right_family

    count_kernel_name, scatter_kernel_name = _KERNEL_PAIRS[(eff_left_family, eff_right_family)]

    left_state = eff_left_owned._ensure_device_state()
    right_state = eff_right_owned._ensure_device_state()

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _shared_paths_kernels()

    left_args, left_types = _family_args(left_state, eff_left_family, runtime)
    right_args, right_types = _family_args(right_state, eff_right_family, runtime)

    # --- Count pass ---
    d_counts = runtime.allocate((sub_count,), np.int32, zero=True)

    count_tail_args = [ptr(eff_left_idx), ptr(eff_right_idx), ptr(d_counts), sub_count]
    count_tail_types = [KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32]

    all_count_args = tuple(left_args + right_args + count_tail_args)
    all_count_types = tuple(left_types + right_types + count_tail_types)

    grid, block = runtime.launch_config(kernels[count_kernel_name], sub_count)
    runtime.launch(
        kernels[count_kernel_name],
        grid=grid,
        block=block,
        params=(all_count_args, all_count_types),
    )

    # --- Prefix sum + total ---
    d_offsets = exclusive_sum(d_counts, synchronize=False)
    total = count_scatter_total(runtime, d_counts, d_offsets)

    if total == 0:
        h_counts = np.zeros(sub_count, dtype=np.int32)
        h_offsets = np.zeros(sub_count, dtype=np.int32)
        runtime.free(d_counts)
        runtime.free(d_offsets)
        return (h_counts, h_offsets,
                np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32))

    # --- Allocate output arrays ---
    d_out_x1 = runtime.allocate((total,), np.float64)
    d_out_y1 = runtime.allocate((total,), np.float64)
    d_out_x2 = runtime.allocate((total,), np.float64)
    d_out_y2 = runtime.allocate((total,), np.float64)
    d_out_dir = runtime.allocate((total,), np.int32, zero=True)

    # --- Scatter pass ---
    scatter_tail_args = [
        ptr(eff_left_idx), ptr(eff_right_idx), ptr(d_offsets),
        ptr(d_out_x1), ptr(d_out_y1), ptr(d_out_x2), ptr(d_out_y2),
        ptr(d_out_dir), sub_count,
    ]
    scatter_tail_types = [
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
    ]

    all_scatter_args = tuple(left_args + right_args + scatter_tail_args)
    all_scatter_types = tuple(left_types + right_types + scatter_tail_types)

    grid, block = runtime.launch_config(kernels[scatter_kernel_name], sub_count)
    runtime.launch(
        kernels[scatter_kernel_name],
        grid=grid,
        block=block,
        params=(all_scatter_args, all_scatter_types),
    )

    # --- Transfer results to host (single sync) ---
    runtime.synchronize()

    h_counts = np.empty(sub_count, dtype=np.int32)
    h_offsets = np.empty(sub_count, dtype=np.int32)
    h_x1 = np.empty(total, dtype=np.float64)
    h_y1 = np.empty(total, dtype=np.float64)
    h_x2 = np.empty(total, dtype=np.float64)
    h_y2 = np.empty(total, dtype=np.float64)
    h_dir = np.empty(total, dtype=np.int32)

    runtime.copy_device_to_host(d_counts, h_counts)
    runtime.copy_device_to_host(d_offsets, h_offsets)
    runtime.copy_device_to_host(d_out_x1, h_x1)
    runtime.copy_device_to_host(d_out_y1, h_y1)
    runtime.copy_device_to_host(d_out_x2, h_x2)
    runtime.copy_device_to_host(d_out_y2, h_y2)
    runtime.copy_device_to_host(d_out_dir, h_dir)

    # Direction classification is symmetric: if A and B share a collinear
    # segment in the same direction, it's "forward" regardless of which
    # operand is called "left" or "right" in the kernel.  Verified against
    # Shapely: shared_paths(LS, MLS) and shared_paths(MLS, LS) both
    # classify direction identically.  No direction flip needed on swap.

    # Cleanup
    runtime.free(d_counts)
    runtime.free(d_offsets)
    runtime.free(d_out_x1)
    runtime.free(d_out_y1)
    runtime.free(d_out_x2)
    runtime.free(d_out_y2)
    runtime.free(d_out_dir)

    return (h_counts, h_offsets, h_x1, h_y1, h_x2, h_y2, h_dir)


@register_kernel_variant(
    "shared_paths",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "shared_paths"),
)
def _shared_paths_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """Pure-GPU element-wise shared_paths.

    Groups rows by (left_tag, right_tag) and dispatches to the appropriate
    NVRTC kernel per group.  Geometry data stays device-resident; only
    the final shared segment coordinates (small output) cross the bus.

    Returns a numpy array of Shapely GeometryCollection objects.
    """
    n = left.row_count
    runtime = get_cuda_runtime()

    # Ensure geometry buffers are device-resident
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="shared_paths: left geometry for GPU kernel",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="shared_paths: right geometry for GPU kernel",
    )

    left_tags = left.tags
    right_tags = right.tags
    left_valid = left.validity
    right_valid = right.validity

    both_valid = left_valid & right_valid
    valid_idx = np.flatnonzero(both_valid)

    results = init_shared_paths_result_array(n)

    if valid_idx.size == 0:
        return results

    valid_left_tags = left_tags[valid_idx]
    valid_right_tags = right_tags[valid_idx]

    all_ok = True

    # Collect all sub-group results then assemble
    all_pair_counts = []
    all_pair_offsets = []
    all_seg_x1 = []
    all_seg_y1 = []
    all_seg_x2 = []
    all_seg_y2 = []
    all_seg_dir = []
    all_pair_to_row = []

    for lt, rt in unique_tag_pairs(valid_left_tags, valid_right_tags):
        lf = TAG_FAMILIES[lt] if lt in TAG_FAMILIES else None
        rf = TAG_FAMILIES[rt] if rt in TAG_FAMILIES else None
        if lf is None or rf is None:
            all_ok = False
            continue

        # Only support lineal geometry families
        if lf not in (_LS, _MLS) or rf not in (_LS, _MLS):
            all_ok = False
            continue

        sub_mask = (valid_left_tags == lt) & (valid_right_tags == rt)
        sub_valid_pos = np.flatnonzero(sub_mask)
        sub_idx = valid_idx[sub_valid_pos]
        sub_count = sub_idx.size
        if sub_count == 0:
            continue

        d_idx = runtime.from_host(sub_idx.astype(np.int32))

        result = _launch_shared_paths_subgroup(
            left, right,
            d_idx, d_idx,  # left_idx == right_idx for element-wise
            sub_count, lf, rf,
        )

        runtime.free(d_idx)

        if result is not None:
            h_counts, h_offsets, h_x1, h_y1, h_x2, h_y2, h_dir = result
            all_pair_counts.append(h_counts)
            all_pair_offsets.append(h_offsets)
            all_seg_x1.append(h_x1)
            all_seg_y1.append(h_y1)
            all_seg_x2.append(h_x2)
            all_seg_y2.append(h_y2)
            all_seg_dir.append(h_dir)
            all_pair_to_row.append(sub_idx)
        else:
            all_ok = False

    # Assemble GeometryCollections from all sub-groups
    for sg_idx in range(len(all_pair_counts)):
        counts = all_pair_counts[sg_idx]
        offsets = all_pair_offsets[sg_idx]
        x1 = all_seg_x1[sg_idx]
        y1 = all_seg_y1[sg_idx]
        x2 = all_seg_x2[sg_idx]
        y2 = all_seg_y2[sg_idx]
        dirs = all_seg_dir[sg_idx]
        pair_rows = all_pair_to_row[sg_idx]

        for pair_idx in range(len(counts)):
            count = counts[pair_idx]
            if count == 0:
                continue

            row_idx = pair_rows[pair_idx]
            offset = offsets[pair_idx]

            forward_segs = []
            backward_segs = []

            for s in range(count):
                idx = offset + s
                seg = [(x1[idx], y1[idx]), (x2[idx], y2[idx])]
                if dirs[idx] == 0:
                    forward_segs.append(seg)
                else:
                    backward_segs.append(seg)

            # Merge with any existing segments from previous sub-groups
            existing = results[row_idx]
            results[row_idx] = merge_shared_paths_segments(
                existing if existing is not None else empty_shared_paths_result(),
                forward_segs,
                backward_segs,
            )

    if not all_ok:
        # Some rows had unsupported family pairs — reject the entire batch
        # and let the dispatch system route to the CPU variant.
        raise NotImplementedError(
            "shared_paths GPU path encountered unsupported geometry families; "
            "falling back to CPU variant"
        )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shared_paths_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Element-wise shared_paths between two OwnedGeometryArrays.

    Returns a numpy array of Shapely GeometryCollection objects.  Each
    GeometryCollection contains two MultiLineStrings:
      - geoms[0]: forward shared paths (same direction)
      - geoms[1]: backward shared paths (opposite direction)

    Supports pairwise (N vs N) and broadcast-right (N vs 1) modes.
    """
    n = left.row_count
    workload = detect_workload_shape(n, right.row_count)

    if workload is WorkloadShape.BROADCAST_RIGHT:
        right = tile_single_row(right, n)

    if n == 0:
        return np.empty(0, dtype=object)

    if isinstance(precision, str):
        precision = PrecisionMode(precision)

    selection = plan_dispatch_selection(
        kernel_name="shared_paths",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=combined_residency(left, right),
    )

    precision_plan = selection.precision_plan

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _shared_paths_gpu(left, right)
            record_dispatch_event(
                surface="shared_paths_owned",
                operation="shared_paths",
                implementation="shared_paths_gpu",
                reason="element-wise shared_paths via owned GPU kernels",
                detail=(
                    f"rows={n}, precision={precision_plan.compute_precision.value}, "
                    f"workload={workload.value}"
                ),
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception as exc:
            record_fallback_event(
                surface="shared_paths_owned",
                reason=f"GPU shared_paths kernel failed: {exc}",
                detail=f"rows={n}, falling back to CPU Shapely path",
                pipeline="shared_paths",
            )

    record_dispatch_event(
        surface="shared_paths_owned",
        operation="shared_paths",
        implementation="shapely_cpu",
        reason="GPU not available or not selected for shared_paths",
        detail=f"rows={n}",
        selected=ExecutionMode.CPU,
    )
    return shared_paths_cpu(left, right)
