from __future__ import annotations

import contextlib
import logging

import numpy as np
import shapely

from vibespatial.cuda.cccl_precompile import request_warmup

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "radix_sort_i32_i32", "radix_sort_i64_i32", "radix_sort_u64_i32",
    "unique_by_key_i32_i32", "unique_by_key_u64_i32",
    "segmented_reduce_sum_f64",
    "segmented_reduce_min_f64", "segmented_reduce_max_f64",
    "select_i32",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.dispatch import record_dispatch_event  # noqa: E402
from vibespatial.runtime.fallbacks import record_fallback_event  # noqa: E402
from vibespatial.runtime.kernel_registry import register_kernel_variant  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import Residency  # noqa: E402
from vibespatial.spatial.segment_primitives import (  # noqa: E402
    DeviceSegmentTable,
    SegmentIntersectionResult,
    _extract_segments_gpu,
)

from .types import (  # noqa: E402, F401  # Re-exported for backward compatibility
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    HalfEdgeGraph,
    HalfEdgeGraphDeviceState,
    OverlayExecutionPlan,
    OverlayFaceDeviceState,
    OverlayFaceTable,
    SplitEventDeviceState,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)

# lyy.22: Number of CUDA streams in the per-group overlay stream pool.
# Each stream processes one overlay pair at a time.  Streams enable
# deferred synchronization: results are collected after all pairs are
# dispatched rather than synchronizing per pair.
#
# NOTE: non_blocking=False is required for correctness.  The overlay
# pipeline interleaves CuPy operations (which respect the CuPy stream
# context) with NVRTC kernel launches via runtime.launch() (which use
# the null/default CUDA stream).  Non-blocking streams do NOT
# synchronize with the null stream, which would allow CuPy ops on the
# stream to race with NVRTC kernels on the null stream within a single
# _overlay_owned call.  Blocking streams (the default) synchronize with
# the null stream on every operation, maintaining correct ordering.
#
# The primary benefit of this pattern is structural: it provides a
# framework for future stream-parallel execution once runtime.launch()
# is plumbed with per-stream support.  In the current architecture,
# GPU work is serialized by the null-stream NVRTC kernels regardless
# of the CuPy stream assignment.
_OVERLAY_STREAM_POOL_SIZE = 2


from vibespatial.overlay.gpu_kernels import (  # noqa: E402
    _BATCH_POINT_IN_RING_KERNEL_NAMES,
    _BATCH_POINT_IN_RING_KERNEL_SOURCE,
    _CONTAINMENT_BYPASS_KERNEL_NAMES,
    _CONTAINMENT_BYPASS_KERNEL_SOURCE,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE,
    _OVERLAY_FACE_LABEL_KERNEL_NAMES,
    _OVERLAY_FACE_LABEL_KERNEL_SOURCE,
    _OVERLAY_FACE_WALK_KERNEL_NAMES,
    _OVERLAY_FACE_WALK_KERNEL_SOURCE,
    _OVERLAY_SPLIT_KERNEL_NAMES,
    _OVERLAY_SPLIT_KERNEL_SOURCE,
)

_OVERLAY_COORDINATE_SCALE = 1_000_000_000.0

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
from vibespatial.overlay.bypass import (  # noqa: E402
    _batched_sh_clip,
    _classify_remainder_sh_eligible,
    _combine_bypass_results,
    _containment_bypass_gpu,
    _is_clip_polygon_sh_eligible,
)

request_nvrtc_warmup([
    ("overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES),
    ("overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES),
    ("overlay-face-label", _OVERLAY_FACE_LABEL_KERNEL_SOURCE, _OVERLAY_FACE_LABEL_KERNEL_NAMES),
    ("overlay-face-assembly", _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE, _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES),
    ("overlay-batch-pip", _BATCH_POINT_IN_RING_KERNEL_SOURCE, _BATCH_POINT_IN_RING_KERNEL_NAMES),
    ("overlay-containment-bypass", _CONTAINMENT_BYPASS_KERNEL_SOURCE, _CONTAINMENT_BYPASS_KERNEL_NAMES),
])


@register_kernel_variant(
    "overlay_split",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "split"),
)
def _overlay_split_kernels():
    return compile_kernel_group("overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES)


@register_kernel_variant(
    "overlay_face_walk",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "face-walk"),
)
def _overlay_face_walk_kernels():
    return compile_kernel_group("overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES)


@register_kernel_variant(
    "overlay_face_label",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "face-label"),
)
def _overlay_face_label_kernels():
    return compile_kernel_group("overlay-face-label", _OVERLAY_FACE_LABEL_KERNEL_SOURCE, _OVERLAY_FACE_LABEL_KERNEL_NAMES)


@register_kernel_variant(
    "overlay_face_assembly",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "face-assembly"),
)
def _overlay_face_assembly_kernels():
    return compile_kernel_group("overlay-face-assembly", _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE, _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES)


@register_kernel_variant(
    "overlay_batch_point_in_ring",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "batch-pip"),
)
def _batch_pip_kernels():
    return compile_kernel_group("overlay-batch-pip", _BATCH_POINT_IN_RING_KERNEL_SOURCE, _BATCH_POINT_IN_RING_KERNEL_NAMES)


@register_kernel_variant(
    "overlay_containment_bypass",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "containment-bypass"),
)
def _containment_bypass_kernels():
    return compile_kernel_group(
        "overlay-containment-bypass",
        _CONTAINMENT_BYPASS_KERNEL_SOURCE,
        _CONTAINMENT_BYPASS_KERNEL_NAMES,
    )


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


# Pipeline stages extracted to separate modules.  Re-export for backward
# compatibility.
from vibespatial.overlay.assemble import (  # noqa: E402, F401
    _axis_aligned_box_bounds,
    _build_device_backed_fixed_polygon_output,
    _build_device_resident_polygon_output,
    _build_polygon_output_from_faces_gpu,
    _empty_polygon_output,
    _gather_coords_vectorised,
    _has_polygonal_families,
    _overlay_intersection_rectangles_gpu,
)
from vibespatial.overlay.faces import (  # noqa: E402, F401
    _assemble_faces_from_device_indices,
    _gpu_label_face_coverage,
    _select_overlay_face_indices_gpu,
    build_gpu_overlay_faces,
)
from vibespatial.overlay.graph import (  # noqa: E402, F401
    _gpu_face_walk,
    _quantize_coordinate,
    build_gpu_half_edge_graph,
)
from vibespatial.overlay.host_fallback import (  # noqa: E402
    _build_polygon_output_from_faces,
)
from vibespatial.overlay.split import (  # noqa: E402, F401
    _free_atomic_edge_excess,
    _free_split_event_device_state,
    _segment_metadata,
    _segment_metadata_gpu,
)


def build_gpu_split_events(
    left,
    right,
    *,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
    require_same_row: bool = False,
) -> SplitEventTable:
    # Delegated to overlay/split.py — this re-export preserves import compatibility.
    from vibespatial.overlay.split import build_gpu_split_events as _impl
    return _impl(
        left, right,
        intersection_result=intersection_result,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
        require_same_row=require_same_row,
    )


def build_gpu_atomic_edges(
    split_events: SplitEventTable,
    *,
    isolate_rows: bool = False,
) -> AtomicEdgeTable:
    # Delegated to overlay/split.py — this re-export preserves import compatibility.
    from vibespatial.overlay.split import build_gpu_atomic_edges as _impl
    return _impl(split_events, isolate_rows=isolate_rows)


def _build_overlay_execution_plan(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _row_isolated: bool = False,
) -> OverlayExecutionPlan:
    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
        require_same_row=_row_isolated,
    )
    atomic_edges = build_gpu_atomic_edges(split_events, isolate_rows=_row_isolated)
    # split_events are fully consumed by build_gpu_atomic_edges.
    _free_split_event_device_state(split_events)
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass

    half_edge_graph = build_gpu_half_edge_graph(atomic_edges, isolate_rows=_row_isolated)
    # half_edge_graph retains the atomic-edge arrays it still needs.
    _free_atomic_edge_excess(atomic_edges)
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass

    faces = build_gpu_overlay_faces(
        left,
        right,
        half_edge_graph=half_edge_graph,
        row_isolated=_row_isolated,
    )
    return OverlayExecutionPlan(
        split_events=None,
        atomic_edges=None,
        half_edge_graph=half_edge_graph,
        faces=faces,
        row_isolated=_row_isolated,
    )


def _materialize_overlay_execution_plan(
    plan: OverlayExecutionPlan,
    *,
    operation: str,
    requested: ExecutionMode,
    preserve_row_count: int | None = None,
) -> tuple[OwnedGeometryArray, ExecutionMode]:
    d_selected_face_indices = _select_overlay_face_indices_gpu(plan.faces, operation=operation)
    try:
        if requested is ExecutionMode.GPU:
            result = _build_polygon_output_from_faces_gpu(
                plan.half_edge_graph,
                plan.faces,
                d_selected_face_indices,
                preserve_row_count=preserve_row_count,
            )
            if result is None:
                raise RuntimeError(
                    "GPU face assembly returned None (device state unavailable) "
                    "despite GPU execution mode being requested"
                )
            return result, ExecutionMode.GPU

        gpu_result: OwnedGeometryArray | None = None
        gpu_failed = False
        gpu_fail_reason = ""
        try:
            gpu_result = _build_polygon_output_from_faces_gpu(
                plan.half_edge_graph,
                plan.faces,
                d_selected_face_indices,
                preserve_row_count=preserve_row_count,
            )
            if gpu_result is None:
                gpu_failed = True
                gpu_fail_reason = "GPU face assembly unavailable (no device state)"
        except Exception as exc:
            gpu_failed = True
            gpu_fail_reason = f"GPU face assembly raised {type(exc).__name__}: {exc}"

        if gpu_failed:
            record_fallback_event(
                surface="overlay.gpu._overlay_owned",
                reason=gpu_fail_reason,
                detail=f"operation={operation}",
                requested=requested,
                selected=ExecutionMode.CPU,
                pipeline="overlay",
                d2h_transfer=True,
            )
            selected_face_indices = cp.asnumpy(d_selected_face_indices)
            result = _build_polygon_output_from_faces(
                plan.half_edge_graph, plan.faces, selected_face_indices,
            )
            return result, ExecutionMode.CPU

        return gpu_result, ExecutionMode.GPU  # type: ignore[return-value]
    finally:
        del d_selected_face_indices
        try:
            get_cuda_runtime().free_pool_memory()
        except Exception:
            pass


def overlay_intersection_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="intersection", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_union_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="union", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="difference", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_symmetric_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="symmetric_difference", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_identity_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left, right, operation="identity", dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def _overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    operation: str,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _row_isolated: bool = False,
) -> OwnedGeometryArray:
    requested = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    _polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    polygon_only = (
        set(left.families) <= _polygonal_families
        and set(right.families) <= _polygonal_families
    )
    if not polygon_only:
        raise NotImplementedError("GPU overlay owned operations currently support polygon/multipolygon inputs")

    if requested is ExecutionMode.CPU:
        # CPU-only mode: explicit CPU request, Shapely operations
        left_values = np.asarray(left.to_shapely(), dtype=object)
        right_values = np.asarray(right.to_shapely(), dtype=object)
        if operation == "intersection":
            values = shapely.intersection(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "union":
            values = shapely.union(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "difference":
            values = shapely.difference(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "symmetric_difference":
            values = shapely.symmetric_difference(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "identity":
            values = [geometry for geometry in left_values.tolist() if geometry is not None and not geometry.is_empty]
        else:
            raise ValueError(f"unsupported overlay operation: {operation}")
        result = from_shapely_geometries(
            values,
            residency=Residency.HOST,
        )
        result.runtime_history.append(
            RuntimeSelection(
                requested=requested,
                selected=ExecutionMode.CPU,
                reason=f"CPU requested for overlay {operation}",
            )
        )
        return result

    if requested is ExecutionMode.GPU and cp is None:
        raise RuntimeError("GPU execution was requested, but CuPy is not available")

    selected = ExecutionMode.GPU if cp is not None else ExecutionMode.CPU
    if requested is ExecutionMode.GPU and selected is not ExecutionMode.GPU:
        raise RuntimeError("GPU execution was requested, but no CUDA runtime is available")
    if requested is ExecutionMode.AUTO and selected is ExecutionMode.CPU:
        # Phase 24: AUTO mode, no GPU available — CPU fallback is expected.
        record_fallback_event(
            surface=f"geopandas.overlay.{operation}",
            reason="AUTO mode: no GPU runtime available",
            detail=f"operation={operation}, left_rows={left.row_count}, right_rows={right.row_count}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_overlay_owned",
            d2h_transfer=False,
        )
        return _overlay_owned(left, right, operation=operation, dispatch_mode=ExecutionMode.CPU)

    if operation == "intersection" and not _row_isolated:
        rectangle_fast_path = _overlay_intersection_rectangles_gpu(left, right, requested=requested)
        if rectangle_fast_path is not None:
            return rectangle_fast_path

    # Phase 20: The 10K row CPU threshold (_GPU_OVERLAY_MAX_ROWS) has been
    # removed.  Phases 7-15 eliminated the serial bottlenecks that made GPU
    # overlay slower than Shapely at high row counts.  For AUTO mode the GPU
    # path is now selected whenever a CUDA runtime is available; input
    # residency is already on-device when the caller used the zero-copy
    # pipeline, so no additional transfer heuristic is needed here — the
    # adaptive runtime handles crossover decisions upstream via
    # plan_dispatch_selection().
    plan = _build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=_cached_right_segments,
        _row_isolated=_row_isolated,
    )
    result, face_assembly_mode = _materialize_overlay_execution_plan(
        plan,
        operation=operation,
        requested=requested,
        preserve_row_count=left.row_count if _row_isolated else None,
    )
    del plan
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass
    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=face_assembly_mode,
            reason=f"GPU overlay {operation}: face assembly on {face_assembly_mode.value}",
        )
    )
    return result


def spatial_overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    how: str = "intersection",
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Spatial overlay: intersect each left geometry with spatially overlapping right geometries.

    Unlike :func:`overlay_intersection_owned` which does row-matched pairwise overlay
    (left[0] vs right[0], left[1] vs right[1], ...), this function performs a spatial
    join to discover which (left_i, right_j) pairs overlap, then runs pairwise overlay
    on each discovered pair, and assembles the results.

    This implements the GeoPandas ``overlay(left, right, how=...)`` semantics on GPU,
    following ADR-0016 (8-stage overlay pipeline) and ADR-0012 (two-layer architecture).

    Parameters
    ----------
    left : OwnedGeometryArray
        Left geometry array (e.g. 10K vegetation polygons).
    right : OwnedGeometryArray
        Right geometry array (e.g. 1 dissolved corridor, or 1K zoning polygons).
    how : str
        Overlay operation: "intersection", "union", "difference", "symmetric_difference".
    dispatch_mode : ExecutionMode
        Execution mode for the pairwise overlay step.

    Returns
    -------
    OwnedGeometryArray
        Result geometries from all overlapping pairs, with empty/null results filtered out.
    """
    from vibespatial.spatial.indexing import generate_bounds_pairs

    requested = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)

    # Stage 1: Spatial join — find candidate (left_i, right_j) pairs via MBR overlap.
    # Uses the GPU sweep-plane kernel when geometry count exceeds threshold (ADR-0033 Tier 1).
    candidate_pairs = generate_bounds_pairs(left, right)
    if candidate_pairs.count == 0:
        result = from_shapely_geometries([shapely.Point()])
        result = result.take(np.asarray([], dtype=np.int64))
        result.runtime_history.append(
            RuntimeSelection(
                requested=requested,
                selected=ExecutionMode.CPU,
                reason=f"spatial_overlay {how}: no candidate pairs found",
            )
        )
        return result

    # lyy.32: Keep spatial join pair indices on GPU when device-resident.
    # When candidate_pairs has device indices (GPU spatial join), all sorting,
    # grouping, and subset selection is done with CuPy on device.  Device
    # index arrays are passed directly to take() which routes to device_take(),
    # avoiding D->H->D round-trips.  When device indices are not available
    # (CPU spatial join), falls back to the existing numpy path.
    _pairs_on_device = (
        cp is not None
        and candidate_pairs.device_left_indices is not None
        and candidate_pairs.device_right_indices is not None
    )

    if _pairs_on_device:
        # Device path: all grouping operations stay on GPU.
        d_left_indices = candidate_pairs.device_left_indices
        d_right_indices = candidate_pairs.device_right_indices

        # Ensure int64 for CuPy argsort stability and take() compatibility.
        d_left_indices = d_left_indices.astype(cp.int64, copy=False)
        d_right_indices = d_right_indices.astype(cp.int64, copy=False)

        # Sort pairs by left index for grouping.  CuPy argsort is
        # stable by default (mergesort), preserving relative order
        # within each group.
        d_sort_order = cp.argsort(d_left_indices)
        d_left_indices = d_left_indices[d_sort_order]
        d_right_indices = d_right_indices[d_sort_order]

        d_unique_left, d_group_starts = cp.unique(d_left_indices, return_index=True)
        d_group_ends = cp.append(d_group_starts[1:], cp.array([d_left_indices.size], dtype=d_group_starts.dtype))
        d_unique_right = cp.unique(d_right_indices)

        # take() with CuPy indices routes to device_take() — zero-copy.
        left_subset = left.take(d_unique_left) if int(d_unique_left.size) < left.row_count else left
        right_subset = right.take(d_unique_right) if int(d_unique_right.size) < right.row_count else right

        # Remap right indices from original row space to right_subset row space.
        d_right_remap = cp.empty(right.row_count, dtype=cp.int64)
        d_right_remap[d_unique_right] = cp.arange(int(d_unique_right.size), dtype=cp.int64)
        d_right_subset_indices = d_right_remap[d_right_indices]

        # Keep group boundaries on device — they are small O(unique_left)
        # arrays used for filtering in the containment/SH bypass paths.
        # They are materialised to host lazily right before the per-group
        # Python loop or segmented_union_all call (FIX-09: ZCOPY001).
        group_starts = d_group_starts  # device CuPy array
        group_ends = d_group_ends  # device CuPy array
        unique_left = d_unique_left  # device array for take()
        right_subset_indices = d_right_subset_indices  # device array for take()

        # Host copies for Shapely fallback path (lazy — only materialised if
        # owned-dispatch fails).  These are set after the owned-dispatch
        # try/except block when needed.
        left_indices = None  # deferred; materialised in fallback
        right_indices = None
    else:
        # CPU path: pair indices are already on host.
        left_indices = candidate_pairs.left_indices
        right_indices = candidate_pairs.right_indices

        # Sort pairs by left index for grouping.
        sort_order = np.argsort(left_indices, kind="mergesort")
        left_indices = left_indices[sort_order]
        right_indices = right_indices[sort_order]

        unique_left, group_starts = np.unique(left_indices, return_index=True)
        group_ends = np.append(group_starts[1:], len(left_indices))

        unique_right = np.unique(right_indices)

        # take() operates at buffer level — no Shapely materialization.
        left_subset = left.take(unique_left) if len(unique_left) < left.row_count else left
        right_subset = right.take(unique_right) if len(unique_right) < right.row_count else right

        # Remap right indices from original row space to right_subset row space.
        right_remap = np.empty(right.row_count, dtype=np.intp)
        right_remap[unique_right] = np.arange(len(unique_right))
        right_subset_indices = right_remap[right_indices]

    # Strategy detection: detect workload shape and select the overlay strategy.
    # broadcast_right + intersection: containment bypass (lyy.16) + batched SH
    # clip (lyy.18) reduce work before per-group overlay.  Other strategies
    # fall through to the existing per-group path.
    from vibespatial.overlay.strategies import select_overlay_strategy

    strategy = select_overlay_strategy(
        left, right, how,
        candidate_pair_count=candidate_pairs.count,
    )

    _ws_label = strategy.workload_shape.value if strategy.workload_shape is not None else strategy.name
    record_dispatch_event(
        surface="geopandas.spatial_overlay",
        operation=f"spatial_overlay_{how}",
        implementation=f"spatial_overlay_{strategy.name}",
        reason=strategy.reason,
        detail=(
            f"left={left.row_count}, right={right.row_count}, "
            f"pairs={candidate_pairs.count}, strategy={strategy.name}, "
            f"workload_shape={_ws_label}"
        ),
        requested=requested,
        selected=ExecutionMode.GPU if cp is not None else ExecutionMode.CPU,
    )

    # Strategy-specific implementations.
    # lyy.16/lyy.11: containment/disjointness bypass for broadcast_right.
    #   intersection: contained -> pass-through, disjoint -> skip.
    #   difference: contained -> empty, disjoint -> pass-through.
    # lyy.18: batched SH clip for boundary-crossing simple polygons.
    _containment_result: OwnedGeometryArray | None = None
    _containment_remainder_mask: cp.ndarray | None = None  # type: ignore[name-defined]
    _sh_clip_result: OwnedGeometryArray | None = None

    _bypass_eligible_ops = ("intersection", "difference")
    if strategy.name == "broadcast_right" and how in _bypass_eligible_ops and cp is not None:
        try:
            _containment_result, _containment_remainder_mask = (
                _containment_bypass_gpu(left_subset, right_subset, how)
            )
        except Exception:
            # If containment bypass fails, fall through to per-group.
            logger.debug(
                "lyy.16 containment bypass failed, falling through to overlay",
                exc_info=True,
            )
            _containment_result = None
            _containment_remainder_mask = None

        if _containment_remainder_mask is not None:
            # Some polygons need overlay — filter left_subset to remainder only.
            # Rebuild group structures for the remainder polygons.
            d_remainder_indices = cp.flatnonzero(_containment_remainder_mask).astype(cp.int64)
            left_subset = left_subset.take(d_remainder_indices)

            # Rebuild pair grouping: filter pairs to only reference remainder rows.
            # The remainder_mask is over the original left_subset row space.
            # We need to remap unique_left, group_starts, group_ends to the
            # filtered left_subset.
            if _pairs_on_device:
                # d_left_indices are in original left row space.  d_unique_left
                # maps to left_subset row space (0..len(unique_left)-1).
                # We need to filter to rows where remainder_mask is True.
                #
                # Approach: rebuild grouping from scratch for the filtered pairs.
                # Only pairs whose left_subset row is in the remainder survive.

                # Build a mapping: old left_subset row -> new left_subset row.
                # _containment_remainder_mask is a bool mask over old left_subset.
                d_old_to_new = cp.full(int(_containment_remainder_mask.size), -1, dtype=cp.int64)
                d_old_to_new[d_remainder_indices] = cp.arange(int(d_remainder_indices.size), dtype=cp.int64)

                # Filter groups: each group corresponds to one unique_left row
                # in the old left_subset.  If that row is in the remainder, keep
                # the group; remap its grp_idx to the new left_subset space.
                # unique_left[grp_idx] is the original left row; grp_idx is
                # the old left_subset row.
                d_grp_mask = _containment_remainder_mask[
                    cp.arange(len(group_starts), dtype=cp.int64)
                ]
                new_group_indices = cp.flatnonzero(d_grp_mask)

                # FIX-09: index device arrays directly — no D2H for
                # group boundary filtering.
                group_starts = group_starts[new_group_indices]
                group_ends = group_ends[new_group_indices]
                unique_left = unique_left[new_group_indices]

                # right_subset_indices are unchanged — they reference right_subset
                # rows which are not affected by left filtering.
            else:
                # CPU path: rebuild grouping.
                h_remainder_mask = cp.asnumpy(_containment_remainder_mask) if cp is not None else _containment_remainder_mask  # zcopy:ok(CPU-path branch; H2D at 5450 is in device branch — mutually exclusive)
                old_to_new = np.full(len(h_remainder_mask), -1, dtype=np.int64)
                h_remainder_indices = np.flatnonzero(h_remainder_mask)
                old_to_new[h_remainder_indices] = np.arange(len(h_remainder_indices), dtype=np.int64)

                grp_mask = h_remainder_mask[np.arange(len(group_starts))]
                new_grp_indices = np.flatnonzero(grp_mask)
                group_starts = group_starts[new_grp_indices]
                group_ends = group_ends[new_grp_indices]
                unique_left = unique_left[new_grp_indices]

        elif _containment_result is not None and _containment_remainder_mask is None:
            # ALL polygons were bypassed — no overlay needed.
            # Skip the entire per-group processing.
            result = _containment_result
            result.runtime_history.append(
                RuntimeSelection(
                    requested=requested,
                    selected=ExecutionMode.GPU,
                    reason=(
                        f"spatial_overlay {how}: all polygons bypassed "
                        f"(containment/disjointness bypass)"
                    ),
                )
            )
            return result

        # lyy.18: Batched SH clip for boundary-crossing simple polygons.
        # After containment bypass, left_subset contains only remainder
        # polygons.  Check if the clip polygon (right_subset) is SH-eligible.
        # If so, batch-clip all SH-eligible remainder polygons in a single
        # polygon_intersection kernel launch, further reducing the number of
        # polygons that fall through to the expensive per-group overlay.
        # SH clip is only applicable to intersection (clipping semantics).
        if how == "intersection" and left_subset.row_count > 0:
            try:
                clip_eligible, clip_vert_count = _is_clip_polygon_sh_eligible(right_subset)
                if clip_eligible:
                    sh_eligible_mask, _complex_mask = _classify_remainder_sh_eligible(
                        left_subset, clip_vert_count,
                    )
                    n_sh = int(sh_eligible_mask.sum())

                    if n_sh > 0:
                        _sh_clip_result = _batched_sh_clip(
                            left_subset, right_subset, sh_eligible_mask,
                        )

                        if _sh_clip_result is not None and n_sh < left_subset.row_count:
                            # Some polygons were SH-clipped; filter left_subset
                            # and grouping structures to only the complex remainder.
                            d_complex_indices = cp.asarray(
                                np.flatnonzero(~sh_eligible_mask)
                            ).astype(cp.int64)
                            left_subset = left_subset.take(d_complex_indices)

                            # Rebuild grouping for the reduced left_subset.
                            if _pairs_on_device:
                                # complex_mask[i] is True for old left_subset
                                # rows that are NOT SH-eligible.  Rebuild the
                                # old-to-new mapping for group filtering.
                                d_old_to_new_sh = cp.full(
                                    len(sh_eligible_mask), -1, dtype=cp.int64,
                                )
                                d_old_to_new_sh[d_complex_indices] = cp.arange(
                                    int(d_complex_indices.size), dtype=cp.int64,
                                )
                                d_sh_grp_mask = cp.asarray(~sh_eligible_mask)[
                                    cp.arange(len(group_starts), dtype=cp.int64)
                                ]
                                new_grp_sh = cp.flatnonzero(d_sh_grp_mask)
                                # FIX-09: index device arrays directly.
                                group_starts = group_starts[new_grp_sh]
                                group_ends = group_ends[new_grp_sh]
                                unique_left = unique_left[new_grp_sh]
                            else:
                                h_complex_mask = ~sh_eligible_mask
                                old_to_new_sh = np.full(
                                    len(sh_eligible_mask), -1, dtype=np.int64,
                                )
                                h_complex_indices = np.flatnonzero(h_complex_mask)
                                old_to_new_sh[h_complex_indices] = np.arange(
                                    len(h_complex_indices), dtype=np.int64,
                                )
                                grp_mask_sh = h_complex_mask[
                                    np.arange(len(group_starts))
                                ]
                                new_grp_sh = np.flatnonzero(grp_mask_sh)
                                group_starts = group_starts[new_grp_sh]
                                group_ends = group_ends[new_grp_sh]
                                unique_left = unique_left[new_grp_sh]

                        elif _sh_clip_result is not None:
                            # ALL remainder polygons were SH-clipped.
                            # No overlay needed for any polygon.
                            left_subset = left_subset.take(
                                np.asarray([], dtype=np.int64)
                            )
                            group_starts = group_starts[:0]
                            group_ends = group_ends[:0]
                            if _pairs_on_device:
                                unique_left = unique_left[:0]
                            else:
                                unique_left = unique_left[:0]

                    else:
                        logger.debug(
                            "lyy.18 SH batch clip: clip polygon eligible but "
                            "no remainder polygons qualify (all have holes or "
                            "too many vertices)"
                        )
                else:
                    logger.debug(
                        "lyy.18 SH batch clip: clip polygon not SH-eligible "
                        "(holes or >%d vertices) -- skipping SH tier",
                        64,
                    )
            except Exception:
                logger.debug(
                    "lyy.18 SH batch clip failed, falling through to overlay",
                    exc_info=True,
                )
                _sh_clip_result = None

    elif strategy.name == "broadcast_left":
        pass  # fall through to per_group

    # Release GPU pool memory after containment bypass and SH batch clip:
    # bounds check, PIP, and clip kernels produce large intermediates that
    # are no longer needed before the per-group overlay loop.
    try:
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup

    # Stage 2: Per-left-group processing.
    #
    # Previous approach gathered ALL pairs into a single batch and ran one
    # global binary_constructive_owned call.  This caused two bugs:
    #   Bug 1 (O(n**2) scaling): segment candidate generation used a GLOBAL
    #     sort-sweep so segments from independent geometry pairs
    #     cross-contaminated, producing O(n**2) segment comparisons.
    #   Bug 2 (incorrect difference): computed L_i - R_j per pair instead of
    #     L_i - union(R_j for all j overlapping L_i), yielding multiple
    #     partial-difference fragments per left geometry.
    #
    # Fix: group pairs by left index and process each left geometry
    # independently.  For difference/symmetric_difference, union all right
    # neighbours first via segmented_union_all (matching the approach in
    # _overlay_difference in api/tools/overlay.py).  For intersection/union,
    # process per-pair within each group to keep segment sets isolated.
    #
    # Performance: selective materialization (ADR-0005).
    # take() subsets participating rows at buffer level — no Shapely
    # round-trip.  Per-group calls each handle O(1) segment pairs, giving
    # overall O(N) scaling instead of O(N**2).

    # Remap left indices: unique_left[i] -> i (left_subset row space).
    # left_subset row i corresponds to unique_left[i] in the original array.
    # For per-group processing we iterate over groups 0..len(unique_left)-1.

    # ------------------------------------------------------------------
    # Owned-dispatch path: per-left-group processing via binary_constructive_owned.
    # For difference/symmetric_difference, uses segmented_union_all to union
    # right neighbours before computing the set operation — matching the
    # correct semantics of L_i - union(R_j for all j).
    # ------------------------------------------------------------------
    _used_owned_dispatch = False
    try:
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        # Force GPU dispatch when a GPU runtime is available: the spatial
        # overlay pipeline has already committed to GPU for spatial join
        # and pair generation.  The pairwise constructive step must also
        # use GPU to avoid the 50K CONSTRUCTIVE crossover threshold
        # routing small pair batches to CPU (which triggers a fallback
        # event per batch and forces a D->H->D round-trip through Shapely).
        _pairwise_selection = plan_dispatch_selection(
            kernel_name="overlay_pairwise",
            kernel_class=KernelClass.CONSTRUCTIVE,
            row_count=candidate_pairs.count,
            requested_mode=ExecutionMode.GPU if _pairs_on_device else requested,
            gpu_available=cp is not None,
        )
        _pairwise_mode = _pairwise_selection.selected

        if how in ("difference", "symmetric_difference"):
            # Union all right neighbours per left group, then compute one
            # set operation per unique left geometry.
            # This mirrors the approach in _overlay_difference (api/tools/overlay.py).
            from vibespatial.kernels.constructive.segmented_union import (
                segmented_union_all,
            )

            # Build CSR-style group offsets for segmented_union_all.
            # segmented_union_all expects host numpy offsets.
            # FIX-09: when pairs are on device, materialise group_starts
            # to host here (single bulk D2H for small metadata).
            if _pairs_on_device:
                h_group_starts = cp.asnumpy(group_starts)  # zcopy:ok(small metadata for segmented_union_all)
                n_rsi = int(right_subset_indices.shape[0])
            else:
                h_group_starts = group_starts  # already host numpy
                n_rsi = len(right_subset_indices)
            group_offsets = np.empty(len(h_group_starts) + 1, dtype=np.int64)
            group_offsets[:-1] = h_group_starts
            group_offsets[-1] = n_rsi

            right_gathered = right_subset.take(right_subset_indices)
            right_unions = segmented_union_all(right_gathered, group_offsets)

            # left_subset has one row per unique left geometry, aligned with
            # right_unions (one unioned geometry per group).
            result_owned = binary_constructive_owned(
                how, left_subset, right_unions,
                dispatch_mode=_pairwise_mode,
            )

        else:
            # intersection / union: process per-pair within each group.
            # Each (L_i, R_j) pair produces an independent result fragment.
            # Processing per-group keeps segment sets isolated, avoiding
            # global O(n**2) cross-contamination.
            result_parts: list[OwnedGeometryArray] = []
            _xp = cp if _pairs_on_device else np  # array module for index construction

            # lyy.15: Cache right-side segment extraction for broadcast_right.
            # In the N-vs-1 pattern, the right geometry (corridor) is identical
            # for every pair.  Pre-extract its segments ONCE and reuse across
            # all iterations, avoiding redundant _extract_segments_gpu calls
            # (2 per iteration: once in build_gpu_split_events, once in
            # classify_segment_intersections).
            _cached_right_segs: DeviceSegmentTable | None = None
            _is_broadcast_right_overlay = (
                strategy.name == "broadcast_right"
                and right_subset.row_count == 1
                and cp is not None
            )
            if _is_broadcast_right_overlay and len(unique_left) > 1:
                try:
                    _cached_right_segs = _extract_segments_gpu(right_subset)
                    logger.debug(
                        "lyy.15: cached right-side segments (%d segments) "
                        "for %d per-group iterations",
                        _cached_right_segs.count, len(unique_left),
                    )
                except Exception:
                    logger.debug(
                        "lyy.15: right-side segment caching failed, "
                        "falling through to per-iteration extraction",
                        exc_info=True,
                    )
                    _cached_right_segs = None

            # lyy.21: Batched overlay for broadcast_right many-vs-one.
            #
            # When all pairs share the same right polygon (broadcast_right),
            # gather all complex left polygons and dispatch them in a single
            # binary_constructive_owned call.  Each row is still processed
            # independently (per-row isolation) because
            # binary_constructive_owned routes to _overlay_owned which runs
            # the 8-stage pipeline per-pair.
            #
            # Optimization: for broadcast_right with a single right polygon,
            # call _overlay_owned directly per pair, bypassing the per-
            # iteration overhead of binary_constructive_owned dispatch
            # (workload detection, SH kernel pre-flight, dispatch selection,
            # precision planning).  Host state for right_subset is
            # materialised once before the loop.
            #
            # Cross-contamination guard: per-pair isolation is maintained
            # because each _overlay_owned call operates on exactly one
            # (L_i, R_0) pair.  Segments from pair i never interact with
            # segments from pair j.
            try:
                # lyy.22: Stream pool for per-group overlay.
                #
                # Create a bounded pool of CUDA streams and assign each
                # overlay pair to a stream in round-robin order.  Each
                # pair's GPU work (CuPy operations) is issued on its
                # assigned stream.  Results are collected after all pairs
                # are dispatched, with per-stream synchronization.
                #
                # IMPORTANT: non_blocking=False (default) is required.
                # See _OVERLAY_STREAM_POOL_SIZE comment for rationale.
                _n_groups = len(unique_left)
                _pool_size = min(_OVERLAY_STREAM_POOL_SIZE, _n_groups)
                _use_stream_pool = cp is not None and _n_groups > 1 and _pool_size > 0
                if _use_stream_pool:
                    _stream_pool = [
                        cp.cuda.Stream(non_blocking=False)
                        for _ in range(_pool_size)
                    ]
                    logger.debug(
                        "lyy.22: created stream pool with %d streams "
                        "for %d per-group iterations",
                        len(_stream_pool), _n_groups,
                    )
                else:
                    _stream_pool = None

                # FIX-09: materialise group boundaries to host once
                # right before the per-group Python loop.  This is the
                # only D2H for these arrays; all upstream filtering
                # (containment bypass, SH bypass) operated on device.
                if _pairs_on_device and _n_groups > 0:
                    _h_group_starts = cp.asnumpy(group_starts)
                    _h_group_ends = cp.asnumpy(group_ends)
                else:
                    _h_group_starts = group_starts
                    _h_group_ends = group_ends
                _group_ranges = list(
                    zip(
                        np.asarray(_h_group_starts).tolist(),
                        np.asarray(_h_group_ends).tolist(),
                    )
                )

                if _is_broadcast_right_overlay and _n_groups > 0:
                    # Materialise right host state once outside the loop.
                    # _overlay_owned calls _ensure_host_state on both
                    # inputs; doing it here for right avoids N redundant
                    # no-op calls.
                    right_subset._ensure_host_state()

                    # lyy.22: Dispatch all pairs across the stream pool,
                    # collecting (stream, result) futures for deferred
                    # synchronization.
                    _futures: list[tuple] = []
                    for grp_idx, (start, end) in enumerate(_group_ranges):
                        n_pairs = end - start
                        _stream = _stream_pool[grp_idx % len(_stream_pool)] if _stream_pool else None
                        with (_stream if _stream is not None else contextlib.nullcontext()):
                            left_row = left_subset.take(
                                _xp.array([grp_idx], dtype=_xp.int64)
                            )
                            if n_pairs == 1:
                                # Common case: one pair (L_i, R_0).
                                # Call _overlay_owned directly, bypassing
                                # binary_constructive_owned dispatch overhead.
                                grp_result = _overlay_owned(
                                    left_row, right_subset,
                                    operation=how,
                                    dispatch_mode=_pairwise_mode,
                                    _cached_right_segments=_cached_right_segs,
                                )
                            else:
                                # Multiple right neighbours: replicate left
                                # row and use binary_constructive_owned for
                                # the full dispatch (rare in broadcast_right).
                                right_rows = right_subset.take(
                                    right_subset_indices[start:end]
                                )
                                left_replicated = left_row.take(
                                    _xp.zeros(n_pairs, dtype=_xp.int64)
                                )
                                grp_result = binary_constructive_owned(
                                    how, left_replicated, right_rows,
                                    dispatch_mode=_pairwise_mode,
                                    _cached_right_segments=_cached_right_segs,
                                )
                            _futures.append((_stream, grp_result))

                    # lyy.22: Synchronize per stream and collect results.
                    # Each stream is synced at most once (the first time
                    # we encounter a result from it), which guarantees
                    # all prior work on that stream has completed.
                    _synced_streams: set[int] = set()
                    for _stream, grp_result in _futures:
                        if _stream is not None:
                            _sid = id(_stream)
                            if _sid not in _synced_streams:
                                _stream.synchronize()
                                _synced_streams.add(_sid)
                        result_parts.append(grp_result)

                else:
                    # General case: multiple right neighbours per group,
                    # or non-broadcast_right strategy.
                    _futures_gen: list[tuple] = []
                    for grp_idx, (start, end) in enumerate(_group_ranges):
                        n_pairs = end - start
                        _stream = _stream_pool[grp_idx % len(_stream_pool)] if _stream_pool else None
                        with (_stream if _stream is not None else contextlib.nullcontext()):
                            left_row = left_subset.take(
                                _xp.array([grp_idx], dtype=_xp.int64)
                            )
                            right_rows = right_subset.take(
                                right_subset_indices[start:end]
                            )
                            # Replicate left row to match the number of
                            # right neighbours.
                            if n_pairs > 1:
                                left_replicated = left_row.take(
                                    _xp.zeros(n_pairs, dtype=_xp.int64)
                                )
                            else:
                                left_replicated = left_row
                            grp_result = binary_constructive_owned(
                                how, left_replicated, right_rows,
                                dispatch_mode=_pairwise_mode,
                                _cached_right_segments=_cached_right_segs,
                            )
                            _futures_gen.append((_stream, grp_result))

                    # lyy.22: Synchronize per stream and collect results.
                    _synced_streams_gen: set[int] = set()
                    for _stream, grp_result in _futures_gen:
                        if _stream is not None:
                            _sid = id(_stream)
                            if _sid not in _synced_streams_gen:
                                _stream.synchronize()
                                _synced_streams_gen.add(_sid)
                        result_parts.append(grp_result)
            finally:
                # lyy.15: Free cached right-side segments after all
                # iterations (or on exception).
                if _cached_right_segs is not None:
                    _rt = get_cuda_runtime()
                    _rt.free(_cached_right_segs.x0)
                    _rt.free(_cached_right_segs.y0)
                    _rt.free(_cached_right_segs.x1)
                    _rt.free(_cached_right_segs.y1)
                    _rt.free(_cached_right_segs.row_indices)
                    _rt.free(_cached_right_segs.segment_indices)
                    if _cached_right_segs.part_indices is not None:
                        _rt.free(_cached_right_segs.part_indices)
                    if _cached_right_segs.ring_indices is not None:
                        _rt.free(_cached_right_segs.ring_indices)
                    _cached_right_segs = None
                # lyy.22: Destroy stream pool to avoid resource leaks.
                # CuPy streams are lightweight (~1-2us each) but should
                # still be cleaned up.
                if _stream_pool is not None:
                    for _s in _stream_pool:
                        # CuPy streams do not require explicit destruction;
                        # they are released when garbage collected.  We
                        # synchronize each one to ensure all dispatched work
                        # is complete before leaving the finally block.
                        try:
                            _s.synchronize()
                        except Exception:
                            pass
                    _stream_pool = None

            if result_parts:
                result_owned = OwnedGeometryArray.concat(result_parts)
            else:
                result_owned = from_shapely_geometries([shapely.Point()])
                result_owned = result_owned.take(np.asarray([], dtype=np.int64))

        # Release GPU pool memory after per-group overlay loop: each
        # iteration's split events, half-edge graphs, and face tables
        # leave freed-but-cached blocks in the CuPy pool.
        try:
            get_cuda_runtime().free_pool_memory()
        except Exception:
            pass  # best-effort cleanup

        # Filter empty/null using owned-level metadata (validity + empty_mask)
        # instead of to_shapely() — avoids D->H->D ping-pong.
        # binary_constructive_owned returns polygon-family results (no
        # GeometryCollections), so collection flattening is unnecessary.
        result_owned._ensure_host_state()
        non_empty = result_owned.validity.copy()
        for family, buf in result_owned.families.items():
            family_rows = (result_owned.tags == FAMILY_TAGS[family])
            non_empty[family_rows] &= ~buf.empty_mask[
                result_owned.family_row_offsets[family_rows]
            ]
        keep_indices = np.flatnonzero(non_empty)
        if keep_indices.size == 0:
            result = from_shapely_geometries([shapely.Point()])
            result = result.take(np.asarray([], dtype=np.int64))
        else:
            result = result_owned.take(keep_indices)

        # lyy.16 + lyy.18: Combine bypass results with overlay results.
        result = _combine_bypass_results(
            _containment_result, _sh_clip_result, result,
        )

        _used_owned_dispatch = True
    except (NotImplementedError, ImportError, ValueError):
        _used_owned_dispatch = False

    if not _used_owned_dispatch:
        # lyy.32: When owned dispatch failed and pairs were on device,
        # materialise grouping arrays to host for the Shapely fallback path.
        # This D->H transfer is acceptable because the Shapely path already
        # materialises the full geometries to host below.
        if _pairs_on_device:
            left_indices = cp.asnumpy(d_left_indices)
            right_indices = cp.asnumpy(d_right_indices)
            group_starts = cp.asnumpy(d_group_starts)
            group_ends = cp.asnumpy(d_group_ends)
            unique_left = cp.asnumpy(d_unique_left)
            right_subset_indices = cp.asnumpy(d_right_subset_indices)

        # Phase 24: Record fallback event for spatial overlay CPU path.
        record_fallback_event(
            surface=f"geopandas.spatial_overlay.{how}",
            reason="owned-path dispatch failed, falling back to Shapely",
            detail=f"how={how}, pairs={len(left_indices)}",
            requested=requested,
            selected=ExecutionMode.CPU,
            pipeline="spatial_overlay_owned",
            d2h_transfer=True,
        )
        # Shapely fallback: materialize participating rows for validation + clipping.
        left_shapely_orig = np.asarray(left_subset.to_shapely(), dtype=object)
        right_shapely_orig = np.asarray(right_subset.to_shapely(), dtype=object)

        # Validate input geometries ONCE before replication (ADR-0019).
        # This avoids running make_valid on 10K replicated copies of the same
        # invalid geometry. Validate on the (smaller) subset, then gather.
        left_invalid_mask = ~np.asarray(shapely.is_valid(left_shapely_orig), dtype=bool)
        right_invalid_mask = ~np.asarray(shapely.is_valid(right_shapely_orig), dtype=bool)
        if np.any(left_invalid_mask):
            left_shapely_orig[left_invalid_mask] = shapely.make_valid(left_shapely_orig[left_invalid_mask])
        if np.any(right_invalid_mask):
            right_shapely_orig[right_invalid_mask] = shapely.make_valid(right_shapely_orig[right_invalid_mask])

    if not _used_owned_dispatch:
        # Shapely fast paths and general case (only when owned dispatch failed).

        # Fast path: many-vs-one intersection with clip_by_rect (ADR-0033 Tier 2).
        _used_clip_by_rect = False
        if how == "intersection" and right.row_count == 1:
            right_geom = right_shapely_orig[0]
            if right_geom is not None and right_geom.geom_type == "Polygon" and not right_geom.is_empty:
                coords = np.asarray(right_geom.exterior.coords)
                if len(coords) == 5:
                    xs, ys = coords[:4, 0], coords[:4, 1]
                    x_vals = np.unique(xs)
                    y_vals = np.unique(ys)
                    if len(x_vals) == 2 and len(y_vals) == 2 and len(right_geom.interiors) == 0:
                        xmin, xmax = float(x_vals[0]), float(x_vals[1])
                        ymin, ymax = float(y_vals[0]), float(y_vals[1])
                        # Clip all participating left geometries against the rectangle.
                        result_geoms = shapely.clip_by_rect(left_shapely_orig, xmin, ymin, xmax, ymax)
                        _used_clip_by_rect = True

        # Fast path: many-vs-one intersection with GPU centroid pre-filter.
        _used_centroid_filter = False
        if (not _used_clip_by_rect
                and how == "intersection"
                and right.row_count == 1
                and left.row_count >= 100
                and _has_polygonal_families(left)):
            right_geom = right_shapely_orig[0]
            if right_geom is not None and not right_geom.is_empty:
                from vibespatial.constructive.polygon import polygon_centroids_owned
                from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
                _centroid_selection = plan_dispatch_selection(
                    kernel_name="polygon_centroid",
                    kernel_class=KernelClass.METRIC,
                    row_count=left_subset.row_count,
                    geometry_families=tuple(
                        sorted(family.value for family in left_subset.families)
                    ),
                    mixed_geometry=len(left_subset.families) > 1,
                    current_residency=left_subset.residency,
                    requested_mode=ExecutionMode.GPU,
                    gpu_available=cp is not None,
                )
                if _centroid_selection.selected is ExecutionMode.GPU:
                    cx, cy = polygon_centroids_owned(left_subset)
                    centroids = shapely.points(cx, cy)
                    inside_mask = np.asarray(shapely.within(centroids, right_geom), dtype=bool)
                    mask_bounds = right_geom.bounds
                    left_bounds = compute_geometry_bounds(left_subset)
                    inside_idx = np.flatnonzero(inside_mask)
                    if inside_idx.size > 0:
                        bbox_fully_inside = (
                            (left_bounds[inside_idx, 0] >= mask_bounds[0])
                            & (left_bounds[inside_idx, 1] >= mask_bounds[1])
                            & (left_bounds[inside_idx, 2] <= mask_bounds[2])
                            & (left_bounds[inside_idx, 3] >= mask_bounds[1])
                        )
                        fully_inside_rows = inside_idx[bbox_fully_inside]
                    else:
                        fully_inside_rows = np.asarray([], dtype=np.intp)
                    all_rows = np.arange(left_subset.row_count)
                    need_clip_rows = np.setdiff1d(all_rows, fully_inside_rows)
                    if need_clip_rows.size < left_subset.row_count:
                        result_parts_shapely: list = []
                        if fully_inside_rows.size > 0:
                            result_parts_shapely.extend(left_shapely_orig[fully_inside_rows].tolist())
                        if need_clip_rows.size > 0:
                            clip_left = left_shapely_orig[need_clip_rows]
                            clip_right = np.full(len(need_clip_rows), right_geom, dtype=object)
                            clipped = shapely.intersection(clip_left, clip_right)
                            result_parts_shapely.extend(clipped.tolist())
                        result_geoms = np.asarray(result_parts_shapely, dtype=object)
                        _used_centroid_filter = True

        if not _used_clip_by_rect and not _used_centroid_filter:
            # Per-left-group Shapely path: process each left geometry against
            # its overlapping right neighbours independently.  For difference,
            # this computes L_i - union(R_j) to produce correct results.
            if how == "difference":
                result_list: list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    if len(right_neighbors) == 1:
                        right_union = right_neighbors[0]
                    else:
                        right_union = shapely.union_all(right_neighbors)
                    diff = shapely.difference(np.array([left_geom], dtype=object),
                                              np.array([right_union], dtype=object))
                    result_list.append(diff[0])
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "symmetric_difference":
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    if len(right_neighbors) == 1:
                        right_union = right_neighbors[0]
                    else:
                        right_union = shapely.union_all(right_neighbors)
                    sd = shapely.symmetric_difference(
                        np.array([left_geom], dtype=object),
                        np.array([right_union], dtype=object),
                    )
                    result_list.append(sd[0])
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "intersection":
                # Per-pair intersection: L_i intersect R_j for each pair.
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    left_arr = np.full(len(right_neighbors), left_geom, dtype=object)
                    inter = shapely.intersection(left_arr, right_neighbors)
                    result_list.extend(inter.tolist())
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            elif how == "union":
                # Per-pair union: L_i union R_j for each pair.
                result_list = []
                for grp_idx in range(len(unique_left)):
                    start, end = group_starts[grp_idx], group_ends[grp_idx]
                    left_geom = left_shapely_orig[grp_idx]
                    right_neighbors = right_shapely_orig[right_subset_indices[start:end]]
                    left_arr = np.full(len(right_neighbors), left_geom, dtype=object)
                    unions = shapely.union(left_arr, right_neighbors)
                    result_list.extend(unions.tolist())
                result_geoms = np.asarray(result_list, dtype=object) if result_list else np.asarray([], dtype=object)
            else:
                raise ValueError(f"unsupported spatial overlay operation: {how}")

    if not _used_owned_dispatch:
        # Stage 3: Filter out empty/null results (Shapely path only).
        # The owned-dispatch path does its own filtering above.
        result_arr = np.asarray(result_geoms, dtype=object) if not isinstance(result_geoms, np.ndarray) else result_geoms
        non_null = result_arr != None  # noqa: E711 — intentional identity check for numpy
        non_empty_mask = np.zeros(len(result_arr), dtype=bool)
        if np.any(non_null):
            non_empty_mask[non_null] = ~shapely.is_empty(result_arr[non_null])
        candidates = result_arr[non_empty_mask]

        # Check for GeometryCollections that need flattening
        valid_geoms = []
        has_collections = False
        for g in candidates:
            if g.geom_type == "GeometryCollection":
                has_collections = True
                parts = shapely.get_parts(np.asarray([g], dtype=object))
                for part in parts:
                    if part.geom_type in (
                        "Point", "LineString", "Polygon",
                        "MultiPoint", "MultiLineString", "MultiPolygon",
                    ) and not part.is_empty:
                        valid_geoms.append(part)
            else:
                valid_geoms.append(g)

        if not has_collections:
            valid_geoms = list(candidates)

        if not valid_geoms:
            result = from_shapely_geometries([shapely.Point()])
            result = result.take(np.asarray([], dtype=np.int64))
        else:
            result = from_shapely_geometries(valid_geoms)

        # lyy.16 + lyy.18: Combine bypass results with Shapely fallback results.
        result = _combine_bypass_results(
            _containment_result, _sh_clip_result, result,
        )

    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=ExecutionMode.GPU if candidate_pairs.pairs_examined > 0 and cp is not None else ExecutionMode.CPU,
            reason=(
                f"spatial_overlay {how}: {candidate_pairs.count} candidate pairs from "
                f"{left.row_count}x{right.row_count} inputs"
            ),
        )
    )
    return result
