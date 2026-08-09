"""GPU-accelerated geometry equality operations.

geom_equals_exact: element-wise coordinate comparison with tolerance.
    Tier 1 NVRTC kernel for per-pair coordinate comparison (ADR-0033).
    ADR-0002: PREDICATE class, dual fp32/fp64 via PrecisionPlan.

geom_equals_identical: strict byte-level coordinate equality (tolerance=0).
    Delegates to geom_equals_exact with tolerance=0 — no separate kernel.

geom_equals: mutual topological coverage with explicit empty-set handling.
    Uses native predicate expressions in both directions and combines them
    on device before the terminal boolean export.
"""

from __future__ import annotations

import logging

import numpy as np
import shapely

from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import AdaptivePlan, plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_pairwise_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.precision import KernelClass, PrecisionMode

from .buffers import GeometryFamily
from .owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)

logger = logging.getLogger(__name__)


def geom_equals_exact_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise geometry equality with tolerance.

    Returns a bool array of shape (row_count,).  GPU path compares
    coordinate buffers directly (no Shapely round-trip).  Falls back
    to Shapely for row count below threshold or when GPU is unavailable.
    """
    row_count = left.row_count
    if row_count != right.row_count:
        raise ValueError(
            f"left and right must have same row count, got {row_count} vs {right.row_count}"
        )
    if row_count == 0:
        return np.empty(0, dtype=bool)

    selection = plan_dispatch_selection(
        kernel_name="geom_equals_exact",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        current_residency=combined_residency(left, right),
        work_estimate=estimate_pairwise_work_from_owned(
            left,
            right,
            output_row_count=row_count,
            primary_unit_name="equals-exact-pair-coordinate",
        ),
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _geom_equals_exact_gpu(left, right, tolerance, selection)
            if result is not None:
                record_dispatch_event(
                    surface="geom_equals_exact",
                    operation="geom_equals_exact",
                    implementation="gpu_nvrtc_equals_exact",
                    reason="NVRTC kernel coordinate comparison on device",
                    detail=f"rows={row_count}, tolerance={tolerance}",
                    selected=ExecutionMode.GPU,
                )
                return result
        except Exception:
            logger.debug("equals_exact GPU path failed, falling back to CPU", exc_info=True)

    return _geom_equals_exact_cpu(left, right, tolerance)


def geom_equals_identical_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise strict geometry identity (bitwise coordinate equality).

    Equivalent to ``geom_equals_exact(..., tolerance=0)`` — delegates to the
    same NVRTC kernel infrastructure with zero tolerance.

    Returns a bool array of shape (row_count,).  Null geometries always
    compare as False (Shapely convention).
    """
    return geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=dispatch_mode)


def _geom_equals_exact_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> np.ndarray:
    """CPU path: delegate to Shapely equals_exact."""
    import shapely

    record_dispatch_event(
        surface="geom_equals_exact",
        operation="geom_equals_exact",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={left.row_count}, tolerance={tolerance}",
        selected=ExecutionMode.CPU,
    )
    left_geoms = np.asarray(left.to_shapely(), dtype=object)
    right_geoms = np.asarray(right.to_shapely(), dtype=object)
    return shapely.equals_exact(left_geoms, right_geoms, tolerance=tolerance)


def _geom_equals_exact_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
    runtime_selection: AdaptivePlan,
) -> np.ndarray | None:
    """GPU path: NVRTC kernel coordinate comparison per geometry pair.

    Zero D2H transfers during computation.  Tag comparison, null masking,
    and per-family kernel launches all happen on device.  A single D2H
    transfer at the end returns the final bool array.

    Returns None if GPU comparison is not feasible, triggering CPU fallback.
    """
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    d_result = _geom_equals_exact_gpu_device(
        left,
        right,
        tolerance,
        runtime_selection,
    )
    if d_result is None:
        return None

    runtime = get_cuda_runtime()
    runtime.synchronize()
    result_host = runtime.copy_device_to_host(
        d_result,
        reason="geometry equality result host export",
        terminal_export=True,
    )
    return result_host.astype(bool, copy=False)


def _geom_equals_exact_gpu_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
    runtime_selection: AdaptivePlan,
):
    """Return structural equality as a device int32 vector."""
    try:
        import cupy as cp
    except ImportError:
        return None

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.kernels.predicates.equals_exact import launch_equals_exact_family
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    runtime = get_cuda_runtime()
    row_count = left.row_count

    # --- ADR-0002: select precision plan ---
    precision_plan = runtime_selection.precision_plan
    compute_type = "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"

    # Ensure both arrays have device state
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()

    # --- Step 1: Tag + validity filtering on device (CuPy Tier 2) ---
    # All operations use device arrays — zero D2H transfers.
    d_left_tags = left_state.tags  # int8 device array
    d_right_tags = right_state.tags  # int8 device array
    d_left_validity = left_state.validity  # bool device array
    d_right_validity = right_state.validity  # bool device array

    # tag_match: True where both are valid and same family tag
    d_tag_match = (d_left_tags == d_right_tags) & d_left_validity & d_right_validity

    # Allocate output on device — zero-filled (False by default)
    d_result = runtime.allocate((row_count,), np.int32, zero=True)

    # --- Step 2: Per-family kernel dispatch ---
    for family_key in GeometryFamily:
        tag = FAMILY_TAGS[family_key]

        # Find rows matching this family on device (CuPy)
        d_family_mask = d_tag_match & (d_left_tags == np.int8(tag))
        d_family_rows = cp.flatnonzero(d_family_mask).astype(cp.int32)

        if d_family_rows.shape[0] == 0:
            continue

        # Get device buffers for this family (dict lookup, not CuPy .get())
        left_buf = left_state.families[family_key] if family_key in left_state.families else None
        right_buf = right_state.families[family_key] if family_key in right_state.families else None
        if left_buf is None or right_buf is None:
            continue

        # Check that required offset buffers are present
        if family_key == GeometryFamily.POLYGON:
            if left_buf.ring_offsets is None or right_buf.ring_offsets is None:
                continue
        elif family_key == GeometryFamily.MULTILINESTRING:
            if left_buf.part_offsets is None or right_buf.part_offsets is None:
                continue
        elif family_key == GeometryFamily.MULTIPOLYGON:
            if (
                left_buf.part_offsets is None
                or right_buf.part_offsets is None
                or left_buf.ring_offsets is None
                or right_buf.ring_offsets is None
            ):
                continue

        # Launch per-family NVRTC kernel — returns device int32 array
        d_family_result = launch_equals_exact_family(
            left_state=left_state,
            right_state=right_state,
            left_buf=left_buf,
            right_buf=right_buf,
            family=family_key,
            row_indices_device=d_family_rows,
            tolerance=tolerance,
            compute_type=compute_type,
        )

        # Scatter results back into the full output buffer on device
        if d_family_result is not None and d_family_result.shape[0] > 0:
            d_result[d_family_rows] = d_family_result

    return d_result


def geom_equals_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise topological geometry equality.

    Equality is mutual coverage for non-empty geometries. Empty geometries of
    every supported family represent the same empty point set and therefore
    compare equal when both rows are valid and empty. The GPU path evaluates
    both coverage directions as native expressions, combines them with the
    empty-set rule on device, and performs one terminal boolean export.

    Returns a bool array of shape (row_count,).  Null geometries always
    compare as False (Shapely convention).
    """
    row_count = left.row_count
    if row_count != right.row_count:
        raise ValueError(
            f"left and right must have same row count, got {row_count} vs {right.row_count}"
        )
    if row_count == 0:
        return np.empty(0, dtype=bool)

    requested_mode = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    selection = plan_dispatch_selection(
        kernel_name="geom_equals",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=requested_mode,
        current_residency=combined_residency(left, right),
        work_estimate=estimate_pairwise_work_from_owned(
            left,
            right,
            output_row_count=row_count,
            primary_unit_name="equals-mutual-coverage-pair",
        ),
    )

    if selection.selected is ExecutionMode.GPU:
        result = _geom_equals_topological_gpu(left, right, selection)
        if result is not None:
            record_dispatch_event(
                surface="geom_equals",
                operation="geom_equals",
                implementation="gpu_mutual_coverage_expression",
                reason="native mutual covered_by expressions with device empty-set handling",
                detail=f"rows={row_count}",
                selected=ExecutionMode.GPU,
            )
            return result
        if requested_mode is ExecutionMode.GPU:
            raise NotImplementedError(
                "geom_equals GPU execution requires native mutual-coverage expressions"
            )

    record_dispatch_event(
        surface="geom_equals",
        operation="geom_equals",
        implementation="shapely",
        reason=(
            selection.reason
            if selection.selected is ExecutionMode.CPU
            else "native mutual-coverage expressions unavailable"
        ),
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    left_geoms = np.asarray(left.to_shapely(), dtype=object)
    right_geoms = np.asarray(right.to_shapely(), dtype=object)
    return shapely.equals(left_geoms, right_geoms).astype(bool, copy=False)


def _geom_equals_topological_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    runtime_selection: AdaptivePlan,
) -> np.ndarray | None:
    """Evaluate topological equality with one terminal device-to-host export."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        device_valid_nonempty_mask,
    )
    from vibespatial.predicates.binary import binary_predicate_expression
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    d_exact = _geom_equals_exact_gpu_device(
        left,
        right,
        0.0,
        runtime_selection,
    )
    if d_exact is None:
        return None

    import cupy as cp

    # Preserve the full row capacity while marking structural matches inactive.
    # Exact round trips then avoid relation refinement without a host candidate
    # discovery or allocation fence.
    d_unresolved = ~cp.asarray(d_exact, dtype=cp.bool_)
    d_rows = cp.arange(left.row_count, dtype=cp.int64)
    left_unresolved = left._device_indexed_take(
        d_rows,
        assume_unique_indices=True,
    )._apply_row_activity(
        d_unresolved,
        assume_active_indices_unique=True,
    )
    right_unresolved = right._device_indexed_take(
        d_rows,
        assume_unique_indices=True,
    )._apply_row_activity(
        d_unresolved,
        assume_active_indices_unique=True,
    )

    left_covered = binary_predicate_expression(
        "covered_by",
        left_unresolved,
        right_unresolved,
        dispatch_mode=ExecutionMode.GPU,
        operation="geom_equals.left_covered_by_right",
    )
    right_covered = binary_predicate_expression(
        "covered_by",
        right_unresolved,
        left_unresolved,
        dispatch_mode=ExecutionMode.GPU,
        operation="geom_equals.right_covered_by_left",
    )
    if left_covered is None or right_covered is None:
        return None

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    d_valid = cp.asarray(left_state.validity, dtype=cp.bool_) & cp.asarray(
        right_state.validity,
        dtype=cp.bool_,
    )
    d_left_nonempty = device_valid_nonempty_mask(left)
    d_right_nonempty = device_valid_nonempty_mask(right)
    d_both_empty = d_valid & ~d_left_nonempty & ~d_right_nonempty
    d_result = d_valid & (
        cp.asarray(d_exact, dtype=cp.bool_)
        | (
            cp.asarray(left_covered.values, dtype=cp.bool_)
            & cp.asarray(right_covered.values, dtype=cp.bool_)
        )
        | d_both_empty
    )

    # Coverage predicates still preserve line and multipart structure in
    # several refine paths.  Point-set equality for lineal and polygonal rows
    # is exactly bidirectional difference emptiness, which also handles
    # redundant vertices, single-part multi representations, and reordered
    # components without host normalization.
    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)
    d_right_tags = cp.asarray(right_state.tags, dtype=cp.int8)
    d_lineal = (
        d_valid
        & d_unresolved
        & (
            (d_left_tags == FAMILY_TAGS[GeometryFamily.LINESTRING])
            | (d_left_tags == FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
        )
        & (
            (d_right_tags == FAMILY_TAGS[GeometryFamily.LINESTRING])
            | (d_right_tags == FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
        )
    )
    d_polygonal = (
        d_valid
        & d_unresolved
        & (
            (d_left_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
            | (d_left_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
        )
        & (
            (d_right_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
            | (d_right_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
        )
    )

    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    for d_dimension_rows in (d_lineal, d_polygonal):
        d_positions = cp.flatnonzero(d_dimension_rows).astype(cp.int64, copy=False)
        if int(d_positions.size) == 0:
            continue
        left_partition = left.device_take(d_positions).physicalize_device_rows(
            allow_capacity_allocation=True,
        )
        right_partition = right.device_take(d_positions).physicalize_device_rows(
            allow_capacity_allocation=True,
        )
        left_difference = binary_constructive_owned(
            "difference",
            left_partition,
            right_partition,
            dispatch_mode=ExecutionMode.GPU,
        )
        right_difference = binary_constructive_owned(
            "difference",
            right_partition,
            left_partition,
            dispatch_mode=ExecutionMode.GPU,
        )
        if left_difference is None or right_difference is None:
            return None
        d_result[d_positions] = ~device_valid_nonempty_mask(
            left_difference,
        ) & ~device_valid_nonempty_mask(right_difference)

    from vibespatial.cuda._runtime import get_cuda_runtime

    return (
        get_cuda_runtime()
        .copy_device_to_host(
            d_result,
            reason="geometry topological equality result host export",
            terminal_export=True,
        )
        .astype(bool, copy=False)
    )
