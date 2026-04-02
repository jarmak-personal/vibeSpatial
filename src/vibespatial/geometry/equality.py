"""GPU-accelerated geometry equality operations.

geom_equals_exact: element-wise coordinate comparison with tolerance.
    Tier 1 NVRTC kernel for per-pair coordinate comparison (ADR-0033).
    ADR-0002: PREDICATE class, dual fp32/fp64 via PrecisionPlan.

geom_equals_identical: strict byte-level coordinate equality (tolerance=0).
    Delegates to geom_equals_exact with tolerance=0 — no separate kernel.

geom_equals: normalize-then-compare (composes normalize + equals_exact).
    Inherits dual-precision from both operations.
"""

from __future__ import annotations

import logging

import numpy as np
import shapely

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import AdaptivePlan, plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.precision import KernelClass, PrecisionMode

from .buffers import GeometryFamily
from .owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)

logger = logging.getLogger(__name__)

_EQUALS_EXACT_GPU_THRESHOLD = 1000


def requires_mixed_family_topology_fallback(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    valid_pairs = np.asarray(left.validity & right.validity, dtype=bool)
    if not valid_pairs.any():
        return False
    return bool(np.any(left.tags[valid_pairs] != right.tags[valid_pairs]))


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
    )

    if selection.selected is ExecutionMode.GPU and row_count >= _EQUALS_EXACT_GPU_THRESHOLD:
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
    compute_type = (
        "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"
    )

    # Ensure both arrays have device state
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()

    # --- Step 1: Tag + validity filtering on device (CuPy Tier 2) ---
    # All operations use device arrays — zero D2H transfers.
    d_left_tags = left_state.tags       # int8 device array
    d_right_tags = right_state.tags     # int8 device array
    d_left_validity = left_state.validity   # bool device array
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
            if (left_buf.part_offsets is None or right_buf.part_offsets is None
                    or left_buf.ring_offsets is None or right_buf.ring_offsets is None):
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

    # --- Step 3: Single D2H transfer for final result ---
    runtime.synchronize()
    result_host = runtime.copy_device_to_host(d_result)
    return result_host.astype(bool, copy=False)


def geom_equals_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise topological geometry equality.

    For same-family pairs, topological equality is implemented as structural
    equality after normalization. Mixed-family valid pairs, such as ``Polygon``
    vs single-part ``MultiPolygon``, require the full Shapely topology path.

    Both sub-operations have GPU paths, so the full pipeline runs on device
    when GPU is available.  Zero D2H transfers until the final bool array
    is returned by ``geom_equals_exact_owned``.

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
    if requires_mixed_family_topology_fallback(left, right):
        record_dispatch_event(
            surface="geom_equals",
            operation="geom_equals",
            implementation="shapely_mixed_family_topology",
            reason="mixed valid geometry families require full topological equality",
            detail=f"rows={row_count}",
            selected=ExecutionMode.CPU,
        )
        left_geoms = np.asarray(left.to_shapely(), dtype=object)
        right_geoms = np.asarray(right.to_shapely(), dtype=object)
        return shapely.equals(left_geoms, right_geoms).astype(bool, copy=False)

    from vibespatial.constructive.normalize import normalize_owned

    left_norm = normalize_owned(left, dispatch_mode=dispatch_mode)
    right_norm = normalize_owned(right, dispatch_mode=dispatch_mode)
    return geom_equals_exact_owned(left_norm, right_norm, tolerance=1e-12, dispatch_mode=dispatch_mode)
