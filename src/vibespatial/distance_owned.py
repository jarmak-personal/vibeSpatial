"""On-device element-wise distance and dwithin for OwnedGeometryArray.

Orchestrates existing GPU distance kernels (point_distance, segment_distance,
spatial_nearest point-point) for the public distance/dwithin API surface.
Zero geometry H/D transfers when GPU is available -- only small index arrays
and the final result array cross the bus.

METRIC kernel class per ADR-0002.  Precision dispatch is forwarded to
point_distance (already METRIC-compliant); segment_distance stays fp64
(compliance gap tracked separately).
"""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.adaptive_runtime import plan_dispatch_selection
from vibespatial.cuda_runtime import get_cuda_runtime
from vibespatial.dispatch import record_dispatch_event
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import (
    OwnedGeometryArray,
    TAG_FAMILIES,
    DiagnosticKind,
)
from vibespatial.precision import KernelClass, PrecisionMode
from vibespatial.residency import Residency, TransferTrigger
from vibespatial.runtime import ExecutionMode


# ---------------------------------------------------------------------------
# Point-distance family support (mirrors point_distance._FAMILY_KERNEL_MAP)
# ---------------------------------------------------------------------------

_POINT_DISTANCE_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.LINESTRING,
    GeometryFamily.MULTILINESTRING,
    GeometryFamily.POLYGON,
    GeometryFamily.MULTIPOLYGON,
})

# Segment-distance family support
_SEGMENT_FAMILIES: frozenset[GeometryFamily] = frozenset({
    GeometryFamily.LINESTRING,
    GeometryFamily.MULTILINESTRING,
    GeometryFamily.POLYGON,
    GeometryFamily.MULTIPOLYGON,
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distance_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Element-wise distance between two OwnedGeometryArrays.

    Returns float64 array of length ``left.row_count``.  NaN for null
    geometries, inf when both geometries are empty.

    On GPU: zero geometry H/D transfers.  Only small index arrays and the
    final float64 result cross the bus.
    """
    n = left.row_count
    if right.row_count != n:
        raise ValueError(
            f"Lengths do not match: left={n}, right={right.row_count}"
        )
    if n == 0:
        return np.empty(0, dtype=np.float64)

    if isinstance(precision, str):
        precision = PrecisionMode(precision)

    selection = plan_dispatch_selection(
        kernel_name="geometry_distance",
        kernel_class=KernelClass.METRIC,
        row_count=n,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        try:
            result = _distance_gpu(left, right, precision)
            record_dispatch_event(
                surface="distance_owned",
                operation="distance",
                implementation="distance_owned_gpu",
                reason="element-wise distance via owned GPU kernels",
                detail=f"rows={n}",
                selected=ExecutionMode.GPU,
            )
            return result
        except Exception:
            pass  # fall through to CPU

    return _distance_cpu(left, right)


def dwithin_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    threshold: float | np.ndarray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Element-wise dwithin: ``distance(left[i], right[i]) <= threshold``.

    Returns boolean array.  Null geometries produce False.
    *threshold* may be a scalar or per-row array.
    """
    distances = distance_owned(
        left, right, dispatch_mode=dispatch_mode, precision=precision,
    )
    # NaN rows (null geometry) -> False for dwithin
    return np.where(np.isnan(distances), False, distances <= threshold)


# ---------------------------------------------------------------------------
# GPU path
# ---------------------------------------------------------------------------

def _distance_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    precision: PrecisionMode,
) -> np.ndarray:
    """Pure-GPU element-wise distance.

    Groups rows by (left_tag, right_tag) and dispatches to the appropriate
    existing distance kernel per group.  Geometry data stays device-resident;
    only small index sub-arrays and per-group result sub-arrays are transferred.
    """
    from vibespatial.point_distance import compute_point_distance_gpu
    from vibespatial.segment_distance import compute_segment_distance_gpu
    from vibespatial.spatial_nearest import (
        _compute_multipoint_distances_gpu,
        _launch_point_point_distance_kernel,
    )

    n = left.row_count
    runtime = get_cuda_runtime()

    # Ensure geometry buffers are device-resident (no-op if already there).
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="distance_owned: left geometry for GPU distance",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="distance_owned: right geometry for GPU distance",
    )

    # Result: NaN for null rows, will be overwritten for valid pairs.
    result = np.full(n, np.nan, dtype=np.float64)

    left_tags = left.tags
    right_tags = right.tags
    left_valid = left.validity
    right_valid = right.validity

    # Only process rows where both sides are valid (non-null).
    both_valid = left_valid & right_valid
    valid_idx = np.flatnonzero(both_valid)
    if valid_idx.size == 0:
        return result

    valid_left_tags = left_tags[valid_idx]
    valid_right_tags = right_tags[valid_idx]

    PT = GeometryFamily.POINT
    MP = GeometryFamily.MULTIPOINT

    # Group valid rows by (left_tag, right_tag).
    unique_pairs = set(zip(valid_left_tags.tolist(), valid_right_tags.tolist()))

    for lt, rt in unique_pairs:
        lf = TAG_FAMILIES.get(lt)
        rf = TAG_FAMILIES.get(rt)
        if lf is None or rf is None:
            continue  # unknown tag -- skip (shouldn't happen)

        sub_mask = (valid_left_tags == lt) & (valid_right_tags == rt)
        sub_valid_pos = np.flatnonzero(sub_mask)
        sub_idx = valid_idx[sub_valid_pos]
        sub_count = sub_idx.size
        if sub_count == 0:
            continue

        # For element-wise distance, left_idx[i] == right_idx[i] == sub_idx[i]
        # because row k of left is paired with row k of right.
        d_idx = runtime.from_host(sub_idx.astype(np.int32))

        ok = False

        if lf == PT and rf == PT:
            # Point x Point
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                _launch_point_point_distance_kernel(
                    left, right, d_idx, d_idx, d_sub_dist, sub_count,
                )
                sub_distances = np.empty(sub_count, dtype=np.float64)
                runtime.copy_device_to_host(d_sub_dist, sub_distances)
                result[sub_idx] = sub_distances
                ok = True
            finally:
                runtime.free(d_sub_dist)

        elif lf == PT and rf in _POINT_DISTANCE_FAMILIES:
            # Point x {LS, MLS, PG, MPG}
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                ok = compute_point_distance_gpu(
                    left, right, d_idx, d_idx, d_sub_dist, sub_count,
                    tree_family=rf, compute_precision=precision,
                )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(d_sub_dist, sub_distances)
                    result[sub_idx] = sub_distances
            finally:
                runtime.free(d_sub_dist)

        elif rf == PT and lf in _POINT_DISTANCE_FAMILIES:
            # {LS, MLS, PG, MPG} x Point -- swap so point is query side
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                ok = compute_point_distance_gpu(
                    right, left, d_idx, d_idx, d_sub_dist, sub_count,
                    tree_family=lf, compute_precision=precision,
                )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(d_sub_dist, sub_distances)
                    result[sub_idx] = sub_distances
            finally:
                runtime.free(d_sub_dist)

        elif lf == MP or rf == MP:
            # MultiPoint involved -- use expansion + segmented min.
            # _compute_multipoint_distances_gpu expects host index arrays.
            if lf == MP and rf == MP:
                # MP x MP: Shapely fallback for this sub-group only.
                pass  # ok stays False -> falls to Shapely below
            elif lf == MP:
                mp_result = _compute_multipoint_distances_gpu(
                    left, right, sub_idx, sub_idx,
                    target_family=rf,
                )
                if mp_result is not None:
                    result[sub_idx] = mp_result
                    ok = True
            else:
                mp_result = _compute_multipoint_distances_gpu(
                    right, left, sub_idx, sub_idx,
                    target_family=lf,
                )
                if mp_result is not None:
                    result[sub_idx] = mp_result
                    ok = True

        elif lf in _SEGMENT_FAMILIES and rf in _SEGMENT_FAMILIES:
            # {LS, MLS, PG, MPG} x {LS, MLS, PG, MPG}
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                ok = compute_segment_distance_gpu(
                    left, right, d_idx, d_idx, d_sub_dist, sub_count,
                    query_family=lf, tree_family=rf,
                )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(d_sub_dist, sub_distances)
                    result[sub_idx] = sub_distances
            finally:
                runtime.free(d_sub_dist)

        # Free shared device index array.
        runtime.free(d_idx)

        # Shapely fallback for this sub-group if GPU kernel was not available.
        if not ok:
            left_shapely = np.asarray(left.to_shapely(), dtype=object)
            right_shapely = np.asarray(right.to_shapely(), dtype=object)
            sub_dists = shapely.distance(
                left_shapely[sub_idx], right_shapely[sub_idx],
            )
            result[sub_idx] = np.asarray(sub_dists, dtype=np.float64)

    return result


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def _distance_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    """CPU distance via Shapely.  Records a diagnostic event."""
    left._record(
        DiagnosticKind.MATERIALIZATION,
        "distance_owned: CPU fallback via Shapely",
        visible=True,
    )
    record_dispatch_event(
        surface="distance_owned",
        operation="distance",
        implementation="shapely_cpu",
        reason="GPU not available or not selected for distance",
        detail=f"rows={left.row_count}",
        selected=ExecutionMode.CPU,
    )
    left_shapely = np.asarray(left.to_shapely(), dtype=object)
    right_shapely = np.asarray(right.to_shapely(), dtype=object)
    return np.asarray(shapely.distance(left_shapely, right_shapely), dtype=np.float64)
