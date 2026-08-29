"""Exact device-native grouped union for simple Point rows.

Physical shape (ADR-0046): ``OwnedGeometryArray[Point] + NativeGrouped`` is
lowered to a segmented point-set reduction.  Work is measured in source
points, groups, unique output coordinates, output bytes, and radix-sort
scratch.  The staging layout is a lexicographically sorted ``(group, x, y)``
partition followed by adjacent duplicate elimination and grouped offset
assembly.  Dynamic output stays in source-row coordinate capacity while the
device group offsets carry the logical unique-coordinate count, avoiding a
host allocation packet.  The native output is an owned device array containing
Point and MultiPoint rows; public materialization remains the dissolve export
boundary.

The operation is constructive but creates no coordinates.  Canonical fp64
storage values are gathered unchanged, so no lower-precision compute variant
is applicable.
"""

from __future__ import annotations

import numpy as np

from vibespatial.api._native_grouped import NativeGrouped
from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    seed_all_validity_cache,
)
from vibespatial.overlay.graph import _fp64_radix_keys, _stable_radix_order_pass
from vibespatial.runtime import ExecutionMode, get_requested_mode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_grouped_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.execution_trace import notify_transfer
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None


request_warmup(
    [
        "radix_sort_i32_i32",
        "radix_sort_u64_i32",
    ]
)


def supports_grouped_point_union(owned: OwnedGeometryArray) -> bool:
    """Return whether direct point-set assembly is structurally possible.

    This support probe is deliberately metadata-only.  The crossover planner
    must be able to reject small host workloads before paying for validity,
    emptiness, or coordinate scans.  Device-native carriers use their trusted
    structural proofs and never materialize host metadata for this probe.
    """
    if owned.row_count == 0:
        return False
    if owned.device_state is not None:
        state = owned.device_state
        return bool(
            state.trusted_all_valid is True
            and state.trusted_all_non_empty is True
            and state.trusted_homogeneous_family is GeometryFamily.POINT
            and GeometryFamily.POINT in state.families
        )
    point_buffer = owned.families.get(GeometryFamily.POINT)
    return bool(
        point_buffer is not None
        and int(point_buffer.row_count) == int(owned.row_count)
        and set(owned.families) == {GeometryFamily.POINT}
    )


def _grouped_point_union_semantically_admissible(owned: OwnedGeometryArray) -> bool:
    """Check exact Point-union semantics after device execution is selected.

    Trusted device metadata is the preferred admissibility contract.  A
    missing proof, or a source-wide negative proof when grouping dropped some
    rows, is resolved over the observed Point rows inside the device executor;
    it is not a reason to decline an otherwise exact native carrier.  Host
    carriers retain the conservative finite, non-null, and non-empty scan used
    by the original implementation.
    """
    state = owned.device_state
    if state is not None:
        return bool(
            state.trusted_all_valid is True
            and state.trusted_all_non_empty is True
            and state.trusted_homogeneous_family is GeometryFamily.POINT
            and GeometryFamily.POINT in state.families
        )

    point_buffer = owned.families.get(GeometryFamily.POINT)
    return bool(
        point_buffer is not None
        and point_buffer.host_materialized
        and int(point_buffer.row_count) == int(owned.row_count)
        and owned._validity is not None
        and owned._tags is not None
        and np.all(owned._validity)
        and np.all(owned._tags == FAMILY_TAGS[GeometryFamily.POINT])
        and not np.any(point_buffer.empty_mask)
        and np.all(np.isfinite(point_buffer.x))
        and np.all(np.isfinite(point_buffer.y))
    )


@register_kernel_variant(
    "segmented_point_union",
    "gpu-cccl-radix",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    geometry_families=("point",),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("cccl", "segmented", "constructive", "native-output"),
)
def _segmented_point_union_gpu(
    grouped: NativeGrouped,
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Sort, deduplicate, and assemble exact grouped Point unions on device."""
    if cp is None or owned.residency is not Residency.DEVICE:
        return None
    if grouped.sorted_order is None or grouped.group_ids is None:
        return None
    group_count = grouped.resolved_group_count
    if group_count <= 0 or int(grouped.row_count or -1) != int(owned.row_count):
        return None
    observed_row_count = int(grouped.sorted_order.size)
    if observed_row_count <= 0 or observed_row_count > int(owned.row_count):
        return None
    observed_group_count = int(grouped.group_ids.size)
    if observed_group_count <= 0 or observed_group_count > group_count:
        return None

    physical = (
        owned.physicalize_device_rows(allow_capacity_allocation=True)
        if owned.is_indexed_view
        else owned
    )
    state = physical._ensure_device_state(preserve_indexed_view=True)
    point_buffer = state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        return None

    source_row_count = int(physical.row_count)
    if hasattr(grouped.group_codes, "__cuda_array_interface__"):
        d_codes = cp.asarray(grouped.group_codes, dtype=cp.int32)
    else:
        # Public dissolve grouping is currently pandas-owned.  This is the one
        # required tabular-ingress transfer for that shape; route it through
        # the runtime rather than letting ``cp.asarray`` hide an unmanaged H2D
        # copy.  Device-native NativeGrouped carriers reuse their codes without
        # any transfer.
        host_codes = np.ascontiguousarray(grouped.group_codes, dtype=np.int32)
        d_codes = get_cuda_runtime().from_host(host_codes)
        notify_transfer(
            direction="h2d",
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST.value,
            reason="grouped Point union dense group-code ingress",
            source="cuda_runtime",
            item_count=source_row_count,
            bytes_transferred=int(host_codes.nbytes),
        )
    if int(d_codes.size) != source_row_count:
        return None
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    if observed_row_count != source_row_count:
        if hasattr(grouped.sorted_order, "__cuda_array_interface__"):
            d_observed_rows = cp.asarray(grouped.sorted_order, dtype=cp.int64)
        else:
            host_observed_rows = np.ascontiguousarray(
                grouped.sorted_order,
                dtype=np.int64,
            )
            d_observed_rows = get_cuda_runtime().from_host(host_observed_rows)
            notify_transfer(
                direction="h2d",
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST.value,
                reason="grouped Point union observed-row ingress",
                source="cuda_runtime",
                item_count=observed_row_count,
                bytes_transferred=int(host_observed_rows.nbytes),
            )
        d_codes = d_codes[d_observed_rows]
        d_family_rows = d_family_rows[d_observed_rows]
    d_point_offsets = cp.asarray(point_buffer.geometry_offsets, dtype=cp.int64)
    d_coord_rows = d_point_offsets[d_family_rows]
    d_x = cp.asarray(point_buffer.x, dtype=cp.float64)[d_coord_rows]
    d_y = cp.asarray(point_buffer.y, dtype=cp.float64)[d_coord_rows]

    if (
        state.trusted_all_finite_coordinates is False
        and observed_row_count == source_row_count
    ):
        return None
    if state.trusted_all_finite_coordinates is not True:
        finite_device = (
            cp.all(cp.isfinite(d_x)) & cp.all(cp.isfinite(d_y))
        ).reshape(1)
        finite_host = get_cuda_runtime().copy_device_to_host(
            finite_device,
            reason="grouped Point union finite-coordinate admission scalar fence",
        )
        all_finite = bool(np.asarray(finite_host).reshape(-1)[0])
        if observed_row_count == source_row_count:
            state.trusted_all_finite_coordinates = all_finite
        if not all_finite:
            return None

    row_count = observed_row_count

    # GEOS treats negative and positive zero as the same coordinate. Normalize
    # before radix key creation so equivalent zeros cannot land in separated
    # sub-partitions during the stable x/y passes.
    d_x = cp.where(d_x == 0.0, cp.float64(0.0), d_x)
    d_y = cp.where(d_y == 0.0, cp.float64(0.0), d_y)

    order = cp.arange(row_count, dtype=cp.int32)
    y_keys = _fp64_radix_keys(d_y)
    order = _stable_radix_order_pass(order, y_keys)
    del y_keys
    x_keys = _fp64_radix_keys(d_x)
    order = _stable_radix_order_pass(order, x_keys)
    del x_keys
    order = _stable_radix_order_pass(order, d_codes)

    sorted_codes = d_codes[order]
    sorted_x = d_x[order]
    sorted_y = d_y[order]
    unique_mask = cp.empty(row_count, dtype=cp.bool_)
    unique_mask[0] = True
    unique_mask[1:] = (
        (sorted_codes[1:] != sorted_codes[:-1])
        | (sorted_x[1:] != sorted_x[:-1])
        | (sorted_y[1:] != sorted_y[:-1])
    )
    # Keep dynamic unique cardinality on device.  CuPy boolean compaction
    # performs a scalar D2H allocation fence for each selected array, and
    # ``bincount`` performs another planning reduction.  A fixed-capacity
    # count/scan/scatter plan has the same O(rows) memory bound, exposes the
    # logical coordinate count through the terminal group offsets, and never
    # needs a host-known intermediate size.
    unique_i32 = unique_mask.astype(cp.int32, copy=False)
    unique_rank = cp.cumsum(unique_i32, dtype=cp.int32) - cp.int32(1)
    scatter_rows = cp.where(unique_mask, unique_rank, cp.int32(row_count))
    unique_x_capacity = cp.empty(row_count + 1, dtype=cp.float64)
    unique_y_capacity = cp.empty(row_count + 1, dtype=cp.float64)
    unique_x_capacity[scatter_rows] = sorted_x
    unique_y_capacity[scatter_rows] = sorted_y
    unique_x = unique_x_capacity[:row_count]
    unique_y = unique_y_capacity[:row_count]

    counts = cp.zeros(group_count, dtype=cp.int32)
    cp.add.at(counts, sorted_codes, unique_i32)
    offsets = cp.empty(group_count + 1, dtype=cp.int32)
    offsets[0] = 0
    cp.cumsum(counts, dtype=cp.int32, out=offsets[1:])

    point_buffer_out = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=unique_x,
        y=unique_y,
        geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
        empty_mask=cp.zeros(row_count, dtype=cp.bool_),
        bounds=None,
    )
    multipoint_buffer_out = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=unique_x,
        y=unique_y,
        geometry_offsets=offsets,
        empty_mask=counts <= 1,
        bounds=None,
    )
    singleton = counts == 1
    output_rows = cp.arange(group_count, dtype=cp.int32)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: point_buffer_out,
            GeometryFamily.MULTIPOINT: multipoint_buffer_out,
        },
        row_count=group_count,
        tags=cp.where(
            singleton,
            cp.int8(FAMILY_TAGS[GeometryFamily.POINT]),
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT]),
        ),
        validity=cp.ones(group_count, dtype=cp.bool_),
        family_row_offsets=cp.where(singleton, offsets[:-1], output_rows).astype(
            cp.int32,
            copy=False,
        ),
        execution_mode="gpu",
    )
    all_groups_observed = observed_group_count == group_count
    result_state = result.device_state
    if result_state is not None:
        result_state.trusted_all_valid = True
        result_state.trusted_all_non_empty = all_groups_observed
        result_state.trusted_family_domain = (
            GeometryFamily.POINT,
            GeometryFamily.MULTIPOINT,
        )
        result_state.trusted_unique_family_rows = False
    seed_all_validity_cache(result)
    result._native_grouped_union_implementation = "native_segmented_point_set_union"
    result._grouped_union_empty_geometry_collection_mask = counts == 0
    return result


def grouped_point_union_owned(
    grouped: NativeGrouped,
    owned: OwnedGeometryArray,
    *,
    _admitted: bool = False,
) -> OwnedGeometryArray | None:
    """Plan and execute the admitted grouped Point union physical shape."""
    if cp is None or not has_gpu_runtime():
        return None
    # ``_admitted`` means the caller already ran the metadata-only structural
    # probe.  Exact semantic admission still happens after crossover planning.
    if not _admitted and not supports_grouped_point_union(owned):
        return None
    group_count = grouped.resolved_group_count
    if group_count <= 0 or int(grouped.row_count or -1) != int(owned.row_count):
        return None
    if grouped.sorted_order is None or grouped.group_ids is None:
        return None
    observed_row_count = int(grouped.sorted_order.size)
    if observed_row_count <= 0 or observed_row_count > int(owned.row_count):
        return None
    observed_group_count = int(grouped.group_ids.size)
    if observed_group_count <= 0 or observed_group_count > group_count:
        return None

    estimate = estimate_grouped_work_from_owned(
        owned,
        grouped=grouped,
        output_row_count=group_count,
        output_byte_count=observed_row_count * 16 + group_count * 40,
        temporary_byte_count=observed_row_count * 48 + group_count * 8,
        primary_unit_name="grouped-point",
    )
    plan = plan_dispatch_selection(
        kernel_name="segmented_point_union",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=int(owned.row_count),
        requested_mode=get_requested_mode(),
        geometry_families=("point",),
        current_residency=owned.residency,
        work_estimate=estimate,
    )
    if plan.selected is not ExecutionMode.GPU:
        record_dispatch_event(
            surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
            operation="grouped_point_union",
            implementation="existing_exact_grouped_union",
            reason=plan.reason,
            detail=estimate.telemetry_detail(),
            requested=plan.requested,
            selected=plan.selected,
        )
        return None

    if not _grouped_point_union_semantically_admissible(owned):
        return None

    if owned.residency is not Residency.DEVICE:
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="segmented grouped Point union selected device execution",
        )
    result = _segmented_point_union_gpu(grouped, owned)
    if result is None:
        return None

    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
        operation="grouped_point_union",
        implementation="native_segmented_point_set_union",
        reason=(
            "exact grouped Point union used device radix partitioning, "
            "coordinate deduplication, and direct Point/MultiPoint assembly"
        ),
        detail=(
            f"{estimate.telemetry_detail()}, coordinate_capacity="
            f"{int(result.device_state.families[GeometryFamily.POINT].x.size)}"
        ),
        requested=plan.requested,
        selected=ExecutionMode.GPU,
    )
    return result


__all__ = ["grouped_point_union_owned", "supports_grouped_point_union"]
