"""Byte-, component-, and coordinate-shaped GPU WKB decode pipeline.

The structural scan is authoritative: every root and embedded header is
validated before any count is used for allocation or coordinate decode.  The
public result is row-aligned, while nested coordinate work runs as uniform
little- or big-endian component/ring tasks.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import (
    request_nvrtc_warmup as _request_nvrtc_warmup,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import DeviceFamilyGeometryBuffer, OwnedGeometryArray
from vibespatial.io.pylibcudf import (
    _build_device_mixed_owned,
    _build_device_single_family_owned,
)
from vibespatial.io.wkb_decode_status import (
    WKB_STATUS_REASONS,
    WKBDecodeStatus,
)
from vibespatial.kernels.core.wkb_decode_source import (
    _WKB_DECODE_KERNEL_NAMES,
    _WKB_DECODE_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

logger = logging.getLogger(__name__)

request_warmup(["exclusive_scan_i32"])
_request_nvrtc_warmup(
    [("wkb-decode-endian-aware", _WKB_DECODE_KERNEL_SOURCE, _WKB_DECODE_KERNEL_NAMES)]
)


def _wkb_decode_kernels() -> dict[str, Any]:
    runtime = get_cuda_runtime()
    return runtime.compile_kernels(
        cache_key=make_kernel_cache_key(
            "wkb-decode-endian-aware",
            _WKB_DECODE_KERNEL_SOURCE,
        ),
        source=_WKB_DECODE_KERNEL_SOURCE,
        kernel_names=_WKB_DECODE_KERNEL_NAMES,
    )


_TAG_TO_FAMILY = {
    0: GeometryFamily.POINT,
    1: GeometryFamily.LINESTRING,
    2: GeometryFamily.POLYGON,
    3: GeometryFamily.MULTIPOINT,
    4: GeometryFamily.MULTILINESTRING,
    5: GeometryFamily.MULTIPOLYGON,
}

_NATIVE_STATUS_VALUES = (
    int(WKBDecodeStatus.NATIVE_LITTLE_ENDIAN),
    int(WKBDecodeStatus.NATIVE_BIG_ENDIAN),
    int(WKBDecodeStatus.NATIVE_MIXED_ENDIAN),
)


@dataclass(frozen=True)
class DeviceWKBStructuralPlan:
    """Device-resident byte proof reused by every family decoder."""

    row_count: int
    payload_bytes: int
    statuses: Any
    family_tags: Any
    root_byte_orders: Any
    empty_flags: Any
    primary_counts: Any
    part_counts: Any
    ring_counts: Any
    coordinate_counts: Any
    input_validity: Any
    native_mask: Any


@dataclass(frozen=True)
class DeviceWKBDecodeResult:
    """Native subset plus structural status for partial compatibility merge."""

    owned: OwnedGeometryArray
    plan: DeviceWKBStructuralPlan
    declined_rows: Any


class WKBDeviceDecodeDeclined(NotImplementedError):
    """Raised before decode when valid rows fall outside the native contract."""

    def __init__(self, detail: str, *, plan: DeviceWKBStructuralPlan, declined_rows: Any):
        super().__init__(detail)
        self.plan = plan
        self.declined_rows = declined_rows


def summarize_wkb_device_plan(plan: DeviceWKBStructuralPlan) -> dict[str, Any]:
    """Return one bounded telemetry packet; geometry bytes never leave device."""
    import cupy as cp

    runtime = get_cuda_runtime()
    packet = cp.zeros(47, dtype=cp.uint64)
    if plan.row_count:
        kernel = _wkb_decode_kernels()["wkb_plan_summary"]
        grid, block = runtime.launch_config(kernel, plan.row_count)
        ptr = runtime.pointer
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(plan.statuses),
                    ptr(plan.family_tags),
                    ptr(plan.native_mask),
                    ptr(plan.part_counts),
                    ptr(plan.ring_counts),
                    ptr(plan.coordinate_counts),
                    ptr(packet),
                    plan.row_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
    host = _device_packet_to_host(
        packet,
        reason="WKB decode bounded aggregate telemetry packet",
    ).astype("int64", copy=False)
    family_counts = {
        family.value: int(host[20 + tag]) for tag, family in _TAG_TO_FAMILY.items()
    }
    family_part_counts = {
        family.value: int(host[29 + tag]) for tag, family in _TAG_TO_FAMILY.items()
    }
    family_ring_counts = {
        family.value: int(host[35 + tag]) for tag, family in _TAG_TO_FAMILY.items()
    }
    family_coordinate_counts = {
        family.value: int(host[41 + tag]) for tag, family in _TAG_TO_FAMILY.items()
    }
    return {
        "rows": plan.row_count,
        "payload_bytes": plan.payload_bytes,
        "native_little_endian_rows": int(host[int(WKBDecodeStatus.NATIVE_LITTLE_ENDIAN)]),
        "native_big_endian_rows": int(host[int(WKBDecodeStatus.NATIVE_BIG_ENDIAN)]),
        "native_mixed_endian_rows": int(host[int(WKBDecodeStatus.NATIVE_MIXED_ENDIAN)]),
        "null_rows": int(host[int(WKBDecodeStatus.NULL)]),
        "declined_rows": int(host[10:20].sum()),
        "dimensional_rows": int(host[int(WKBDecodeStatus.DIMENSIONAL_WKB)]),
        "family_counts": family_counts,
        "family_part_counts": family_part_counts,
        "family_ring_counts": family_ring_counts,
        "family_coordinate_counts": family_coordinate_counts,
        "part_count": int(host[26]),
        "ring_count": int(host[27]),
        "coordinate_count": int(host[28]),
    }


def _coordinate_capacity(payload_device) -> int:
    return int(getattr(payload_device, "size", 0)) // 16 + 1


def _structural_capacity(payload_device) -> int:
    return int(getattr(payload_device, "size", 0)) // 4 + 1


def _compact_prefix_capacity(values, logical_size):
    import cupy as cp

    return values[cp.arange(int(values.size), dtype=cp.int32) < logical_size]


def _compact_offsets_capacity(values, logical_last_index):
    import cupy as cp

    return values[cp.arange(int(values.size), dtype=cp.int32) <= logical_last_index]


def _offsets_from_counts(counts):
    import cupy as cp

    count = int(counts.size)
    offsets = cp.empty(count + 1, dtype=cp.int32)
    offsets[0] = 0
    if count:
        bases = exclusive_sum(counts, synchronize=False)
        offsets[1:] = bases + counts
        return offsets, bases
    return offsets, cp.empty(0, dtype=cp.int32)


def scan_wkb_device_structural_plan(
    payload_device,
    record_offsets_device,
    record_count: int,
    *,
    validity_device=None,
) -> DeviceWKBStructuralPlan:
    """Prove complete canonical 2D WKB structure without decoding coordinates."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    offsets = cp.asarray(record_offsets_device, dtype=cp.int64)
    if int(offsets.size) != int(record_count) + 1:
        raise ValueError("WKB record offsets must have record_count + 1 elements")
    if validity_device is None:
        validity = offsets[1:] > offsets[:-1]
    else:
        validity = cp.asarray(validity_device, dtype=cp.bool_)
        if int(validity.size) != int(record_count):
            raise ValueError("WKB validity must be row-aligned")

    statuses = cp.full(record_count, int(WKBDecodeStatus.TRUNCATED_OR_MALFORMED), dtype=cp.uint8)
    family_tags = cp.full(record_count, -1, dtype=cp.int8)
    root_byte_orders = cp.full(record_count, 255, dtype=cp.uint8)
    empty_flags = cp.zeros(record_count, dtype=cp.uint8)
    primary_counts = cp.zeros(record_count, dtype=cp.int32)
    part_counts = cp.zeros(record_count, dtype=cp.int32)
    ring_counts = cp.zeros(record_count, dtype=cp.int32)
    coordinate_counts = cp.zeros(record_count, dtype=cp.int32)

    if record_count:
        kernel = kernels["wkb_structural_scan"]
        grid, block = runtime.launch_config(kernel, record_count)
        ptr = runtime.pointer
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(payload_device),
                    int(getattr(payload_device, "size", 0)),
                    ptr(offsets),
                    ptr(validity),
                    1,
                    ptr(statuses),
                    ptr(family_tags),
                    ptr(root_byte_orders),
                    ptr(empty_flags),
                    ptr(primary_counts),
                    ptr(part_counts),
                    ptr(ring_counts),
                    ptr(coordinate_counts),
                    record_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I64,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
    native_mask = cp.isin(statuses, cp.asarray(_NATIVE_STATUS_VALUES, dtype=cp.uint8))
    return DeviceWKBStructuralPlan(
        row_count=int(record_count),
        payload_bytes=int(getattr(payload_device, "size", 0)),
        statuses=statuses,
        family_tags=family_tags,
        root_byte_orders=root_byte_orders,
        empty_flags=empty_flags,
        primary_counts=primary_counts,
        part_counts=part_counts,
        ring_counts=ring_counts,
        coordinate_counts=coordinate_counts,
        input_validity=validity,
        native_mask=native_mask,
    )


def _device_packet_to_host(values, *, reason: str):
    runtime = get_cuda_runtime()
    return runtime.copy_device_to_host(values, reason=reason)


def _decline_detail(plan: DeviceWKBStructuralPlan, declined_rows) -> str:
    import cupy as cp

    statuses = _device_packet_to_host(
        cp.asarray(plan.statuses[declined_rows], dtype=cp.uint8),
        reason="WKB decode bounded decline status packet",
    )
    rows = _device_packet_to_host(
        cp.asarray(declined_rows, dtype=cp.int32),
        reason="WKB decode bounded decline row packet",
    )
    counts: dict[str, int] = {}
    for raw in statuses:
        status = WKBDecodeStatus(int(raw))
        reason = WKB_STATUS_REASONS[status]
        counts[reason] = counts.get(reason, 0) + 1
    summary = "; ".join(f"{count}x {reason}" for reason, count in sorted(counts.items()))
    first = int(rows[0]) if len(rows) else -1
    return f"{len(rows)} WKB rows declined native decode; first_row={first}; {summary}"


def _apply_semantic_invalid_policy(plan: DeviceWKBStructuralPlan, on_invalid: str):
    import cupy as cp

    semantic_rows = cp.flatnonzero(plan.statuses == int(WKBDecodeStatus.SEMANTIC_INVALID)).astype(
        cp.int32, copy=False
    )
    if int(semantic_rows.size) == 0:
        return semantic_rows
    if on_invalid == "raise":
        first = _device_packet_to_host(
            semantic_rows[:1],
            reason="WKB decode semantic-invalid row packet",
        )
        raise ValueError(f"point array must contain 0 or >1 elements (row {int(first[0])})")
    if on_invalid == "warn":
        warnings.warn("point array must contain 0 or >1 elements", UserWarning, stacklevel=3)
    elif on_invalid != "ignore":
        raise ValueError("on_invalid must be one of 'raise', 'warn', or 'ignore'")
    return semantic_rows


def _native_partitions(plan: DeviceWKBStructuralPlan) -> dict[GeometryFamily, Any]:
    import cupy as cp

    partitions: dict[GeometryFamily, Any] = {}
    for tag, family in _TAG_TO_FAMILY.items():
        rows = cp.flatnonzero(plan.native_mask & (plan.family_tags == tag)).astype(
            cp.int32,
            copy=False,
        )
        if int(rows.size):
            partitions[family] = rows
    return partitions


def _launch_row_kernel(kernel, params, count: int) -> None:
    runtime = get_cuda_runtime()
    if not count:
        return
    grid, block = runtime.launch_config(kernel, count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _launch_block_tasks(kernel, params, count: int) -> None:
    runtime = get_cuda_runtime()
    if not count:
        return
    block_size = min(int(runtime.optimal_block_size(kernel)), 256)
    runtime.launch(
        kernel,
        grid=(count, 1, 1),
        block=(block_size, 1, 1),
        params=params,
    )


def _decode_point_family(payload, offsets, rows, plan) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    count = int(rows.size)
    x_all = cp.empty(count, dtype=cp.float64)
    y_all = cp.empty(count, dtype=cp.float64)
    orders = plan.root_byte_orders[rows]
    ptr = runtime.pointer
    for order, suffix in ((1, "le"), (0, "be")):
        positions = cp.flatnonzero(orders == order).astype(cp.int32, copy=False)
        selected_rows = rows[positions]
        n = int(positions.size)
        if n:
            kernel = kernels[f"decode_point_rows_{suffix}"]
            _launch_row_kernel(
                kernel,
                (
                    (
                        ptr(payload),
                        ptr(offsets),
                        ptr(selected_rows),
                        ptr(positions),
                        ptr(x_all),
                        ptr(y_all),
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                    ),
                ),
                n,
            )
    empty = plan.empty_flags[rows].astype(cp.bool_, copy=False)
    nonempty = ~empty
    geometry_offsets, _ = _offsets_from_counts(nonempty.astype(cp.int32, copy=False))
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=x_all[nonempty],
        y=y_all[nonempty],
        geometry_offsets=geometry_offsets,
        empty_mask=empty,
        bounds=None,
    )


def _decode_dense_point_family(
    payload,
    offsets,
    row_count: int,
    *,
    byte_order: int,
) -> tuple[Any, DeviceFamilyGeometryBuffer]:
    """Decode an all-valid, non-empty, one-endian point column in final shape."""
    import cupy as cp

    runtime = get_cuda_runtime()
    rows = cp.arange(row_count, dtype=cp.int32)
    x = cp.empty(row_count, dtype=cp.float64)
    y = cp.empty(row_count, dtype=cp.float64)
    kernel = _wkb_decode_kernels()[
        "decode_point_rows_le" if byte_order == 1 else "decode_point_rows_be"
    ]
    _launch_row_kernel(
        kernel,
        (
            (
                runtime.pointer(payload),
                runtime.pointer(offsets),
                runtime.pointer(rows),
                runtime.pointer(rows),
                runtime.pointer(x),
                runtime.pointer(y),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
        row_count,
    )
    return rows, DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=x,
        y=y,
        geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
        empty_mask=cp.zeros(row_count, dtype=cp.bool_),
        bounds=None,
    )


def _decode_linestring_family(
    payload,
    offsets,
    rows,
    plan,
    *,
    exact_coordinate_count: int | None = None,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    counts = plan.coordinate_counts[rows].astype(cp.int32, copy=False)
    geometry_offsets, _ = _offsets_from_counts(counts)
    capacity = (
        _coordinate_capacity(payload)
        if exact_coordinate_count is None
        else int(exact_coordinate_count)
    )
    x = cp.empty(capacity, dtype=cp.float64)
    y = cp.empty(capacity, dtype=cp.float64)
    orders = plan.root_byte_orders[rows]
    ptr = runtime.pointer
    for order, suffix in ((1, "le"), (0, "be")):
        positions = cp.flatnonzero(orders == order).astype(cp.int32, copy=False)
        selected_rows = rows[positions]
        n = int(positions.size)
        if n:
            kernel = kernels[f"decode_linestring_rows_{suffix}"]
            _launch_block_tasks(
                kernel,
                (
                    (
                        ptr(payload),
                        ptr(offsets),
                        ptr(selected_rows),
                        ptr(positions),
                        ptr(geometry_offsets),
                        ptr(x),
                        ptr(y),
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                    ),
                ),
                n,
            )
    total = geometry_offsets[-1]
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=x if exact_coordinate_count is not None else _compact_prefix_capacity(x, total),
        y=y if exact_coordinate_count is not None else _compact_prefix_capacity(y, total),
        geometry_offsets=geometry_offsets,
        empty_mask=counts == 0,
        bounds=None,
    )


def _allocate_tasks(payload, task_capacity: int):
    import cupy as cp

    return (
        cp.empty(task_capacity, dtype=cp.int64),
        cp.empty(task_capacity, dtype=cp.int32),
        cp.empty(task_capacity, dtype=cp.int32),
        cp.empty(task_capacity, dtype=cp.uint8),
    )


def _decode_emitted_tasks(
    payload,
    task_arrays,
    logical_tasks,
    x,
    y,
    *,
    exact_task_count: int | None = None,
) -> None:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    byte_offsets, counts, output_offsets, byte_orders = task_arrays
    if exact_task_count is None:
        keep = cp.arange(int(byte_offsets.size), dtype=cp.int32) < logical_tasks
        byte_offsets = byte_offsets[keep]
        counts = counts[keep]
        output_offsets = output_offsets[keep]
        byte_orders = byte_orders[keep]
    ptr = runtime.pointer
    for order, suffix in ((1, "le"), (0, "be")):
        indexes = cp.flatnonzero(byte_orders == order).astype(cp.int32, copy=False)
        n = int(indexes.size)
        if n:
            kernel = kernels[f"decode_coordinate_tasks_{suffix}"]
            _launch_block_tasks(
                kernel,
                (
                    (
                        ptr(payload),
                        ptr(byte_offsets),
                        ptr(counts),
                        ptr(output_offsets),
                        ptr(indexes),
                        ptr(x),
                        ptr(y),
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                    ),
                ),
                n,
            )


def _decode_polygon_family(
    payload,
    offsets,
    rows,
    plan,
    *,
    exact_ring_count: int | None = None,
    exact_coordinate_count: int | None = None,
    coordinate_output: tuple[Any, Any] | None = None,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    ring_counts = plan.ring_counts[rows].astype(cp.int32, copy=False)
    coord_counts = plan.coordinate_counts[rows].astype(cp.int32, copy=False)
    geometry_offsets, ring_bases = _offsets_from_counts(ring_counts)
    coord_offsets, coord_bases = _offsets_from_counts(coord_counts)
    task_capacity = (
        _structural_capacity(payload)
        if exact_ring_count is None
        else int(exact_ring_count)
    )
    tasks = _allocate_tasks(payload, task_capacity)
    ring_offsets = cp.empty(
        task_capacity + int(exact_ring_count is not None),
        dtype=cp.int32,
    )
    n = int(rows.size)
    ptr = runtime.pointer
    if n:
        kernel = kernels["emit_polygon_ring_tasks"]
        _launch_row_kernel(
            kernel,
            (
                (
                    ptr(payload),
                    ptr(offsets),
                    ptr(rows),
                    ptr(ring_bases),
                    ptr(coord_bases),
                    ptr(tasks[0]),
                    ptr(tasks[1]),
                    ptr(tasks[2]),
                    ptr(tasks[3]),
                    ptr(ring_offsets),
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
            n,
        )
    coord_capacity = (
        _coordinate_capacity(payload)
        if exact_coordinate_count is None
        else int(exact_coordinate_count)
    )
    if coordinate_output is None:
        x = cp.empty(coord_capacity, dtype=cp.float64)
        y = cp.empty(coord_capacity, dtype=cp.float64)
    else:
        x, y = coordinate_output
        if int(x.size) != coord_capacity or int(y.size) != coord_capacity:
            raise ValueError("exact Polygon coordinate output must match structural total")
    total_rings = geometry_offsets[-1]
    total_coords = coord_offsets[-1]
    _decode_emitted_tasks(
        payload,
        tasks,
        total_rings,
        x,
        y,
        exact_task_count=exact_ring_count,
    )
    ring_offsets[total_rings] = total_coords
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=x if exact_coordinate_count is not None else _compact_prefix_capacity(x, total_coords),
        y=y if exact_coordinate_count is not None else _compact_prefix_capacity(y, total_coords),
        geometry_offsets=geometry_offsets,
        empty_mask=ring_counts == 0,
        ring_offsets=(
            ring_offsets
            if exact_ring_count is not None
            else _compact_offsets_capacity(ring_offsets, total_rings)
        ),
        bounds=None,
        dense_single_ring_width=None,
    )


def _decode_multipoint_family(
    payload,
    offsets,
    rows,
    plan,
    *,
    exact_coordinate_count: int | None = None,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    counts = plan.coordinate_counts[rows].astype(cp.int32, copy=False)
    geometry_offsets, coord_bases = _offsets_from_counts(counts)
    capacity = (
        _coordinate_capacity(payload)
        if exact_coordinate_count is None
        else int(exact_coordinate_count)
    )
    tasks = _allocate_tasks(payload, capacity)
    n = int(rows.size)
    ptr = runtime.pointer
    if n:
        kernel = kernels["emit_multipoint_tasks"]
        _launch_row_kernel(
            kernel,
            (
                (
                    ptr(payload),
                    ptr(offsets),
                    ptr(rows),
                    ptr(coord_bases),
                    ptr(tasks[0]),
                    ptr(tasks[1]),
                    ptr(tasks[2]),
                    ptr(tasks[3]),
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
            n,
        )
    x = cp.empty(capacity, dtype=cp.float64)
    y = cp.empty(capacity, dtype=cp.float64)
    total = geometry_offsets[-1]
    _decode_emitted_tasks(
        payload,
        tasks,
        total,
        x,
        y,
        exact_task_count=exact_coordinate_count,
    )
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=x if exact_coordinate_count is not None else _compact_prefix_capacity(x, total),
        y=y if exact_coordinate_count is not None else _compact_prefix_capacity(y, total),
        geometry_offsets=geometry_offsets,
        empty_mask=counts == 0,
        bounds=None,
    )


def _decode_multilinestring_family(
    payload,
    offsets,
    rows,
    plan,
    *,
    exact_part_count: int | None = None,
    exact_coordinate_count: int | None = None,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    part_counts = plan.part_counts[rows].astype(cp.int32, copy=False)
    coord_counts = plan.coordinate_counts[rows].astype(cp.int32, copy=False)
    geometry_offsets, part_bases = _offsets_from_counts(part_counts)
    coord_offsets, coord_bases = _offsets_from_counts(coord_counts)
    task_capacity = (
        _structural_capacity(payload)
        if exact_part_count is None
        else int(exact_part_count)
    )
    tasks = _allocate_tasks(payload, task_capacity)
    part_offsets = cp.empty(
        task_capacity + int(exact_part_count is not None),
        dtype=cp.int32,
    )
    n = int(rows.size)
    ptr = runtime.pointer
    if n:
        kernel = kernels["emit_multilinestring_part_tasks"]
        _launch_row_kernel(
            kernel,
            (
                (
                    ptr(payload),
                    ptr(offsets),
                    ptr(rows),
                    ptr(part_bases),
                    ptr(coord_bases),
                    ptr(tasks[0]),
                    ptr(tasks[1]),
                    ptr(tasks[2]),
                    ptr(tasks[3]),
                    ptr(part_offsets),
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
            n,
        )
    coord_capacity = (
        _coordinate_capacity(payload)
        if exact_coordinate_count is None
        else int(exact_coordinate_count)
    )
    x = cp.empty(coord_capacity, dtype=cp.float64)
    y = cp.empty(coord_capacity, dtype=cp.float64)
    total_parts = geometry_offsets[-1]
    total_coords = coord_offsets[-1]
    _decode_emitted_tasks(
        payload,
        tasks,
        total_parts,
        x,
        y,
        exact_task_count=exact_part_count,
    )
    part_offsets[total_parts] = total_coords
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=x if exact_coordinate_count is not None else _compact_prefix_capacity(x, total_coords),
        y=y if exact_coordinate_count is not None else _compact_prefix_capacity(y, total_coords),
        geometry_offsets=geometry_offsets,
        empty_mask=part_counts == 0,
        part_offsets=(
            part_offsets
            if exact_part_count is not None
            else _compact_offsets_capacity(part_offsets, total_parts)
        ),
        bounds=None,
    )


def _decode_multipolygon_family(
    payload,
    offsets,
    rows,
    plan,
    *,
    exact_part_count: int | None = None,
    exact_ring_count: int | None = None,
    exact_coordinate_count: int | None = None,
    coordinate_output: tuple[Any, Any] | None = None,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    part_counts = plan.part_counts[rows].astype(cp.int32, copy=False)
    ring_counts = plan.ring_counts[rows].astype(cp.int32, copy=False)
    coord_counts = plan.coordinate_counts[rows].astype(cp.int32, copy=False)
    geometry_offsets, part_bases = _offsets_from_counts(part_counts)
    ring_offsets_by_row, ring_bases = _offsets_from_counts(ring_counts)
    coord_offsets, coord_bases = _offsets_from_counts(coord_counts)
    task_capacity = (
        _structural_capacity(payload)
        if exact_ring_count is None
        else int(exact_ring_count)
    )
    part_capacity = (
        _structural_capacity(payload)
        if exact_part_count is None
        else int(exact_part_count)
    )
    tasks = _allocate_tasks(payload, task_capacity)
    part_offsets = cp.empty(
        part_capacity + int(exact_part_count is not None),
        dtype=cp.int32,
    )
    ring_offsets = cp.empty(
        task_capacity + int(exact_ring_count is not None),
        dtype=cp.int32,
    )
    n = int(rows.size)
    ptr = runtime.pointer
    if n:
        kernel = kernels["emit_multipolygon_ring_tasks"]
        _launch_row_kernel(
            kernel,
            (
                (
                    ptr(payload),
                    ptr(offsets),
                    ptr(rows),
                    ptr(part_bases),
                    ptr(ring_bases),
                    ptr(coord_bases),
                    ptr(tasks[0]),
                    ptr(tasks[1]),
                    ptr(tasks[2]),
                    ptr(tasks[3]),
                    ptr(part_offsets),
                    ptr(ring_offsets),
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
            n,
        )
    coord_capacity = (
        _coordinate_capacity(payload)
        if exact_coordinate_count is None
        else int(exact_coordinate_count)
    )
    if coordinate_output is None:
        x = cp.empty(coord_capacity, dtype=cp.float64)
        y = cp.empty(coord_capacity, dtype=cp.float64)
    else:
        x, y = coordinate_output
        if int(x.size) != coord_capacity or int(y.size) != coord_capacity:
            raise ValueError("exact MultiPolygon coordinate output must match structural total")
    total_parts = geometry_offsets[-1]
    total_rings = ring_offsets_by_row[-1]
    total_coords = coord_offsets[-1]
    _decode_emitted_tasks(
        payload,
        tasks,
        total_rings,
        x,
        y,
        exact_task_count=exact_ring_count,
    )
    part_offsets[total_parts] = total_rings
    ring_offsets[total_rings] = total_coords
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=x if exact_coordinate_count is not None else _compact_prefix_capacity(x, total_coords),
        y=y if exact_coordinate_count is not None else _compact_prefix_capacity(y, total_coords),
        geometry_offsets=geometry_offsets,
        empty_mask=part_counts == 0,
        part_offsets=(
            part_offsets
            if exact_part_count is not None
            else _compact_offsets_capacity(part_offsets, total_parts)
        ),
        ring_offsets=(
            ring_offsets
            if exact_ring_count is not None
            else _compact_offsets_capacity(ring_offsets, total_rings)
        ),
        bounds=None,
    )


def _assemble_owned(partitions, buffers, plan) -> OwnedGeometryArray:
    import cupy as cp

    validity = plan.native_mask.astype(cp.bool_, copy=False)
    if not buffers:
        result = _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.empty(0, dtype=cp.bool_),
            detail="created all-null device WKB result after authoritative admission",
        )
    elif len(buffers) == 1:
        family = next(iter(buffers))
        result = _build_device_single_family_owned(
            family=family,
            validity_device=validity,
            x_device=buffers[family].x,
            y_device=buffers[family].y,
            geometry_offsets_device=buffers[family].geometry_offsets,
            empty_mask_device=buffers[family].empty_mask,
            part_offsets_device=buffers[family].part_offsets,
            ring_offsets_device=buffers[family].ring_offsets,
            dense_single_ring_width=buffers[family].dense_single_ring_width,
            detail="created device-resident owned geometry from endian-aware WKB decode",
            all_valid=int(partitions[family].size) == plan.row_count,
            valid_count=int(partitions[family].size),
        )
    else:
        family_row_offsets = cp.full(plan.row_count, -1, dtype=cp.int32)
        for family, rows in partitions.items():
            family_row_offsets[rows] = cp.arange(int(rows.size), dtype=cp.int32)
        tags = cp.where(validity, plan.family_tags, cp.int8(-1)).astype(cp.int8, copy=False)
        result = _build_device_mixed_owned(
            validity_device=validity,
            tags_device=tags,
            family_row_offsets_device=family_row_offsets,
            family_devices=buffers,
            detail="created mixed device geometry from endian-aware WKB decode",
            all_valid=sum(int(rows.size) for rows in partitions.values())
            == plan.row_count,
        )
    result._wkb_structural_plan = plan
    return result


@register_kernel_variant(
    "decode_wkb",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("wkb", "decode", "nvrtc", "cccl", "big-endian"),
)
def decode_wkb_device_pipeline(
    payload_device,
    record_offsets_device,
    record_count: int,
    *,
    validity_device=None,
    on_invalid: str = "raise",
    allow_declined: bool = False,
    structural_plan: DeviceWKBStructuralPlan | None = None,
    structural_summary: dict[str, Any] | None = None,
) -> OwnedGeometryArray | DeviceWKBDecodeResult:
    """Decode canonical 2D WKB directly to device owned buffers.

    Integer and coordinate bytes are decoded exactly in their owning record's
    endian order.  Precision dispatch does not downcast WKB coordinates:
    storage and decoded values remain fp64 by contract.
    """
    import cupy as cp

    plan = structural_plan
    if plan is None:
        plan = scan_wkb_device_structural_plan(
            payload_device,
            record_offsets_device,
            record_count,
            validity_device=validity_device,
        )
    elif plan.row_count != int(record_count):
        raise ValueError("WKB structural plan must match record_count")
    _apply_semantic_invalid_policy(plan, on_invalid)
    declined_mask = (
        plan.input_validity
        & ~plan.native_mask
        & (plan.statuses != int(WKBDecodeStatus.SEMANTIC_INVALID))
    )
    declined_rows = cp.flatnonzero(declined_mask).astype(cp.int32, copy=False)
    if int(declined_rows.size) and not allow_declined:
        raise WKBDeviceDecodeDeclined(
            _decline_detail(plan, declined_rows),
            plan=plan,
            declined_rows=declined_rows,
        )

    offsets = cp.asarray(record_offsets_device, dtype=cp.int64)
    dense_point_order = None
    if structural_summary is not None and record_count:
        all_points = (
            structural_summary["family_counts"][GeometryFamily.POINT.value]
            == record_count
        )
        all_nonempty = structural_summary["coordinate_count"] == record_count
        if all_points and all_nonempty and structural_summary["null_rows"] == 0:
            if structural_summary["native_little_endian_rows"] == record_count:
                dense_point_order = 1
            elif structural_summary["native_big_endian_rows"] == record_count:
                dense_point_order = 0
    if dense_point_order is not None:
        point_rows, point_buffer = _decode_dense_point_family(
            payload_device,
            offsets,
            record_count,
            byte_order=dense_point_order,
        )
        partitions = {GeometryFamily.POINT: point_rows}
        buffers = {GeometryFamily.POINT: point_buffer}
    else:
        partitions = _native_partitions(plan)
        buffers: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        polygonal_coordinate_outputs: dict[
            GeometryFamily, tuple[Any, Any]
        ] = {}
        if structural_summary is not None and set(partitions) == {
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
        }:
            polygon_coordinates = structural_summary["family_coordinate_counts"][
                GeometryFamily.POLYGON.value
            ]
            multipolygon_coordinates = structural_summary["family_coordinate_counts"][
                GeometryFamily.MULTIPOLYGON.value
            ]
            total_polygonal_coordinates = polygon_coordinates + multipolygon_coordinates
            shared_x = cp.empty(total_polygonal_coordinates, dtype=cp.float64)
            shared_y = cp.empty(total_polygonal_coordinates, dtype=cp.float64)
            polygonal_coordinate_outputs = {
                GeometryFamily.POLYGON: (
                    shared_x[:polygon_coordinates],
                    shared_y[:polygon_coordinates],
                ),
                GeometryFamily.MULTIPOLYGON: (
                    shared_x[polygon_coordinates:],
                    shared_y[polygon_coordinates:],
                ),
            }
        for family, rows in partitions.items():
            exact_parts = (
                None
                if structural_summary is None
                else structural_summary["family_part_counts"][family.value]
            )
            exact_rings = (
                None
                if structural_summary is None
                else structural_summary["family_ring_counts"][family.value]
            )
            exact_coordinates = (
                None
                if structural_summary is None
                else structural_summary["family_coordinate_counts"][family.value]
            )
            if family is GeometryFamily.POINT:
                buffers[family] = _decode_point_family(payload_device, offsets, rows, plan)
            elif family is GeometryFamily.LINESTRING:
                buffers[family] = _decode_linestring_family(
                    payload_device,
                    offsets,
                    rows,
                    plan,
                    exact_coordinate_count=exact_coordinates,
                )
            elif family is GeometryFamily.POLYGON:
                buffers[family] = _decode_polygon_family(
                    payload_device,
                    offsets,
                    rows,
                    plan,
                    exact_ring_count=exact_rings,
                    exact_coordinate_count=exact_coordinates,
                    coordinate_output=polygonal_coordinate_outputs.get(family),
                )
            elif family is GeometryFamily.MULTIPOINT:
                buffers[family] = _decode_multipoint_family(
                    payload_device,
                    offsets,
                    rows,
                    plan,
                    exact_coordinate_count=exact_coordinates,
                )
            elif family is GeometryFamily.MULTILINESTRING:
                buffers[family] = _decode_multilinestring_family(
                    payload_device,
                    offsets,
                    rows,
                    plan,
                    exact_part_count=exact_parts,
                    exact_coordinate_count=exact_coordinates,
                )
            elif family is GeometryFamily.MULTIPOLYGON:
                buffers[family] = _decode_multipolygon_family(
                    payload_device,
                    offsets,
                    rows,
                    plan,
                    exact_part_count=exact_parts,
                    exact_ring_count=exact_rings,
                    exact_coordinate_count=exact_coordinates,
                    coordinate_output=polygonal_coordinate_outputs.get(family),
                )
    owned = _assemble_owned(partitions, buffers, plan)
    if allow_declined:
        return DeviceWKBDecodeResult(owned=owned, plan=plan, declined_rows=declined_rows)
    return owned
