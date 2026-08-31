"""Reusable exact y-edge directory for point-in-polygon refinement.

This is a physical metadata carrier, not a public spatial index. It is cached
on immutable polygon ``OwnedGeometryDeviceState`` and reused by public point-
region consumers. Compile-time-uniform widths preserve the hot exact kernel,
while a device-memory tier and concrete allocation admission choose how much
of the GPU to use without changing predicate semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from vibespatial.api._native_state import NativeStreamReadiness
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    cuda_stream_identity,
    get_cuda_completion_retainer,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import attach_work_amplification, hotpath_stage

from .point_location_index_kernels import (
    PART_Y_BIN_COUNT,
    POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
    POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES,
    SUPPORTED_PART_Y_BIN_COUNTS,
    coverage_grid_width_for_bin_count,
    point_location_part_y_index_profile_source,
    point_location_part_y_index_source,
)

_MIN_PREPARED_COORDINATES = 1_000_000
_INDEXABLE_FAMILIES = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_POINT_FAMILIES = (GeometryFamily.POINT, GeometryFamily.MULTIPOINT)

_GIB = 1 << 30
_NOMINAL_VRAM_SNAP_FRACTION = 0.95
_FUTURE_POINT_REGION_RESERVE_BYTES = 64 << 20
_FUTURE_POINT_REGION_RESERVE_FRACTION = 0.05

request_nvrtc_warmup(
    [
        (
            f"point-location-part-y-index-b{bin_count}",
            point_location_part_y_index_source(bin_count),
            POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
        )
        for bin_count in SUPPORTED_PART_Y_BIN_COUNTS
    ]
)


@dataclass(frozen=True, slots=True)
class PointLocationIndexCacheKey:
    """Exact identity of one compile-time-uniform prepared representation."""

    family: GeometryFamily
    bin_count: int
    coverage_grid_width: int


@dataclass(frozen=True, slots=True)
class PointLocationIndexDecision:
    """Bounded host-visible selection and admission telemetry."""

    family: GeometryFamily
    total_device_bytes: int
    nominal_vram_class_gib: int
    target_bin_count: int
    admitted_bin_count: int | None
    coverage_grid_width: int
    coverage_decline_reason: str | None
    decline_reason: str | None
    edge_membership_count: int
    persistent_bytes: int
    peak_build_bytes: int
    cache_hit: bool = False


@dataclass(frozen=True)
class PreparedPolygonPartYIndex:
    """Device-resident edge memberships grouped by polygon-part y bin."""

    family: GeometryFamily
    geometry_count: int
    part_count: int
    bin_count: int
    target_bin_count: int
    nominal_vram_class_gib: int
    edge_membership_count: int
    peak_build_bytes: int
    decline_reason: str | None
    coverage_grid_width: int
    coverage_decline_reason: str | None
    part_xmin: object | None
    part_xmax: object | None
    coverage: object | None
    part_ymin: object
    part_ymax: object
    counts: object
    offsets: object
    entries: object
    readiness: NativeStreamReadiness

    @property
    def cache_key(self) -> PointLocationIndexCacheKey:
        return PointLocationIndexCacheKey(
            self.family,
            self.bin_count,
            self.coverage_grid_width,
        )

    @property
    def device_bytes(self) -> int:
        return sum(
            int(getattr(value, "nbytes", 0))
            for value in (
                self.part_xmin,
                self.part_xmax,
                self.coverage,
                self.part_ymin,
                self.part_ymax,
                self.counts,
                self.offsets,
                self.entries,
            )
        )


def _snap_nominal_vram_gib(total_device_bytes: int) -> float:
    """Snap slightly under-reported marketed capacities to their nominal tier."""
    capacity = max(int(total_device_bytes), 0) / _GIB
    for boundary in (8, 16, 24, 48, 100):
        if boundary * _NOMINAL_VRAM_SNAP_FRACTION <= capacity < boundary:
            return float(boundary)
    return capacity


def point_location_bin_policy(total_device_bytes: int) -> tuple[int, int]:
    """Return ``(nominal class GiB, first attempted bins)`` for one device."""
    capacity = _snap_nominal_vram_gib(total_device_bytes)
    nominal = int(round(capacity))
    if capacity <= 8:
        return nominal, 8
    if capacity <= 16:
        return nominal, 16
    if capacity < 24:
        return nominal, 32
    if capacity < 48:
        return nominal, 64
    if capacity < 100:
        return nominal, 128
    return nominal, 256


def _admission_widths(target_bin_count: int) -> tuple[int, ...]:
    target = int(target_bin_count)
    if target not in SUPPORTED_PART_Y_BIN_COUNTS:
        raise ValueError(
            f"unsupported polygon part-y target {target}; "
            f"expected one of {SUPPORTED_PART_Y_BIN_COUNTS}"
        )
    return tuple(width for width in reversed(SUPPORTED_PART_Y_BIN_COUNTS) if width <= target)


def _future_point_region_reserve(total_device_bytes: int, remaining_bytes: int) -> int:
    desired = max(
        _FUTURE_POINT_REGION_RESERVE_BYTES,
        int(total_device_bytes * _FUTURE_POINT_REGION_RESERVE_FRACTION),
    )
    return min(desired, max(int(remaining_bytes), 0) // 3)


def _structural_peak_bytes(part_count: int, ring_count: int, bin_count: int) -> int:
    """Peak bytes before the exact membership total is host-visible."""
    bin_slots = int(part_count) * int(bin_count)
    return int(part_count) * 16 + int(ring_count) * 4 + bin_slots * 20


def _complete_peak_bytes(
    part_count: int,
    ring_count: int,
    bin_count: int,
    membership_count: int,
) -> int:
    bin_slots = int(part_count) * int(bin_count)
    scatter_peak = (
        int(part_count) * 16
        + int(ring_count) * 4
        + bin_slots * 16
        + int(membership_count) * 4
    )
    return max(
        _structural_peak_bytes(part_count, ring_count, bin_count),
        scatter_peak,
    )


def _coverage_grid_bytes(part_count: int, coverage_grid_width: int) -> int:
    """Persistent fp64 x bounds plus one byte per conservative grid cell."""
    return int(part_count) * (
        16 + int(coverage_grid_width) * int(coverage_grid_width)
    )


def _record_index_readiness() -> NativeStreamReadiness:
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    event = cp.cuda.Event(disable_timing=True)
    event.record(stream)
    return NativeStreamReadiness(stream=stream, event=event, ready=False)


def wait_for_polygon_part_y_index(readiness: NativeStreamReadiness) -> None:
    """Order a cross-stream consumer after the completed immutable index."""
    if readiness.ready or readiness.event is None:
        return
    import cupy as cp

    consumer = cp.cuda.get_current_stream()
    if readiness.stream is not None and (
        cuda_stream_identity(readiness.stream) == cuda_stream_identity(consumer)
    ):
        return
    consumer.wait_event(readiness.event)
    get_cuda_completion_retainer().defer(
        consumer,
        readiness.event,
        lambda _event: None,
    )


def cached_polygon_part_y_index(state, family: GeometryFamily):
    """Return the selected completion-ready index for ``family`` if present."""
    decision = state.point_location_index_decisions.get(family)
    if decision is None or decision.admitted_bin_count is None:
        return None
    key = PointLocationIndexCacheKey(
        family,
        decision.admitted_bin_count,
        decision.coverage_grid_width,
    )
    prepared = state.point_location_indexes.get(key)
    if prepared is None:
        raise RuntimeError("point-location decision references a missing prepared index")
    wait_for_polygon_part_y_index(prepared.readiness)
    return prepared


def _point_location_preparation_metrics(
    prepared,
    *,
    built: bool,
    cache_hit: bool,
    cache_miss: bool,
    declined: bool,
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
    """Return bounded cache evidence from the prepared carrier's host fields."""
    max_metrics = {}
    if prepared is not None:
        part_count = int(prepared.part_count)
        max_metrics.update(
            {
                "source_geometries": int(prepared.geometry_count),
                "source_parts": part_count,
                "part_y_bin_slots": part_count * int(prepared.bin_count),
                "target_bin_count": int(prepared.target_bin_count),
                "admitted_bin_count": int(prepared.bin_count),
                "nominal_vram_class_gib": int(prepared.nominal_vram_class_gib),
                "edge_memberships": int(prepared.edge_membership_count),
                "persistent_bytes": int(prepared.device_bytes),
                "peak_build_bytes": int(prepared.peak_build_bytes),
                "coverage_grid_width": int(prepared.coverage_grid_width),
                "coverage_cells": int(prepared.part_count)
                * int(prepared.coverage_grid_width) ** 2,
            }
        )
    unavailable = ["build_seconds", "avoidable_rebuild_seconds", "invalidation_reason"]
    if prepared is None:
        unavailable.extend(
            (
                "source_geometries",
                "source_parts",
                "part_y_bin_slots",
                "target_bin_count",
                "admitted_bin_count",
                "nominal_vram_class_gib",
                "edge_memberships",
                "persistent_bytes",
                "peak_build_bytes",
                "coverage_grid_width",
                "coverage_cells",
            )
        )
    return (
        {
            "preparation_requests": 1,
            "build_count": int(built),
            "cache_hits": int(cache_hit),
            "cache_misses": int(cache_miss),
            "declined_preparations": int(declined),
        },
        max_metrics,
        tuple(unavailable),
    )


def point_location_part_y_index_kernels(bin_count: int = PART_Y_BIN_COUNT):
    source = point_location_part_y_index_source(bin_count)
    return compile_kernel_group(
        f"point-location-part-y-index-b{int(bin_count)}",
        source,
        POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
    )


def point_location_part_y_index_profile_kernels(bin_count: int = PART_Y_BIN_COUNT):
    """Compile profiler-only entry points outside production warmup."""
    source = point_location_part_y_index_profile_source(bin_count)
    return compile_kernel_group(
        f"point-location-part-y-index-profile-b{int(bin_count)}",
        source,
        POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES,
    )


def prepare_polygon_part_y_index(
    owned,
    family: GeometryFamily,
    *,
    _target_bin_count: int | None = None,
):
    """Build or return the cached exact part-y edge directory."""
    if family not in _INDEXABLE_FAMILIES:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    runtime = get_cuda_runtime()
    budget = runtime.query_memory_budget()
    nominal_class, policy_target = point_location_bin_policy(budget.total_device_bytes)
    target = policy_target if _target_bin_count is None else int(_target_bin_count)
    widths = _admission_widths(target)
    cached = next(
        (
            value
            for width in widths
            for coverage_width in (
                coverage_grid_width_for_bin_count(width),
                0,
            )
            if (
                value := state.point_location_indexes.get(
                    PointLocationIndexCacheKey(family, width, coverage_width)
                )
            )
            is not None
        ),
        None,
    )
    with hotpath_stage(
        "predicate.point_location_part_y_index.prepare",
        category="setup",
    ) as stage_metadata:
        prepared = _prepare_polygon_part_y_index_impl(
            owned,
            family,
            state=state,
            cached=cached,
            nominal_vram_class_gib=nominal_class,
            target_bin_count=target,
        )
        if stage_metadata is not None:
            cache_hit = cached is not None
            built = cached is None and prepared is not None
            sums, maxima, unavailable = _point_location_preparation_metrics(
                prepared,
                built=built,
                cache_hit=cache_hit,
                cache_miss=cached is None,
                declined=prepared is None,
            )
            attach_work_amplification(
                stage_metadata,
                operation="prepare_point_region_y_index",
                metric_family="rebuild",
                sums=sums,
                maxima=maxima,
                unavailable=unavailable,
                physical_shape="reusable_polygon_part_y_index",
                consumer_kind="point_region_exact_refinement",
                semantic_contract={
                    "cache_scope": "OwnedGeometryDeviceState point-location indexes",
                    "cache_identity_exported": True,
                    "device_logical_counts_read": False,
                },
            )
        return prepared


def _prepare_polygon_part_y_index_impl(
    owned,
    family: GeometryFamily,
    *,
    state,
    cached,
    nominal_vram_class_gib: int,
    target_bin_count: int,
):
    from .point_region_profile import current_point_region_profile

    profile = current_point_region_profile()
    if cached is not None:
        wait_for_polygon_part_y_index(cached.readiness)
        decision = state.point_location_index_decisions.get(family)
        if decision is not None:
            state.point_location_index_decisions[family] = replace(decision, cache_hit=True)
        if profile is not None:
            profile.note_index_cache_hit(cached)
        return cached

    buffer = state.families.get(family)
    force_prepared = profile is not None and profile.force_prepared_index
    if buffer is None or (
        int(buffer.x.size) < _MIN_PREPARED_COORDINATES and not force_prepared
    ):
        return None
    if int(buffer.x.size) >= (1 << 31) or int(buffer.ring_offsets.size - 1) >= (1 << 31):
        return None

    import cupy as cp

    build_started = perf_counter() if profile is not None else None
    part_ring_offsets = (
        buffer.geometry_offsets if family is GeometryFamily.POLYGON else buffer.part_offsets
    )
    if part_ring_offsets is None or buffer.ring_offsets is None:
        return None
    part_count = int(part_ring_offsets.size - 1)
    ring_count = int(buffer.ring_offsets.size - 1)
    coordinate_count = int(buffer.y.size)
    runtime = get_cuda_runtime()
    total_device_bytes = int(runtime.query_memory_budget().total_device_bytes)
    ptr = runtime.pointer
    decline_reasons: list[str] = []

    part_ymin = None
    part_ymax = None
    ring_parts = None
    bounds_ready = False

    for bin_count in _admission_widths(target_bin_count):
        remaining = runtime.query_memory_remaining_bytes()
        future_reserve = _future_point_region_reserve(total_device_bytes, remaining)
        structural_peak = _structural_peak_bytes(part_count, ring_count, bin_count)
        preflight = runtime.admit_device_memory(
            stage=f"predicate.point_location_part_y_index.b{bin_count}.structural",
            required_bytes=structural_peak + future_reserve,
            requested_units=part_count * bin_count,
        )
        if not preflight.admitted:
            decline_reasons.append(
                f"b{bin_count}: structural peak {structural_peak} plus future "
                f"reserve {future_reserve} exceeds {preflight.remaining_bytes} bytes"
            )
            continue

        counts = None
        counts_i64 = None
        offsets = None
        cursors = None
        entries = None
        part_xmin = None
        part_xmax = None
        coverage = None
        coverage_grid_width = 0
        coverage_decline_reason = None
        try:
            kernels = point_location_part_y_index_kernels(bin_count)
            if not bounds_ready:
                part_ymin = cp.empty(part_count, dtype=cp.float64)
                part_ymax = cp.empty(part_count, dtype=cp.float64)
                ring_parts = cp.empty(ring_count, dtype=cp.int32)
                bounds_kernel = kernels["compute_polygon_part_y_bounds"]
                grid, block = runtime.launch_config(bounds_kernel, part_count)
                runtime.launch(
                    bounds_kernel,
                    grid=grid,
                    block=block,
                    params=(
                        (
                            part_count,
                            ptr(part_ring_offsets),
                            ptr(buffer.ring_offsets),
                            ptr(buffer.y),
                            ptr(part_ymin),
                            ptr(part_ymax),
                        ),
                        (
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                        ),
                    ),
                )
                ring_kernel = kernels["map_polygon_rings_to_parts"]
                grid, block = runtime.launch_config(ring_kernel, ring_count)
                runtime.launch(
                    ring_kernel,
                    grid=grid,
                    block=block,
                    params=(
                        (ring_count, part_count, ptr(part_ring_offsets), ptr(ring_parts)),
                        (
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                        ),
                    ),
                )
                bounds_ready = True

            bin_slots = part_count * bin_count
            counts = cp.zeros(bin_slots, dtype=cp.uint32)
            count_kernel = kernels["count_polygon_edge_y_bin_memberships"]
            grid, block = runtime.launch_config(count_kernel, coordinate_count)
            runtime.launch(
                count_kernel,
                grid=grid,
                block=block,
                params=(
                    (
                        coordinate_count,
                        ring_count,
                        ptr(buffer.ring_offsets),
                        ptr(ring_parts),
                        ptr(buffer.y),
                        ptr(part_ymin),
                        ptr(part_ymax),
                        ptr(counts),
                    ),
                    (
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                    ),
                ),
            )
            counts_i64 = counts.astype(cp.int64)
            offsets = exclusive_sum(counts_i64, synchronize=False)
            membership_count = count_scatter_total(
                runtime,
                counts_i64,
                offsets,
                reason="point-location y-edge membership allocation fence",
            )
            del counts_i64

            peak_build_bytes = _complete_peak_bytes(
                part_count,
                ring_count,
                bin_count,
                membership_count,
            )
            additional_bytes = bin_slots * 4 + membership_count * 4 + future_reserve
            final_admission = runtime.admit_device_memory(
                stage=f"predicate.point_location_part_y_index.b{bin_count}.complete",
                required_bytes=additional_bytes,
                requested_units=membership_count,
            )
            if not final_admission.admitted:
                decline_reasons.append(
                    f"b{bin_count}: {membership_count} memberships require "
                    f"{additional_bytes} additional bytes with "
                    f"{final_admission.remaining_bytes} available"
                )
                del counts, offsets
                continue

            cursors = cp.zeros(bin_slots, dtype=cp.uint32)
            entries = cp.empty(membership_count, dtype=cp.uint32)
            scatter_kernel = kernels["scatter_polygon_edge_y_bin_memberships"]
            grid, block = runtime.launch_config(scatter_kernel, coordinate_count)
            runtime.launch(
                scatter_kernel,
                grid=grid,
                block=block,
                params=(
                    (
                        coordinate_count,
                        ring_count,
                        ptr(buffer.ring_offsets),
                        ptr(ring_parts),
                        ptr(buffer.y),
                        ptr(part_ymin),
                        ptr(part_ymax),
                        ptr(offsets),
                        ptr(cursors),
                        ptr(entries),
                    ),
                    (
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                    ),
                ),
            )
            target_coverage_grid_width = coverage_grid_width_for_bin_count(
                bin_count
            )
            coverage_bytes = _coverage_grid_bytes(
                part_count,
                target_coverage_grid_width,
            )
            coverage_admission = runtime.admit_device_memory(
                stage=(
                    f"predicate.point_location_part_y_index.b{bin_count}."
                    "coverage_grid"
                ),
                required_bytes=coverage_bytes + future_reserve,
                requested_units=(
                    part_count
                    * target_coverage_grid_width
                    * target_coverage_grid_width
                ),
            )
            if coverage_admission.admitted:
                try:
                    part_xmin = cp.empty(part_count, dtype=cp.float64)
                    part_xmax = cp.empty(part_count, dtype=cp.float64)
                    coverage = cp.empty(
                        part_count
                        * target_coverage_grid_width
                        * target_coverage_grid_width,
                        dtype=cp.uint8,
                    )
                    x_bounds_kernel = kernels["compute_polygon_part_x_bounds"]
                    grid, block = runtime.launch_config(x_bounds_kernel, part_count)
                    runtime.launch(
                        x_bounds_kernel,
                        grid=grid,
                        block=block,
                        params=(
                            (
                                part_count,
                                ptr(part_ring_offsets),
                                ptr(buffer.ring_offsets),
                                ptr(buffer.x),
                                ptr(part_xmin),
                                ptr(part_xmax),
                            ),
                            (
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                            ),
                        ),
                    )
                    initialize_coverage = kernels[
                        "initialize_polygon_part_coverage_cells"
                    ]
                    coverage_cell_count = (
                        part_count
                        * target_coverage_grid_width
                        * target_coverage_grid_width
                    )
                    grid, block = runtime.launch_config(
                        initialize_coverage,
                        coverage_cell_count,
                    )
                    runtime.launch(
                        initialize_coverage,
                        grid=grid,
                        block=block,
                        params=(
                            (
                                part_count,
                                ptr(buffer.ring_offsets),
                                ptr(buffer.x),
                                ptr(buffer.y),
                                ptr(part_xmin),
                                ptr(part_xmax),
                                ptr(part_ymin),
                                ptr(part_ymax),
                                ptr(counts),
                                ptr(offsets),
                                ptr(entries),
                                ptr(coverage),
                            ),
                            (
                                KERNEL_PARAM_I32,
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
                            ),
                        ),
                    )
                    mark_coverage = kernels["mark_polygon_edge_coverage_cells"]
                    grid, block = runtime.launch_config(
                        mark_coverage,
                        coordinate_count,
                    )
                    runtime.launch(
                        mark_coverage,
                        grid=grid,
                        block=block,
                        params=(
                            (
                                coordinate_count,
                                ring_count,
                                ptr(buffer.ring_offsets),
                                ptr(ring_parts),
                                ptr(buffer.x),
                                ptr(buffer.y),
                                ptr(part_xmin),
                                ptr(part_xmax),
                                ptr(part_ymin),
                                ptr(part_ymax),
                                ptr(coverage),
                            ),
                            (
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                            ),
                        ),
                    )
                    coverage_grid_width = target_coverage_grid_width
                    peak_build_bytes += coverage_bytes
                except MemoryError:
                    part_xmin = None
                    part_xmax = None
                    coverage = None
                    coverage_decline_reason = (
                        "device allocation failed before coverage publication"
                    )
            else:
                coverage_decline_reason = (
                    f"coverage grid requires {coverage_bytes} bytes plus future "
                    f"reserve {future_reserve} with "
                    f"{coverage_admission.remaining_bytes} available"
                )
            readiness = _record_index_readiness()
            get_cuda_completion_retainer().defer(
                readiness.stream,
                (ring_parts, cursors),
                lambda _owners: None,
            )
            decline_reason = "; ".join(decline_reasons) or None
            prepared = PreparedPolygonPartYIndex(
                family=family,
                geometry_count=int(buffer.geometry_offsets.size - 1),
                part_count=part_count,
                bin_count=bin_count,
                target_bin_count=target_bin_count,
                nominal_vram_class_gib=nominal_vram_class_gib,
                edge_membership_count=membership_count,
                peak_build_bytes=peak_build_bytes,
                decline_reason=decline_reason,
                coverage_grid_width=coverage_grid_width,
                coverage_decline_reason=coverage_decline_reason,
                part_xmin=part_xmin,
                part_xmax=part_xmax,
                coverage=coverage,
                part_ymin=part_ymin,
                part_ymax=part_ymax,
                counts=counts,
                offsets=offsets,
                entries=entries,
                readiness=readiness,
            )
            key = prepared.cache_key
            published = state.point_location_indexes.setdefault(key, prepared)
            if published is not prepared:
                get_cuda_completion_retainer().defer(
                    readiness.stream,
                    prepared,
                    lambda _prepared: None,
                )
                wait_for_polygon_part_y_index(published.readiness)
                prepared = published
            decision = PointLocationIndexDecision(
                family=family,
                total_device_bytes=total_device_bytes,
                nominal_vram_class_gib=nominal_vram_class_gib,
                target_bin_count=target_bin_count,
                admitted_bin_count=int(prepared.bin_count),
                coverage_grid_width=int(prepared.coverage_grid_width),
                coverage_decline_reason=prepared.coverage_decline_reason,
                decline_reason=decline_reason,
                edge_membership_count=int(prepared.edge_membership_count),
                persistent_bytes=int(prepared.device_bytes),
                peak_build_bytes=int(prepared.peak_build_bytes),
            )
            state.point_location_index_decisions[family] = decision
            if profile is not None:
                runtime.synchronize()
                assert build_started is not None
                profile.note_index_build(prepared, perf_counter() - build_started)
            record_dispatch_event(
                surface="vibespatial.predicates.point_location_index",
                operation="prepare_point_location",
                implementation="polygon_part_y_edge_directory_gpu",
                reason=(
                    f"VRAM class {nominal_vram_class_gib} GiB targeted "
                    f"{target_bin_count} bins and admitted {prepared.bin_count}; "
                    f"prepared {part_count} parts with {membership_count} exact "
                    f"edge memberships, {prepared.device_bytes} persistent bytes, "
                    f"{prepared.coverage_grid_width}x"
                    f"{prepared.coverage_grid_width} conservative coverage cells, "
                    f"and {peak_build_bytes} peak build bytes"
                ),
                selected=ExecutionMode.GPU,
            )
            return prepared
        except MemoryError:
            counts = None
            offsets = None
            cursors = None
            entries = None
            decline_reasons.append(f"b{bin_count}: device allocation failed before publication")
            continue

    decision = PointLocationIndexDecision(
        family=family,
        total_device_bytes=total_device_bytes,
        nominal_vram_class_gib=nominal_vram_class_gib,
        target_bin_count=target_bin_count,
        admitted_bin_count=None,
        coverage_grid_width=0,
        coverage_decline_reason="part-y index was not admitted",
        decline_reason="; ".join(decline_reasons) or "no compiled width admitted",
        edge_membership_count=0,
        persistent_bytes=0,
        peak_build_bytes=0,
    )
    state.point_location_index_decisions[family] = decision
    record_dispatch_event(
        surface="vibespatial.predicates.point_location_index",
        operation="prepare_point_location",
        implementation="exact_point_region_tiled_gpu",
        reason=(
            f"prepared index declined from target {target_bin_count}: "
            f"{decision.decline_reason}; exact GPU refinement remains active"
        ),
        selected=ExecutionMode.GPU,
    )
    return None


def prepare_point_region_y_indexes(left_owned, right_owned) -> None:
    """Prepare large polygon sides before candidate arrays consume the budget."""
    left_families = set(left_owned.families)
    right_families = set(right_owned.families)
    if left_families & set(_POINT_FAMILIES):
        for family in right_families & set(_INDEXABLE_FAMILIES):
            prepare_polygon_part_y_index(right_owned, family)
    if right_families & set(_POINT_FAMILIES):
        for family in left_families & set(_INDEXABLE_FAMILIES):
            prepare_polygon_part_y_index(left_owned, family)
