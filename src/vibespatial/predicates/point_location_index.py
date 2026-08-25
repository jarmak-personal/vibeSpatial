"""Reusable exact y-edge directory for point-in-polygon refinement.

This is a physical metadata carrier, not a public spatial index.  It is cached
on the polygon ``OwnedGeometryDeviceState`` and reused by repeated public
``sindex.query`` calls.  Eight y bins per polygon part reduce exact ray-cast
work without duplicating geometry coordinates or changing predicate semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

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
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import attach_work_amplification, hotpath_stage

from .point_location_index_kernels import (
    _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE,
    _POINT_LOCATION_PART_Y_INDEX_SOURCE,
    PART_Y_BIN_COUNT,
    POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
    POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES,
)

_MIN_PREPARED_COORDINATES = 1_000_000
_INDEXABLE_FAMILIES = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_POINT_FAMILIES = (GeometryFamily.POINT, GeometryFamily.MULTIPOINT)

request_nvrtc_warmup(
    [
        (
            "point-location-part-y-index",
            _POINT_LOCATION_PART_Y_INDEX_SOURCE,
            POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
        )
    ]
)


@dataclass(frozen=True)
class PreparedPolygonPartYIndex:
    """Device-resident edge memberships grouped by polygon-part y bin."""

    family: GeometryFamily
    geometry_count: int
    part_count: int
    bin_count: int
    edge_membership_count: int
    part_ymin: object
    part_ymax: object
    counts: object
    offsets: object
    entries: object

    @property
    def device_bytes(self) -> int:
        return sum(
            int(getattr(value, "nbytes", 0))
            for value in (
                self.part_ymin,
                self.part_ymax,
                self.counts,
                self.offsets,
                self.entries,
            )
        )


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
                "edge_memberships": int(prepared.edge_membership_count),
                "persistent_bytes": int(prepared.device_bytes),
            }
        )
    unavailable = [
        "build_seconds",
        "avoidable_rebuild_seconds",
        "invalidation_reason",
    ]
    if prepared is None:
        unavailable.extend(
            (
                "source_geometries",
                "source_parts",
                "part_y_bin_slots",
                "edge_memberships",
                "persistent_bytes",
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


def point_location_part_y_index_kernels():
    return compile_kernel_group(
        "point-location-part-y-index",
        _POINT_LOCATION_PART_Y_INDEX_SOURCE,
        POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
    )


def point_location_part_y_index_profile_kernels():
    """Compile profiler-only entry points outside production warmup."""
    return compile_kernel_group(
        "point-location-part-y-index-profile",
        _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE,
        POINT_LOCATION_PART_Y_INDEX_PROFILE_KERNEL_NAMES,
    )


def prepare_polygon_part_y_index(owned, family: GeometryFamily):
    """Build or return the cached exact part-y edge directory."""
    if family not in _INDEXABLE_FAMILIES:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    cached = state.point_location_indexes.get(family)
    with hotpath_stage(
        "predicate.point_location_part_y_index.prepare",
        category="setup",
    ) as stage_metadata:
        prepared = _prepare_polygon_part_y_index_impl(
            owned,
            family,
            state=state,
            cached=cached,
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
                    "cache_identity_exported": False,
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
):
    if cached is not None:
        from .point_region_profile import current_point_region_profile

        profile = current_point_region_profile()
        if profile is not None:
            profile.note_index_cache_hit(cached)
        return cached
    buffer = state.families.get(family)
    from .point_region_profile import current_point_region_profile

    profile = current_point_region_profile()
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
        buffer.geometry_offsets
        if family is GeometryFamily.POLYGON
        else buffer.part_offsets
    )
    if part_ring_offsets is None or buffer.ring_offsets is None:
        return None
    part_count = int(part_ring_offsets.size - 1)
    bin_slots = part_count * PART_Y_BIN_COUNT
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = point_location_part_y_index_kernels()

    part_ymin = cp.empty(part_count, dtype=cp.float64)
    part_ymax = cp.empty(part_count, dtype=cp.float64)
    counts = cp.empty(bin_slots, dtype=cp.uint32)
    count_kernel = kernels["count_polygon_part_y_bins"]
    grid, block = runtime.launch_config(count_kernel, part_count)
    runtime.launch(
        count_kernel,
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
                ptr(counts),
            ),
            (
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
    admission = runtime.admit_device_memory(
        stage="predicate.point_location_part_y_index",
        required_bytes=membership_count * np.dtype(np.uint32).itemsize,
        requested_units=membership_count,
    )
    if not admission.admitted:
        return None
    entries = cp.empty(membership_count, dtype=cp.uint32)
    scatter_kernel = kernels["scatter_polygon_part_y_bins"]
    grid, block = runtime.launch_config(scatter_kernel, part_count)
    runtime.launch(
        scatter_kernel,
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
                ptr(offsets),
                ptr(entries),
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
            ),
        ),
    )
    prepared = PreparedPolygonPartYIndex(
        family=family,
        geometry_count=int(buffer.geometry_offsets.size - 1),
        part_count=part_count,
        bin_count=PART_Y_BIN_COUNT,
        edge_membership_count=membership_count,
        part_ymin=part_ymin,
        part_ymax=part_ymax,
        counts=counts,
        offsets=offsets,
        entries=entries,
    )
    state.point_location_indexes[family] = prepared
    if profile is not None:
        # Production stays fully asynchronous.  Profiling pays one explicit
        # completion fence so the reported wall interval includes scatter,
        # whose entries are part of the completed index.
        runtime.synchronize()
        assert build_started is not None
        profile.note_index_build(prepared, perf_counter() - build_started)
    record_dispatch_event(
        surface="vibespatial.predicates.point_location_index",
        operation="prepare_point_location",
        implementation="polygon_part_y_edge_directory_gpu",
        reason=(
            f"prepared {part_count} polygon parts into {bin_slots} y bins "
            f"with {membership_count} exact edge memberships"
        ),
        selected=ExecutionMode.GPU,
    )
    return prepared


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
