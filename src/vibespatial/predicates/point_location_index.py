"""Reusable exact y-edge directory for point-in-polygon refinement.

This is a physical metadata carrier, not a public spatial index.  It is cached
on the polygon ``OwnedGeometryDeviceState`` and reused by repeated public
``sindex.query`` calls.  Eight y bins per polygon part reduce exact ray-cast
work without duplicating geometry coordinates or changing predicate semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

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

from .point_location_index_kernels import (
    _POINT_LOCATION_PART_Y_INDEX_SOURCE,
    PART_Y_BIN_COUNT,
    POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
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


def point_location_part_y_index_kernels():
    return compile_kernel_group(
        "point-location-part-y-index",
        _POINT_LOCATION_PART_Y_INDEX_SOURCE,
        POINT_LOCATION_PART_Y_INDEX_KERNEL_NAMES,
    )


def prepare_polygon_part_y_index(owned, family: GeometryFamily):
    """Build or return the cached exact part-y edge directory."""
    if family not in _INDEXABLE_FAMILIES:
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    cached = state.point_location_indexes.get(family)
    if cached is not None:
        return cached
    buffer = state.families.get(family)
    if buffer is None or int(buffer.x.size) < _MIN_PREPARED_COORDINATES:
        return None
    if int(buffer.x.size) >= (1 << 31) or int(buffer.ring_offsets.size - 1) >= (1 << 31):
        return None

    import cupy as cp

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
