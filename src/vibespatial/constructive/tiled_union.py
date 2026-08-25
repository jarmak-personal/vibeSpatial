"""Exact device collective union with bounded topology tiles.

The logical operation is one polygonal unary union.  Its physical work is
segment, tile, face, and boundary-atom shaped: small arrangements use one
collective topology plan, while large arrangements are clipped into disjoint
tiles, solved locally, and stitched by noded coverage assembly.

ADR-0002: constructive coordinates and topology remain fp64.
ADR-0044: inputs and outputs remain device-resident ``OwnedGeometryArray``.
ADR-0046: dispatch uses segment-peer pressure rather than public row count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - GPU-only implementation
    cp = None

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    PairSortStrategy,
    compact_indices,
    exclusive_sum,
    sort_pairs,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.crossover import estimate_physical_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import (
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray


request_warmup(
    [
        "exclusive_scan_i32",
        "radix_sort_u64_i32",
    ]
)


# Face propagation becomes nonlinear once a dense collective arrangement has
# enough source-segment peer pressure.  This estimate is deliberately based on
# physical segment capacity and fan-in, not a public row threshold.
_DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE = 64 * 1024 * 1024
# Persistent split events, atomic edges, and faces, rather than bbox candidates,
# set the local topology limit. The online seam reducer bounds aggregate child
# output, so local plans target two million conservative segment peers.
_TARGET_TILE_SEGMENT_PEER_PRESSURE = 8 * 1024 * 1024
_MIN_TOPOLOGY_TILE_COUNT = 8
_TOPOLOGY_SEAM_FAN_IN = 4
# Aggregate segment-peer pressure bounds topology work. This second bound keeps
# grouped output slots to one compact planning page even when active tiles are
# individually sparse.
_TOPOLOGY_CONSTRUCTIVE_BATCH_TILES = 32


@dataclass(frozen=True)
class _TopologyTilePlan:
    tile_bounds: object
    source_bounds: object
    relation: object
    host_offsets: np.ndarray
    host_segment_peer_pressure: np.ndarray
    full_tile_mask: object
    cols: int
    rows: int
    max_segment_peer_pressure: int


def _sync_hotpath() -> None:
    if hotpath_timing_enabled():
        cp.cuda.get_current_stream().synchronize()


def _single_group_collective_union_gpu(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    from vibespatial.constructive.binary_constructive import (
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )

    row_count = int(owned.row_count)
    result = _regroup_native_grouped_parts_with_grouped_union_gpu(
        owned,
        cp.arange(row_count, dtype=cp.int64),
        cp.asarray([0, row_count], dtype=cp.int64),
        cp.asarray([0], dtype=cp.int64),
        output_row_count=1,
        dispatch_mode=ExecutionMode.GPU,
        allow_direct_disjoint_pack=False,
        use_same_row_fast_path=True,
        group_size_max=row_count,
    )
    if result is None or result.row_count != 1:
        raise RuntimeError("collective polygon union did not produce its logical output row")
    return result


def _topology_tile_count(segment_peer_pressure: int) -> int:
    ratio = max(
        float(segment_peer_pressure) / float(_TARGET_TILE_SEGMENT_PEER_PRESSURE),
        1.0,
    )
    # A 2-D grid partitions boundary segments by both axes but elongated
    # polygon rows can still span one full axis.  The conservative 2/3 power
    # models one axis reducing segment work and the other reducing row fan-in;
    # the measured relation below keeps refining if skew violates that model.
    required = max(_MIN_TOPOLOGY_TILE_COUNT, int(math.ceil(ratio ** (2.0 / 3.0))))
    power_of_two = 1 << int(math.ceil(math.log2(required)))
    return power_of_two


def _device_topology_tile_bounds(
    owned: OwnedGeometryArray,
    *,
    tile_count: int,
    source_bounds=None,
):
    """Build a row-major power-of-two grid from a compact device plan."""
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )
    from vibespatial.spatial.indexing import RegularGridRectIndex

    d_bounds = (
        cp.asarray(source_bounds, dtype=cp.float64).reshape(owned.row_count, 4)
        if source_bounds is not None
        else cp.asarray(
            compute_geometry_bounds_device(owned, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(owned.row_count, 4)
    )
    d_global = cp.stack(
        (
            cp.min(d_bounds[:, 0]),
            cp.min(d_bounds[:, 1]),
            cp.max(d_bounds[:, 2]),
            cp.max(d_bounds[:, 3]),
        )
    ).astype(cp.float64, copy=False)
    d_width = d_global[2] - d_global[0]
    d_height = d_global[3] - d_global[1]
    d_x_pressure = cp.sum(
        (d_bounds[:, 2] - d_bounds[:, 0]) / cp.maximum(d_width, np.finfo(np.float64).tiny),
        dtype=cp.float64,
    )
    d_y_pressure = cp.sum(
        (d_bounds[:, 3] - d_bounds[:, 1])
        / cp.maximum(d_height, np.finfo(np.float64).tiny),
        dtype=cp.float64,
    )
    tile_exponent = int(math.log2(tile_count))
    if 1 << tile_exponent != tile_count:
        raise ValueError("collective topology tile count must be a power of two")
    major_count = 1 << ((tile_exponent + 1) // 2)
    minor_count = 1 << (tile_exponent // 2)
    planning_packet = get_cuda_runtime().copy_device_to_host(
        cp.concatenate((d_global, cp.stack((d_x_pressure, d_y_pressure)))),
        reason="collective union topology-grid planning packet",
    )
    major_is_y = bool(float(planning_packet[5]) > float(planning_packet[4]))
    cols = minor_count if major_is_y else major_count
    rows = major_count if major_is_y else minor_count
    origin_x = float(planning_packet[0])
    origin_y = float(planning_packet[1])
    width = float(planning_packet[2]) - origin_x
    height = float(planning_packet[3]) - origin_y
    if not width > 0.0 or not height > 0.0:
        raise ValueError("collective polygon topology requires positive 2-D extent")
    cell_width = width / float(cols)
    cell_height = height / float(rows)

    d_tile_ids = cp.arange(tile_count, dtype=cp.int32)
    d_row_ids = d_tile_ids // np.int32(cols)
    d_col_ids = d_tile_ids % np.int32(cols)
    d_tile_bounds = cp.empty((tile_count, 4), dtype=cp.float64)
    d_tile_bounds[:, 0] = origin_x + d_col_ids * cell_width
    d_tile_bounds[:, 1] = origin_y + d_row_ids * cell_height
    d_tile_bounds[:, 2] = origin_x + (d_col_ids + np.int32(1)) * cell_width
    d_tile_bounds[:, 3] = origin_y + (d_row_ids + np.int32(1)) * cell_height
    metadata = RegularGridRectIndex(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=rows,
        size=tile_count,
    )
    return d_tile_bounds, d_bounds, metadata


def _compact_unique_topology_pair_keys(keys):
    """Return sorted unique ``tile,row`` uint64 keys on the device."""
    d_keys = cp.asarray(keys, dtype=cp.uint64)
    item_count = int(d_keys.size)
    if item_count == 0:
        return cp.empty(0, dtype=cp.uint64)
    sorted_result = sort_pairs(
        d_keys,
        cp.arange(item_count, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    d_sorted = sorted_result.keys
    d_unique = cp.empty(item_count, dtype=cp.uint8)
    d_unique[0] = 1
    if item_count > 1:
        d_unique[1:] = (d_sorted[1:] != d_sorted[:-1]).astype(cp.uint8, copy=False)
    return d_sorted[compact_indices(d_unique).values]


def _topology_scanline_center_pair_keys(segments, metadata):
    """Return source rows whose polygon contains each grid-cell center.

    Horizontal scanlines use a half-open edge rule.  Sorting crossings by
    ``(source row, grid row, x)`` then pairing alternating crossings gives the
    exact even-odd interior intervals for valid Polygon and MultiPolygon rows.
    """
    from vibespatial.cuda._runtime import count_scatter_total, get_cuda_runtime
    from vibespatial.overlay.graph import _fp64_radix_keys, _stable_radix_order_pass

    segment_count = int(segments.count)
    if segment_count == 0:
        return cp.empty(0, dtype=cp.uint64)
    d_x0 = cp.asarray(segments.x0, dtype=cp.float64)
    d_y0 = cp.asarray(segments.y0, dtype=cp.float64)
    d_x1 = cp.asarray(segments.x1, dtype=cp.float64)
    d_y1 = cp.asarray(segments.y1, dtype=cp.float64)
    d_lower_y = cp.minimum(d_y0, d_y1)
    d_upper_y = cp.maximum(d_y0, d_y1)
    d_start_rows = cp.ceil(
        (d_lower_y - metadata.origin_y) / metadata.cell_height - 0.5
    ).astype(cp.int32)
    d_stop_rows = cp.ceil(
        (d_upper_y - metadata.origin_y) / metadata.cell_height - 0.5
    ).astype(cp.int32)
    d_start_rows = cp.clip(d_start_rows, 0, metadata.rows)
    d_stop_rows = cp.clip(d_stop_rows, 0, metadata.rows)
    d_counts = cp.where(
        d_y0 != d_y1,
        cp.maximum(d_stop_rows - d_start_rows, np.int32(0)),
        np.int32(0),
    ).astype(cp.int32, copy=False)
    d_offsets = exclusive_sum(d_counts, synchronize=False)
    crossing_count = count_scatter_total(
        get_cuda_runtime(),
        d_counts,
        d_offsets,
        reason="collective union tile-center scanline event allocation fence",
    )
    if crossing_count == 0:
        return cp.empty(0, dtype=cp.uint64)
    d_slots = cp.arange(crossing_count, dtype=cp.int32)
    d_segment_ids = cp.searchsorted(
        d_offsets + d_counts,
        d_slots,
        side="right",
    ).astype(cp.int32, copy=False)
    d_grid_rows = (
        d_start_rows[d_segment_ids]
        + d_slots
        - d_offsets[d_segment_ids]
    ).astype(cp.int32, copy=False)
    d_scan_y = metadata.origin_y + (
        d_grid_rows.astype(cp.float64) + 0.5
    ) * metadata.cell_height
    d_cross_x = d_x0[d_segment_ids] + (
        (d_scan_y - d_y0[d_segment_ids])
        * (d_x1[d_segment_ids] - d_x0[d_segment_ids])
        / (d_y1[d_segment_ids] - d_y0[d_segment_ids])
    )
    d_source_rows = cp.asarray(segments.row_indices, dtype=cp.int32)[d_segment_ids]
    d_group_keys = (
        d_source_rows.astype(cp.uint64) << cp.uint64(32)
    ) | d_grid_rows.astype(cp.uint32).astype(cp.uint64)
    d_order = cp.arange(crossing_count, dtype=cp.int32)
    d_order = _stable_radix_order_pass(d_order, _fp64_radix_keys(d_cross_x))
    d_order = _stable_radix_order_pass(d_order, d_group_keys)
    d_sorted_groups = d_group_keys[d_order]
    d_sorted_x = d_cross_x[d_order]
    d_group_starts = cp.empty(crossing_count, dtype=cp.bool_)
    d_group_starts[0] = True
    if crossing_count > 1:
        d_group_starts[1:] = d_sorted_groups[1:] != d_sorted_groups[:-1]
    d_positions = cp.arange(crossing_count, dtype=cp.int32)
    d_group_start_positions = d_positions[d_group_starts]
    d_group_ids = (
        cp.cumsum(d_group_starts.astype(cp.int32), dtype=cp.int32) - np.int32(1)
    )
    d_local_positions = d_positions - d_group_start_positions[d_group_ids]
    d_interval_active = cp.zeros(crossing_count, dtype=cp.bool_)
    if crossing_count > 1:
        d_interval_active[:-1] = (
            (d_sorted_groups[:-1] == d_sorted_groups[1:])
            & ((d_local_positions[:-1] & np.int32(1)) == 0)
            & (d_sorted_x[:-1] < d_sorted_x[1:])
        )
    d_interval_ids = compact_indices(
        d_interval_active.astype(cp.uint8, copy=False)
    ).values
    interval_count = int(d_interval_ids.size)
    if interval_count == 0:
        return cp.empty(0, dtype=cp.uint64)

    d_left_x = d_sorted_x[d_interval_ids]
    d_right_x = d_sorted_x[d_interval_ids + 1]
    d_start_cols = cp.floor(
        (d_left_x - metadata.origin_x) / metadata.cell_width - 0.5
    ).astype(cp.int32) + np.int32(1)
    d_stop_cols = cp.ceil(
        (d_right_x - metadata.origin_x) / metadata.cell_width - 0.5
    ).astype(cp.int32)
    d_start_cols = cp.clip(d_start_cols, 0, metadata.cols)
    d_stop_cols = cp.clip(d_stop_cols, 0, metadata.cols)
    d_column_counts = cp.maximum(
        d_stop_cols - d_start_cols,
        np.int32(0),
    ).astype(cp.int32, copy=False)
    d_column_offsets = exclusive_sum(d_column_counts, synchronize=False)
    center_pair_count = count_scatter_total(
        get_cuda_runtime(),
        d_column_counts,
        d_column_offsets,
        reason="collective union tile-center relation allocation fence",
    )
    if center_pair_count == 0:
        return cp.empty(0, dtype=cp.uint64)
    d_center_slots = cp.arange(center_pair_count, dtype=cp.int32)
    d_expanded_intervals = cp.searchsorted(
        d_column_offsets + d_column_counts,
        d_center_slots,
        side="right",
    ).astype(cp.int32, copy=False)
    d_cols = (
        d_start_cols[d_expanded_intervals]
        + d_center_slots
        - d_column_offsets[d_expanded_intervals]
    ).astype(cp.int32, copy=False)
    d_interval_groups = d_sorted_groups[d_interval_ids][d_expanded_intervals]
    d_rows = (d_interval_groups & cp.uint64(0xFFFFFFFF)).astype(
        cp.int32,
        copy=False,
    )
    d_sources = (d_interval_groups >> cp.uint64(32)).astype(
        cp.uint32,
        copy=False,
    )
    d_tiles = d_rows * np.int32(metadata.cols) + d_cols
    return _compact_unique_topology_pair_keys(
        (d_tiles.astype(cp.uint64) << cp.uint64(32))
        | d_sources.astype(cp.uint64)
    )


def _build_topology_tile_candidate_relation(
    owned: OwnedGeometryArray,
    segments,
    d_tile_bounds,
    metadata,
):
    """Return an exact segment-shaped tile/source relation and full-tile proof."""
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial.indexing import FlatSpatialIndex
    from vibespatial.spatial.query_box import _query_regular_grid_rect_box_index

    tile_count = int(d_tile_bounds.shape[0])
    source_row_count = int(owned.row_count)
    d_segment_bounds = cp.empty((segments.count, 4), dtype=cp.float64)
    d_segment_bounds[:, 0] = cp.minimum(segments.x0, segments.x1)
    d_segment_bounds[:, 1] = cp.minimum(segments.y0, segments.y1)
    d_segment_bounds[:, 2] = cp.maximum(segments.x0, segments.x1)
    d_segment_bounds[:, 3] = cp.maximum(segments.y0, segments.y1)
    tile_owned = _build_device_boxes_from_bounds(
        d_tile_bounds,
        row_count=tile_count,
    )
    tile_index = FlatSpatialIndex(
        geometry_array=tile_owned,
        _host_order=None,
        _host_morton_keys=None,
        _host_bounds=None,
        total_bounds=(
            metadata.origin_x,
            metadata.origin_y,
            metadata.origin_x + metadata.cols * metadata.cell_width,
            metadata.origin_y + metadata.rows * metadata.cell_height,
        ),
        regular_grid=metadata,
        device_order=cp.arange(tile_count, dtype=cp.int32),
        device_bounds=d_tile_bounds,
    )
    segment_pairs = _query_regular_grid_rect_box_index(
        tile_index,
        d_segment_bounds,
        predicate=None,
    )
    if segment_pairs is None or not hasattr(segment_pairs, "d_left"):
        raise RuntimeError("segment-grid topology relation did not remain device-resident")
    d_raw_tile_ids = cp.asarray(segment_pairs.d_right, dtype=cp.int32)
    d_raw_source_rows = cp.asarray(segments.row_indices, dtype=cp.int32)[
        cp.asarray(segment_pairs.d_left, dtype=cp.int32)
    ]
    d_boundary_keys = _compact_unique_topology_pair_keys(
        (d_raw_tile_ids.astype(cp.uint64) << cp.uint64(32))
        | d_raw_source_rows.astype(cp.uint32).astype(cp.uint64)
    )
    d_center_keys = _topology_scanline_center_pair_keys(segments, metadata)
    if int(d_center_keys.size) > 0:
        d_boundary_positions = cp.searchsorted(d_boundary_keys, d_center_keys)
        d_safe_positions = cp.minimum(
            d_boundary_positions,
            np.int64(max(int(d_boundary_keys.size) - 1, 0)),
        )
        d_has_boundary = (
            d_boundary_positions < int(d_boundary_keys.size)
        ) & (
            d_boundary_keys[d_safe_positions] == d_center_keys
            if int(d_boundary_keys.size) > 0
            else cp.zeros(d_center_keys.size, dtype=cp.bool_)
        )
        d_full_tile_ids = cp.unique(
            (d_center_keys[~d_has_boundary] >> cp.uint64(32)).astype(
                cp.int32,
                copy=False,
            )
        )
    else:
        d_full_tile_ids = cp.empty(0, dtype=cp.int32)
    d_full_tile_mask = cp.zeros(tile_count, dtype=cp.bool_)
    d_full_tile_mask[d_full_tile_ids] = True

    d_tile_ids = (d_boundary_keys >> cp.uint64(32)).astype(cp.int32, copy=False)
    d_source_rows = (d_boundary_keys & cp.uint64(0xFFFFFFFF)).astype(
        cp.int32,
        copy=False,
    )
    d_keep = ~d_full_tile_mask[d_tile_ids]
    d_tile_ids = d_tile_ids[d_keep]
    d_source_rows = d_source_rows[d_keep]
    d_counts = cp.bincount(d_tile_ids, minlength=tile_count).astype(
        cp.int64,
        copy=False,
    )
    d_offsets = cp.empty(tile_count + 1, dtype=cp.int64)
    d_offsets[0] = 0
    d_offsets[1:] = cp.cumsum(d_counts, dtype=cp.int64)
    d_segment_counts = cp.bincount(
        d_raw_tile_ids,
        minlength=tile_count,
    ).astype(cp.int64, copy=False)
    d_segment_peer_pressure = d_segment_counts * d_counts
    host_plan = np.asarray(
        get_cuda_runtime().copy_device_to_host(
            cp.concatenate((d_offsets, d_segment_peer_pressure)),
            reason="collective union segment-tile grouped planning packet",
        ),
        dtype=np.int64,
    )
    host_offsets = host_plan[: tile_count + 1]
    host_segment_peer_pressure = host_plan[tile_count + 1 :]
    max_pressure = int(np.max(host_segment_peer_pressure, initial=0))
    relation = NativeRelation(
        left_indices=d_tile_ids.astype(cp.int64, copy=False),
        right_indices=d_source_rows.astype(cp.int64, copy=False),
        predicate="bbox_intersects",
        left_row_count=tile_count,
        right_row_count=source_row_count,
        sorted_by_left=True,
        left_group_offsets=d_offsets,
    )
    return (
        relation,
        host_offsets,
        host_segment_peer_pressure,
        d_full_tile_mask,
        max_pressure,
    )


def _topology_constructive_batch_spans(
    host_offsets: np.ndarray,
    host_segment_peer_pressure: np.ndarray,
    *,
    max_tiles_per_batch: int,
) -> tuple[tuple[int, int], ...]:
    """Pack active contiguous tile ranges under one aggregate topology budget."""
    offsets = np.asarray(host_offsets, dtype=np.int64)
    pressure = np.asarray(host_segment_peer_pressure, dtype=np.int64)
    tile_count = int(pressure.size)
    if offsets.shape != (tile_count + 1,):
        raise ValueError("topology batch offsets and pressure must align")
    if max_tiles_per_batch <= 0:
        raise ValueError("topology batch tile capacity must be positive")
    active_tile_ids = np.flatnonzero(np.diff(offsets) > 0)
    batches: list[tuple[int, int]] = []
    active_index = 0
    while active_index < int(active_tile_ids.size):
        tile_start = int(active_tile_ids[active_index])
        tile_end = tile_start + 1
        batch_pressure = int(pressure[tile_start])
        active_index += 1
        while active_index < int(active_tile_ids.size):
            next_tile = int(active_tile_ids[active_index])
            next_end = next_tile + 1
            if next_end - tile_start > max_tiles_per_batch:
                break
            next_pressure = int(pressure[next_tile])
            if (
                batch_pressure > 0
                and next_pressure > 0
                and batch_pressure + next_pressure
                > _TARGET_TILE_SEGMENT_PEER_PRESSURE
            ):
                break
            tile_end = next_end
            batch_pressure += next_pressure
            active_index += 1
        batches.append((tile_start, tile_end))
    return tuple(batches)


def _clip_topology_tile_candidate_batch(
    owned: OwnedGeometryArray,
    d_source_rows,
    d_candidate_tile_ids,
    d_tile_bounds,
):
    """Clip one tile batch's bbox-related source capacity."""
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds

    d_source_rows = cp.asarray(d_source_rows, dtype=cp.int64)
    candidate_count = int(d_source_rows.size)
    source_capacity = owned._device_indexed_take(
        d_source_rows,
        assume_unique_indices=False,
    )
    d_pair_bounds = cp.asarray(d_tile_bounds, dtype=cp.float64)[
        cp.asarray(d_candidate_tile_ids, dtype=cp.int64)
    ]
    rectangle_capacity = _build_device_boxes_from_bounds(
        d_pair_bounds,
        row_count=candidate_count,
    )
    clipped = _binary_constructive_gpu(
        "intersection",
        source_capacity,
        rectangle_capacity,
        dispatch_mode=ExecutionMode.GPU,
    )
    if clipped is None or clipped.row_count != candidate_count:
        raise RuntimeError("topology tile clip violated candidate-relation capacity")
    return clipped


def _normalize_collective_polygon_parts(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Explode polygonal rows once so every topology tile clips Polygon parts."""
    if set(owned.families) == {GeometryFamily.POLYGON}:
        return owned
    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )

    parts = _explode_polygonal_rows_to_polygon_capacity_gpu(owned)
    if parts is None:
        raise RuntimeError("collective union could not physicalize polygon parts")
    d_active_positions = cp.flatnonzero(
        parts.selection.active_capacity_mask()
    ).astype(cp.int64, copy=False)
    if int(d_active_positions.size) == 0:
        raise RuntimeError("collective union received no active polygon parts")
    normalized = parts.geometry._device_indexed_take(
        d_active_positions,
        assume_unique_indices=True,
    )
    state = normalized._ensure_device_state(preserve_indexed_view=True)
    state.trusted_all_valid = True
    state.trusted_all_non_empty = True
    state.trusted_homogeneous_family = GeometryFamily.POLYGON
    state.trusted_polygonal_only = True
    return normalized


def _stitch_topology_coverage_rows(
    coverage: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Union one bounded coverage fan-in through exact grouped topology."""
    from vibespatial.constructive.binary_constructive import (
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )

    row_count = int(coverage.row_count)
    if row_count <= 1:
        return coverage
    if row_count > _TOPOLOGY_SEAM_FAN_IN:
        raise ValueError("topology seam stitch exceeded its bounded fan-in")

    result = _regroup_native_grouped_parts_with_grouped_union_gpu(
        coverage,
        cp.arange(row_count, dtype=cp.int64),
        cp.asarray([0, row_count], dtype=cp.int64),
        cp.asarray([0], dtype=cp.int64),
        output_row_count=1,
        dispatch_mode=ExecutionMode.GPU,
        allow_direct_disjoint_pack=False,
        use_same_row_fast_path=False,
        group_size_max=row_count,
    )
    if result is None or result.row_count != 1:
        raise RuntimeError("topology seam assembly did not produce one output row")
    return _physicalize_topology_coverage_output(result)


def _physicalize_topology_coverage_output(
    coverage: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Compact emitted topology rows before they enter the seam carrier."""
    from vibespatial.geometry.owned import (
        device_physicalize_owned_row_selections_exact,
    )

    (physical,) = device_physicalize_owned_row_selections_exact(
        [(coverage, cp.ones(coverage.row_count, dtype=cp.bool_))],
        reason="collective union emitted topology exact-allocation packet",
        compact_concrete_prefix=True,
    )
    if physical is None or physical.row_count != coverage.row_count:
        raise RuntimeError("topology output physicalization lost concrete rows")
    return physical


def _reduce_topology_tile_coverage(
    tile_results: list[OwnedGeometryArray],
) -> OwnedGeometryArray:
    """Hierarchically stitch power-of-two sibling coverages on device."""
    from vibespatial.geometry.owned import OwnedGeometryArray

    level = tile_results
    while len(level) > 1:
        next_level: list[OwnedGeometryArray] = []
        for start in range(0, len(level), _TOPOLOGY_SEAM_FAN_IN):
            siblings = OwnedGeometryArray.concat(
                level[start : start + _TOPOLOGY_SEAM_FAN_IN]
            )
            next_level.append(_stitch_topology_coverage_rows(siblings))
        level = next_level
    if not level:
        raise RuntimeError("topology tile coverage produced no rows")
    return level[0]


def _append_topology_coverage_row(
    levels: list[list[OwnedGeometryArray]],
    row: OwnedGeometryArray,
    *,
    level_index: int = 0,
) -> None:
    """Carry one tile row through an online base-fan-in seam reduction."""
    carry = row
    while True:
        while level_index >= len(levels):
            levels.append([])
        level = levels[level_index]
        level.append(carry)
        if len(level) < _TOPOLOGY_SEAM_FAN_IN:
            return
        from vibespatial.geometry.owned import OwnedGeometryArray

        carry = _stitch_topology_coverage_rows(OwnedGeometryArray.concat(level))
        level.clear()
        level_index += 1


def _finish_topology_coverage_levels(
    levels: list[list[OwnedGeometryArray]],
) -> OwnedGeometryArray:
    """Stitch the ordered residual chunks from an online coverage reduction."""
    from vibespatial.geometry.owned import OwnedGeometryArray

    ordered_chunks: list[OwnedGeometryArray] = []
    for level in reversed(levels):
        if not level:
            continue
        siblings = OwnedGeometryArray.concat(level)
        ordered_chunks.append(_stitch_topology_coverage_rows(siblings))
    return _reduce_topology_tile_coverage(ordered_chunks)


def _plan_topology_tiles(
    owned: OwnedGeometryArray,
    *,
    initial_tile_count: int,
    force_tile_count: bool,
) -> _TopologyTilePlan:
    """Refine a segment-grid relation until every local topology is bounded."""
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )
    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    d_source_bounds = cp.asarray(
        compute_geometry_bounds_device(owned, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(owned.row_count, 4)
    segments = _extract_segments_gpu(owned)
    tile_count = int(initial_tile_count)
    try:
        while True:
            d_tile_bounds, _, metadata = _device_topology_tile_bounds(
                owned,
                tile_count=tile_count,
                source_bounds=d_source_bounds,
            )
            (
                relation,
                host_offsets,
                host_segment_peer_pressure,
                d_full_tile_mask,
                max_pressure,
            ) = (
                _build_topology_tile_candidate_relation(
                    owned,
                    segments,
                    d_tile_bounds,
                    metadata,
                )
            )
            if force_tile_count or max_pressure <= _TARGET_TILE_SEGMENT_PEER_PRESSURE:
                return _TopologyTilePlan(
                    tile_bounds=d_tile_bounds,
                    source_bounds=d_source_bounds,
                    relation=relation,
                    host_offsets=host_offsets,
                    host_segment_peer_pressure=host_segment_peer_pressure,
                    full_tile_mask=d_full_tile_mask,
                    cols=metadata.cols,
                    rows=metadata.rows,
                    max_segment_peer_pressure=max_pressure,
                )
            del (
                relation,
                host_offsets,
                host_segment_peer_pressure,
                d_full_tile_mask,
                d_tile_bounds,
            )
            tile_count *= 2
            if tile_count >= 1 << 31:
                raise RuntimeError(
                    "collective union topology grid exceeded int32 tile capacity"
                )
    finally:
        from vibespatial.cuda._runtime import get_cuda_completion_retainer

        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            segments,
            lambda retained_segments: retained_segments.free(),
        )


def _assemble_full_topology_tiles(
    d_tile_bounds,
    d_full_tile_mask,
    *,
    cols: int,
    rows: int,
) -> OwnedGeometryArray | None:
    """Assemble a dense regular-grid coverage from its exposed edge contour."""
    from vibespatial.overlay.boundary_graph import (
        build_polygon_output_from_boundary_segments_gpu,
    )
    from vibespatial.runtime import RuntimeSelection

    d_full = cp.asarray(d_full_tile_mask, dtype=cp.bool_).reshape(rows, cols)
    d_tile_ids = cp.arange(cols * rows, dtype=cp.int32).reshape(rows, cols)
    d_left = cp.flatnonzero(
        d_full & cp.concatenate(
            (cp.ones((rows, 1), dtype=cp.bool_), ~d_full[:, :-1]),
            axis=1,
        )
    )
    d_right = cp.flatnonzero(
        d_full & cp.concatenate(
            (~d_full[:, 1:], cp.ones((rows, 1), dtype=cp.bool_)),
            axis=1,
        )
    )
    d_bottom = cp.flatnonzero(
        d_full & cp.concatenate(
            (cp.ones((1, cols), dtype=cp.bool_), ~d_full[:-1, :]),
            axis=0,
        )
    )
    d_top = cp.flatnonzero(
        d_full & cp.concatenate(
            (~d_full[1:, :], cp.ones((1, cols), dtype=cp.bool_)),
            axis=0,
        )
    )
    edge_count = sum(map(int, (d_left.size, d_right.size, d_bottom.size, d_top.size)))
    if edge_count == 0:
        return None
    d_bounds = cp.asarray(d_tile_bounds, dtype=cp.float64)
    left_bounds = d_bounds[d_tile_ids.ravel()[d_left]]
    right_bounds = d_bounds[d_tile_ids.ravel()[d_right]]
    bottom_bounds = d_bounds[d_tile_ids.ravel()[d_bottom]]
    top_bounds = d_bounds[d_tile_ids.ravel()[d_top]]
    start_x = cp.concatenate(
        (left_bounds[:, 0], right_bounds[:, 2], bottom_bounds[:, 0], top_bounds[:, 2])
    )
    start_y = cp.concatenate(
        (left_bounds[:, 3], right_bounds[:, 1], bottom_bounds[:, 1], top_bounds[:, 3])
    )
    end_x = cp.concatenate(
        (left_bounds[:, 0], right_bounds[:, 2], bottom_bounds[:, 2], top_bounds[:, 0])
    )
    end_y = cp.concatenate(
        (left_bounds[:, 1], right_bounds[:, 3], bottom_bounds[:, 1], top_bounds[:, 3])
    )
    return build_polygon_output_from_boundary_segments_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        row_indices=cp.zeros(edge_count, dtype=cp.int32),
        row_count=1,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="regular-grid full-tile coverage assembled from exposed edges",
        ),
        d_valid_empty_rows=cp.ones(1, dtype=cp.bool_),
    )


def _tiled_single_group_collective_union_gpu(
    owned: OwnedGeometryArray,
    *,
    tile_count: int,
    force_tile_count: bool,
    assemble_union: bool = True,
) -> tuple[OwnedGeometryArray | None, OwnedGeometryArray, int, int]:
    from vibespatial.constructive.binary_constructive import (
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )
    from vibespatial.geometry.owned import build_empty_polygon_rows_device

    topology_plan = _plan_topology_tiles(
        owned,
        initial_tile_count=tile_count,
        force_tile_count=force_tile_count,
    )
    d_tile_bounds = topology_plan.tile_bounds
    tile_relation = topology_plan.relation
    host_offsets = topology_plan.host_offsets
    host_segment_peer_pressure = topology_plan.host_segment_peer_pressure
    tile_count = int(topology_plan.cols * topology_plan.rows)
    max_segment_peer_pressure = topology_plan.max_segment_peer_pressure
    coverage_levels: list[list[OwnedGeometryArray]] = []
    coverage_parts: list[OwnedGeometryArray] = []
    full_tile_coverage = _assemble_full_topology_tiles(
        d_tile_bounds,
        topology_plan.full_tile_mask,
        cols=topology_plan.cols,
        rows=topology_plan.rows,
    )
    if full_tile_coverage is not None:
        if assemble_union:
            _append_topology_coverage_row(coverage_levels, full_tile_coverage)
        else:
            coverage_parts.append(full_tile_coverage)

    active_tile_batches = _topology_constructive_batch_spans(
        host_offsets,
        host_segment_peer_pressure,
        max_tiles_per_batch=_TOPOLOGY_CONSTRUCTIVE_BATCH_TILES,
    )
    for tile_start, tile_end in active_tile_batches:
        batch_tile_count = tile_end - tile_start
        row_start = int(host_offsets[tile_start])
        row_end = int(host_offsets[tile_end])
        batch_candidate_count = row_end - row_start
        if batch_candidate_count <= 0:
            batch_result = build_empty_polygon_rows_device(batch_tile_count)
        else:
            d_candidate_tile_ids = tile_relation.left_indices[row_start:row_end]
            group_size_max = int(
                np.max(
                    np.diff(host_offsets[tile_start : tile_end + 1]),
                    initial=0,
                )
            )
            _sync_hotpath()
            with hotpath_stage(
                "constructive.union.tile_clip",
                category="setup",
            ) as amplification_metadata:
                clipped_batch = _clip_topology_tile_candidate_batch(
                    owned,
                    tile_relation.right_indices[row_start:row_end],
                    d_candidate_tile_ids,
                    d_tile_bounds,
                )
                if amplification_metadata is not None:
                    attach_work_amplification(
                        amplification_metadata,
                        operation="constructive.union.tile_clip",
                        metric_family="group_compression",
                        sums={
                            "input_rows": int(batch_candidate_count),
                            "pre_reduction_fragments": int(clipped_batch.row_count),
                            "output_groups": int(batch_tile_count),
                        },
                        maxima={
                            "max_group_size": int(group_size_max),
                            "tiles_per_batch": int(batch_tile_count),
                        },
                        unavailable=(
                            "input_segments",
                            "input_coordinates",
                            "output_parts",
                            "output_coordinates",
                        ),
                    )
            from vibespatial.geometry.owned import device_valid_nonempty_mask

            d_live_positions = cp.flatnonzero(
                device_valid_nonempty_mask(clipped_batch)
            ).astype(cp.int64, copy=False)
            live_count = int(d_live_positions.size)
            clipped_batch = clipped_batch._device_indexed_take(
                d_live_positions,
                assume_unique_indices=True,
            )
            if live_count == 0:
                batch_result = build_empty_polygon_rows_device(batch_tile_count)
            else:
                d_live_group_ids = (
                    cp.asarray(d_candidate_tile_ids, dtype=cp.int64)[
                        d_live_positions
                    ]
                    - np.int64(tile_start)
                )
                d_batch_counts = cp.zeros(batch_tile_count, dtype=cp.int32)
                cp.add.at(d_batch_counts, d_live_group_ids, np.int32(1))
                d_observed_group_ids = cp.flatnonzero(d_batch_counts > 0).astype(
                    cp.int64,
                    copy=False,
                )
                d_observed_offsets = cp.empty(
                    int(d_observed_group_ids.size) + 1,
                    dtype=cp.int64,
                )
                d_observed_offsets[0] = 0
                d_observed_offsets[1:] = cp.cumsum(
                    d_batch_counts[d_observed_group_ids],
                    dtype=cp.int64,
                )
                _sync_hotpath()
                with hotpath_stage(
                    "constructive.union.tile_topology",
                    category="refine",
                ) as amplification_metadata:
                    batch_result = _regroup_native_grouped_parts_with_grouped_union_gpu(
                        clipped_batch,
                        cp.arange(live_count, dtype=cp.int64),
                        d_observed_offsets,
                        d_observed_group_ids,
                        output_row_count=batch_tile_count,
                        dispatch_mode=ExecutionMode.GPU,
                        allow_direct_disjoint_pack=False,
                        use_same_row_fast_path=True,
                        group_size_max=group_size_max,
                        empty_output=build_empty_polygon_rows_device(batch_tile_count),
                    )
                    if amplification_metadata is not None:
                        topology_sums = {
                            "input_rows": int(live_count),
                            "pre_reduction_fragments": int(clipped_batch.row_count),
                            "observed_groups": int(d_observed_group_ids.size),
                        }
                        topology_unavailable = [
                            "input_segments",
                            "input_coordinates",
                            "output_parts",
                            "output_coordinates",
                        ]
                        if batch_result is None:
                            topology_unavailable.append("output_groups")
                        else:
                            topology_sums["output_groups"] = int(batch_result.row_count)
                        attach_work_amplification(
                            amplification_metadata,
                            operation="constructive.union.tile_topology",
                            metric_family="group_compression",
                            sums=topology_sums,
                            maxima={
                                "max_group_size": int(group_size_max),
                                "tiles_per_batch": int(batch_tile_count),
                            },
                            unavailable=tuple(topology_unavailable),
                        )
                _sync_hotpath()
        if batch_result is None or batch_result.row_count != batch_tile_count:
            raise RuntimeError("grouped topology batch violated tile-row capacity")
        batch_result = _physicalize_topology_coverage_output(batch_result)
        if assemble_union:
            if batch_tile_count == _TOPOLOGY_SEAM_FAN_IN:
                _append_topology_coverage_row(
                    coverage_levels,
                    _stitch_topology_coverage_rows(batch_result),
                    level_index=1,
                )
            else:
                for local_tile in range(batch_tile_count):
                    _append_topology_coverage_row(
                        coverage_levels,
                        batch_result._device_indexed_take(
                            cp.asarray([local_tile], dtype=cp.int64),
                            assume_unique_indices=True,
                        ),
                    )
        else:
            coverage_parts.append(batch_result)
        if batch_candidate_count > 0:
            del clipped_batch
    del tile_relation, host_offsets, topology_plan

    from vibespatial.geometry.owned import OwnedGeometryArray

    result = None
    if assemble_union:
        _sync_hotpath()
        with hotpath_stage(
            "constructive.union.tile_seam_stitch",
            category="assemble",
        ) as amplification_metadata:
            result = _finish_topology_coverage_levels(coverage_levels)
            if amplification_metadata is not None:
                attach_work_amplification(
                    amplification_metadata,
                    operation="constructive.union.tile_seam_stitch",
                    metric_family="group_compression",
                    sums={
                        "input_rows": int(tile_count),
                        "output_groups": int(result.row_count),
                    },
                    maxima={
                        "tile_count": int(tile_count),
                        "max_segment_peer_pressure": int(max_segment_peer_pressure),
                    },
                    unavailable=(
                        "max_group_size",
                        "input_segments",
                        "input_coordinates",
                        "pre_reduction_fragments",
                        "output_parts",
                        "output_coordinates",
                    ),
                )
        _sync_hotpath()
        coverage = result
    else:
        if not coverage_parts:
            raise RuntimeError("collective topology produced no coverage rows")
        coverage = OwnedGeometryArray.concat(coverage_parts)
    return result, coverage, tile_count, max_segment_peer_pressure


def single_group_polygon_collective_coverage_gpu(
    owned: OwnedGeometryArray,
    *,
    force_tile_count: int | None = None,
) -> OwnedGeometryArray | None:
    """Return exact interior-disjoint tile coverage for a large polygon union.

    The carrier is intended for downstream relation/grouped consumers. Its row
    union is the exact logical union, but shared tile seams remain until the
    consumer's grouped constructive reduction or final seam stitch.
    """
    if cp is None or owned.residency is not Residency.DEVICE:
        return None
    if owned.row_count <= 1 or not set(owned.families).issubset(
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    ):
        return None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if state.trusted_all_valid is not True or state.trusted_all_non_empty is not True:
        return None
    normalized = _normalize_collective_polygon_parts(owned)
    work = estimate_physical_work_from_owned(normalized)
    segment_capacity = max(int(work.segment_count), int(work.coordinate_count), 1)
    segment_peer_pressure = segment_capacity * max(int(normalized.row_count) - 1, 1)
    if (
        force_tile_count is None
        and segment_peer_pressure <= _DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE
    ):
        return None
    tile_count = (
        int(force_tile_count)
        if force_tile_count is not None
        else _topology_tile_count(segment_peer_pressure)
    )
    _result, coverage, resolved_tile_count, max_tile_pressure = (
        _tiled_single_group_collective_union_gpu(
            normalized,
            tile_count=tile_count,
            force_tile_count=force_tile_count is not None,
            assemble_union=False,
        )
    )
    record_dispatch_event(
        surface="vibespatial.constructive.collective_union",
        operation="union_coverage",
        implementation="gpu_single_group_tiled_collective_coverage",
        reason=(
            "large logical polygon union retained exact tile coverage for a "
            "downstream relation-grouped consumer"
        ),
        detail=(
            f"rows={normalized.row_count}; coverage_rows={coverage.row_count}; "
            f"tiles={resolved_tile_count}; "
            f"max_tile_segment_peer_pressure={max_tile_pressure}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return coverage


def single_group_polygon_collective_union_gpu(
    owned: OwnedGeometryArray,
    *,
    force_tile_count: int | None = None,
) -> OwnedGeometryArray:
    """Union one all-valid polygon group through bounded collective topology.

    The tiled variant clips source geometry into mutually interior-disjoint
    cells.  Local unions therefore never feed overlapping complex aggregate
    polygons into another binary overlay.  The final operation only nodes and
    removes shared tile seams from a proven coverage.
    """
    if cp is None or owned.residency is not Residency.DEVICE:
        raise RuntimeError("collective polygon union requires device residency")
    if owned.row_count <= 1:
        return owned
    if not set(owned.families).issubset(
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    ):
        raise ValueError("collective polygon union requires polygonal rows")
    state = owned._ensure_device_state(preserve_indexed_view=True)
    if state.trusted_all_valid is not True or state.trusted_all_non_empty is not True:
        raise ValueError("collective polygon union requires all-valid non-empty proof")
    owned = _normalize_collective_polygon_parts(owned)

    work = estimate_physical_work_from_owned(owned)
    segment_capacity = max(int(work.segment_count), int(work.coordinate_count), 1)
    segment_peer_pressure = segment_capacity * max(int(owned.row_count) - 1, 1)
    tile_count = (
        int(force_tile_count)
        if force_tile_count is not None
        else _topology_tile_count(segment_peer_pressure)
    )
    if tile_count <= 1:
        raise ValueError("collective topology tile count must be greater than one")

    if force_tile_count is None and (
        segment_peer_pressure <= _DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE
    ):
        result = _single_group_collective_union_gpu(owned)
        implementation = "gpu_single_group_collective_topology"
        detail = (
            f"rows={owned.row_count}; segment_capacity={segment_capacity}; "
            f"segment_peer_pressure={segment_peer_pressure}; tiles=1"
        )
    else:
        result, _coverage, tile_count, max_tile_pressure = (
            _tiled_single_group_collective_union_gpu(
                owned,
                tile_count=tile_count,
                force_tile_count=force_tile_count is not None,
            )
        )
        if result is None:
            raise RuntimeError("collective topology did not assemble its union row")
        implementation = "gpu_single_group_tiled_collective_topology"
        detail = (
            f"rows={owned.row_count}; segment_capacity={segment_capacity}; "
            f"segment_peer_pressure={segment_peer_pressure}; tiles={tile_count}; "
            f"max_tile_segment_peer_pressure={max_tile_pressure}; grid=device-selected"
        )

    record_dispatch_event(
        surface="vibespatial.constructive.collective_union",
        operation="union_all",
        implementation=implementation,
        reason=(
            "single logical polygon group reduced through bounded collective "
            "topology without recursive complex-union inputs"
        ),
        detail=detail,
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return result
