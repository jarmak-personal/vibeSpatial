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
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - GPU-only implementation
    cp = None

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.crossover import estimate_physical_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray


# Face propagation becomes nonlinear once a dense collective arrangement has
# enough source-segment peer pressure.  This estimate is deliberately based on
# physical segment capacity and fan-in, not a public row threshold.
_DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE = 64 * 1024 * 1024
_TARGET_TILE_SEGMENT_PEER_PRESSURE = 2 * 1024 * 1024
_MIN_TOPOLOGY_TILE_COUNT = 8
_MAX_TOPOLOGY_TILE_COUNT = 64
_TOPOLOGY_CLIP_PAGE_TILES = 8


def _sync_hotpath() -> None:
    if hotpath_trace_enabled():
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
    return min(
        _MAX_TOPOLOGY_TILE_COUNT,
        max(_MIN_TOPOLOGY_TILE_COUNT, int(math.ceil(math.sqrt(ratio)))),
    )


def _device_topology_tile_bounds(
    owned: OwnedGeometryArray,
    *,
    tile_count: int,
):
    """Build tile bounds and choose the lower-duplication axis on device."""
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )

    d_bounds = cp.asarray(
        compute_geometry_bounds_device(owned, preserve_indexed_view=True),
        dtype=cp.float64,
    ).reshape(owned.row_count, 4)
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
    d_use_y = d_y_pressure <= d_x_pressure

    d_axis_min = cp.where(d_use_y, d_global[1], d_global[0])
    d_axis_max = cp.where(d_use_y, d_global[3], d_global[2])
    d_edge_positions = cp.arange(tile_count + 1, dtype=cp.float64) / np.float64(
        tile_count
    )
    d_edges = d_axis_min + (d_axis_max - d_axis_min) * d_edge_positions
    d_tile_bounds = cp.empty((tile_count, 4), dtype=cp.float64)
    d_tile_bounds[:, 0] = cp.where(d_use_y, d_global[0], d_edges[:-1])
    d_tile_bounds[:, 1] = cp.where(d_use_y, d_edges[:-1], d_global[1])
    d_tile_bounds[:, 2] = cp.where(d_use_y, d_global[2], d_edges[1:])
    d_tile_bounds[:, 3] = cp.where(d_use_y, d_edges[1:], d_global[3])
    return d_tile_bounds, d_use_y


def _clip_topology_tile_page(
    owned: OwnedGeometryArray,
    d_tile_bounds,
) -> OwnedGeometryArray:
    """Clip a fixed tile page at source-row capacity without count export."""
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds

    source_row_count = int(owned.row_count)
    page_tile_count = int(d_tile_bounds.shape[0])
    d_source_rows = cp.tile(
        cp.arange(source_row_count, dtype=cp.int64),
        page_tile_count,
    )
    d_pair_bounds = cp.repeat(
        cp.asarray(d_tile_bounds, dtype=cp.float64),
        source_row_count,
        axis=0,
    )
    source_capacity = owned._device_indexed_take(
        d_source_rows,
        assume_unique_indices=False,
    )
    rectangle_capacity = _build_device_boxes_from_bounds(
        d_pair_bounds,
        row_count=page_tile_count * source_row_count,
    )
    clipped = _binary_constructive_gpu(
        "intersection",
        source_capacity,
        rectangle_capacity,
        dispatch_mode=ExecutionMode.GPU,
    )
    if clipped is None or clipped.row_count != page_tile_count * source_row_count:
        raise RuntimeError("topology tile clip violated source-row capacity")
    return clipped


def _tiled_single_group_collective_union_gpu(
    owned: OwnedGeometryArray,
    *,
    tile_count: int,
) -> OwnedGeometryArray:
    from vibespatial.constructive.binary_constructive import (
        _assemble_noded_polygon_coverage_split_events_gpu,
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.overlay.split import build_gpu_split_events

    d_tile_bounds, d_use_y = _device_topology_tile_bounds(
        owned,
        tile_count=tile_count,
    )
    source_row_count = int(owned.row_count)
    tile_results: list[OwnedGeometryArray] = []

    for page_start in range(0, tile_count, _TOPOLOGY_CLIP_PAGE_TILES):
        page_end = min(page_start + _TOPOLOGY_CLIP_PAGE_TILES, tile_count)
        _sync_hotpath()
        with hotpath_stage("constructive.union.tile_clip", category="setup"):
            clipped_page = _clip_topology_tile_page(
                owned,
                d_tile_bounds[page_start:page_end],
            )
        _sync_hotpath()

        for local_tile in range(page_end - page_start):
            row_start = local_tile * source_row_count
            d_tile_rows = cp.arange(
                row_start,
                row_start + source_row_count,
                dtype=cp.int64,
            )
            clipped_tile = clipped_page._device_indexed_take(
                d_tile_rows,
                assume_unique_indices=True,
            )
            _sync_hotpath()
            with hotpath_stage("constructive.union.tile_topology", category="refine"):
                tile_result = _regroup_native_grouped_parts_with_grouped_union_gpu(
                    clipped_tile,
                    cp.arange(source_row_count, dtype=cp.int64),
                    cp.asarray([0, source_row_count], dtype=cp.int64),
                    cp.asarray([0], dtype=cp.int64),
                    output_row_count=1,
                    dispatch_mode=ExecutionMode.GPU,
                    allow_direct_disjoint_pack=False,
                    use_same_row_fast_path=True,
                    group_size_max=source_row_count,
                )
            _sync_hotpath()
            if tile_result is None or tile_result.row_count != 1:
                raise RuntimeError("topology tile union did not produce one tile row")
            tile_results.append(tile_result)
        del clipped_page

    tiled = OwnedGeometryArray.concat(tile_results)
    from vibespatial.geometry.owned import build_empty_polygon_rows_device

    empty_left = build_empty_polygon_rows_device(1)
    _sync_hotpath()
    with hotpath_stage("constructive.union.tile_seam_stitch", category="assemble"):
        split_events = build_gpu_split_events(
            empty_left,
            tiled,
            dispatch_mode=ExecutionMode.GPU,
            require_same_row=True,
            use_same_row_fast_path=False,
            right_geometry_source_rows=cp.zeros(tile_count, dtype=cp.int32),
            include_same_side_splits=True,
        )
        result = _assemble_noded_polygon_coverage_split_events_gpu(
            split_events,
            output_row_count=1,
            d_valid_empty_rows=cp.ones(1, dtype=cp.bool_),
        )
    _sync_hotpath()
    if result is None or result.row_count != 1:
        raise RuntimeError("topology tile seam assembly did not produce one output row")
    return result


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
        result = _tiled_single_group_collective_union_gpu(
            owned,
            tile_count=tile_count,
        )
        implementation = "gpu_single_group_tiled_collective_topology"
        detail = (
            f"rows={owned.row_count}; segment_capacity={segment_capacity}; "
            f"segment_peer_pressure={segment_peer_pressure}; tiles={tile_count}; "
            "axis=device-selected"
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
