"""Collective lineal/polygonal constructive assembly on device."""

from __future__ import annotations

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.api._native_rowset import NativeDeviceSelection
from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event


def _capacity_points(d_x, d_y, d_active) -> OwnedGeometryArray:
    capacity = int(d_active.size)
    return build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=d_x,
                y=d_y,
                geometry_offsets=cp.arange(capacity + 1, dtype=cp.int32),
                empty_mask=~d_active,
                bounds=None,
            )
        },
        row_count=capacity,
        tags=cp.full(
            capacity,
            FAMILY_TAGS[GeometryFamily.POINT],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )


def _capacity_lines(d_src_x, d_src_y, d_dst_x, d_dst_y, d_active) -> OwnedGeometryArray:
    """Represent split intervals as exact two-coordinate device rows."""
    capacity = int(d_active.size)
    d_x = cp.empty(capacity * 2, dtype=cp.float64)
    d_y = cp.empty(capacity * 2, dtype=cp.float64)
    d_x[0::2] = d_src_x
    d_y[0::2] = d_src_y
    d_x[1::2] = d_dst_x
    d_y[1::2] = d_dst_y
    return build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_x,
                y=d_y,
                geometry_offsets=(cp.arange(capacity + 1, dtype=cp.int32) * cp.int32(2)),
                empty_mask=~d_active,
                bounds=None,
            )
        },
        row_count=capacity,
        tags=cp.full(
            capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=d_active,
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )


def _scatter_selected_line_family(
    *,
    d_edge_mask,
    d_rows,
    d_fragment_ids,
    d_src_x,
    d_src_y,
    d_dst_x,
    d_dst_y,
    row_count: int,
    family: GeometryFamily,
) -> DeviceFamilyGeometryBuffer:
    """Pack one line family from edge capacity without a logical-count read."""
    capacity = int(d_edge_mask.size)
    selection = NativeDeviceSelection.from_mask(d_edge_mask)
    d_active = selection.active_capacity_mask()
    d_selected_rows = selection.gather_capacity(d_rows, fill_value=0).astype(
        cp.int32,
        copy=False,
    )
    d_selected_fragments = selection.gather_capacity(
        d_fragment_ids,
        fill_value=0,
    ).astype(cp.int64, copy=False)
    d_selected_src_x = selection.gather_capacity(d_src_x, fill_value=0.0)
    d_selected_src_y = selection.gather_capacity(d_src_y, fill_value=0.0)
    d_selected_dst_x = selection.gather_capacity(d_dst_x, fill_value=0.0)
    d_selected_dst_y = selection.gather_capacity(d_dst_y, fill_value=0.0)

    d_new_part = d_active.copy()
    if capacity > 1:
        d_new_part[1:] &= ~d_active[:-1] | (d_selected_fragments[1:] != d_selected_fragments[:-1])
    d_part_ids = cp.cumsum(d_new_part, dtype=cp.int64) - 1
    d_coord_counts = cp.where(
        d_active,
        cp.where(d_new_part, cp.int32(2), cp.int32(1)),
        cp.int32(0),
    )
    d_coord_offsets = cp.zeros(capacity + 1, dtype=cp.int64)
    cp.cumsum(d_coord_counts, out=d_coord_offsets[1:])
    d_coord_ends = d_coord_offsets[1:] - 1
    d_safe_coord_ends = cp.where(d_active, d_coord_ends, cp.int64(0))
    d_safe_coord_starts = cp.where(
        d_new_part,
        d_coord_ends - 1,
        cp.int64(0),
    )

    coord_capacity = capacity * 2
    d_x = cp.zeros(coord_capacity, dtype=cp.float64)
    d_y = cp.zeros(coord_capacity, dtype=cp.float64)
    cp.add.at(
        d_x,
        d_safe_coord_ends,
        cp.where(d_active, d_selected_dst_x, cp.float64(0.0)),
    )
    cp.add.at(
        d_y,
        d_safe_coord_ends,
        cp.where(d_active, d_selected_dst_y, cp.float64(0.0)),
    )
    cp.add.at(
        d_x,
        d_safe_coord_starts,
        cp.where(d_new_part, d_selected_src_x, cp.float64(0.0)),
    )
    cp.add.at(
        d_y,
        d_safe_coord_starts,
        cp.where(d_new_part, d_selected_src_y, cp.float64(0.0)),
    )

    # CuPy scatter-add does not support signed int64 targets. Geometry spans
    # are int32-addressable throughout OwnedGeometryArray, so accumulate the
    # bounded per-row counts in int32 and widen only for prefix arithmetic.
    d_row_coord_counts = cp.zeros(row_count, dtype=cp.int32)
    cp.add.at(
        d_row_coord_counts,
        d_selected_rows,
        d_coord_counts,
    )
    d_row_part_counts = cp.zeros(row_count, dtype=cp.int32)
    cp.add.at(
        d_row_part_counts,
        d_selected_rows,
        d_new_part.astype(cp.int32, copy=False),
    )

    if family is GeometryFamily.LINESTRING:
        d_geometry_offsets = cp.zeros(row_count + 1, dtype=cp.int64)
        cp.cumsum(d_row_coord_counts, out=d_geometry_offsets[1:])
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x,
            y=d_y,
            geometry_offsets=d_geometry_offsets.astype(cp.int32, copy=False),
            empty_mask=d_row_part_counts != 1,
            bounds=None,
        )

    d_geometry_offsets = cp.zeros(row_count + 1, dtype=cp.int64)
    cp.cumsum(d_row_part_counts, out=d_geometry_offsets[1:])
    d_logical_coord_total = d_coord_offsets[-1].astype(cp.int32, copy=False)
    d_part_offsets = cp.zeros(capacity + 1, dtype=cp.int32)
    d_part_offsets += d_logical_coord_total
    d_safe_part_ids = cp.where(d_new_part, d_part_ids, cp.int64(0))
    cp.minimum.at(
        d_part_offsets,
        d_safe_part_ids,
        cp.where(
            d_new_part,
            d_safe_coord_starts.astype(cp.int32, copy=False),
            d_logical_coord_total,
        ),
    )
    return DeviceFamilyGeometryBuffer(
        family=family,
        x=d_x,
        y=d_y,
        geometry_offsets=d_geometry_offsets.astype(cp.int32, copy=False),
        empty_mask=d_row_part_counts <= 1,
        part_offsets=d_part_offsets,
        bounds=None,
    )


def _assemble_lineal_capacity(
    *,
    d_keep,
    d_break_before,
    d_rows,
    d_source_parts,
    d_src_x,
    d_src_y,
    d_dst_x,
    d_dst_y,
    d_output_validity,
    row_count: int,
) -> OwnedGeometryArray:
    selection = NativeDeviceSelection.from_mask(d_keep)
    capacity = selection.capacity
    d_active = selection.active_capacity_mask()
    d_rows = selection.gather_capacity(d_rows, fill_value=0).astype(
        cp.int32,
        copy=False,
    )
    d_source_parts = selection.gather_capacity(
        d_source_parts,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    d_src_x = selection.gather_capacity(d_src_x, fill_value=0.0)
    d_src_y = selection.gather_capacity(d_src_y, fill_value=0.0)
    d_dst_x = selection.gather_capacity(d_dst_x, fill_value=0.0)
    d_dst_y = selection.gather_capacity(d_dst_y, fill_value=0.0)
    d_break_before = selection.gather_capacity(
        d_break_before,
        fill_value=True,
    ).astype(cp.bool_, copy=False)

    d_continues = cp.zeros(capacity, dtype=cp.bool_)
    if capacity > 1:
        d_continues[1:] = (
            d_active[1:]
            & d_active[:-1]
            & (d_rows[1:] == d_rows[:-1])
            & (d_source_parts[1:] == d_source_parts[:-1])
            & (d_src_x[1:] == d_dst_x[:-1])
            & (d_src_y[1:] == d_dst_y[:-1])
            & ~d_break_before[1:]
        )
    d_new_fragment = d_active & ~d_continues
    d_fragment_ids = cp.cumsum(d_new_fragment, dtype=cp.int64) - 1

    # Segment extraction groups physical families before public rows. Restore
    # public-row order while preserving source order within each row so row
    # offsets address the coordinate prefix they describe.
    d_lane_ids = cp.arange(capacity, dtype=cp.uint64)
    d_sort_keys = (d_rows.astype(cp.uint64, copy=False) << cp.uint64(32)) | d_lane_ids
    d_sort_keys = cp.where(
        d_active,
        d_sort_keys,
        cp.uint64(0xFFFFFFFFFFFFFFFF),
    )
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values
    d_active = d_active[d_order]
    d_rows = d_rows[d_order]
    d_new_fragment = d_new_fragment[d_order]
    d_fragment_ids = d_fragment_ids[d_order]
    d_src_x = d_src_x[d_order]
    d_src_y = d_src_y[d_order]
    d_dst_x = d_dst_x[d_order]
    d_dst_y = d_dst_y[d_order]

    d_row_part_counts = cp.zeros(row_count, dtype=cp.int32)
    cp.add.at(
        d_row_part_counts,
        d_rows,
        d_new_fragment.astype(cp.int32, copy=False),
    )
    d_output_validity = cp.asarray(d_output_validity, dtype=cp.bool_)
    d_line_rows = d_output_validity & (d_row_part_counts <= 1)
    d_multi_rows = d_output_validity & (d_row_part_counts > 1)

    line_buffer = _scatter_selected_line_family(
        d_edge_mask=d_active & d_line_rows[d_rows],
        d_rows=d_rows,
        d_fragment_ids=d_fragment_ids,
        d_src_x=d_src_x,
        d_src_y=d_src_y,
        d_dst_x=d_dst_x,
        d_dst_y=d_dst_y,
        row_count=row_count,
        family=GeometryFamily.LINESTRING,
    )
    multi_buffer = _scatter_selected_line_family(
        d_edge_mask=d_active & d_multi_rows[d_rows],
        d_rows=d_rows,
        d_fragment_ids=d_fragment_ids,
        d_src_x=d_src_x,
        d_src_y=d_src_y,
        d_dst_x=d_dst_x,
        d_dst_y=d_dst_y,
        row_count=row_count,
        family=GeometryFamily.MULTILINESTRING,
    )
    d_validity = d_line_rows | d_multi_rows
    d_tags = cp.where(
        d_line_rows,
        cp.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]),
        cp.where(
            d_multi_rows,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING]),
            cp.int8(-1),
        ),
    )
    d_family_rows = cp.where(
        d_validity,
        cp.arange(row_count, dtype=cp.int32),
        cp.int32(-1),
    )
    return build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: line_buffer,
            GeometryFamily.MULTILINESTRING: multi_buffer,
        },
        row_count=row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )


def _lineal_split_constructive_topology_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    operation: str,
    right_shape: str,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    crs=None,
):
    """Classify split line intervals against aligned right-hand geometry."""
    if cp is None or left.row_count != right.row_count:
        return None
    if left.row_count == 0:
        return None
    if operation not in {"intersection", "difference"}:
        raise ValueError(f"unsupported lineal split operation: {operation}")

    from vibespatial.overlay.split import (
        _free_split_event_device_state,
        build_gpu_split_events,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    split_events = build_gpu_split_events(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=True,
    )
    try:
        device = split_events.device_state
        interval_capacity = max(int(split_events.count) - 1, 0)
        if interval_capacity > 1_073_741_823 or left.row_count > 2_147_483_647:
            return None

        left_state = left._ensure_device_state(preserve_indexed_view=True)
        right_state = right._ensure_device_state(preserve_indexed_view=True)
        d_output_validity = cp.asarray(left_state.validity, dtype=cp.bool_) & cp.asarray(
            right_state.validity,
            dtype=cp.bool_,
        )
        if interval_capacity == 0:
            empty_i32 = cp.empty(0, dtype=cp.int32)
            empty_f64 = cp.empty(0, dtype=cp.float64)
            result = _assemble_lineal_capacity(
                d_keep=cp.empty(0, dtype=cp.bool_),
                d_break_before=cp.empty(0, dtype=cp.bool_),
                d_rows=empty_i32,
                d_source_parts=empty_i32,
                d_src_x=empty_f64,
                d_src_y=empty_f64,
                d_dst_x=empty_f64,
                d_dst_y=empty_f64,
                d_output_validity=d_output_validity,
                row_count=left.row_count,
            )
        else:
            d_source_ids = cp.asarray(device.source_segment_ids, dtype=cp.int32)
            d_rows = cp.asarray(device.row_indices, dtype=cp.int32)[:-1]
            d_source_parts = cp.asarray(device.part_indices, dtype=cp.int32)[:-1]
            d_src_x = cp.asarray(device.x, dtype=cp.float64)[:-1]
            d_src_y = cp.asarray(device.y, dtype=cp.float64)[:-1]
            d_dst_x = cp.asarray(device.x, dtype=cp.float64)[1:]
            d_dst_y = cp.asarray(device.y, dtype=cp.float64)[1:]
            # Split-event packed keys are ordered by (source segment, t). This
            # makes adjacent lanes the exact atomic intervals of one source.
            d_interval_active = (d_source_ids[:-1] == d_source_ids[1:]) & (
                d_source_ids[:-1] < int(split_events.left_segment_count)
            )
            d_dx = d_dst_x - d_src_x
            d_dy = d_dst_y - d_src_y
            d_interval_active &= (d_dx != cp.float64(0.0)) | (d_dy != cp.float64(0.0))

            d_event_active = d_source_ids[:-1] < int(split_events.left_segment_count)
            interval_midpoints = _capacity_points(
                d_src_x + (d_dst_x - d_src_x) * cp.float64(0.5),
                d_src_y + (d_dst_y - d_src_y) * cp.float64(0.5),
                d_interval_active,
            )
            d_safe_rows = cp.where(d_interval_active, d_rows, cp.int32(0))
            d_event_rows = cp.where(d_event_active, d_rows, cp.int32(0))
            interval_right = right._device_indexed_take(
                d_safe_rows.astype(cp.int64, copy=False),
            )
            interval_covered = binary_predicate_expression(
                "covered_by",
                interval_midpoints,
                interval_right,
                dispatch_mode=ExecutionMode.GPU,
                operation=f"constructive.lineal_{right_shape}.interval_covered_by",
            )
            event_points = _capacity_points(d_src_x, d_src_y, d_event_active)
            event_right = right._device_indexed_take(
                d_event_rows.astype(cp.int64, copy=False),
            )
            event_covered = binary_predicate_expression(
                "covered_by",
                event_points,
                event_right,
                dispatch_mode=ExecutionMode.GPU,
                operation=f"constructive.lineal_{right_shape}.event_covered_by",
            )
            if interval_covered is None or event_covered is None:
                return None
            d_interval_covered = cp.asarray(
                interval_covered.values,
                dtype=cp.bool_,
            )
            d_event_covered = cp.asarray(event_covered.values, dtype=cp.bool_)
            if (
                int(d_interval_covered.size) != interval_capacity
                or int(d_event_covered.size) != interval_capacity
            ):
                return None
            keep_inside = operation == "intersection"
            result = _assemble_lineal_capacity(
                d_keep=(
                    d_interval_active & d_interval_covered
                    if keep_inside
                    else d_interval_active & ~d_interval_covered
                ),
                d_break_before=(
                    cp.zeros(interval_capacity, dtype=cp.bool_)
                    if keep_inside
                    else d_interval_active & d_event_covered
                ),
                d_rows=d_safe_rows,
                d_source_parts=d_source_parts,
                d_src_x=d_src_x,
                d_src_y=d_src_y,
                d_dst_x=d_dst_x,
                d_dst_y=d_dst_y,
                d_output_validity=d_output_validity,
                row_count=left.row_count,
            )

            if operation == "intersection":
                from vibespatial.api._native_results import (
                    _geometry_composition_from_owned_parts_at_capacity,
                )
                from vibespatial.api._native_rowset import NativeDeviceSelection
                from vibespatial.constructive.binary_constructive import (
                    PointPartCapacitySelection,
                )
                from vibespatial.constructive.grouped_mixed_union import (
                    unique_points_from_part_capacity_device,
                )
                from vibespatial.geometry.owned import device_mask_owned_capacity

                d_point_active = d_event_active & d_event_covered
                point_capacity = _capacity_points(d_src_x, d_src_y, d_point_active)
                line_capacity = result._device_indexed_take(
                    d_event_rows.astype(cp.int64, copy=False),
                )
                point_on_line = binary_predicate_expression(
                    "covered_by",
                    point_capacity,
                    line_capacity,
                    dispatch_mode=ExecutionMode.GPU,
                    operation="constructive.lineal_polygonal.point_covered_by_line",
                )
                if point_on_line is None:
                    return None
                d_point_active &= ~cp.asarray(point_on_line.values, dtype=cp.bool_)
                point_selection = NativeDeviceSelection.from_mask(d_point_active)
                point_parts = PointPartCapacitySelection(
                    geometry=point_capacity._device_indexed_take(
                        point_selection.safe_capacity_positions(),
                    ),
                    source_rows=point_selection.gather_capacity(
                        d_event_rows,
                        fill_value=0,
                    ).astype(cp.int32, copy=False),
                    selection=point_selection.as_capacity_prefix(),
                )
                unique_points = unique_points_from_part_capacity_device(
                    point_parts,
                    point_parts.source_rows,
                    output_row_count=left.row_count,
                )
                output_rows = cp.arange(left.row_count, dtype=cp.int64)
                capacity_parts = [(result, output_rows)]
                if unique_points is not None:
                    point_owned, d_point_keep = unique_points
                    capacity_parts.append(
                        (
                            device_mask_owned_capacity(point_owned, d_point_keep),
                            output_rows,
                        )
                    )
                result = _geometry_composition_from_owned_parts_at_capacity(
                    tuple(capacity_parts),
                    row_count=left.row_count,
                    crs=crs,
                )
                if result is None:
                    raise RuntimeError("lineal/polygonal intersection lost native composition")
    finally:
        _free_split_event_device_state(split_events)

    record_dispatch_event(
        surface="vibespatial.constructive.binary",
        operation=operation,
        implementation=f"lineal_{right_shape}_collective_split_topology_gpu",
        reason=(
            f"lineal/{right_shape} {operation} split every source segment against "
            "all aligned right-hand segments and reconstructed classified intervals"
        ),
        detail=(
            f"rows={left.row_count}; interval_capacity={interval_capacity}; "
            "workload_shape=segment_split_interval_predicate_capacity_assembly"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return result


def lineal_polygonal_constructive_topology_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    operation: str,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    crs=None,
):
    """Intersect or subtract aligned polygon topology from lineal rows."""
    return _lineal_split_constructive_topology_gpu(
        left,
        right,
        operation=operation,
        right_shape="polygonal",
        dispatch_mode=dispatch_mode,
        crs=crs,
    )


def lineal_lineal_difference_topology_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Subtract aligned lineal rows through split-interval classification."""
    return _lineal_split_constructive_topology_gpu(
        left,
        right,
        operation="difference",
        right_shape="lineal",
        dispatch_mode=dispatch_mode,
    )


def lineal_polygonal_difference_topology_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
) -> OwnedGeometryArray | None:
    """Compatibility entry point for the collective difference carrier."""
    return lineal_polygonal_constructive_topology_gpu(
        left,
        right,
        operation="difference",
        dispatch_mode=dispatch_mode,
    )


__all__ = [
    "lineal_lineal_difference_topology_gpu",
    "lineal_polygonal_constructive_topology_gpu",
    "lineal_polygonal_difference_topology_gpu",
]
