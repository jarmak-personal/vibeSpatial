"""Device-native grouped union for mixed constructive remnants.

Physical shape contract (ADR-0046): relation-capacity constructive rows are
reduced by source group. Polygon area remains a segmented grouped constructive
reduction; linework is an atomic-edge dynamic-output assembly; point cleanup is
a warp-per-point, source-segment reduction. Inputs and outputs stay owned/native
until the public clip export boundary. Constructive coordinates remain fp64.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    _device_gather_xy_offset_slices,
    build_device_resident_owned,
    device_mask_owned_capacity,
)
from vibespatial.kernels.constructive.grouped_mixed_union import (
    _GROUPED_MIXED_UNION_KERNEL_NAMES,
    _GROUPED_MIXED_UNION_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - GPU-only module
    cp = None


request_nvrtc_warmup(
    [
        (
            "grouped-mixed-union-fp64",
            _GROUPED_MIXED_UNION_KERNEL_SOURCE,
            _GROUPED_MIXED_UNION_KERNEL_NAMES,
        )
    ]
)


def _compile_grouped_mixed_union_kernels():
    return compile_kernel_group(
        "grouped-mixed-union-fp64",
        _GROUPED_MIXED_UNION_KERNEL_SOURCE,
        _GROUPED_MIXED_UNION_KERNEL_NAMES,
    )


@dataclass(frozen=True)
class GroupedMixedUnionCapacity:
    """Fixed source-row capacity plus a device logical-output mask."""

    geometry: object
    keep_mask: DeviceArray


@dataclass(frozen=True)
class _GroupedAtomicLineCapacity:
    owned: OwnedGeometryArray
    keep_mask: DeviceArray
    edge_group_offsets: DeviceArray
    edge_x0: DeviceArray
    edge_y0: DeviceArray
    edge_x1: DeviceArray
    edge_y1: DeviceArray


def sorted_part_capacity_plan(part_selection, output_rows, output_row_count: int):
    """Sort selected part capacity by output row and derive group offsets."""
    if cp is None:  # pragma: no cover - GPU-only caller
        raise RuntimeError("CuPy is required for grouped part assembly")
    capacity = int(part_selection.capacity)
    d_output_rows = cp.asarray(output_rows, dtype=cp.int64)
    if int(d_output_rows.size) != capacity:
        raise ValueError("part output rows must match part capacity")
    d_active = part_selection.selection.active_capacity_mask()
    d_safe_output_rows = cp.where(
        d_active,
        d_output_rows,
        cp.int64(output_row_count),
    )
    d_slots = cp.arange(capacity, dtype=cp.int64)
    d_order = cp.argsort(
        d_safe_output_rows * cp.int64(max(capacity, 1) + 1) + d_slots,
    ).astype(cp.int64, copy=False)
    d_sorted_active = d_active[d_order]
    sorted_geometry = part_selection.geometry._device_indexed_take(
        d_order,
    )._apply_row_activity(d_sorted_active)
    d_counts = cp.bincount(
        d_safe_output_rows,
        weights=d_active.astype(cp.int32, copy=False),
        minlength=output_row_count + 1,
    )[:output_row_count].astype(cp.int32, copy=False)
    d_group_offsets = cp.empty(output_row_count + 1, dtype=cp.int32)
    d_group_offsets[0] = 0
    cp.cumsum(d_counts, out=d_group_offsets[1:])
    return sorted_geometry, d_sorted_active, d_counts, d_group_offsets


def pack_line_part_capacity_device(
    part_selection,
    output_rows,
    *,
    output_row_count: int,
) -> OwnedGeometryArray | None:
    """Pack LineString part capacity into row-aligned lineal capacity."""
    (
        sorted_geometry,
        d_sorted_active,
        d_counts,
        d_group_offsets,
    ) = sorted_part_capacity_plan(
        part_selection,
        output_rows,
        output_row_count,
    )
    state = sorted_geometry._ensure_device_state(preserve_indexed_view=True)
    line_buffer = state.families.get(GeometryFamily.LINESTRING)
    if line_buffer is None:
        return None
    d_family_rows = cp.where(
        d_sorted_active,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_line_x, d_line_y, d_line_offsets = _device_gather_xy_offset_slices(
        cp.asarray(line_buffer.x, dtype=cp.float64),
        cp.asarray(line_buffer.y, dtype=cp.float64),
        cp.asarray(line_buffer.geometry_offsets, dtype=cp.int64),
        d_family_rows,
        allocation_capacity=int(part_selection.coord_capacity),
        active_row_count=d_group_offsets[-1:],
    )
    gathered_line_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=d_line_x,
        y=d_line_y,
        geometry_offsets=d_line_offsets,
        empty_mask=~d_sorted_active,
        bounds=None,
    )
    d_single = d_counts == 1
    d_multi = d_counts > 1
    d_validity = d_single | d_multi
    d_rows = cp.arange(output_row_count, dtype=cp.int32)
    d_tags = cp.where(
        d_single,
        cp.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]),
        cp.where(
            d_multi,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING]),
            cp.int8(-1),
        ),
    )
    d_family_rows = cp.where(
        d_single,
        d_group_offsets[:-1],
        cp.where(d_multi, d_rows, cp.int32(-1)),
    ).astype(cp.int32, copy=False)
    multiline_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=d_line_x,
        y=d_line_y,
        geometry_offsets=d_group_offsets,
        empty_mask=~d_multi,
        part_offsets=d_line_offsets,
        bounds=None,
    )
    return build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: gathered_line_buffer,
            GeometryFamily.MULTILINESTRING: multiline_buffer,
        },
        row_count=output_row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )


def pack_point_part_capacity_device(
    part_selection,
    output_rows,
    *,
    output_row_count: int,
) -> OwnedGeometryArray | None:
    """Pack Point part capacity into row-aligned pointlike capacity."""
    (
        sorted_geometry,
        d_sorted_active,
        d_counts,
        d_group_offsets,
    ) = sorted_part_capacity_plan(
        part_selection,
        output_rows,
        output_row_count,
    )
    state = sorted_geometry._ensure_device_state(preserve_indexed_view=True)
    point_buffer = state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        return None
    d_family_rows = cp.where(
        d_sorted_active,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_point_offsets = cp.asarray(point_buffer.geometry_offsets, dtype=cp.int64)
    d_coord_rows = d_point_offsets[d_family_rows]
    d_point_x = cp.asarray(point_buffer.x, dtype=cp.float64)[d_coord_rows].copy()
    d_point_y = cp.asarray(point_buffer.y, dtype=cp.float64)[d_coord_rows].copy()
    gathered_point_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=d_point_x,
        y=d_point_y,
        geometry_offsets=cp.arange(int(part_selection.capacity) + 1, dtype=cp.int32),
        empty_mask=~d_sorted_active,
        bounds=None,
    )
    d_single = d_counts == 1
    d_multi = d_counts > 1
    d_validity = d_single | d_multi
    d_rows = cp.arange(output_row_count, dtype=cp.int32)
    d_tags = cp.where(
        d_single,
        cp.int8(FAMILY_TAGS[GeometryFamily.POINT]),
        cp.where(
            d_multi,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT]),
            cp.int8(-1),
        ),
    )
    d_family_rows = cp.where(
        d_single,
        d_group_offsets[:-1],
        cp.where(d_multi, d_rows, cp.int32(-1)),
    ).astype(cp.int32, copy=False)
    multipoint_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=d_point_x,
        y=d_point_y,
        geometry_offsets=d_group_offsets,
        empty_mask=~d_multi,
        bounds=None,
    )
    return build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: gathered_point_buffer,
            GeometryFamily.MULTIPOINT: multipoint_buffer,
        },
        row_count=output_row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )


def _valid_nonempty_mask(owned: OwnedGeometryArray):
    from vibespatial.geometry.owned import device_valid_nonempty_mask

    return cp.asarray(device_valid_nonempty_mask(owned), dtype=cp.bool_)


def polygon_pair_boundary_capacity_parts_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    pair_active=None,
):
    """Return aligned line/point boundary intersections for polygon pairs.

    Physical shape: polygon boundary segments -> aligned line/polygon
    constructive capacity -> concrete native composition parts. The returned
    output-row vectors map each part row back to its original pair-capacity row.
    """
    if cp is None or left.row_count != right.row_count:
        return None
    from vibespatial.constructive.boundary import boundary_owned
    from vibespatial.geometry.owned import device_mask_owned_capacity
    from vibespatial.kernels.constructive.nonpolygon_binary import (
        linestring_polygon_intersection,
    )

    if pair_active is not None:
        d_pair_active = cp.asarray(pair_active, dtype=cp.bool_)
        if int(d_pair_active.size) != int(left.row_count):
            raise ValueError("polygon boundary pair activity must match pair capacity")
        left = device_mask_owned_capacity(left, d_pair_active)
        right = device_mask_owned_capacity(right, d_pair_active)
    boundary = boundary_owned(left, dispatch_mode=ExecutionMode.GPU)
    native = linestring_polygon_intersection(boundary, right)
    if native is None:
        return None
    if native.owned is not None:
        if native.owned.row_count != left.row_count:
            return None
        return ((native.owned, cp.arange(left.row_count, dtype=cp.int64)),)
    if native.composition is None:
        return None

    parts = []
    for part in native.composition.parts:
        owned = part.geometry.owned
        if owned is None:
            return None
        d_output_rows = cp.asarray(part.output_rows, dtype=cp.int64)
        if d_output_rows.shape[0] != owned.row_count:
            return None
        parts.append((owned, d_output_rows))
    return tuple(parts)


def _concat_line_part_capacities(parts):
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import LinePartCapacitySelection

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return (
        LinePartCapacitySelection(
            geometry=OwnedGeometryArray.concat([part.geometry for part, _rows in parts]),
            source_rows=cp.concatenate(
                [cp.asarray(part.source_rows, dtype=cp.int32) for part, _rows in parts]
            ),
            selection=NativeDeviceSelection.from_mask(
                cp.concatenate([part.selection.active_capacity_mask() for part, _rows in parts])
            ),
            coord_capacity=sum(int(part.coord_capacity) for part, _rows in parts),
        ),
        cp.concatenate([cp.asarray(rows, dtype=cp.int64) for _part, rows in parts]),
    )


def _concat_point_part_capacities(parts):
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import PointPartCapacitySelection

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return (
        PointPartCapacitySelection(
            geometry=OwnedGeometryArray.concat([part.geometry for part, _rows in parts]),
            source_rows=cp.concatenate(
                [cp.asarray(part.source_rows, dtype=cp.int32) for part, _rows in parts]
            ),
            selection=NativeDeviceSelection.from_mask(
                cp.concatenate([part.selection.active_capacity_mask() for part, _rows in parts])
            ),
        ),
        cp.concatenate([cp.asarray(rows, dtype=cp.int64) for _part, rows in parts]),
    )


def _line_parts_outside_area(
    pair_parts,
    group_rows,
    pair_active,
    polygon_capacity: OwnedGeometryArray,
):
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        LinePartCapacitySelection,
        _dispatch_lineal_polygonal_difference_gpu,
        _explode_lineal_rows_to_line_capacity_gpu,
    )

    d_pair_active = cp.asarray(pair_active, dtype=cp.bool_)
    d_group_rows = cp.asarray(group_rows, dtype=cp.int64)
    candidates = []
    for pair_geometry, pair_output_rows in pair_parts:
        line_parts = _explode_lineal_rows_to_line_capacity_gpu(pair_geometry)
        if line_parts is None:
            continue
        d_part_rows = cp.asarray(line_parts.source_rows, dtype=cp.int64)
        d_pair_rows = cp.asarray(pair_output_rows, dtype=cp.int64)[d_part_rows]
        d_active = line_parts.selection.active_capacity_mask() & d_pair_active[d_pair_rows]
        d_output_rows = cp.where(d_active, d_group_rows[d_pair_rows], cp.int64(0))
        candidates.append(
            (
                LinePartCapacitySelection(
                    geometry=line_parts.geometry,
                    source_rows=d_output_rows.astype(cp.int32, copy=False),
                    selection=NativeDeviceSelection.from_mask(d_active),
                    coord_capacity=line_parts.coord_capacity,
                ),
                d_output_rows,
            )
        )
    concatenated = _concat_line_part_capacities(candidates)
    if concatenated is None:
        return None
    candidate_parts, d_candidate_output_rows = concatenated
    grouped_lines = pack_line_part_capacity_device(
        candidate_parts,
        d_candidate_output_rows,
        output_row_count=int(polygon_capacity.row_count),
    )
    if grouped_lines is None:
        return None
    difference = _dispatch_lineal_polygonal_difference_gpu(
        grouped_lines,
        polygon_capacity,
        dispatch_mode=ExecutionMode.GPU,
    )
    if difference is None or int(difference.row_count) != int(polygon_capacity.row_count):
        raise RuntimeError("grouped line remnants declined row-aligned area subtraction")
    remainder_parts = _explode_lineal_rows_to_line_capacity_gpu(difference)
    if remainder_parts is None:
        return None
    d_remainder_output_rows = cp.asarray(remainder_parts.source_rows, dtype=cp.int64)
    return (
        LinePartCapacitySelection(
            geometry=remainder_parts.geometry,
            source_rows=d_remainder_output_rows.astype(cp.int32, copy=False),
            selection=NativeDeviceSelection.from_mask(
                remainder_parts.selection.active_capacity_mask()
            ),
            coord_capacity=remainder_parts.coord_capacity,
        ),
        d_remainder_output_rows,
    )


def _grouped_atomic_line_union(
    pair_parts,
    group_rows,
    pair_active,
    polygon_capacity: OwnedGeometryArray,
    *,
    output_row_count: int,
) -> _GroupedAtomicLineCapacity | None:
    line_parts = _line_parts_outside_area(
        pair_parts,
        group_rows,
        pair_active,
        polygon_capacity,
    )
    if line_parts is None:
        return None
    part_selection, d_output_rows = line_parts
    return atomic_line_union_from_part_capacity_device(
        part_selection,
        d_output_rows,
        output_row_count=output_row_count,
    )


def atomic_line_union_from_part_capacity_device(
    part_selection,
    output_rows,
    *,
    output_row_count: int,
    preserve_source_orientation: bool = False,
) -> _GroupedAtomicLineCapacity | None:
    """Node and deduplicate line-part capacity within logical output rows.

    ``preserve_source_orientation`` keeps the traversal of the deterministic
    source representative for directional operations such as shared paths.
    Ordinary set union remains canonically oriented.
    """
    grouped_lines = pack_line_part_capacity_device(
        part_selection,
        output_rows,
        output_row_count=output_row_count,
    )
    if grouped_lines is None:
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import LinePartCapacitySelection
    from vibespatial.overlay.split import build_gpu_atomic_edges, build_gpu_split_events

    split_events = build_gpu_split_events(
        grouped_lines,
        grouped_lines,
        dispatch_mode=ExecutionMode.GPU,
        require_same_row=True,
        use_same_row_fast_path=True,
        include_same_side_splits=False,
    )
    atomic_edges = build_gpu_atomic_edges(
        split_events,
        isolate_rows=True,
        preserve_source_orientation=preserve_source_orientation,
    )
    atomic = atomic_edges.device_state
    if atomic is None or atomic_edges.count == 0:
        return None

    edge_capacity = atomic_edges.count // 2
    d_forward = cp.arange(edge_capacity, dtype=cp.int64) * cp.int64(2)
    d_raw_rows = cp.asarray(atomic.row_indices, dtype=cp.int64)[d_forward]
    d_order_key = d_raw_rows * cp.int64(max(edge_capacity, 1) + 1) + cp.arange(
        edge_capacity,
        dtype=cp.int64,
    )
    d_order = cp.argsort(d_order_key).astype(cp.int64, copy=False)
    d_edge_rows = d_raw_rows[d_order].astype(cp.int32, copy=False)
    d_edge_x0 = cp.asarray(atomic.src_x, dtype=cp.float64)[d_forward][d_order].copy()
    d_edge_y0 = cp.asarray(atomic.src_y, dtype=cp.float64)[d_forward][d_order].copy()
    d_edge_x1 = cp.asarray(atomic.dst_x, dtype=cp.float64)[d_forward][d_order].copy()
    d_edge_y1 = cp.asarray(atomic.dst_y, dtype=cp.float64)[d_forward][d_order].copy()
    d_edge_counts = cp.bincount(
        d_edge_rows,
        minlength=output_row_count,
    ).astype(cp.int64, copy=False)
    d_edge_offsets = cp.empty(output_row_count + 1, dtype=cp.int64)
    d_edge_offsets[0] = 0
    cp.cumsum(d_edge_counts, out=d_edge_offsets[1:])

    d_line_x = cp.empty(edge_capacity * 2, dtype=cp.float64)
    d_line_y = cp.empty(edge_capacity * 2, dtype=cp.float64)
    d_line_x[0::2] = d_edge_x0
    d_line_y[0::2] = d_edge_y0
    d_line_x[1::2] = d_edge_x1
    d_line_y[1::2] = d_edge_y1
    line_geometry = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_line_x,
                y=d_line_y,
                geometry_offsets=(cp.arange(edge_capacity + 1, dtype=cp.int32) * cp.int32(2)),
                empty_mask=cp.zeros(edge_capacity, dtype=cp.bool_),
                bounds=None,
            )
        },
        row_count=edge_capacity,
        tags=cp.full(
            edge_capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=cp.ones(edge_capacity, dtype=cp.bool_),
        family_row_offsets=cp.arange(edge_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    edge_selection = NativeDeviceSelection.from_mask(
        cp.ones(edge_capacity, dtype=cp.bool_),
    )
    noded_parts = LinePartCapacitySelection(
        geometry=line_geometry,
        source_rows=d_edge_rows,
        selection=edge_selection,
        coord_capacity=edge_capacity * 2,
    )
    noded_owned = pack_line_part_capacity_device(
        noded_parts,
        d_edge_rows,
        output_row_count=output_row_count,
    )
    if noded_owned is None:
        raise RuntimeError("grouped atomic line assembly lost line capacity")
    d_keep = _valid_nonempty_mask(noded_owned)
    return _GroupedAtomicLineCapacity(
        owned=device_mask_owned_capacity(noded_owned, d_keep),
        keep_mask=d_keep,
        edge_group_offsets=d_edge_offsets,
        edge_x0=d_edge_x0,
        edge_y0=d_edge_y0,
        edge_x1=d_edge_x1,
        edge_y1=d_edge_y1,
    )


def _points_covered_by_atomic_lines(point_parts, lines: _GroupedAtomicLineCapacity):
    capacity = int(point_parts.capacity)
    if capacity == 0:
        return cp.empty(0, dtype=cp.bool_)
    state = point_parts.geometry._ensure_device_state(preserve_indexed_view=True)
    point_buffer = state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        raise RuntimeError("grouped point cleanup lost Point storage")
    d_active = point_parts.selection.active_capacity_mask()
    d_family_rows = cp.where(
        d_active,
        cp.asarray(state.family_row_offsets, dtype=cp.int64),
        cp.int64(0),
    )
    d_offsets = cp.asarray(point_buffer.geometry_offsets, dtype=cp.int64)
    d_coord_rows = d_offsets[d_family_rows]
    d_point_x = cp.asarray(point_buffer.x, dtype=cp.float64)[d_coord_rows]
    d_point_y = cp.asarray(point_buffer.y, dtype=cp.float64)[d_coord_rows]
    d_groups = cp.asarray(point_parts.source_rows, dtype=cp.int32)
    d_covered = cp.zeros(capacity, dtype=cp.uint8)
    runtime = get_cuda_runtime()
    kernel = _compile_grouped_mixed_union_kernels()["grouped_points_on_atomic_edges"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, capacity * 32)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_point_x),
                ptr(d_point_y),
                ptr(d_groups),
                ptr(d_active),
                ptr(lines.edge_group_offsets),
                ptr(lines.edge_x0),
                ptr(lines.edge_y0),
                ptr(lines.edge_x1),
                ptr(lines.edge_y1),
                ptr(d_covered),
                capacity,
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
    )
    return d_covered.astype(cp.bool_, copy=False)


def unique_points_from_part_capacity_device(
    part_selection,
    output_rows,
    *,
    output_row_count: int,
    covering_lines: _GroupedAtomicLineCapacity | None = None,
):
    """Deduplicate point-part capacity and suppress points covered by lines."""
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        PointPartCapacitySelection,
        _explode_point_rows_to_point_capacity_gpu,
    )
    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_owned,
    )

    grouped_points = pack_point_part_capacity_device(
        part_selection,
        output_rows,
        output_row_count=output_row_count,
    )
    if grouped_points is None:
        return None
    unique_points = extract_unique_points_owned(
        grouped_points,
        dispatch_mode=ExecutionMode.GPU,
    )
    unique_parts = _explode_point_rows_to_point_capacity_gpu(unique_points)
    if unique_parts is None:
        return None
    d_unique_active = unique_parts.selection.active_capacity_mask()
    if covering_lines is not None:
        d_unique_active &= ~_points_covered_by_atomic_lines(
            unique_parts,
            covering_lines,
        )
    filtered_selection = NativeDeviceSelection.from_mask(d_unique_active)
    d_filtered_active = filtered_selection.active_capacity_mask()
    filtered_parts = PointPartCapacitySelection(
        geometry=unique_parts.geometry._device_indexed_take(
            filtered_selection.partition_capacity_positions(),
            assume_unique_indices=True,
        )._apply_row_activity(d_filtered_active),
        source_rows=filtered_selection.gather_capacity(
            unique_parts.source_rows,
            fill_value=0,
        ).astype(cp.int32, copy=False),
        selection=filtered_selection.as_capacity_prefix(),
    )
    filtered_points = pack_point_part_capacity_device(
        filtered_parts,
        filtered_parts.source_rows,
        output_row_count=output_row_count,
    )
    if filtered_points is None:
        return None
    d_keep = _valid_nonempty_mask(filtered_points)
    return device_mask_owned_capacity(filtered_points, d_keep), d_keep


def _grouped_point_union(
    pair_parts,
    group_rows,
    pair_active,
    polygon_capacity: OwnedGeometryArray,
    lines: _GroupedAtomicLineCapacity | None,
    *,
    output_row_count: int,
) -> tuple[OwnedGeometryArray, DeviceArray] | None:
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        PointPartCapacitySelection,
        _difference_point_polygon_gpu,
        _explode_point_rows_to_point_capacity_gpu,
    )

    d_pair_active = cp.asarray(pair_active, dtype=cp.bool_)
    d_group_rows = cp.asarray(group_rows, dtype=cp.int64)
    outside_capacities = []
    for pair_geometry, pair_output_rows in pair_parts:
        point_parts = _explode_point_rows_to_point_capacity_gpu(pair_geometry)
        if point_parts is None:
            continue
        d_part_rows = cp.asarray(point_parts.source_rows, dtype=cp.int64)
        d_pair_rows = cp.asarray(pair_output_rows, dtype=cp.int64)[d_part_rows]
        d_active = point_parts.selection.active_capacity_mask() & d_pair_active[d_pair_rows]
        d_output_rows = cp.where(d_active, d_group_rows[d_pair_rows], cp.int64(0))
        point_capacity = device_mask_owned_capacity(point_parts.geometry, d_active)
        area_capacity = polygon_capacity.device_take_capacity(d_output_rows, d_active)
        outside_area = _difference_point_polygon_gpu(point_capacity, area_capacity)
        if int(outside_area.row_count) != point_parts.capacity:
            raise RuntimeError("grouped point remnants lost pair capacity")
        d_outside_area = _valid_nonempty_mask(outside_area)
        outside_capacities.append(
            (
                PointPartCapacitySelection(
                    geometry=outside_area,
                    source_rows=d_output_rows.astype(cp.int32, copy=False),
                    selection=NativeDeviceSelection.from_mask(d_outside_area),
                ),
                d_output_rows,
            )
        )
    concatenated = _concat_point_part_capacities(outside_capacities)
    if concatenated is None:
        return None
    outside_parts, d_output_rows = concatenated
    return unique_points_from_part_capacity_device(
        outside_parts,
        d_output_rows,
        output_row_count=output_row_count,
        covering_lines=lines,
    )


def grouped_mixed_union_capacity_device(
    pair_intersections: OwnedGeometryArray,
    group_rows,
    pair_active,
    polygon_capacity: OwnedGeometryArray,
    polygon_keep,
    *,
    output_row_count: int,
    crs,
    pair_boundary_parts=(),
) -> GroupedMixedUnionCapacity:
    """Reduce mixed pair intersections to exact source-row native capacity."""
    if cp is None:  # pragma: no cover - GPU-only caller
        raise RuntimeError("CuPy is required for grouped mixed union")
    from vibespatial.api._native_results import (
        GeometryNativeResult,
        _geometry_composition_from_owned_parts_at_capacity,
    )

    d_pair_active = cp.asarray(pair_active, dtype=cp.bool_)
    d_polygon_keep = cp.asarray(polygon_keep, dtype=cp.bool_)
    if int(d_pair_active.size) != int(pair_intersections.row_count):
        raise ValueError("grouped mixed pair activity must match pair capacity")
    if int(d_polygon_keep.size) != output_row_count:
        raise ValueError("grouped polygon keep mask must match source capacity")

    polygon_part = device_mask_owned_capacity(polygon_capacity, d_polygon_keep)
    pair_parts = (
        (pair_intersections, cp.arange(pair_intersections.row_count, dtype=cp.int64)),
        *tuple(pair_boundary_parts),
    )
    lines = _grouped_atomic_line_union(
        pair_parts,
        group_rows,
        d_pair_active,
        polygon_capacity,
        output_row_count=output_row_count,
    )
    points = _grouped_point_union(
        pair_parts,
        group_rows,
        d_pair_active,
        polygon_capacity,
        lines,
        output_row_count=output_row_count,
    )

    d_keep = d_polygon_keep.copy()
    capacity_parts = [(polygon_part, cp.arange(output_row_count, dtype=cp.int64))]
    if lines is not None:
        d_keep |= cp.asarray(lines.keep_mask, dtype=cp.bool_)
        capacity_parts.append((lines.owned, cp.arange(output_row_count, dtype=cp.int64)))
    if points is not None:
        point_owned, d_point_keep = points
        d_keep |= cp.asarray(d_point_keep, dtype=cp.bool_)
        capacity_parts.append((point_owned, cp.arange(output_row_count, dtype=cp.int64)))

    if len(capacity_parts) == 1:
        geometry = GeometryNativeResult.from_owned(polygon_part, crs=crs)
    else:
        geometry = _geometry_composition_from_owned_parts_at_capacity(
            tuple(capacity_parts),
            row_count=output_row_count,
            crs=crs,
        )
        if geometry is None:
            raise RuntimeError("grouped mixed union failed native composition")
    return GroupedMixedUnionCapacity(geometry=geometry, keep_mask=d_keep)


__all__ = [
    "GroupedMixedUnionCapacity",
    "atomic_line_union_from_part_capacity_device",
    "grouped_mixed_union_capacity_device",
    "pack_line_part_capacity_device",
    "pack_point_part_capacity_device",
    "polygon_pair_boundary_capacity_parts_device",
    "sorted_part_capacity_plan",
    "unique_points_from_part_capacity_device",
]
