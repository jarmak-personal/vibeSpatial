from __future__ import annotations

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    NULL_TAG,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency, TransferTrigger


def point_membership_rows_to_owned(
    points: OwnedGeometryArray,
    *,
    output_validity,
    keep_rows,
) -> OwnedGeometryArray:
    """Assemble aligned Point results, preserving valid empty rows."""
    import cupy as cp

    state = points._ensure_device_state(preserve_indexed_view=True)
    source_buffer = state.families[GeometryFamily.POINT]
    d_validity = cp.asarray(output_validity, dtype=cp.bool_)
    d_keep = cp.asarray(keep_rows, dtype=cp.bool_) & d_validity
    d_point_rows = cp.flatnonzero(d_validity).astype(cp.int64, copy=False)
    d_kept_rows = cp.flatnonzero(d_keep).astype(cp.int64, copy=False)
    d_tags = cp.full(points.row_count, NULL_TAG, dtype=cp.int8)
    d_family_rows = cp.full(points.row_count, -1, dtype=cp.int32)
    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    if int(d_point_rows.size):
        d_counts = d_keep[d_point_rows].astype(cp.int32, copy=False)
        d_offsets = cp.empty(int(d_point_rows.size) + 1, dtype=cp.int32)
        d_offsets[0] = 0
        cp.cumsum(d_counts, out=d_offsets[1:])
        d_source_family_rows = state.family_row_offsets[d_kept_rows].astype(
            cp.int64,
            copy=False,
        )
        d_source_coord_rows = source_buffer.geometry_offsets[d_source_family_rows].astype(
            cp.int64, copy=False
        )
        device_families[GeometryFamily.POINT] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=source_buffer.x[d_source_coord_rows].copy(),
            y=source_buffer.y[d_source_coord_rows].copy(),
            geometry_offsets=d_offsets,
            empty_mask=d_counts == 0,
            bounds=None,
        )
        d_tags[d_point_rows] = FAMILY_TAGS[GeometryFamily.POINT]
        d_family_rows[d_point_rows] = cp.arange(
            int(d_point_rows.size),
            dtype=cp.int32,
        )

    return build_device_resident_owned(
        device_families=device_families,
        row_count=points.row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )


def _deduplicate_selected_points(point_x, point_y, source_rows):
    """Return exact unique `(source row, x, y)` point tuples on device."""
    import cupy as cp

    d_source_rows = cp.asarray(source_rows, dtype=cp.int64)
    if int(d_source_rows.size) <= 1:
        return point_x, point_y, d_source_rows
    d_order = cp.lexsort(cp.stack((point_y, point_x, d_source_rows)))
    d_sorted_source = d_source_rows[d_order]
    d_sorted_x = point_x[d_order]
    d_sorted_y = point_y[d_order]
    d_unique = cp.empty(int(d_order.size), dtype=cp.bool_)
    d_unique[0] = True
    d_unique[1:] = (
        (d_sorted_source[1:] != d_sorted_source[:-1])
        | (d_sorted_x[1:] != d_sorted_x[:-1])
        | (d_sorted_y[1:] != d_sorted_y[:-1])
    )
    return (
        d_sorted_x[d_unique],
        d_sorted_y[d_unique],
        d_sorted_source[d_unique],
    )


def _selected_point_rows_to_owned(
    *,
    point_x,
    point_y,
    source_rows,
    output_validity,
    row_count: int,
) -> OwnedGeometryArray:
    """Pack selected point rows into aligned Point/MultiPoint device buffers."""
    import cupy as cp

    point_x, point_y, d_source_rows = _deduplicate_selected_points(
        cp.asarray(point_x, dtype=cp.float64),
        cp.asarray(point_y, dtype=cp.float64),
        source_rows,
    )
    d_counts = cp.zeros(row_count, dtype=cp.int32)
    if int(d_source_rows.size):
        cp.add.at(
            d_counts,
            d_source_rows,
            cp.ones(int(d_source_rows.size), dtype=cp.int32),
        )
    d_validity = cp.asarray(output_validity, dtype=cp.bool_)
    d_point_rows = cp.flatnonzero(d_validity & (d_counts <= 1)).astype(
        cp.int64,
        copy=False,
    )
    d_multipoint_rows = cp.flatnonzero(d_validity & (d_counts > 1)).astype(
        cp.int64,
        copy=False,
    )

    d_tags = cp.full(row_count, NULL_TAG, dtype=cp.int8)
    d_family_rows = cp.full(row_count, -1, dtype=cp.int32)
    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    if int(d_point_rows.size):
        d_point_counts = d_counts[d_point_rows]
        d_point_offsets = cp.empty(int(d_point_rows.size) + 1, dtype=cp.int32)
        d_point_offsets[0] = 0
        cp.cumsum(d_point_counts, out=d_point_offsets[1:])
        d_single_coord_mask = d_counts[d_source_rows] == 1
        device_families[GeometryFamily.POINT] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=point_x[d_single_coord_mask].copy(),
            y=point_y[d_single_coord_mask].copy(),
            geometry_offsets=d_point_offsets,
            empty_mask=d_point_counts == 0,
            bounds=None,
        )
        d_tags[d_point_rows] = FAMILY_TAGS[GeometryFamily.POINT]
        d_family_rows[d_point_rows] = cp.arange(
            int(d_point_rows.size),
            dtype=cp.int32,
        )

    if int(d_multipoint_rows.size):
        d_multipoint_counts = d_counts[d_multipoint_rows]
        d_multipoint_offsets = cp.empty(
            int(d_multipoint_rows.size) + 1,
            dtype=cp.int32,
        )
        d_multipoint_offsets[0] = 0
        cp.cumsum(d_multipoint_counts, out=d_multipoint_offsets[1:])
        d_multi_coord_mask = d_counts[d_source_rows] > 1
        device_families[GeometryFamily.MULTIPOINT] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=point_x[d_multi_coord_mask].copy(),
            y=point_y[d_multi_coord_mask].copy(),
            geometry_offsets=d_multipoint_offsets,
            empty_mask=cp.zeros(int(d_multipoint_rows.size), dtype=cp.bool_),
            bounds=None,
        )
        d_tags[d_multipoint_rows] = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
        d_family_rows[d_multipoint_rows] = cp.arange(
            int(d_multipoint_rows.size),
            dtype=cp.int32,
        )

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )


def _multipoint_polygon_constructive(
    operation: str,
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Evaluate aligned MultiPoint/Polygon set membership entirely on device."""
    import cupy as cp

    if operation not in {"intersection", "difference"}:
        raise ValueError(f"unsupported MultiPoint-Polygon operation: {operation}")
    if multipoints.row_count != polygons.row_count:
        raise ValueError("MultiPoint-Polygon constructive inputs must be row-aligned")

    multipoints.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"multipoint_polygon_{operation} GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"multipoint_polygon_{operation} GPU",
    )
    mp_state = multipoints._ensure_device_state(preserve_indexed_view=True)
    polygon_state = polygons._ensure_device_state(preserve_indexed_view=True)
    d_output_validity = cp.asarray(mp_state.validity, dtype=cp.bool_) & cp.asarray(
        polygon_state.validity,
        dtype=cp.bool_,
    )

    # Reuse the canonical point-family capacity carrier. Physical point slots
    # and source rows stay allocated at source capacity while the active prefix
    # remains device-resident.
    from vibespatial.constructive.binary_constructive import (
        _explode_point_rows_to_point_capacity_gpu,
    )

    point_parts = _explode_point_rows_to_point_capacity_gpu(multipoints)
    if point_parts is None:
        return _selected_point_rows_to_owned(
            point_x=cp.empty(0, dtype=cp.float64),
            point_y=cp.empty(0, dtype=cp.float64),
            source_rows=cp.empty(0, dtype=cp.int64),
            output_validity=d_output_validity,
            row_count=multipoints.row_count,
        )
    point_rows = point_parts.geometry
    d_source_rows = cp.asarray(point_parts.source_rows, dtype=cp.int64)
    d_active = point_parts.selection.active_capacity_mask()

    aligned_polygons = polygons.device_take(
        d_source_rows,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    intersects = binary_predicate_expression(
        "intersects",
        point_rows,
        aligned_polygons,
        dispatch_mode=ExecutionMode.GPU,
        operation=f"constructive.multipoint_polygon.{operation}",
    )
    if intersects is None or not hasattr(
        intersects.values,
        "__cuda_array_interface__",
    ):
        raise RuntimeError("MultiPoint-Polygon membership left native execution")

    d_hits = cp.asarray(intersects.values, dtype=cp.bool_)
    d_keep = d_hits if operation == "intersection" else ~d_hits
    d_keep &= d_active & d_output_validity[d_source_rows]

    point_state = point_rows._ensure_device_state(preserve_indexed_view=True)
    point_buffer = point_state.families[GeometryFamily.POINT]
    d_point_family_rows = cp.asarray(
        point_state.family_row_offsets,
        dtype=cp.int64,
    )
    d_point_coord_rows = cp.asarray(
        point_buffer.geometry_offsets,
        dtype=cp.int64,
    )[d_point_family_rows]
    return _selected_point_rows_to_owned(
        point_x=cp.asarray(point_buffer.x)[d_point_coord_rows[d_keep]],
        point_y=cp.asarray(point_buffer.y)[d_point_coord_rows[d_keep]],
        source_rows=d_source_rows[d_keep],
        output_validity=d_output_validity,
        row_count=multipoints.row_count,
    )


def multipoint_polygon_intersection(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon intersection with device-resident point selection."""
    return _multipoint_polygon_constructive("intersection", multipoints, polygons)


def multipoint_polygon_difference(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon difference with device-resident point selection."""
    return _multipoint_polygon_constructive("difference", multipoints, polygons)
