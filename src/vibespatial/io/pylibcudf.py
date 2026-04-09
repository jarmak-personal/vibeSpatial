from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime.residency import Residency

from .wkb import (
    WKB_TYPE_IDS,
    DeviceWKBHeaderScan,
)

# CCCL warmup (ADR-0034)
request_warmup(["exclusive_scan_i32"])


def _build_lazy_host_family_stub(
    family: GeometryFamily,
    *,
    row_count: int,
) -> FamilyGeometryBuffer:
    """Create an unmaterialized host placeholder for a device family buffer."""
    return FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=np.bool_),
        bounds=None,
        host_materialized=False,
    )

def _pylibcudf_buffer_view(column, dtype: np.dtype[Any]):
    import cupy as cp

    raw = cp.asarray(column.data())
    return raw.view(dtype)


def _require_pylibcudf_zero_offset(column, detail: str) -> None:
    if column.offset() != 0:
        raise NotImplementedError(f"offset pylibcudf {detail} columns are not supported yet")


def _pylibcudf_validity_mask(column):
    import cupy as cp

    row_count = int(column.size())
    if column.null_count() == 0:
        return cp.ones(row_count, dtype=cp.bool_)
    bitmap = cp.asarray(column.null_mask()).view(cp.uint8)
    indexes = cp.arange(row_count, dtype=cp.int32)
    return ((bitmap[indexes // 8] >> (indexes % 8)) & 1).astype(cp.bool_)


def _pylibcudf_point_xy_children(column):
    if column.num_children() != 2:
        raise NotImplementedError("GeoArrow point columns must expose x/y children")
    return (
        _pylibcudf_buffer_view(column.child(0), np.float64),
        _pylibcudf_buffer_view(column.child(1), np.float64),
    )


def _pylibcudf_can_adopt_zero_copy(column) -> bool:
    """True when column.offset() == 0 and column.null_count() == 0."""
    return column.offset() == 0 and column.null_count() == 0


def _device_narrow_offsets_int64_to_int32(offsets_int64):
    """Narrow int64 offsets to int32 on device via CuPy cast with overflow check."""
    import cupy as cp
    max_val = int(cp.asnumpy(offsets_int64.max()))
    if max_val > np.iinfo(np.int32).max:
        raise OverflowError(f"GeoArrow offsets exceed int32 range (max={max_val})")
    return offsets_int64.astype(cp.int32)


def _pylibcudf_list_offsets_adopt(column):
    """Adopt list offsets from pylibcudf column, narrowing int64->int32 if needed.
    For int32: returns zero-copy CuPy view. For int64: returns narrowed int32 copy."""
    offsets_child = column.child(0)
    element_count = int(offsets_child.size())
    # Try int32 first via buffer view
    raw = _pylibcudf_buffer_view(offsets_child, np.int32)
    if raw.nbytes == element_count * 8:  # int64 offsets
        raw64 = _pylibcudf_buffer_view(offsets_child, np.int64)
        return _device_narrow_offsets_int64_to_int32(raw64[:element_count])
    return raw[:element_count]


def _pylibcudf_list_offsets(column):
    if column.num_children() != 2:
        raise NotImplementedError("GeoArrow list columns must expose offsets and values children")
    return _pylibcudf_buffer_view(column.child(0), np.int32)


def _device_compact_offsets(lengths):
    import cupy as cp

    n = int(lengths.size)
    compact_offsets = cp.empty(n + 1, dtype=cp.int32)
    compact_offsets[0] = 0
    if n > 0:
        # CCCL exclusive_sum (ADR-0033): 1.8-3.7x faster than CuPy cumsum
        offsets_excl = exclusive_sum(lengths, synchronize=False)
        compact_offsets[1:] = offsets_excl + lengths
    return compact_offsets


def _device_child_selection_mask(offsets, parent_mask):
    import cupy as cp

    lengths = offsets[1:] - offsets[:-1]
    selected_lengths = lengths[parent_mask]
    compact_offsets = _device_compact_offsets(selected_lengths)
    child_count = int(cp.asnumpy(offsets[-1]))
    if child_count == 0:
        return compact_offsets, cp.zeros(0, dtype=cp.bool_)
    starts = offsets[:-1][parent_mask]
    ends = offsets[1:][parent_mask]
    diff = cp.zeros(child_count + 1, dtype=cp.int32)
    cp.add.at(diff, starts, 1)
    cp.add.at(diff, ends, -1)
    return compact_offsets, cp.cumsum(diff[:-1], dtype=cp.int32) > 0


def _device_select_true(mask):
    import cupy as cp

    # CuPy wins on this decode-shaped path and avoids the current CCCL bool-mask
    # JIT issue in select predicates.
    return cp.flatnonzero(mask).astype(cp.int32, copy=False)


def _device_mask_count(mask) -> int:
    import cupy as cp

    return int(cp.asnumpy(mask.astype(cp.int32).sum()))

def _pylibcudf_wkb_offsets(column):
    if column.num_children() != 1:
        raise NotImplementedError("WKB binary columns must expose one offsets child")
    return _pylibcudf_buffer_view(column.child(0), np.int32)


def _pylibcudf_wkb_payload(column):
    import cupy as cp

    return cp.asarray(column.data()).view(cp.uint8)


def _pylibcudf_unpack_le_uint32(payload, starts):
    import cupy as cp

    return (
        payload[starts].astype(cp.uint32)
        | (payload[starts + 1].astype(cp.uint32) << 8)
        | (payload[starts + 2].astype(cp.uint32) << 16)
        | (payload[starts + 3].astype(cp.uint32) << 24)
    )


def _pylibcudf_unpack_le_float64(payload, starts):
    import cupy as cp

    shifts = cp.asarray([0, 8, 16, 24, 32, 40, 48, 56], dtype=cp.uint64)
    byte_matrix = payload[starts[:, None] + cp.arange(8, dtype=cp.int32)]
    packed = cp.sum(byte_matrix.astype(cp.uint64) << shifts[None, :], axis=1, dtype=cp.uint64)
    return packed.view(cp.float64)

def _scan_pylibcudf_wkb_headers(column) -> DeviceWKBHeaderScan:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, "WKB")
    validity = _pylibcudf_validity_mask(column)
    row_count = int(column.size())
    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    lengths = offsets[1:] - offsets[:-1]
    header_ready = validity & (lengths >= 5)
    header_rows = _device_select_true(header_ready)
    type_ids = cp.full(row_count, -1, dtype=cp.int32)
    family_tags = cp.full(row_count, -1, dtype=cp.int8)
    native_mask = cp.zeros(row_count, dtype=cp.bool_)
    point_mask = cp.zeros(row_count, dtype=cp.bool_)

    if int(header_rows.size):
        starts = offsets[:-1][header_rows]
        byteorder = payload[starts]
        type_values = _pylibcudf_unpack_le_uint32(payload, starts + 1)
        little_endian = byteorder == 1
        native_types = cp.isin(
            type_values,
            cp.asarray(
                [
                    WKB_TYPE_IDS[GeometryFamily.POINT],
                    WKB_TYPE_IDS[GeometryFamily.LINESTRING],
                    WKB_TYPE_IDS[GeometryFamily.POLYGON],
                    WKB_TYPE_IDS[GeometryFamily.MULTIPOINT],
                    WKB_TYPE_IDS[GeometryFamily.MULTILINESTRING],
                    WKB_TYPE_IDS[GeometryFamily.MULTIPOLYGON],
                ],
                dtype=cp.uint32,
            ),
        )
        native_rows = header_rows[little_endian & native_types]
        type_ids[header_rows] = type_values.astype(cp.int32)
        native_mask[native_rows] = True
        for family, tag in FAMILY_TAGS.items():
            type_id = WKB_TYPE_IDS[family]
            family_rows = header_rows[little_endian & (type_values == type_id)]
            if int(family_rows.size):
                family_tags[family_rows] = np.int8(tag)
        point_rows = header_rows[little_endian & (type_values == WKB_TYPE_IDS[GeometryFamily.POINT])]
        if int(point_rows.size):
            point_mask[point_rows] = True

    fallback_mask = validity & ~native_mask
    return DeviceWKBHeaderScan(
        row_count=row_count,
        valid_count=int(cp.asnumpy(validity.astype(cp.int32).sum())),
        native_count=int(cp.asnumpy(native_mask.astype(cp.int32).sum())),
        fallback_count=int(cp.asnumpy(fallback_mask.astype(cp.int32).sum())),
        validity=validity,
        type_ids=type_ids,
        family_tags=family_tags,
        native_mask=native_mask,
        fallback_mask=fallback_mask,
        point_mask=point_mask,
    )

def _build_device_single_family_owned(
    *,
    family: GeometryFamily,
    validity_device,
    x_device,
    y_device,
    geometry_offsets_device,
    empty_mask_device,
    part_offsets_device=None,
    ring_offsets_device=None,
    detail: str,
) -> OwnedGeometryArray:
    import cupy as cp

    row_count = int(validity_device.size)
    valid_count = int(cp.count_nonzero(validity_device).item())
    tags_device = cp.where(validity_device, np.int8(FAMILY_TAGS[family]), np.int8(-1)).astype(cp.int8)
    family_row_offsets_device = cp.full(row_count, -1, dtype=cp.int32)
    if valid_count:
        family_row_offsets_device[validity_device] = cp.arange(valid_count, dtype=cp.int32)
    buffer = _build_lazy_host_family_stub(family, row_count=valid_count)
    owned = OwnedGeometryArray(
        validity=None,
        tags=None,
        family_row_offsets=None,
        families={family: buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=validity_device,
            tags=tags_device,
            family_row_offsets=family_row_offsets_device,
            families={
                family: DeviceFamilyGeometryBuffer(
                    family=family,
                    x=x_device,
                    y=y_device,
                    geometry_offsets=geometry_offsets_device,
                    empty_mask=empty_mask_device,
                    part_offsets=part_offsets_device,
                    ring_offsets=ring_offsets_device,
                    bounds=None,
                )
            },
        ),
        _row_count=row_count,
    )
    owned._record(
        DiagnosticKind.CREATED,
        detail,
        visible=True,
    )
    return owned

def _build_device_mixed_owned(
    *,
    validity_device,
    tags_device,
    family_row_offsets_device,
    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer],
    detail: str,
) -> OwnedGeometryArray:
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, device_buffer in family_devices.items():
        row_count = int(device_buffer.empty_mask.size)
        families[family] = _build_lazy_host_family_stub(family, row_count=row_count)
    owned = OwnedGeometryArray(
        validity=None,
        tags=None,
        family_row_offsets=None,
        families=families,
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=validity_device,
            tags=tags_device,
            family_row_offsets=family_row_offsets_device,
            families=family_devices,
        ),
        _row_count=int(validity_device.size),
    )
    owned._record(
        DiagnosticKind.CREATED,
        detail,
        visible=True,
    )
    return owned

def _build_device_wkb_linestring_family(column, row_indexes):
    import cupy as cp

    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][row_indexes]
    lengths = offsets[1:][row_indexes] - starts
    point_counts = _pylibcudf_unpack_le_uint32(payload, starts + 5).astype(cp.int32, copy=False)
    expected_lengths = 9 + (point_counts * 16)
    if bool(cp.asnumpy(cp.any(lengths != expected_lengths))):
        raise NotImplementedError(
            "pylibcudf device WKB linestring decode currently supports only canonical little-endian 2D records"
        )

    geometry_offsets_device = _device_compact_offsets(point_counts)
    total_points = int(cp.asnumpy(geometry_offsets_device[-1]))
    if total_points:
        coord_indexes = cp.arange(total_points, dtype=cp.int32)
        row_ids = cp.searchsorted(geometry_offsets_device[1:], coord_indexes, side="right")
        coord_rows = starts[row_ids] + 9
        local_offsets = coord_indexes - geometry_offsets_device[row_ids]
        x_starts = coord_rows + (local_offsets * 16)
        y_starts = x_starts + 8
        x_device = _pylibcudf_unpack_le_float64(payload, x_starts)
        y_device = _pylibcudf_unpack_le_float64(payload, y_starts)
    else:
        x_device = cp.empty(0, dtype=cp.float64)
        y_device = cp.empty(0, dtype=cp.float64)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=x_device,
        y=y_device,
        geometry_offsets=geometry_offsets_device,
        empty_mask=point_counts == 0,
        bounds=None,
    )

def _build_device_wkb_polygon_family(column, row_indexes):
    """GPU WKB polygon decode: header(5) + ring_count(4) + rings[ring_count(4) + coords[16*n]]."""
    import cupy as cp

    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][row_indexes]
    n_rows = int(row_indexes.size)

    # Read ring count per polygon (offset 5 in WKB record).
    ring_counts = _pylibcudf_unpack_le_uint32(payload, starts + 5).astype(cp.int32, copy=False)
    geometry_offsets_device = _device_compact_offsets(ring_counts)
    total_rings = int(cp.asnumpy(geometry_offsets_device[-1]))

    if total_rings == 0:
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=geometry_offsets_device,
            empty_mask=ring_counts == 0,
            ring_offsets=cp.zeros(1, dtype=cp.int32),
            bounds=None,
        )

    # Build per-ring byte offsets within the WKB payload. Each ring is preceded
    # by a uint32 point count, then n_pts * 16 bytes of x/y float64 pairs.
    # We need to scan cumulatively within each polygon record.
    ring_indexes = cp.arange(total_rings, dtype=cp.int32)
    ring_row_ids = cp.searchsorted(geometry_offsets_device[1:], ring_indexes, side="right")
    _local_ring_idx = ring_indexes - geometry_offsets_device[ring_row_ids]

    # First ring of each polygon starts at byte offset starts[row] + 9
    # (5 header + 4 ring_count). Subsequent rings start after the
    # preceding ring's point_count(4) + coords(n_pts*16). We compute
    # cumulative byte offsets per ring.
    #
    # Strategy: read ring point counts by scanning forward through rings.
    # We must do this iteratively per ring index within each polygon.
    # For GPU efficiency, we flatten: emit ring starts in a single pass
    # by prefix-summing per-ring byte sizes.
    #
    # First pass: compute starting byte of ring 0 for each polygon.
    poly_ring0_starts = starts + 9  # byte offset of first ring's point_count
    ring_byte_starts = cp.empty(total_rings, dtype=cp.int64)
    ring_byte_starts[geometry_offsets_device[:-1][ring_counts > 0]] = poly_ring0_starts[ring_counts > 0]

    # Read point counts for each ring. To do this we need the byte position
    # of each ring header. We compute this via a sequential scan over rings
    # within polygons, accumulating sizes.
    # For a vectorized approach: two-pass. Pass 1: read all ring point counts
    # at known positions. Pass 2: exclusive scan of ring byte sizes within
    # each polygon to get per-ring positions.
    #
    # Position of ring i within polygon p:
    #   ring_start_byte = poly_start + 9 + sum_{j<i}(4 + 16 * n_pts_j)
    # This is an exclusive prefix sum of (4 + 16*n_pts) within each polygon.
    # We can compute this using a two-pass approach:
    # 1. Start all ring_byte_starts at ring 0 of each polygon.
    # 2. Read the point count at ring 0 position.
    # 3. Compute the cumulative byte offset for subsequent rings.

    # Iterative ring position resolution — loops over max_rings which is
    # typically small (most polygons have 1-3 rings: exterior + holes).
    max_rings = int(cp.asnumpy(ring_counts.max())) if n_rows else 0
    ring_point_counts = cp.zeros(total_rings, dtype=cp.int32)

    # Initialize: ring 0 of each polygon starts at poly_ring0_starts.
    current_positions = poly_ring0_starts.copy()  # per-polygon cursor

    for ring_idx in range(max_rings):
        # Which polygons have a ring at this index?
        # No early-exit guard needed: loop bounded by max_rings (from .max()),
        # so has_ring is guaranteed non-empty for all ring_idx < max_rings.
        # Masked ops below are no-ops for polygons with fewer rings.
        has_ring = ring_counts > ring_idx

        # Global ring positions for ring_idx of each polygon
        global_ring_idxs = geometry_offsets_device[:-1][has_ring] + ring_idx
        ring_byte_starts[global_ring_idxs] = current_positions[has_ring]

        # Read point count at the current position
        pts = _pylibcudf_unpack_le_uint32(payload, current_positions[has_ring]).astype(cp.int32, copy=False)
        ring_point_counts[global_ring_idxs] = pts

        # Advance cursor past this ring: 4 (point count) + n_pts * 16 (coords)
        current_positions[has_ring] += 4 + pts.astype(cp.int64) * 16

    # Build ring offsets (cumulative point counts per polygon).
    ring_offsets_device = _device_compact_offsets(ring_point_counts)
    total_points = int(cp.asnumpy(ring_offsets_device[-1]))

    if total_points:
        coord_indexes = cp.arange(total_points, dtype=cp.int32)
        ring_ids = cp.searchsorted(ring_offsets_device[1:], coord_indexes, side="right")
        local_offsets = coord_indexes - ring_offsets_device[ring_ids]
        # Byte position of each coordinate: ring_byte_start + 4 (point count header) + local_offset * 16
        x_byte_starts = ring_byte_starts[ring_ids] + 4 + local_offsets.astype(cp.int64) * 16
        y_byte_starts = x_byte_starts + 8
        x_device = _pylibcudf_unpack_le_float64(payload, x_byte_starts)
        y_device = _pylibcudf_unpack_le_float64(payload, y_byte_starts)
    else:
        x_device = cp.empty(0, dtype=cp.float64)
        y_device = cp.empty(0, dtype=cp.float64)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=x_device,
        y=y_device,
        geometry_offsets=geometry_offsets_device,
        empty_mask=ring_counts == 0,
        ring_offsets=ring_offsets_device,
        bounds=None,
    )

def _build_device_wkb_multipoint_family(column, row_indexes):
    """GPU WKB multipoint decode: header(5) + part_count(4) + parts[header(5) + x(8) + y(8)]."""
    import cupy as cp

    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][row_indexes]

    # Read part count per multipoint (offset 5 in WKB record).
    part_counts = _pylibcudf_unpack_le_uint32(payload, starts + 5).astype(cp.int32, copy=False)
    geometry_offsets_device = _device_compact_offsets(part_counts)
    total_points = int(cp.asnumpy(geometry_offsets_device[-1]))

    if total_points:
        coord_indexes = cp.arange(total_points, dtype=cp.int32)
        row_ids = cp.searchsorted(geometry_offsets_device[1:], coord_indexes, side="right")
        local_offsets = coord_indexes - geometry_offsets_device[row_ids]
        # Each sub-point is 21 bytes: header(5) + x(8) + y(8)
        # byte_pos = starts[row] + 9 + local * 21
        byte_pos = starts[row_ids] + 9 + local_offsets.astype(cp.int64) * 21
        x_device = _pylibcudf_unpack_le_float64(payload, byte_pos + 5)
        y_device = _pylibcudf_unpack_le_float64(payload, byte_pos + 13)
    else:
        x_device = cp.empty(0, dtype=cp.float64)
        y_device = cp.empty(0, dtype=cp.float64)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=x_device,
        y=y_device,
        geometry_offsets=geometry_offsets_device,
        empty_mask=part_counts == 0,
        bounds=None,
    )


def _build_device_wkb_multilinestring_family(column, row_indexes):
    """GPU WKB multilinestring decode: header(5) + part_count(4) + parts[header(5) + pt_count(4) + coords[16*n]].

    One-level iterative scan identical to the polygon ring scan pattern.
    """
    import cupy as cp

    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][row_indexes]
    n_rows = int(row_indexes.size)

    # Read part count per multilinestring (offset 5 in WKB record).
    part_counts = _pylibcudf_unpack_le_uint32(payload, starts + 5).astype(cp.int32, copy=False)
    geometry_offsets_device = _device_compact_offsets(part_counts)
    total_parts = int(cp.asnumpy(geometry_offsets_device[-1]))

    if total_parts == 0:
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=geometry_offsets_device,
            empty_mask=part_counts == 0,
            part_offsets=cp.zeros(1, dtype=cp.int32),
            bounds=None,
        )

    # Iterative per-part-index scan (mirrors polygon ring scan pattern).
    # Each sub-linestring: header(5) + pt_count(4) + coords(n*16) = 9 + n*16 bytes.
    max_parts = int(cp.asnumpy(part_counts.max())) if n_rows else 0
    part_point_counts = cp.zeros(total_parts, dtype=cp.int32)
    part_byte_starts = cp.empty(total_parts, dtype=cp.int64)

    # First part of each row starts at byte offset starts[row] + 9
    current_positions = (starts + 9).copy()

    for part_idx in range(max_parts):
        # No early-exit guard needed: loop bounded by max_parts (from .max()),
        # so has_part is guaranteed non-empty for all part_idx < max_parts.
        # Masked ops below are no-ops for rows with fewer parts.
        has_part = part_counts > part_idx

        global_part_idxs = geometry_offsets_device[:-1][has_part] + part_idx
        part_byte_starts[global_part_idxs] = current_positions[has_part]

        # Read point count at current_position + 5 (skip sub-linestring header)
        pts = _pylibcudf_unpack_le_uint32(payload, current_positions[has_part] + 5).astype(cp.int32, copy=False)
        part_point_counts[global_part_idxs] = pts

        # Advance cursor past this part: 9 (header + pt_count) + n_pts * 16 (coords)
        current_positions[has_part] += 9 + pts.astype(cp.int64) * 16

    # Build part offsets (cumulative point counts).
    part_offsets_device = _device_compact_offsets(part_point_counts)
    total_points = int(cp.asnumpy(part_offsets_device[-1]))

    if total_points:
        coord_indexes = cp.arange(total_points, dtype=cp.int32)
        part_ids = cp.searchsorted(part_offsets_device[1:], coord_indexes, side="right")
        local_offsets = coord_indexes - part_offsets_device[part_ids]
        # Byte position: part_byte_start + 9 (header + pt_count) + local_offset * 16
        x_byte_starts = part_byte_starts[part_ids] + 9 + local_offsets.astype(cp.int64) * 16
        y_byte_starts = x_byte_starts + 8
        x_device = _pylibcudf_unpack_le_float64(payload, x_byte_starts)
        y_device = _pylibcudf_unpack_le_float64(payload, y_byte_starts)
    else:
        x_device = cp.empty(0, dtype=cp.float64)
        y_device = cp.empty(0, dtype=cp.float64)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=x_device,
        y=y_device,
        geometry_offsets=geometry_offsets_device,
        empty_mask=part_counts == 0,
        part_offsets=part_offsets_device,
        bounds=None,
    )


def _build_device_wkb_multipolygon_family(column, row_indexes):
    """GPU WKB multipolygon decode: two-level iterative scan.

    Outer level: polygons within each multipolygon row.
    Inner level: rings within each polygon (reuses polygon scan pattern).
    """
    import cupy as cp

    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][row_indexes]
    n_rows = int(row_indexes.size)

    # Read polygon count per multipolygon (offset 5 in WKB record).
    polygon_counts = _pylibcudf_unpack_le_uint32(payload, starts + 5).astype(cp.int32, copy=False)
    geometry_offsets_device = _device_compact_offsets(polygon_counts)
    total_polygons = int(cp.asnumpy(geometry_offsets_device[-1]))

    if total_polygons == 0:
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=geometry_offsets_device,
            empty_mask=polygon_counts == 0,
            part_offsets=cp.zeros(1, dtype=cp.int32),
            ring_offsets=cp.zeros(1, dtype=cp.int32),
            bounds=None,
        )

    # --- Outer loop: walk polygons within each multipolygon row ---
    max_polygons = int(cp.asnumpy(polygon_counts.max())) if n_rows else 0
    poly_ring_counts = cp.zeros(total_polygons, dtype=cp.int32)
    poly_ring0_starts = cp.empty(total_polygons, dtype=cp.int64)

    # First polygon of each row starts at byte offset starts[row] + 9
    current_positions = (starts + 9).copy()

    # We need to track each polygon's ring0 start for the inner loop.
    # For each polygon we read: header(5) + ring_count(4), then walk rings.
    # But we only need to record ring_count and ring0_start per polygon here,
    # then advance past the entire polygon record.

    for poly_idx in range(max_polygons):
        # No early-exit guard needed: loop bounded by max_polygons (from .max()),
        # so has_poly is guaranteed non-empty for all poly_idx < max_polygons.
        # Masked ops below are no-ops for rows with fewer polygons.
        has_poly = polygon_counts > poly_idx

        global_poly_idxs = geometry_offsets_device[:-1][has_poly] + poly_idx

        # Read ring count at current_position + 5 (skip polygon sub-header)
        ring_counts_here = _pylibcudf_unpack_le_uint32(
            payload, current_positions[has_poly] + 5
        ).astype(cp.int32, copy=False)
        poly_ring_counts[global_poly_idxs] = ring_counts_here

        # Ring 0 starts at current_position + 9 (polygon header)
        poly_ring0_starts[global_poly_idxs] = current_positions[has_poly] + 9

        # Advance past the entire polygon: walk all rings to find byte size.
        # We need to scan each ring's point_count to skip its coords.
        # Removed per-ring-index cp.asnumpy(has_ring.any()) D2H early-exit;
        # masked ops are no-ops for polygons with fewer rings.
        # NOTE: int(.max()) still performs one D2H scalar read per polygon
        # batch (unavoidable — Python needs a host int for range()).
        # has_poly guaranteed non-empty: outer loop bounded by max_polygons
        # (from .max()), so at least one row has polygon_counts == max_polygons.
        max_rings_here = int(ring_counts_here.max())
        poly_cursors = current_positions[has_poly] + 9  # start of ring data

        for ring_idx in range(max_rings_here):
            has_ring = ring_counts_here > ring_idx
            pts = _pylibcudf_unpack_le_uint32(
                payload, poly_cursors[has_ring]
            ).astype(cp.int64, copy=False)
            poly_cursors[has_ring] += 4 + pts * 16

        # Advance main cursor past this polygon
        current_positions[has_poly] = poly_cursors

    # --- Build part_offsets (polygon -> ring) ---
    part_offsets_device = _device_compact_offsets(poly_ring_counts)
    total_rings = int(cp.asnumpy(part_offsets_device[-1]))

    if total_rings == 0:
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=geometry_offsets_device,
            empty_mask=polygon_counts == 0,
            part_offsets=part_offsets_device,
            ring_offsets=cp.zeros(1, dtype=cp.int32),
            bounds=None,
        )

    # --- Inner loop: walk rings within each polygon ---
    max_rings = int(cp.asnumpy(poly_ring_counts.max())) if total_polygons else 0
    ring_point_counts = cp.zeros(total_rings, dtype=cp.int32)
    ring_byte_starts = cp.empty(total_rings, dtype=cp.int64)

    # Per-polygon cursor starting at ring0
    poly_cursors = poly_ring0_starts.copy()

    for ring_idx in range(max_rings):
        # No early-exit guard needed: loop bounded by max_rings (from .max()),
        # so has_ring is guaranteed non-empty for all ring_idx < max_rings.
        # Masked ops below are no-ops for polygons with fewer rings.
        has_ring = poly_ring_counts > ring_idx

        global_ring_idxs = part_offsets_device[:-1][has_ring] + ring_idx
        ring_byte_starts[global_ring_idxs] = poly_cursors[has_ring]

        pts = _pylibcudf_unpack_le_uint32(
            payload, poly_cursors[has_ring]
        ).astype(cp.int32, copy=False)
        ring_point_counts[global_ring_idxs] = pts

        # Advance cursor past this ring: 4 (point count) + n_pts * 16 (coords)
        poly_cursors[has_ring] += 4 + pts.astype(cp.int64) * 16

    # Build ring offsets (cumulative point counts).
    ring_offsets_device = _device_compact_offsets(ring_point_counts)
    total_points = int(cp.asnumpy(ring_offsets_device[-1]))

    if total_points:
        coord_indexes = cp.arange(total_points, dtype=cp.int32)
        ring_ids = cp.searchsorted(ring_offsets_device[1:], coord_indexes, side="right")
        local_offsets = coord_indexes - ring_offsets_device[ring_ids]
        # Byte position: ring_byte_start + 4 (point count header) + local_offset * 16
        x_byte_starts = ring_byte_starts[ring_ids] + 4 + local_offsets.astype(cp.int64) * 16
        y_byte_starts = x_byte_starts + 8
        x_device = _pylibcudf_unpack_le_float64(payload, x_byte_starts)
        y_device = _pylibcudf_unpack_le_float64(payload, y_byte_starts)
    else:
        x_device = cp.empty(0, dtype=cp.float64)
        y_device = cp.empty(0, dtype=cp.float64)

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=x_device,
        y=y_device,
        geometry_offsets=geometry_offsets_device,
        empty_mask=polygon_counts == 0,
        part_offsets=part_offsets_device,
        ring_offsets=ring_offsets_device,
        bounds=None,
    )


def _decode_pylibcudf_wkb_multipoint_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    multipoint_mask = header_scan.family_tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
    if header_scan.native_count != header_scan.valid_count or _device_mask_count(multipoint_mask) != header_scan.valid_count:
        raise NotImplementedError(
            "pylibcudf device WKB decode currently supports only multipoint-only columns; "
            "mixed-family WKB rows still use the staged host bridge"
        )

    row_indexes = _device_select_true(multipoint_mask)
    family_buffer = _build_device_wkb_multipoint_family(column, row_indexes)
    return _build_device_single_family_owned(
        family=GeometryFamily.MULTIPOINT,
        validity_device=header_scan.validity,
        x_device=family_buffer.x,
        y_device=family_buffer.y,
        geometry_offsets_device=family_buffer.geometry_offsets,
        empty_mask_device=family_buffer.empty_mask,
        detail="created device-resident owned geometry array from pylibcudf WKB multipoint scan",
    )


def _decode_pylibcudf_wkb_homogeneous_nested_column_to_owned(
    column,
    family: GeometryFamily,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    family_mask = header_scan.family_tags == np.int8(FAMILY_TAGS[family])
    if header_scan.native_count != header_scan.valid_count or _device_mask_count(family_mask) != header_scan.valid_count:
        raise NotImplementedError(
            f"pylibcudf device WKB decode currently supports only {family.value.lower()}-only columns; "
            "mixed-family WKB rows still use the staged device pipeline"
        )

    from vibespatial.kernels.core.wkb_decode import decode_wkb_device_pipeline

    return decode_wkb_device_pipeline(
        _pylibcudf_wkb_payload(column),
        _pylibcudf_wkb_offsets(column),
        header_scan.row_count,
    )


def _decode_pylibcudf_wkb_multilinestring_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    return _decode_pylibcudf_wkb_homogeneous_nested_column_to_owned(
        column,
        GeometryFamily.MULTILINESTRING,
        scan=scan,
    )


def _decode_pylibcudf_wkb_multipolygon_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    return _decode_pylibcudf_wkb_homogeneous_nested_column_to_owned(
        column,
        GeometryFamily.MULTIPOLYGON,
        scan=scan,
    )


def _decode_pylibcudf_wkb_polygon_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    return _decode_pylibcudf_wkb_homogeneous_nested_column_to_owned(
        column,
        GeometryFamily.POLYGON,
        scan=scan,
    )


def _decode_pylibcudf_wkb_general_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    """GPU WKB decode for any supported homogeneous or mixed-family column."""

    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    if header_scan.native_count != header_scan.valid_count:
        raise NotImplementedError(
            "pylibcudf device WKB decode requires all valid rows to be native supported types"
        )

    # Preserve the lightweight fast paths for point-only and point/linestring
    # columns. Heavier polygon-family decode should use the staged kernel
    # pipeline instead of the older Python-orchestrated pylibcudf helpers.
    point_count = _device_mask_count(header_scan.point_mask)
    linestring_mask = header_scan.family_tags == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING])
    linestring_count = _device_mask_count(linestring_mask)
    valid = header_scan.valid_count
    if valid == point_count:
        return _decode_pylibcudf_wkb_point_column_to_owned(column, scan=header_scan)
    if valid == linestring_count:
        return _decode_pylibcudf_wkb_linestring_column_to_owned(column, scan=header_scan)
    if valid == point_count + linestring_count:
        return _decode_pylibcudf_wkb_point_linestring_column_to_owned(column, scan=header_scan)

    from vibespatial.kernels.core.wkb_decode import decode_wkb_device_pipeline

    return decode_wkb_device_pipeline(
        _pylibcudf_wkb_payload(column),
        _pylibcudf_wkb_offsets(column),
        header_scan.row_count,
    )


def _decode_pylibcudf_point_geoarrow_column_to_owned(column) -> OwnedGeometryArray:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, "point")

    if _pylibcudf_can_adopt_zero_copy(column):
        row_count = int(column.size())
        x_device, y_device = _pylibcudf_point_xy_children(column)
        x_device = x_device[:row_count]
        y_device = y_device[:row_count]
        validity_device = cp.ones(row_count, dtype=cp.bool_)
        empty_mask_device = cp.isnan(x_device) | cp.isnan(y_device)
        nonempty = ~empty_mask_device
        geometry_offsets_device = _device_compact_offsets(nonempty.astype(cp.int32))
        owned = _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity_device,
            x_device=x_device[nonempty],
            y_device=y_device[nonempty],
            geometry_offsets_device=geometry_offsets_device,
            empty_mask_device=empty_mask_device,
            detail="zero-copy adopted device-resident point buffers from pylibcudf GeoArrow column",
        )
        owned.device_adopted = True
        if owned.device_state is not None:
            owned.device_state._column_refs = [column]
        return owned

    validity_device = _pylibcudf_validity_mask(column)
    x_all, y_all = _pylibcudf_point_xy_children(column)
    x_device = x_all[validity_device]
    y_device = y_all[validity_device]
    empty_mask_device = cp.isnan(x_device) | cp.isnan(y_device)
    nonempty = ~empty_mask_device
    geometry_offsets_device = _device_compact_offsets(nonempty.astype(cp.int32))
    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=validity_device,
        x_device=x_device[nonempty],
        y_device=y_device[nonempty],
        geometry_offsets_device=geometry_offsets_device,
        empty_mask_device=empty_mask_device,
        detail="created device-resident owned geometry array from pylibcudf GeoParquet point scan",
    )


def _decode_pylibcudf_wkb_point_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    import cupy as cp

    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    if header_scan.native_count != header_scan.valid_count or _device_mask_count(header_scan.point_mask) != header_scan.valid_count:
        raise NotImplementedError(
            "pylibcudf device WKB decode currently supports only point-only columns; "
            "mixed-family WKB rows still use the staged host bridge"
        )

    point_rows = _device_select_true(header_scan.point_mask)
    offsets = _pylibcudf_wkb_offsets(column)
    payload = _pylibcudf_wkb_payload(column)
    starts = offsets[:-1][point_rows]
    x_values = _pylibcudf_unpack_le_float64(payload, starts + 5)
    y_values = _pylibcudf_unpack_le_float64(payload, starts + 13)
    empty_mask = cp.isnan(x_values) | cp.isnan(y_values)
    nonempty = ~empty_mask
    geometry_offsets = _device_compact_offsets(nonempty.astype(cp.int32))
    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=header_scan.validity,
        x_device=x_values[nonempty],
        y_device=y_values[nonempty],
        geometry_offsets_device=geometry_offsets,
        empty_mask_device=empty_mask,
        detail="created device-resident owned geometry array from pylibcudf WKB point scan",
    )


def _decode_pylibcudf_wkb_linestring_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    linestring_mask = header_scan.family_tags == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING])
    if header_scan.native_count != header_scan.valid_count or _device_mask_count(linestring_mask) != header_scan.valid_count:
        raise NotImplementedError(
            "pylibcudf device WKB decode currently supports only linestring-only columns; "
            "mixed-family WKB rows still use the staged host bridge"
        )

    row_indexes = _device_select_true(linestring_mask)
    family_buffer = _build_device_wkb_linestring_family(column, row_indexes)
    return _build_device_single_family_owned(
        family=GeometryFamily.LINESTRING,
        validity_device=header_scan.validity,
        x_device=family_buffer.x,
        y_device=family_buffer.y,
        geometry_offsets_device=family_buffer.geometry_offsets,
        empty_mask_device=family_buffer.empty_mask,
        detail="created device-resident owned geometry array from pylibcudf WKB linestring scan",
    )


def _decode_pylibcudf_wkb_point_linestring_column_to_owned(
    column,
    *,
    scan: DeviceWKBHeaderScan | None = None,
) -> OwnedGeometryArray:
    import cupy as cp

    header_scan = _scan_pylibcudf_wkb_headers(column) if scan is None else scan
    linestring_mask = header_scan.family_tags == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING])
    supported_mask = header_scan.point_mask | linestring_mask
    supported_count = _device_mask_count(supported_mask)
    if header_scan.native_count != header_scan.valid_count or supported_count != header_scan.valid_count:
        raise NotImplementedError(
            "pylibcudf device WKB mixed decode currently supports only point and linestring rows"
        )

    row_count = int(header_scan.validity.size)
    point_rows = _device_select_true(header_scan.point_mask)
    point_offsets = _pylibcudf_wkb_offsets(column)
    point_payload = _pylibcudf_wkb_payload(column)
    point_starts = point_offsets[:-1][point_rows]
    point_x = _pylibcudf_unpack_le_float64(point_payload, point_starts + 5)
    point_y = _pylibcudf_unpack_le_float64(point_payload, point_starts + 13)
    point_empty = cp.isnan(point_x) | cp.isnan(point_y)
    point_nonempty = ~point_empty
    point_geometry_offsets = _device_compact_offsets(point_nonempty.astype(cp.int32, copy=False))
    point_family = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=point_x[point_nonempty],
        y=point_y[point_nonempty],
        geometry_offsets=point_geometry_offsets,
        empty_mask=point_empty,
        bounds=None,
    )

    linestring_rows = _device_select_true(linestring_mask)
    linestring_family = _build_device_wkb_linestring_family(column, linestring_rows)

    tags_device = cp.where(header_scan.validity, header_scan.family_tags, np.int8(-1)).astype(cp.int8, copy=False)
    family_row_offsets_device = cp.full(row_count, -1, dtype=cp.int32)
    if int(point_rows.size):
        family_row_offsets_device[point_rows] = cp.arange(int(point_rows.size), dtype=cp.int32)
    if int(linestring_rows.size):
        family_row_offsets_device[linestring_rows] = cp.arange(int(linestring_rows.size), dtype=cp.int32)
    return _build_device_mixed_owned(
        validity_device=header_scan.validity,
        tags_device=tags_device,
        family_row_offsets_device=family_row_offsets_device,
        family_devices={
            GeometryFamily.POINT: point_family,
            GeometryFamily.LINESTRING: linestring_family,
        },
        detail="created device-resident owned geometry array from pylibcudf mixed WKB point/linestring scan",
    )


def _decode_pylibcudf_linestring_like_geoarrow_column_to_owned(column, family: GeometryFamily) -> OwnedGeometryArray:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, family.value)

    if _pylibcudf_can_adopt_zero_copy(column):
        row_count = int(column.size())
        geometry_offsets_device = _pylibcudf_list_offsets_adopt(column)
        coord_count = int(cp.asnumpy(geometry_offsets_device[row_count]))
        x_all, y_all = _pylibcudf_point_xy_children(column.child(1))
        x_device = x_all[:coord_count]
        y_device = y_all[:coord_count]
        geometry_offsets_device = geometry_offsets_device[:row_count + 1]
        validity_device = cp.ones(row_count, dtype=cp.bool_)
        empty_mask_device = (geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0
        owned = _build_device_single_family_owned(
            family=family,
            validity_device=validity_device,
            x_device=x_device,
            y_device=y_device,
            geometry_offsets_device=geometry_offsets_device,
            empty_mask_device=empty_mask_device,
            detail=f"zero-copy adopted device-resident {family.value} buffers from pylibcudf GeoArrow column",
        )
        owned.device_adopted = True
        if owned.device_state is not None:
            owned.device_state._column_refs = [column]
        return owned

    validity_device = _pylibcudf_validity_mask(column)
    geometry_offsets_device, coord_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(column),
        validity_device,
    )
    x_all, y_all = _pylibcudf_point_xy_children(column.child(1))
    return _build_device_single_family_owned(
        family=family,
        validity_device=validity_device,
        x_device=x_all[coord_mask],
        y_device=y_all[coord_mask],
        geometry_offsets_device=geometry_offsets_device,
        empty_mask_device=(geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0,
        detail=f"created device-resident owned geometry array from pylibcudf GeoParquet {family.value} scan",
    )


def _decode_pylibcudf_polygon_geoarrow_column_to_owned(column) -> OwnedGeometryArray:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, "polygon")

    if _pylibcudf_can_adopt_zero_copy(column):
        row_count = int(column.size())
        geometry_offsets_device = _pylibcudf_list_offsets_adopt(column)
        geometry_offsets_device = geometry_offsets_device[:row_count + 1]
        ring_column = column.child(1)
        ring_count = int(cp.asnumpy(geometry_offsets_device[row_count]))
        ring_offsets_device = _pylibcudf_list_offsets_adopt(ring_column)
        ring_offsets_device = ring_offsets_device[:ring_count + 1]
        coord_count = int(cp.asnumpy(ring_offsets_device[ring_count]))
        x_all, y_all = _pylibcudf_point_xy_children(ring_column.child(1))
        x_device = x_all[:coord_count]
        y_device = y_all[:coord_count]
        validity_device = cp.ones(row_count, dtype=cp.bool_)
        empty_mask_device = (geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0
        owned = _build_device_single_family_owned(
            family=GeometryFamily.POLYGON,
            validity_device=validity_device,
            x_device=x_device,
            y_device=y_device,
            geometry_offsets_device=geometry_offsets_device,
            empty_mask_device=empty_mask_device,
            ring_offsets_device=ring_offsets_device,
            detail="zero-copy adopted device-resident polygon buffers from pylibcudf GeoArrow column",
        )
        owned.device_adopted = True
        if owned.device_state is not None:
            owned.device_state._column_refs = [column]
        return owned

    validity_device = _pylibcudf_validity_mask(column)
    geometry_offsets_device, ring_parent_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(column),
        validity_device,
    )
    ring_column = column.child(1)
    ring_offsets_device, coord_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(ring_column),
        ring_parent_mask,
    )
    x_all, y_all = _pylibcudf_point_xy_children(ring_column.child(1))
    return _build_device_single_family_owned(
        family=GeometryFamily.POLYGON,
        validity_device=validity_device,
        x_device=x_all[coord_mask],
        y_device=y_all[coord_mask],
        geometry_offsets_device=geometry_offsets_device,
        empty_mask_device=(geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0,
        ring_offsets_device=ring_offsets_device,
        detail="created device-resident owned geometry array from pylibcudf GeoParquet polygon scan",
    )


def _decode_pylibcudf_multilinestring_geoarrow_column_to_owned(column) -> OwnedGeometryArray:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, "multilinestring")

    if _pylibcudf_can_adopt_zero_copy(column):
        row_count = int(column.size())
        geometry_offsets_device = _pylibcudf_list_offsets_adopt(column)
        geometry_offsets_device = geometry_offsets_device[:row_count + 1]
        part_column = column.child(1)
        part_count = int(cp.asnumpy(geometry_offsets_device[row_count]))
        part_offsets_device = _pylibcudf_list_offsets_adopt(part_column)
        part_offsets_device = part_offsets_device[:part_count + 1]
        coord_count = int(cp.asnumpy(part_offsets_device[part_count]))
        x_all, y_all = _pylibcudf_point_xy_children(part_column.child(1))
        x_device = x_all[:coord_count]
        y_device = y_all[:coord_count]
        validity_device = cp.ones(row_count, dtype=cp.bool_)
        empty_mask_device = (geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0
        owned = _build_device_single_family_owned(
            family=GeometryFamily.MULTILINESTRING,
            validity_device=validity_device,
            x_device=x_device,
            y_device=y_device,
            geometry_offsets_device=geometry_offsets_device,
            empty_mask_device=empty_mask_device,
            part_offsets_device=part_offsets_device,
            detail="zero-copy adopted device-resident multilinestring buffers from pylibcudf GeoArrow column",
        )
        owned.device_adopted = True
        if owned.device_state is not None:
            owned.device_state._column_refs = [column]
        return owned

    validity_device = _pylibcudf_validity_mask(column)
    geometry_offsets_device, part_parent_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(column),
        validity_device,
    )
    part_column = column.child(1)
    part_offsets_device, coord_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(part_column),
        part_parent_mask,
    )
    x_all, y_all = _pylibcudf_point_xy_children(part_column.child(1))
    return _build_device_single_family_owned(
        family=GeometryFamily.MULTILINESTRING,
        validity_device=validity_device,
        x_device=x_all[coord_mask],
        y_device=y_all[coord_mask],
        geometry_offsets_device=geometry_offsets_device,
        empty_mask_device=(geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0,
        part_offsets_device=part_offsets_device,
        detail="created device-resident owned geometry array from pylibcudf GeoParquet multilinestring scan",
    )


def _decode_pylibcudf_multipolygon_geoarrow_column_to_owned(column) -> OwnedGeometryArray:
    import cupy as cp

    _require_pylibcudf_zero_offset(column, "multipolygon")

    if _pylibcudf_can_adopt_zero_copy(column):
        row_count = int(column.size())
        geometry_offsets_device = _pylibcudf_list_offsets_adopt(column)
        geometry_offsets_device = geometry_offsets_device[:row_count + 1]
        polygon_column = column.child(1)
        polygon_count = int(cp.asnumpy(geometry_offsets_device[row_count]))
        part_offsets_device = _pylibcudf_list_offsets_adopt(polygon_column)
        part_offsets_device = part_offsets_device[:polygon_count + 1]
        ring_column = polygon_column.child(1)
        ring_count = int(cp.asnumpy(part_offsets_device[polygon_count]))
        ring_offsets_device = _pylibcudf_list_offsets_adopt(ring_column)
        ring_offsets_device = ring_offsets_device[:ring_count + 1]
        coord_count = int(cp.asnumpy(ring_offsets_device[ring_count]))
        x_all, y_all = _pylibcudf_point_xy_children(ring_column.child(1))
        x_device = x_all[:coord_count]
        y_device = y_all[:coord_count]
        validity_device = cp.ones(row_count, dtype=cp.bool_)
        empty_mask_device = (geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0
        owned = _build_device_single_family_owned(
            family=GeometryFamily.MULTIPOLYGON,
            validity_device=validity_device,
            x_device=x_device,
            y_device=y_device,
            geometry_offsets_device=geometry_offsets_device,
            empty_mask_device=empty_mask_device,
            part_offsets_device=part_offsets_device,
            ring_offsets_device=ring_offsets_device,
            detail="zero-copy adopted device-resident multipolygon buffers from pylibcudf GeoArrow column",
        )
        owned.device_adopted = True
        if owned.device_state is not None:
            owned.device_state._column_refs = [column]
        return owned

    validity_device = _pylibcudf_validity_mask(column)
    geometry_offsets_device, polygon_parent_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(column),
        validity_device,
    )
    polygon_column = column.child(1)
    part_offsets_device, ring_parent_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(polygon_column),
        polygon_parent_mask,
    )
    ring_column = polygon_column.child(1)
    ring_offsets_device, coord_mask = _device_child_selection_mask(
        _pylibcudf_list_offsets(ring_column),
        ring_parent_mask,
    )
    x_all, y_all = _pylibcudf_point_xy_children(ring_column.child(1))
    return _build_device_single_family_owned(
        family=GeometryFamily.MULTIPOLYGON,
        validity_device=validity_device,
        x_device=x_all[coord_mask],
        y_device=y_all[coord_mask],
        geometry_offsets_device=geometry_offsets_device,
        empty_mask_device=(geometry_offsets_device[1:] - geometry_offsets_device[:-1]) == 0,
        part_offsets_device=part_offsets_device,
        ring_offsets_device=ring_offsets_device,
        detail="created device-resident owned geometry array from pylibcudf GeoParquet multipolygon scan",
    )


def _decode_pylibcudf_geoparquet_column_to_owned(column, encoding) -> OwnedGeometryArray:
    normalized_encoding = None if encoding is None else str(encoding).lower()
    if normalized_encoding == "point":
        return _decode_pylibcudf_point_geoarrow_column_to_owned(column)
    if normalized_encoding in {"linestring", "multipoint"}:
        family = GeometryFamily.LINESTRING if normalized_encoding == "linestring" else GeometryFamily.MULTIPOINT
        return _decode_pylibcudf_linestring_like_geoarrow_column_to_owned(column, family)
    if normalized_encoding == "polygon":
        return _decode_pylibcudf_polygon_geoarrow_column_to_owned(column)
    if normalized_encoding == "multilinestring":
        return _decode_pylibcudf_multilinestring_geoarrow_column_to_owned(column)
    if normalized_encoding == "multipolygon":
        return _decode_pylibcudf_multipolygon_geoarrow_column_to_owned(column)
    if normalized_encoding == "wkb":
        return _decode_pylibcudf_wkb_general_column_to_owned(column)
    raise NotImplementedError(f"pylibcudf device decode does not support GeoParquet encoding {encoding!r} yet")


def _decode_pylibcudf_geoparquet_table_to_owned(
    table,
    geo_metadata: dict[str, Any],
    *,
    column_index: int | None = None,
) -> OwnedGeometryArray:
    primary = geo_metadata["primary_column"]
    encoding = geo_metadata["columns"][primary].get("encoding")
    index = 0 if column_index is None else int(column_index)
    column = table.columns()[index]
    return _decode_pylibcudf_geoparquet_column_to_owned(column, encoding)

def _is_pylibcudf_table(table) -> bool:
    return table.__class__.__module__.startswith("pylibcudf")
