from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial.geometry.buffers import GeometryFamily

if TYPE_CHECKING:
    from vibespatial.geometry.owned import FamilyGeometryBuffer, OwnedGeometryArray


def _empty_geometry_for_family(family: GeometryFamily):
    return {
        GeometryFamily.POINT: Point(),
        GeometryFamily.LINESTRING: LineString(),
        GeometryFamily.POLYGON: Polygon(),
        GeometryFamily.MULTIPOINT: MultiPoint([]),
        GeometryFamily.MULTILINESTRING: MultiLineString([]),
        GeometryFamily.MULTIPOLYGON: MultiPolygon([]),
    }[family]


def _xy_view(buffer: FamilyGeometryBuffer, coord_count: int) -> np.ndarray:
    return np.column_stack((buffer.x[:coord_count], buffer.y[:coord_count]))


def materialize_family_row(buffer: FamilyGeometryBuffer, row_index: int):
    if bool(buffer.empty_mask[row_index]):
        return _empty_geometry_for_family(buffer.family)

    if buffer.family in {
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    }:
        start = int(buffer.geometry_offsets[row_index])
        end = int(buffer.geometry_offsets[row_index + 1])
        coords = list(zip(buffer.x[start:end], buffer.y[start:end], strict=True))
        if buffer.family is GeometryFamily.POINT:
            x, y = coords[0]
            return Point(float(x), float(y))
        if buffer.family is GeometryFamily.LINESTRING:
            return LineString(coords)
        return MultiPoint(coords)

    if buffer.family is GeometryFamily.POLYGON:
        ring_start = int(buffer.geometry_offsets[row_index])
        ring_end = int(buffer.geometry_offsets[row_index + 1])
        rings = []
        for ring_index in range(ring_start, ring_end):
            coord_start = int(buffer.ring_offsets[ring_index])
            coord_end = int(buffer.ring_offsets[ring_index + 1])
            rings.append(
                list(
                    zip(
                        buffer.x[coord_start:coord_end],
                        buffer.y[coord_start:coord_end],
                        strict=True,
                    )
                )
            )
        valid_holes = [ring for ring in rings[1:] if len(ring) >= 3]
        return Polygon(rings[0], holes=valid_holes)

    if buffer.family is GeometryFamily.MULTILINESTRING:
        part_start = int(buffer.geometry_offsets[row_index])
        part_end = int(buffer.geometry_offsets[row_index + 1])
        parts = []
        for part_index in range(part_start, part_end):
            coord_start = int(buffer.part_offsets[part_index])
            coord_end = int(buffer.part_offsets[part_index + 1])
            parts.append(
                list(
                    zip(
                        buffer.x[coord_start:coord_end],
                        buffer.y[coord_start:coord_end],
                        strict=True,
                    )
                )
            )
        return MultiLineString(parts)

    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_start = int(buffer.geometry_offsets[row_index])
        polygon_end = int(buffer.geometry_offsets[row_index + 1])
        polygons = []
        for polygon_index in range(polygon_start, polygon_end):
            ring_start = int(buffer.part_offsets[polygon_index])
            ring_end = int(buffer.part_offsets[polygon_index + 1])
            rings = []
            for ring_index in range(ring_start, ring_end):
                coord_start = int(buffer.ring_offsets[ring_index])
                coord_end = int(buffer.ring_offsets[ring_index + 1])
                rings.append(
                    list(
                        zip(
                            buffer.x[coord_start:coord_end],
                            buffer.y[coord_start:coord_end],
                            strict=True,
                        )
                    )
                )
            valid_holes = [ring for ring in rings[1:] if len(ring) >= 3]
            polygons.append(Polygon(rings[0], holes=valid_holes))
        return MultiPolygon(polygons)

    raise NotImplementedError(f"unsupported geometry family: {buffer.family.value}")


def _materialize_point_family(buffer: FamilyGeometryBuffer, family_rows: np.ndarray) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty = ~empty_mask
    if nonempty.any():
        coord_rows = np.asarray(buffer.geometry_offsets[family_rows[nonempty]], dtype=np.int64)
        result[nonempty] = np.asarray(
            shapely.points(buffer.x[coord_rows], buffer.y[coord_rows]),
            dtype=object,
        )
    return result


def _materialize_full_point_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    coord_rows = np.asarray(buffer.geometry_offsets[nonempty_positions], dtype=np.int64)
    result[nonempty_positions] = np.asarray(
        shapely.points(buffer.x[coord_rows], buffer.y[coord_rows]),
        dtype=object,
    )
    return result


def _materialize_linestring_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result

    flat_xy: list[list[float]] = []
    line_indices: list[int] = []
    for output_position, local_position in enumerate(nonempty_positions):
        family_row = int(family_rows[local_position])
        start = int(buffer.geometry_offsets[family_row])
        end = int(buffer.geometry_offsets[family_row + 1])
        for coord_index in range(start, end):
            flat_xy.append([float(buffer.x[coord_index]), float(buffer.y[coord_index])])
            line_indices.append(output_position)
    result[nonempty_positions] = np.asarray(
        shapely.linestrings(
            np.asarray(flat_xy, dtype=np.float64),
            indices=np.asarray(line_indices, dtype=np.int32),
        ),
        dtype=object,
    )
    return result


def _materialize_full_linestring_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    line_sizes = np.diff(buffer.geometry_offsets)[nonempty_positions]
    coord_count = int(buffer.geometry_offsets[-1])
    result[nonempty_positions] = np.asarray(
        shapely.linestrings(
            _xy_view(buffer, coord_count),
            indices=np.repeat(np.arange(nonempty_positions.size, dtype=np.int32), line_sizes),
        ),
        dtype=object,
    )
    return result


def _materialize_multipoint_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result

    flat_xy: list[list[float]] = []
    geom_indices: list[int] = []
    for output_position, local_position in enumerate(nonempty_positions):
        family_row = int(family_rows[local_position])
        start = int(buffer.geometry_offsets[family_row])
        end = int(buffer.geometry_offsets[family_row + 1])
        for coord_index in range(start, end):
            flat_xy.append([float(buffer.x[coord_index]), float(buffer.y[coord_index])])
            geom_indices.append(output_position)
    result[nonempty_positions] = np.asarray(
        shapely.multipoints(
            np.asarray(flat_xy, dtype=np.float64),
            indices=np.asarray(geom_indices, dtype=np.int32),
        ),
        dtype=object,
    )
    return result


def _materialize_full_multipoint_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    point_sizes = np.diff(buffer.geometry_offsets)[nonempty_positions]
    coord_count = int(buffer.geometry_offsets[-1])
    result[nonempty_positions] = np.asarray(
        shapely.multipoints(
            _xy_view(buffer, coord_count),
            indices=np.repeat(np.arange(nonempty_positions.size, dtype=np.int32), point_sizes),
        ),
        dtype=object,
    )
    return result


def _materialize_polygon_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result

    flat_xy: list[list[float]] = []
    ring_indices: list[int] = []
    polygon_indices: list[int] = []
    ring_position = 0
    for output_position, local_position in enumerate(nonempty_positions):
        family_row = int(family_rows[local_position])
        ring_start = int(buffer.geometry_offsets[family_row])
        ring_end = int(buffer.geometry_offsets[family_row + 1])
        for ring_index in range(ring_start, ring_end):
            coord_start = int(buffer.ring_offsets[ring_index])
            coord_end = int(buffer.ring_offsets[ring_index + 1])
            for coord_index in range(coord_start, coord_end):
                flat_xy.append([float(buffer.x[coord_index]), float(buffer.y[coord_index])])
                ring_indices.append(ring_position)
            polygon_indices.append(output_position)
            ring_position += 1
    rings = shapely.linearrings(
        np.asarray(flat_xy, dtype=np.float64),
        indices=np.asarray(ring_indices, dtype=np.int32),
    )
    result[nonempty_positions] = np.asarray(
        shapely.polygons(rings, indices=np.asarray(polygon_indices, dtype=np.int32)),
        dtype=object,
    )
    return result


def _materialize_full_polygon_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    ring_counts = np.diff(buffer.geometry_offsets)[nonempty_positions]
    coord_count = int(buffer.ring_offsets[-1])
    rings = shapely.linearrings(
        _xy_view(buffer, coord_count),
        indices=np.repeat(
            np.arange(buffer.ring_offsets.size - 1, dtype=np.int32),
            np.diff(buffer.ring_offsets),
        ),
    )
    result[nonempty_positions] = np.asarray(
        shapely.polygons(
            rings,
            indices=np.repeat(np.arange(nonempty_positions.size, dtype=np.int32), ring_counts),
        ),
        dtype=object,
    )
    return result


def _materialize_multilinestring_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result

    flat_xy: list[list[float]] = []
    line_indices: list[int] = []
    multiline_indices: list[int] = []
    line_position = 0
    for output_position, local_position in enumerate(nonempty_positions):
        family_row = int(family_rows[local_position])
        part_start = int(buffer.geometry_offsets[family_row])
        part_end = int(buffer.geometry_offsets[family_row + 1])
        for part_index in range(part_start, part_end):
            coord_start = int(buffer.part_offsets[part_index])
            coord_end = int(buffer.part_offsets[part_index + 1])
            for coord_index in range(coord_start, coord_end):
                flat_xy.append([float(buffer.x[coord_index]), float(buffer.y[coord_index])])
                line_indices.append(line_position)
            multiline_indices.append(output_position)
            line_position += 1
    lines = shapely.linestrings(
        np.asarray(flat_xy, dtype=np.float64),
        indices=np.asarray(line_indices, dtype=np.int32),
    )
    result[nonempty_positions] = np.asarray(
        shapely.multilinestrings(lines, indices=np.asarray(multiline_indices, dtype=np.int32)),
        dtype=object,
    )
    return result


def _materialize_full_multilinestring_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    coord_count = int(buffer.part_offsets[-1])
    lines = shapely.linestrings(
        _xy_view(buffer, coord_count),
        indices=np.repeat(
            np.arange(buffer.part_offsets.size - 1, dtype=np.int32),
            np.diff(buffer.part_offsets),
        ),
    )
    line_counts = np.diff(buffer.geometry_offsets)[nonempty_positions]
    result[nonempty_positions] = np.asarray(
        shapely.multilinestrings(
            lines,
            indices=np.repeat(np.arange(nonempty_positions.size, dtype=np.int32), line_counts),
        ),
        dtype=object,
    )
    return result


def _materialize_multipolygon_family(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> np.ndarray:
    result = np.empty(family_rows.size, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask[family_rows], dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result

    flat_xy: list[list[float]] = []
    ring_indices: list[int] = []
    polygon_indices: list[int] = []
    multipolygon_indices: list[int] = []
    ring_position = 0
    polygon_position = 0
    for output_position, local_position in enumerate(nonempty_positions):
        family_row = int(family_rows[local_position])
        polygon_start = int(buffer.geometry_offsets[family_row])
        polygon_end = int(buffer.geometry_offsets[family_row + 1])
        for polygon_index in range(polygon_start, polygon_end):
            part_start = int(buffer.part_offsets[polygon_index])
            part_end = int(buffer.part_offsets[polygon_index + 1])
            for ring_index in range(part_start, part_end):
                coord_start = int(buffer.ring_offsets[ring_index])
                coord_end = int(buffer.ring_offsets[ring_index + 1])
                for coord_index in range(coord_start, coord_end):
                    flat_xy.append([float(buffer.x[coord_index]), float(buffer.y[coord_index])])
                    ring_indices.append(ring_position)
                polygon_indices.append(polygon_position)
                ring_position += 1
            multipolygon_indices.append(output_position)
            polygon_position += 1
    rings = shapely.linearrings(
        np.asarray(flat_xy, dtype=np.float64),
        indices=np.asarray(ring_indices, dtype=np.int32),
    )
    polygons = shapely.polygons(rings, indices=np.asarray(polygon_indices, dtype=np.int32))
    result[nonempty_positions] = np.asarray(
        shapely.multipolygons(polygons, indices=np.asarray(multipolygon_indices, dtype=np.int32)),
        dtype=object,
    )
    return result


def _materialize_full_multipolygon_family(buffer: FamilyGeometryBuffer) -> np.ndarray:
    result = np.empty(buffer.row_count, dtype=object)
    empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
    if empty_mask.any():
        result[empty_mask] = _empty_geometry_for_family(buffer.family)
    nonempty_positions = np.flatnonzero(~empty_mask)
    if nonempty_positions.size == 0:
        return result
    coord_count = int(buffer.ring_offsets[-1])
    rings = shapely.linearrings(
        _xy_view(buffer, coord_count),
        indices=np.repeat(
            np.arange(buffer.ring_offsets.size - 1, dtype=np.int32),
            np.diff(buffer.ring_offsets),
        ),
    )
    polygons = shapely.polygons(
        rings,
        indices=np.repeat(
            np.arange(buffer.part_offsets.size - 1, dtype=np.int32),
            np.diff(buffer.part_offsets),
        ),
    )
    polygon_counts = np.diff(buffer.geometry_offsets)[nonempty_positions]
    result[nonempty_positions] = np.asarray(
        shapely.multipolygons(
            polygons,
            indices=np.repeat(np.arange(nonempty_positions.size, dtype=np.int32), polygon_counts),
        ),
        dtype=object,
    )
    return result


def _is_full_family_selection(buffer: FamilyGeometryBuffer, family_rows: np.ndarray) -> bool:
    return (
        family_rows.size == buffer.row_count
        and bool(np.array_equal(family_rows, np.arange(buffer.row_count, dtype=np.int32)))
    )


def _materialize_family_rows(buffer: FamilyGeometryBuffer, family_rows: np.ndarray) -> np.ndarray:
    family_rows = np.asarray(family_rows, dtype=np.int32)
    if family_rows.size == 0:
        return np.empty(0, dtype=object)
    if _is_full_family_selection(buffer, family_rows):
        if buffer.family is GeometryFamily.POINT:
            return _materialize_full_point_family(buffer)
        if buffer.family is GeometryFamily.LINESTRING:
            return _materialize_full_linestring_family(buffer)
        if buffer.family is GeometryFamily.POLYGON:
            return _materialize_full_polygon_family(buffer)
        if buffer.family is GeometryFamily.MULTIPOINT:
            return _materialize_full_multipoint_family(buffer)
        if buffer.family is GeometryFamily.MULTILINESTRING:
            return _materialize_full_multilinestring_family(buffer)
        if buffer.family is GeometryFamily.MULTIPOLYGON:
            return _materialize_full_multipolygon_family(buffer)
    if buffer.family is GeometryFamily.POINT:
        return _materialize_point_family(buffer, family_rows)
    if buffer.family is GeometryFamily.LINESTRING:
        return _materialize_linestring_family(buffer, family_rows)
    if buffer.family is GeometryFamily.POLYGON:
        return _materialize_polygon_family(buffer, family_rows)
    if buffer.family is GeometryFamily.MULTIPOINT:
        return _materialize_multipoint_family(buffer, family_rows)
    if buffer.family is GeometryFamily.MULTILINESTRING:
        return _materialize_multilinestring_family(buffer, family_rows)
    if buffer.family is GeometryFamily.MULTIPOLYGON:
        return _materialize_multipolygon_family(buffer, family_rows)
    raise NotImplementedError(f"unsupported geometry family: {buffer.family.value}")


def owned_to_shapely(
    owned: OwnedGeometryArray,
    *,
    rows: np.ndarray | None = None,
    record_event: bool = True,
) -> np.ndarray:
    from vibespatial.geometry import owned as owned_module

    owned._ensure_host_state()
    if rows is None:
        rows = np.arange(owned.row_count, dtype=np.intp)
        detail = "materialized shapely geometries via explicit host bridge"
    else:
        rows = np.asarray(rows, dtype=np.intp)
        detail = f"materialized {rows.size} shapely geometries via explicit host bridge"
    if record_event:
        owned._record(owned_module.DiagnosticKind.MATERIALIZATION, detail, visible=True)

    result = np.empty(rows.size, dtype=object)
    if rows.size == 0:
        return result

    validity = np.asarray(owned.validity[rows], dtype=bool)
    result[~validity] = None
    if not validity.any():
        return result

    tags = np.asarray(owned.tags[rows], dtype=np.int8)
    family_row_offsets = np.asarray(owned.family_row_offsets[rows], dtype=np.int32)
    for family, buffer in owned.families.items():
        family_tag = owned_module.FAMILY_TAGS[family]
        family_mask = validity & (tags == family_tag)
        if not family_mask.any():
            continue
        result[np.flatnonzero(family_mask)] = _materialize_family_rows(
            buffer,
            family_row_offsets[family_mask],
        )
    return result
