from __future__ import annotations

from vibespatial import (
    GEOMETRY_BUFFER_SCHEMAS,
    GeometryFamily,
    PrecisionMode,
    get_geometry_buffer_schema,
)


def test_all_primary_geometry_families_have_concrete_schema() -> None:
    assert set(GEOMETRY_BUFFER_SCHEMAS) == {
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOINT,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    }


def test_all_schemas_use_canonical_fp64_separated_coordinates() -> None:
    for schema in GEOMETRY_BUFFER_SCHEMAS.values():
        assert schema.coord_precision is PrecisionMode.FP64
        assert schema.coord_layout == "separated-xy"
        assert schema.x.dtype == "float64"
        assert schema.y.dtype == "float64"


def test_nulls_are_validity_bitmaps_and_empties_use_zero_spans() -> None:
    for schema in GEOMETRY_BUFFER_SCHEMAS.values():
        assert schema.validity.dtype == "bitmask"
        assert schema.empty_via_zero_span is True


def test_point_schema_uses_row_to_coordinate_offsets_for_empty_points() -> None:
    point = get_geometry_buffer_schema(GeometryFamily.POINT)

    assert point.geometry_offsets is not None
    assert point.geometry_offsets.level == "row->coordinate"


def test_polygon_schema_uses_row_to_ring_and_ring_to_coordinate_offsets() -> None:
    polygon = get_geometry_buffer_schema(GeometryFamily.POLYGON)

    assert polygon.geometry_offsets is not None
    assert polygon.geometry_offsets.level == "row->ring"
    assert polygon.part_offsets is None
    assert polygon.ring_offsets is not None
    assert polygon.ring_offsets.level == "ring->coordinate"


def test_multipolygon_schema_uses_three_level_hierarchy() -> None:
    multipolygon = get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON)

    assert multipolygon.geometry_offsets is not None
    assert multipolygon.part_offsets is not None
    assert multipolygon.ring_offsets is not None
    assert multipolygon.geometry_offsets.level == "row->polygon"
    assert multipolygon.part_offsets.level == "polygon->ring"
    assert multipolygon.ring_offsets.level == "ring->coordinate"
