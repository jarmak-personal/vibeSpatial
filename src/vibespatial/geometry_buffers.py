from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vibespatial.precision import PrecisionMode


class GeometryFamily(StrEnum):
    POINT = "point"
    LINESTRING = "linestring"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"
    MULTILINESTRING = "multilinestring"
    MULTIPOLYGON = "multipolygon"


class BufferKind(StrEnum):
    VALIDITY = "validity"
    TAG = "tag"
    OFFSET = "offset"
    COORDINATE = "coordinate"
    BOUNDS = "bounds"


@dataclass(frozen=True)
class BufferSpec:
    name: str
    kind: BufferKind
    dtype: str
    level: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class GeometryBufferSchema:
    family: GeometryFamily
    coord_precision: PrecisionMode
    coord_layout: str
    validity: BufferSpec
    x: BufferSpec
    y: BufferSpec
    geometry_offsets: BufferSpec | None = None
    part_offsets: BufferSpec | None = None
    ring_offsets: BufferSpec | None = None
    bounds: BufferSpec | None = None
    supports_mixed_parent: bool = True
    empty_via_zero_span: bool = True
    notes: tuple[str, ...] = ()

    @property
    def coordinate_buffers(self) -> tuple[BufferSpec, BufferSpec]:
        return (self.x, self.y)

    @property
    def offset_buffers(self) -> tuple[BufferSpec, ...]:
        return tuple(
            buffer
            for buffer in (self.geometry_offsets, self.part_offsets, self.ring_offsets)
            if buffer is not None
        )


VALIDITY_SPEC = BufferSpec(
    name="validity",
    kind=BufferKind.VALIDITY,
    dtype="bitmask",
    level="row",
    description="Arrow-style validity bitmap; nulls are invalid rows and never share representation with empties.",
)

X_SPEC = BufferSpec(
    name="x",
    kind=BufferKind.COORDINATE,
    dtype="float64",
    level="coordinate",
    description="Separated authoritative x coordinates stored in canonical fp64 precision.",
)

Y_SPEC = BufferSpec(
    name="y",
    kind=BufferKind.COORDINATE,
    dtype="float64",
    level="coordinate",
    description="Separated authoritative y coordinates stored in canonical fp64 precision.",
)

BOUNDS_SPEC = BufferSpec(
    name="bounds",
    kind=BufferKind.BOUNDS,
    dtype="float64[4]",
    level="row",
    required=False,
    description="Optional cached minx/miny/maxx/maxy bounds; derivable and invalidated independently of geometry payload.",
)

POINT_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.POINT,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->coordinate",
        description="Prefix offsets into x/y coordinate buffers; valid points use one coordinate and empty points use a zero-length span.",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "Each valid non-empty row owns exactly one x/y pair.",
        "Empty points are represented by a valid row with a zero-length coordinate span.",
    ),
)

LINESTRING_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.LINESTRING,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->coordinate",
        description="Prefix offsets into x/y coordinate buffers; row i uses coordinates[geometry_offsets[i]:geometry_offsets[i+1]].",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "Single-part lines do not need a separate part-offset buffer.",
        "Empties are valid rows with geometry_offsets[i] == geometry_offsets[i+1].",
    ),
)

POLYGON_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.POLYGON,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->ring",
        description="Prefix offsets into ring_offsets; row i uses rings[geometry_offsets[i]:geometry_offsets[i+1]].",
    ),
    ring_offsets=BufferSpec(
        name="ring_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="ring->coordinate",
        description="Prefix offsets into x/y coordinate buffers; ring j uses coordinates[ring_offsets[j]:ring_offsets[j+1]].",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "Exterior and interior rings share one ring_offsets buffer and preserve source order.",
        "Empties are valid rows with zero rings.",
    ),
)

MULTIPOINT_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.MULTIPOINT,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->coordinate",
        description="Prefix offsets into x/y coordinate buffers; row i uses coordinates[geometry_offsets[i]:geometry_offsets[i+1]].",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "MultiPoint shares the same payload shape as LineString but keeps different semantics and kernel dispatch.",
        "Empties are valid rows with zero coordinates.",
    ),
)

MULTILINESTRING_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.MULTILINESTRING,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->part",
        description="Prefix offsets into part_offsets; row i uses parts[geometry_offsets[i]:geometry_offsets[i+1]].",
    ),
    part_offsets=BufferSpec(
        name="part_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="part->coordinate",
        description="Prefix offsets into x/y coordinate buffers; part j uses coordinates[part_offsets[j]:part_offsets[j+1]].",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "Each part is a LineString payload slice.",
        "Empties are valid rows with zero parts.",
    ),
)

MULTIPOLYGON_SCHEMA = GeometryBufferSchema(
    family=GeometryFamily.MULTIPOLYGON,
    coord_precision=PrecisionMode.FP64,
    coord_layout="separated-xy",
    validity=VALIDITY_SPEC,
    x=X_SPEC,
    y=Y_SPEC,
    geometry_offsets=BufferSpec(
        name="geometry_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="row->polygon",
        description="Prefix offsets into part_offsets; row i uses polygons[geometry_offsets[i]:geometry_offsets[i+1]].",
    ),
    part_offsets=BufferSpec(
        name="part_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="polygon->ring",
        description="Prefix offsets into ring_offsets; polygon j uses rings[part_offsets[j]:part_offsets[j+1]].",
    ),
    ring_offsets=BufferSpec(
        name="ring_offsets",
        kind=BufferKind.OFFSET,
        dtype="int32",
        level="ring->coordinate",
        description="Prefix offsets into x/y coordinate buffers; ring k uses coordinates[ring_offsets[k]:ring_offsets[k+1]].",
    ),
    bounds=BOUNDS_SPEC,
    notes=(
        "Multipart polygon layout is row -> polygon part -> ring -> coordinate.",
        "Empties are valid rows with zero polygon parts.",
    ),
)

GEOMETRY_BUFFER_SCHEMAS: dict[GeometryFamily, GeometryBufferSchema] = {
    schema.family: schema
    for schema in (
        POINT_SCHEMA,
        LINESTRING_SCHEMA,
        POLYGON_SCHEMA,
        MULTIPOINT_SCHEMA,
        MULTILINESTRING_SCHEMA,
        MULTIPOLYGON_SCHEMA,
    )
}


def get_geometry_buffer_schema(family: GeometryFamily | str) -> GeometryBufferSchema:
    normalized = family if isinstance(family, GeometryFamily) else GeometryFamily(family)
    return GEOMETRY_BUFFER_SCHEMAS[normalized]
