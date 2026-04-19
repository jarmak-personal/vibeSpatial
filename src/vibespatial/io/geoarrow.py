from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    BufferSharingMode,
    DiagnosticKind,
    FamilyGeometryBuffer,
    MixedGeoArrowView,
    OwnedGeometryArray,
    from_geoarrow,
    from_shapely_geometries,
    from_wkb,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency, TransferTrigger

from .support import IOFormat, IOOperation, IOPathKind, plan_io_support
from .wkb import (
    _encode_native_wkb,
    _encode_owned_geoarrow_column_device,
    _encode_owned_wkb_array,
    decode_wkb_arrow_array_owned,
    decode_wkb_owned,
    plan_wkb_partition,
)

_TAG_TO_GEOM_TYPE_NAME = {
    FAMILY_TAGS[GeometryFamily.POINT]: "Point",
    FAMILY_TAGS[GeometryFamily.LINESTRING]: "LineString",
    FAMILY_TAGS[GeometryFamily.POLYGON]: "Polygon",
    FAMILY_TAGS[GeometryFamily.MULTIPOINT]: "MultiPoint",
    FAMILY_TAGS[GeometryFamily.MULTILINESTRING]: "MultiLineString",
    FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]: "MultiPolygon",
}


@dataclass(frozen=True)
class GeoArrowCodecPlan:
    operation: IOOperation
    selected_path: IOPathKind
    canonical_gpu: bool
    device_codec_available: bool
    zero_copy_adoption: bool
    lazy_materialization: bool
    reason: str

@dataclass(frozen=True)
class GeoArrowBridgeBenchmark:
    operation: str
    sharing: str
    geometry_type: str
    rows: int
    elapsed_seconds: float
    shares_memory: bool

@dataclass(frozen=True)
class NativeGeometryBenchmark:
    operation: str
    geometry_type: str
    implementation: str
    rows: int
    elapsed_seconds: float
    rows_per_second: float

@dataclass(frozen=True)
class WKBBridgeBenchmark:
    operation: str
    geometry_type: str
    implementation: str
    rows: int
    fallback_rows: int
    elapsed_seconds: float
    rows_per_second: float


class _GeoArrowNativeCompatibilityRoute(RuntimeError):
    """Signal that GeoArrow import should use the explicit compatibility adapter."""


def plan_geoarrow_codec(operation: IOOperation | str) -> GeoArrowCodecPlan:
    normalized = operation if isinstance(operation, IOOperation) else IOOperation(operation)
    plan = plan_io_support(IOFormat.GEOARROW, normalized)
    return GeoArrowCodecPlan(
        operation=normalized,
        selected_path=plan.selected_path,
        canonical_gpu=plan.canonical_gpu,
        device_codec_available=False,
        zero_copy_adoption=True,
        lazy_materialization=True,
        reason=(
            "GeoArrow is the canonical GPU geometry interchange; aligned buffers should "
            "be adopted zero-copy and only materialize host geometry objects on demand "
            "until the device-side codec lands."
        ),
    )

def _geoarrow_field_metadata(*, extension_name: str, crs: Any | None = None) -> dict[bytes, bytes]:
    import json

    metadata = {
        b"ARROW:extension:name": extension_name.encode(),
        b"ARROW:extension:metadata": b"{}",
    }
    if crs is not None:
        metadata[b"ARROW:extension:metadata"] = json.dumps({"crs": crs.to_json_dict()}).encode()
    return metadata

def encode_owned_geoarrow(array: OwnedGeometryArray) -> MixedGeoArrowView:
    record_dispatch_event(
        surface="vibespatial.io.geoarrow",
        operation="encode",
        implementation="owned_geoarrow_compat_boundary",
        reason=(
            "explicit GeoArrow compatibility export boundary: expose an aligned "
            "host-visible GeoArrow view from owned buffers"
        ),
        selected=ExecutionMode.CPU,
    )
    return array.to_geoarrow(sharing=BufferSharingMode.SHARE)


def decode_owned_geoarrow(view: MixedGeoArrowView) -> OwnedGeometryArray:
    record_dispatch_event(
        surface="vibespatial.io.geoarrow",
        operation="decode",
        implementation="owned_geoarrow_compat_boundary",
        reason=(
            "explicit GeoArrow compatibility import boundary: adopt aligned "
            "host-visible GeoArrow buffers into the owned model"
        ),
        selected=ExecutionMode.CPU,
    )
    return from_geoarrow(view, sharing=BufferSharingMode.AUTO)

def _full_offsets_from_local(
    array: OwnedGeometryArray,
    buffer: FamilyGeometryBuffer,
) -> np.ndarray:
    offsets = np.zeros(array.row_count + 1, dtype=np.int32)
    for row_index in range(array.row_count):
        if not bool(array.validity[row_index]):
            offsets[row_index + 1] = offsets[row_index]
            continue
        local = int(array.family_row_offsets[row_index])
        length = int(buffer.geometry_offsets[local + 1] - buffer.geometry_offsets[local])
        offsets[row_index + 1] = offsets[row_index] + length
    return offsets

def _encode_point_family(buffer: FamilyGeometryBuffer, array: OwnedGeometryArray, *, field_name: str, crs: Any | None, interleaved: bool):
    import pyarrow as pa

    x_full = np.full(array.row_count, np.nan, dtype=np.float64)
    y_full = np.full(array.row_count, np.nan, dtype=np.float64)
    valid_rows = np.flatnonzero(array.validity)
    locals_ = array.family_row_offsets[valid_rows]
    non_empty_rows = ~buffer.empty_mask[locals_]
    if bool(non_empty_rows.any()):
        coord_indices = buffer.geometry_offsets[locals_[non_empty_rows]]
        x_full[valid_rows[non_empty_rows]] = buffer.x[coord_indices]
        y_full[valid_rows[non_empty_rows]] = buffer.y[coord_indices]
    mask = None if bool(array.validity.all()) else pa.array(~array.validity, type=pa.bool_())
    if interleaved:
        point_type = pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)
        values = pa.array(np.column_stack([x_full, y_full]).ravel(), type=pa.float64())
        geom_arr = pa.FixedSizeListArray.from_arrays(values, type=point_type, mask=mask)
    else:
        geom_arr = pa.StructArray.from_arrays(
            [pa.array(x_full), pa.array(y_full)],
            fields=[pa.field("x", pa.float64(), nullable=False), pa.field("y", pa.float64(), nullable=False)],
            mask=mask,
        )
    field = pa.field(
        field_name,
        geom_arr.type,
        nullable=True,
        metadata=_geoarrow_field_metadata(extension_name="geoarrow.point", crs=crs),
    )
    return field, geom_arr

def _encode_list_family(
    *,
    extension_name: str,
    buffer: FamilyGeometryBuffer,
    array: OwnedGeometryArray,
    field_name: str,
    crs: Any | None,
    interleaved: bool,
    nested_kind: str,
):
    import pyarrow as pa

    from vibespatial.api.io._geoarrow import (
        _linestring_type,
        _multilinestring_type,
        _multipoint_type,
        _multipolygon_type,
        _polygon_type,
    )

    mask = None if bool(array.validity.all()) else pa.array(~array.validity, type=pa.bool_())
    if interleaved:
        point_type = pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)
        point_values = pa.FixedSizeListArray.from_arrays(
            pa.array(np.column_stack([buffer.x, buffer.y]).ravel(), type=pa.float64()),
            type=point_type,
        )
    else:
        point_values = pa.StructArray.from_arrays(
            [pa.array(buffer.x), pa.array(buffer.y)],
            fields=[pa.field("x", pa.float64(), nullable=False), pa.field("y", pa.float64(), nullable=False)],
        )

    if nested_kind == "linestring":
        geom_offsets = _full_offsets_from_local(array, buffer)
        geom_arr = pa.ListArray.from_arrays(pa.array(geom_offsets), point_values, type=_linestring_type(point_values.type), mask=mask)
    elif nested_kind == "polygon":
        geom_offsets = _full_offsets_from_local(array, buffer)
        rings = pa.ListArray.from_arrays(pa.array(buffer.ring_offsets), point_values)
        geom_arr = pa.ListArray.from_arrays(pa.array(geom_offsets), rings, mask=mask).cast(_polygon_type(point_values.type))
    elif nested_kind == "multipoint":
        geom_offsets = _full_offsets_from_local(array, buffer)
        geom_arr = pa.ListArray.from_arrays(pa.array(geom_offsets), point_values, type=_multipoint_type(point_values.type), mask=mask)
    elif nested_kind == "multilinestring":
        geom_offsets = _full_offsets_from_local(array, buffer)
        parts = pa.ListArray.from_arrays(pa.array(buffer.part_offsets), point_values)
        geom_arr = pa.ListArray.from_arrays(pa.array(geom_offsets), parts, mask=mask).cast(_multilinestring_type(point_values.type))
    elif nested_kind == "multipolygon":
        geom_offsets = _full_offsets_from_local(array, buffer)
        rings = pa.ListArray.from_arrays(pa.array(buffer.ring_offsets), point_values)
        polygons = pa.ListArray.from_arrays(pa.array(buffer.part_offsets), rings)
        geom_arr = pa.ListArray.from_arrays(pa.array(geom_offsets), polygons, mask=mask).cast(_multipolygon_type(point_values.type))
    else:
        raise ValueError(f"Unsupported nested_kind: {nested_kind}")
    field = pa.field(
        field_name,
        geom_arr.type,
        nullable=True,
        metadata=_geoarrow_field_metadata(extension_name=extension_name, crs=crs),
    )
    return field, geom_arr

def encode_owned_geoarrow_array(
    array: OwnedGeometryArray,
    *,
    field_name: str = "geometry",
    crs: Any | None = None,
    interleaved: bool = True,
):
    validity, tags, family_row_offsets, families = _authoritative_geoarrow_host_view(array)
    family, requires_promotion = _geoarrow_export_family_from_tags(validity, tags)
    if requires_promotion:
        buffer, view = _promote_supported_geoarrow_mix(
            export_family=family,
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=families,
        )
    else:
        buffer = families[family]
        view = SimpleNamespace(
            row_count=int(validity.size),
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
        )
    if family.value == "point":
        return _encode_point_family(buffer, view, field_name=field_name, crs=crs, interleaved=interleaved)
    mapping = {
        "linestring": ("geoarrow.linestring", "linestring"),
        "polygon": ("geoarrow.polygon", "polygon"),
        "multipoint": ("geoarrow.multipoint", "multipoint"),
        "multilinestring": ("geoarrow.multilinestring", "multilinestring"),
        "multipolygon": ("geoarrow.multipolygon", "multipolygon"),
    }
    extension_name, nested_kind = mapping[family.value]
    return _encode_list_family(
        extension_name=extension_name,
        buffer=buffer,
        array=view,
        field_name=field_name,
        crs=crs,
        interleaved=interleaved,
        nested_kind=nested_kind,
    )

def _owned_geoarrow_fast_path_reason(series, *, include_z: bool | None) -> str | None:
    import shapely

    if include_z is True:
        return "requested z-dimension output requires upstream GeoArrow constructor semantics"

    # DeviceGeometryArray: check eligibility from owned buffers, no Shapely.
    arr = series.array
    if isinstance(arr, DeviceGeometryArray):
        return _owned_geoarrow_fast_path_reason_from_owned(arr.to_owned(), include_z=include_z)

    values = np.asarray(arr)
    if len(values) == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    missing = shapely.is_missing(values)
    if bool(missing.all()):
        return "all-missing geometry column requires upstream GeoArrow constructor semantics"
    present = values[~missing]
    if bool(shapely.has_z(present).any()):
        return "3D geometry rows require upstream GeoArrow constructor semantics"
    try:
        _geoarrow_export_family_from_family_set(_owned_geoarrow_family_set(arr.to_owned()))
    except ValueError as exc:
        return str(exc)
    return None


def _owned_geoarrow_fast_path_reason_from_owned(
    owned: OwnedGeometryArray,
    *,
    include_z: bool | None,
) -> str | None:
    if include_z is True:
        return "requested z-dimension output requires upstream GeoArrow constructor semantics"
    if owned.row_count == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    if not bool(owned.validity.any()):
        return "all-missing geometry column requires upstream GeoArrow constructor semantics"
    try:
        _geoarrow_export_family_from_family_set(_owned_geoarrow_family_set(owned))
    except ValueError as exc:
        return str(exc)
    return None


def _authoritative_geoarrow_host_view(
    array: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[GeometryFamily, FamilyGeometryBuffer]]:
    if array.device_state is None:
        return array.validity, array.tags, array.family_row_offsets, array.families

    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    state = array.device_state
    validity = np.asarray(runtime.copy_device_to_host(state.validity), dtype=np.bool_)
    tags = np.asarray(runtime.copy_device_to_host(state.tags), dtype=np.int8)
    family_row_offsets = np.asarray(
        runtime.copy_device_to_host(state.family_row_offsets),
        dtype=np.int32,
    )
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, buffer in array.families.items():
        device_buffer = state.families.get(family)
        if device_buffer is None:
            families[family] = buffer
            continue
        families[family] = FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=np.asarray(runtime.copy_device_to_host(device_buffer.x), dtype=np.float64),
            y=np.asarray(runtime.copy_device_to_host(device_buffer.y), dtype=np.float64),
            geometry_offsets=np.asarray(
                runtime.copy_device_to_host(device_buffer.geometry_offsets),
                dtype=np.int32,
            ),
            empty_mask=np.asarray(
                runtime.copy_device_to_host(device_buffer.empty_mask),
                dtype=np.bool_,
            ),
            part_offsets=(
                None
                if device_buffer.part_offsets is None
                else np.asarray(
                    runtime.copy_device_to_host(device_buffer.part_offsets),
                    dtype=np.int32,
                )
            ),
            ring_offsets=(
                None
                if device_buffer.ring_offsets is None
                else np.asarray(
                    runtime.copy_device_to_host(device_buffer.ring_offsets),
                    dtype=np.int32,
                )
            ),
            bounds=(
                None
                if device_buffer.bounds is None
                else np.asarray(runtime.copy_device_to_host(device_buffer.bounds), dtype=np.float64)
            ),
            host_materialized=False,
        )
    return validity, tags, family_row_offsets, families


_SUPPORTED_GEOARROW_MIXES = {
    frozenset({"Point", "MultiPoint"}),
    frozenset({"LineString", "MultiLineString"}),
    frozenset({"Polygon", "MultiPolygon"}),
}

_SUPPORTED_GEOARROW_PROMOTIONS = {
    frozenset({GeometryFamily.POINT, GeometryFamily.MULTIPOINT}): GeometryFamily.MULTIPOINT,
    frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}): GeometryFamily.MULTILINESTRING,
    frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}): GeometryFamily.MULTIPOLYGON,
}


def _geoarrow_export_family_from_family_set(
    family_set: frozenset[GeometryFamily],
) -> tuple[GeometryFamily, bool]:
    if len(family_set) == 1:
        return next(iter(family_set)), False
    promoted_family = _SUPPORTED_GEOARROW_PROMOTIONS.get(family_set)
    if promoted_family is not None:
        return promoted_family, True
    raise ValueError("Geometry type combination is not supported for native GeoArrow encoding")


def _owned_geoarrow_family_set(owned: OwnedGeometryArray) -> frozenset[GeometryFamily]:
    if owned.device_state is not None:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
            cp = None
    else:
        cp = None

    if cp is not None:
        state = owned._ensure_device_state()
        validity = cp.asarray(state.validity)
        valid_count = int(cp.count_nonzero(validity).item())
        if valid_count == 0:
            return frozenset()
        valid_tags = cp.asarray(state.tags)[validity]
        unique_tags = tuple(int(tag) for tag in cp.unique(valid_tags).get())
    else:
        valid_tags = owned.tags[owned.validity]
        if valid_tags.size == 0:
            return frozenset()
        unique_tags = tuple(int(tag) for tag in np.unique(valid_tags))
    return frozenset(TAG_FAMILIES[tag] for tag in unique_tags)


def _geoarrow_export_family_from_tags(
    validity: np.ndarray,
    tags: np.ndarray,
) -> tuple[GeometryFamily, bool]:
    valid_tags = np.asarray(tags[validity], dtype=np.int8)
    if valid_tags.size == 0:
        raise ValueError("GeoArrow export requires at least one valid geometry row")
    family_set = frozenset(TAG_FAMILIES[int(tag)] for tag in np.unique(valid_tags))
    return _geoarrow_export_family_from_family_set(family_set)


def _promoted_geoarrow_row_view(validity: np.ndarray):
    family_row_offsets = np.full(validity.size, -1, dtype=np.int32)
    if bool(validity.any()):
        family_row_offsets[validity] = np.arange(int(validity.sum()), dtype=np.int32)
    return SimpleNamespace(
        row_count=int(validity.size),
        validity=validity,
        family_row_offsets=family_row_offsets,
    )


def _promote_point_multipoint_mix(
    *,
    validity: np.ndarray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    families: dict[GeometryFamily, FamilyGeometryBuffer],
):
    point_buffer = families.get(GeometryFamily.POINT)
    multipoint_buffer = families.get(GeometryFamily.MULTIPOINT)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    geometry_offsets = [0]
    empty_mask: list[bool] = []
    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    multipoint_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    for row_index in range(validity.size):
        if not bool(validity[row_index]):
            continue
        local_row = int(family_row_offsets[row_index])
        tag = int(tags[row_index])
        if tag == point_tag:
            assert point_buffer is not None
            if bool(point_buffer.empty_mask[local_row]):
                empty_mask.append(True)
                geometry_offsets.append(geometry_offsets[-1])
                continue
            coord_index = int(point_buffer.geometry_offsets[local_row])
            x_chunks.append(point_buffer.x[coord_index : coord_index + 1])
            y_chunks.append(point_buffer.y[coord_index : coord_index + 1])
            empty_mask.append(False)
            geometry_offsets.append(geometry_offsets[-1] + 1)
            continue
        if tag == multipoint_tag:
            assert multipoint_buffer is not None
            start = int(multipoint_buffer.geometry_offsets[local_row])
            end = int(multipoint_buffer.geometry_offsets[local_row + 1])
            count = end - start
            if count:
                x_chunks.append(multipoint_buffer.x[start:end])
                y_chunks.append(multipoint_buffer.y[start:end])
            empty_mask.append(count == 0)
            geometry_offsets.append(geometry_offsets[-1] + count)
            continue
        raise ValueError(f"Unsupported GeoArrow promotion tag: {tag}")

    return (
        SimpleNamespace(
            family=GeometryFamily.MULTIPOINT,
            x=np.concatenate(x_chunks) if x_chunks else np.empty(0, dtype=np.float64),
            y=np.concatenate(y_chunks) if y_chunks else np.empty(0, dtype=np.float64),
            geometry_offsets=np.asarray(geometry_offsets, dtype=np.int32),
            empty_mask=np.asarray(empty_mask, dtype=np.bool_),
            part_offsets=None,
            ring_offsets=None,
        ),
        _promoted_geoarrow_row_view(validity),
    )


def _promote_linestring_multilinestring_mix(
    *,
    validity: np.ndarray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    families: dict[GeometryFamily, FamilyGeometryBuffer],
):
    linestring_buffer = families.get(GeometryFamily.LINESTRING)
    multilinestring_buffer = families.get(GeometryFamily.MULTILINESTRING)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    geometry_offsets = [0]
    part_offsets = [0]
    empty_mask: list[bool] = []
    linestring_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    multilinestring_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    for row_index in range(validity.size):
        if not bool(validity[row_index]):
            continue
        local_row = int(family_row_offsets[row_index])
        tag = int(tags[row_index])
        if tag == linestring_tag:
            assert linestring_buffer is not None
            start = int(linestring_buffer.geometry_offsets[local_row])
            end = int(linestring_buffer.geometry_offsets[local_row + 1])
            count = end - start
            if count:
                x_chunks.append(linestring_buffer.x[start:end])
                y_chunks.append(linestring_buffer.y[start:end])
                part_offsets.append(part_offsets[-1] + count)
                geometry_offsets.append(geometry_offsets[-1] + 1)
                empty_mask.append(False)
            else:
                geometry_offsets.append(geometry_offsets[-1])
                empty_mask.append(True)
            continue
        if tag == multilinestring_tag:
            assert multilinestring_buffer is not None
            start_part = int(multilinestring_buffer.geometry_offsets[local_row])
            end_part = int(multilinestring_buffer.geometry_offsets[local_row + 1])
            part_count = end_part - start_part
            if part_count:
                coord_start = int(multilinestring_buffer.part_offsets[start_part])
                coord_end = int(multilinestring_buffer.part_offsets[end_part])
                x_chunks.append(multilinestring_buffer.x[coord_start:coord_end])
                y_chunks.append(multilinestring_buffer.y[coord_start:coord_end])
                promoted_part_offsets = (
                    multilinestring_buffer.part_offsets[start_part : end_part + 1]
                    - coord_start
                    + part_offsets[-1]
                )
                part_offsets.extend(promoted_part_offsets[1:])
            geometry_offsets.append(geometry_offsets[-1] + part_count)
            empty_mask.append(part_count == 0)
            continue
        raise ValueError(f"Unsupported GeoArrow promotion tag: {tag}")

    return (
        SimpleNamespace(
            family=GeometryFamily.MULTILINESTRING,
            x=np.concatenate(x_chunks) if x_chunks else np.empty(0, dtype=np.float64),
            y=np.concatenate(y_chunks) if y_chunks else np.empty(0, dtype=np.float64),
            geometry_offsets=np.asarray(geometry_offsets, dtype=np.int32),
            empty_mask=np.asarray(empty_mask, dtype=np.bool_),
            part_offsets=np.asarray(part_offsets, dtype=np.int32),
            ring_offsets=None,
        ),
        _promoted_geoarrow_row_view(validity),
    )


def _promote_polygon_multipolygon_mix(
    *,
    validity: np.ndarray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    families: dict[GeometryFamily, FamilyGeometryBuffer],
):
    polygon_buffer = families.get(GeometryFamily.POLYGON)
    multipolygon_buffer = families.get(GeometryFamily.MULTIPOLYGON)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    geometry_offsets = [0]
    part_offsets = [0]
    ring_offsets = [0]
    empty_mask: list[bool] = []
    polygon_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    multipolygon_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    for row_index in range(validity.size):
        if not bool(validity[row_index]):
            continue
        local_row = int(family_row_offsets[row_index])
        tag = int(tags[row_index])
        if tag == polygon_tag:
            assert polygon_buffer is not None
            start_ring = int(polygon_buffer.geometry_offsets[local_row])
            end_ring = int(polygon_buffer.geometry_offsets[local_row + 1])
            ring_count = end_ring - start_ring
            if ring_count:
                coord_start = int(polygon_buffer.ring_offsets[start_ring])
                coord_end = int(polygon_buffer.ring_offsets[end_ring])
                x_chunks.append(polygon_buffer.x[coord_start:coord_end])
                y_chunks.append(polygon_buffer.y[coord_start:coord_end])
                promoted_ring_offsets = (
                    polygon_buffer.ring_offsets[start_ring : end_ring + 1]
                    - coord_start
                    + ring_offsets[-1]
                )
                ring_offsets.extend(promoted_ring_offsets[1:])
                part_offsets.append(part_offsets[-1] + ring_count)
                geometry_offsets.append(geometry_offsets[-1] + 1)
                empty_mask.append(False)
            else:
                geometry_offsets.append(geometry_offsets[-1])
                empty_mask.append(True)
            continue
        if tag == multipolygon_tag:
            assert multipolygon_buffer is not None
            start_polygon = int(multipolygon_buffer.geometry_offsets[local_row])
            end_polygon = int(multipolygon_buffer.geometry_offsets[local_row + 1])
            polygon_count = end_polygon - start_polygon
            if polygon_count:
                start_ring = int(multipolygon_buffer.part_offsets[start_polygon])
                end_ring = int(multipolygon_buffer.part_offsets[end_polygon])
                coord_start = int(multipolygon_buffer.ring_offsets[start_ring])
                coord_end = int(multipolygon_buffer.ring_offsets[end_ring])
                x_chunks.append(multipolygon_buffer.x[coord_start:coord_end])
                y_chunks.append(multipolygon_buffer.y[coord_start:coord_end])
                promoted_ring_offsets = (
                    multipolygon_buffer.ring_offsets[start_ring : end_ring + 1]
                    - coord_start
                    + ring_offsets[-1]
                )
                ring_offsets.extend(promoted_ring_offsets[1:])
                promoted_part_offsets = (
                    multipolygon_buffer.part_offsets[start_polygon : end_polygon + 1]
                    - start_ring
                    + part_offsets[-1]
                )
                part_offsets.extend(promoted_part_offsets[1:])
            geometry_offsets.append(geometry_offsets[-1] + polygon_count)
            empty_mask.append(polygon_count == 0)
            continue
        raise ValueError(f"Unsupported GeoArrow promotion tag: {tag}")

    return (
        SimpleNamespace(
            family=GeometryFamily.MULTIPOLYGON,
            x=np.concatenate(x_chunks) if x_chunks else np.empty(0, dtype=np.float64),
            y=np.concatenate(y_chunks) if y_chunks else np.empty(0, dtype=np.float64),
            geometry_offsets=np.asarray(geometry_offsets, dtype=np.int32),
            empty_mask=np.asarray(empty_mask, dtype=np.bool_),
            part_offsets=np.asarray(part_offsets, dtype=np.int32),
            ring_offsets=np.asarray(ring_offsets, dtype=np.int32),
        ),
        _promoted_geoarrow_row_view(validity),
    )


def _promote_supported_geoarrow_mix(
    *,
    export_family: GeometryFamily,
    validity: np.ndarray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    families: dict[GeometryFamily, FamilyGeometryBuffer],
):
    if export_family is GeometryFamily.MULTIPOINT:
        return _promote_point_multipoint_mix(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=families,
        )
    if export_family is GeometryFamily.MULTILINESTRING:
        return _promote_linestring_multilinestring_mix(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=families,
        )
    if export_family is GeometryFamily.MULTIPOLYGON:
        return _promote_polygon_multipolygon_mix(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=families,
        )
    raise ValueError(f"Unsupported GeoArrow promotion target: {export_family.value}")


def _device_geoarrow_constructor_fallback_reason(owned: OwnedGeometryArray) -> str | None:
    if owned.device_state is not None:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
            cp = None
    else:
        cp = None

    if cp is not None:
        state = owned._ensure_device_state()
        validity = cp.asarray(state.validity)
        valid_count = int(cp.count_nonzero(validity).item())
        if valid_count == 0:
            return None
        valid_tags = cp.asarray(state.tags)[validity]
        unique_tags = tuple(int(tag) for tag in cp.unique(valid_tags).get())
    else:
        validity = owned.validity
        valid_tags = owned.tags[validity]
        if valid_tags.size == 0:
            return None
        unique_tags = tuple(int(tag) for tag in np.unique(valid_tags))

    geom_types = frozenset(_TAG_TO_GEOM_TYPE_NAME[tag] for tag in unique_tags)
    if len(geom_types) <= 1 or geom_types in _SUPPORTED_GEOARROW_MIXES:
        return None
    return "Geometry type combination is not supported for native GeoArrow encoding"


def _geoarrow_constructor_fallback_reason(series) -> str | None:
    import shapely

    values = np.asarray(series.array)
    if len(values) == 0:
        return None
    missing = shapely.is_missing(values)
    if bool(missing.all()):
        return None
    present = values[~missing]
    geom_types = {geometry.geom_type for geometry in present}
    if len(geom_types) == 2 and geom_types in (
        {"Point", "MultiPoint"},
        {"LineString", "MultiLineString"},
        {"Polygon", "MultiPolygon"},
    ):
        return None
    if len(geom_types) != 1:
        return "Geometry type combination is not supported for native GeoArrow encoding"
    geom_type = next(iter(geom_types))
    if geom_type not in {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
    }:
        return f"Geometry type combination is not supported for native GeoArrow encoding: {geom_type}"
    return None

def _construct_geoarrow_array_with_explicit_fallback(
    series,
    *,
    field_name: str,
    interleaved: bool,
    include_z: bool | None,
    surface: str,
    fallback_to_wkb_on_error: bool,
    return_mode: bool = False,
):
    from vibespatial.api.io._geoarrow import construct_geometry_array, construct_wkb_array

    if len(series) == 0 or bool(series.isna().all()):
        raise NotImplementedError(
            "GeoArrow export requires at least one non-missing geometry to infer the geometry type"
        )

    arr = series.array
    owned = arr.to_owned() if isinstance(arr, DeviceGeometryArray) else None
    fast_path_reason = (
        _owned_geoarrow_fast_path_reason_from_owned(owned, include_z=include_z)
        if owned is not None
        else _owned_geoarrow_fast_path_reason(series, include_z=include_z)
    )
    if fast_path_reason is None:
        try:
            if owned is not None:
                family, requires_promotion = _geoarrow_export_family_from_family_set(
                    _owned_geoarrow_family_set(owned)
                )
                if not requires_promotion:
                    result = _encode_owned_geoarrow_array_device(
                        owned,
                        family=family,
                        field_name=field_name,
                        crs=series.crs,
                        interleaved=interleaved,
                    )
                    return (*result, ExecutionMode.GPU) if return_mode else result
            result = encode_owned_geoarrow_array(
                owned if owned is not None else series.array.to_owned(),
                field_name=field_name,
                crs=series.crs,
                interleaved=interleaved,
            )
            return (*result, ExecutionMode.CPU) if return_mode else result
        except Exception as exc:
            fast_path_reason = str(exc)
    if owned is not None:
        constructor_fallback_reason = _device_geoarrow_constructor_fallback_reason(owned)
    else:
        constructor_fallback_reason = _geoarrow_constructor_fallback_reason(series)
    if constructor_fallback_reason is not None:
        if not fallback_to_wkb_on_error:
            raise ValueError(constructor_fallback_reason)
        if owned is not None:
            record_dispatch_event(
                surface=surface,
                operation="to_arrow",
                implementation="native_wkb_compatibility_bridge",
                reason=(
                    "unsupported GeoArrow geometry mix exported through the repo-owned "
                    "WKB compatibility bridge instead of the host constructor"
                ),
                selected=ExecutionMode.GPU,
            )
            return _encode_owned_wkb_array(
                owned,
                field_name=field_name,
                crs=series.crs,
                return_mode=return_mode,
            )
        record_fallback_event(
            surface=surface,
            reason="explicit CPU fallback to WKB until native GeoArrow encoder covers this geometry mix",
            detail=constructor_fallback_reason,
            selected=ExecutionMode.CPU,
            pipeline="io/geoarrow_encode",
            d2h_transfer=True,
        )
        values = np.asarray(arr)
        result = construct_wkb_array(
            values,
            field_name=field_name,
            crs=series.crs,
        )
        return (*result, ExecutionMode.CPU) if return_mode else result
    if owned is not None:
        record_fallback_event(
            surface=surface,
            reason="explicit CPU compatibility export for GeoArrow materialization",
            detail=fast_path_reason or "native GeoArrow constructor semantics require host materialization",
            selected=ExecutionMode.CPU,
            pipeline="io/geoarrow_encode",
            d2h_transfer=True,
        )
    values = np.asarray(arr)
    result = construct_geometry_array(
        values,
        include_z=include_z,
        field_name=field_name,
        crs=series.crs,
        interleaved=interleaved,
    )
    return (*result, ExecutionMode.CPU) if return_mode else result


def _series_owned_geometry(series):
    arr = series.array
    if isinstance(arr, DeviceGeometryArray):
        return arr.to_owned()
    return getattr(arr, "_owned", None)


def _geoarrow_point_type(*, interleaved: bool):
    import pyarrow as pa

    if interleaved:
        return pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)
    return pa.struct(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )


def _geoarrow_target_type(
    family: GeometryFamily,
    *,
    interleaved: bool,
):
    from vibespatial.api.io._geoarrow import (
        _linestring_type,
        _multilinestring_type,
        _multipoint_type,
        _multipolygon_type,
        _polygon_type,
    )

    point_type = _geoarrow_point_type(interleaved=interleaved)
    if family is GeometryFamily.POINT:
        return point_type
    if family is GeometryFamily.LINESTRING:
        return _linestring_type(point_type)
    if family is GeometryFamily.POLYGON:
        return _polygon_type(point_type)
    if family is GeometryFamily.MULTIPOINT:
        return _multipoint_type(point_type)
    if family is GeometryFamily.MULTILINESTRING:
        return _multilinestring_type(point_type)
    if family is GeometryFamily.MULTIPOLYGON:
        return _multipolygon_type(point_type)
    raise ValueError(f"Unsupported geometry family for device GeoArrow encode: {family}")


def _rebuild_arrow_array_with_type(
    source_array,
    target_type,
    *,
    interleaved_point_child=None,
):
    import pyarrow as pa

    if pa.types.is_fixed_size_list(target_type):
        if interleaved_point_child is None:
            raise ValueError("interleaved GeoArrow export requires a prepared point child array")
        return pa.Array.from_buffers(
            target_type,
            len(source_array),
            source_array.buffers()[: target_type.num_buffers],
            null_count=source_array.null_count,
            offset=source_array.offset,
            children=[interleaved_point_child],
        )
    if source_array.type == target_type:
        return source_array

    children = None
    if pa.types.is_struct(target_type):
        children = [
            _rebuild_arrow_array_with_type(
                source_array.field(index),
                target_type[index].type,
                interleaved_point_child=interleaved_point_child,
            )
            for index in range(target_type.num_fields)
        ]
    elif pa.types.is_list(target_type) or pa.types.is_large_list(target_type):
        children = [
            _rebuild_arrow_array_with_type(
                source_array.values,
                target_type.value_type,
                interleaved_point_child=interleaved_point_child,
            )
        ]
    return pa.Array.from_buffers(
        target_type,
        len(source_array),
        source_array.buffers()[: target_type.num_buffers],
        null_count=source_array.null_count,
        offset=source_array.offset,
        children=children,
    )


def _device_interleaved_point_child(plc_column, *, family: GeometryFamily):
    import pyarrow as pa
    import pylibcudf as plc

    if family is GeometryFamily.POINT:
        x_column = plc_column.child(0)
        y_column = plc_column.child(1)
    else:
        raise RuntimeError("device interleaved GeoArrow child must be built from owned buffers")

    values = plc.reshape.interleave_columns(plc.Table([x_column, y_column])).to_arrow()
    return pa.Array.from_buffers(
        pa.float64(),
        len(values),
        values.buffers(),
        null_count=values.null_count,
        offset=values.offset,
    )


def _encode_owned_geoarrow_array_device(
    owned: OwnedGeometryArray,
    *,
    family: GeometryFamily,
    field_name: str,
    crs: Any | None,
    interleaved: bool,
):
    import pyarrow as pa
    import pylibcudf as plc

    plc_column, encoding_name = _encode_owned_geoarrow_column_device(owned)
    source_array = plc_column.to_arrow()
    target_type = _geoarrow_target_type(family, interleaved=interleaved)
    interleaved_point_child = None
    if interleaved:
        if family is GeometryFamily.POINT:
            interleaved_point_child = _device_interleaved_point_child(plc_column, family=family)
        else:
            state = owned._ensure_device_state()
            device_buffer = state.families[family]
            x_column = plc.Column.from_cuda_array_interface(device_buffer.x)
            y_column = plc.Column.from_cuda_array_interface(device_buffer.y)
            values = plc.reshape.interleave_columns(plc.Table([x_column, y_column])).to_arrow()
            interleaved_point_child = pa.Array.from_buffers(
                pa.float64(),
                len(values),
                values.buffers(),
                null_count=values.null_count,
                offset=values.offset,
            )
    geom_arr = _rebuild_arrow_array_with_type(
        source_array,
        target_type,
        interleaved_point_child=interleaved_point_child,
    )
    field = pa.field(
        field_name,
        geom_arr.type,
        nullable=True,
        metadata=_geoarrow_field_metadata(
            extension_name=f"geoarrow.{encoding_name}",
            crs=crs,
        ),
    )
    return field, geom_arr

def _arrow_validity_mask(array) -> np.ndarray:
    if array.null_count == 0:
        return np.ones(len(array), dtype=bool)
    return np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)


def _child_selection_mask(offsets: np.ndarray, parent_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.diff(offsets)
    selected_lengths = lengths[parent_mask]
    compact_offsets = np.empty(selected_lengths.size + 1, dtype=np.int32)
    compact_offsets[0] = 0
    if selected_lengths.size:
        compact_offsets[1:] = np.cumsum(selected_lengths, dtype=np.int32)
    starts = offsets[:-1][parent_mask]
    ends = offsets[1:][parent_mask]
    diff = np.zeros(int(offsets[-1]) + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return compact_offsets, np.cumsum(diff[:-1]) > 0


def _extract_point_xy(values) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow as pa

    if isinstance(values, pa.FixedSizeListArray):
        width = values.type.list_size
        coords = np.asarray(values.values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, width)
        return coords[:, 0], coords[:, 1]
    if isinstance(values, pa.StructArray):
        field_names = [field.name for field in values.type]
        if "x" in field_names and "y" in field_names:
            x_values = values.field("x")
            y_values = values.field("y")
        else:
            x_values = values.field(0)
            y_values = values.field(1)
        return (
            np.asarray(x_values.to_numpy(zero_copy_only=False), dtype=np.float64),
            np.asarray(y_values.to_numpy(zero_copy_only=False), dtype=np.float64),
        )
    raise TypeError(f"Unsupported GeoArrow point storage: {type(values)!r}")

def _arrow_primitive_numpy(array, *, dtype, zero_copy: bool = False) -> np.ndarray:
    try:
        values = array.to_numpy(zero_copy_only=zero_copy)
    except Exception:
        if zero_copy:
            raise
        values = array.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def _extract_point_xy_zero_copy(values) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow as pa

    if isinstance(values, pa.FixedSizeListArray):
        width = values.type.list_size
        coords = _arrow_primitive_numpy(values.values, dtype=np.float64, zero_copy=True).reshape(-1, width)
        return coords[:, 0], coords[:, 1]
    if isinstance(values, pa.StructArray):
        field_names = [field.name for field in values.type]
        if "x" in field_names and "y" in field_names:
            x_values = values.field("x")
            y_values = values.field("y")
        else:
            x_values = values.field(0)
            y_values = values.field(1)
        return (
            _arrow_primitive_numpy(x_values, dtype=np.float64, zero_copy=True),
            _arrow_primitive_numpy(y_values, dtype=np.float64, zero_copy=True),
        )
    raise TypeError(f"Unsupported GeoArrow point storage: {type(values)!r}")


def _build_single_family_owned(
    *,
    family,
    validity,
    tags,
    family_row_offsets,
    view,
    sharing: BufferSharingMode = BufferSharingMode.AUTO,
    shares_memory: bool = False,
) -> OwnedGeometryArray:
    from vibespatial.geometry.owned import MixedGeoArrowView

    mixed = MixedGeoArrowView(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={family: view},
        shares_memory=shares_memory,
    )
    return from_geoarrow(mixed, sharing=sharing)


def _normalize_direct_vector(values: np.ndarray, *, dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == dtype and array.ndim == 1 and bool(array.flags.c_contiguous):
        return array
    return np.ascontiguousarray(array, dtype=dtype)


def _build_single_family_owned_direct(
    *,
    family,
    validity,
    tags,
    family_row_offsets,
    view,
    shares_memory: bool = False,
) -> OwnedGeometryArray:
    buffer = FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=int(view.empty_mask.size),
        x=_normalize_direct_vector(view.x, dtype=np.float64),
        y=_normalize_direct_vector(view.y, dtype=np.float64),
        geometry_offsets=_normalize_direct_vector(view.geometry_offsets, dtype=np.int32),
        empty_mask=_normalize_direct_vector(view.empty_mask, dtype=np.bool_),
        part_offsets=None
        if view.part_offsets is None
        else _normalize_direct_vector(view.part_offsets, dtype=np.int32),
        ring_offsets=None
        if view.ring_offsets is None
        else _normalize_direct_vector(view.ring_offsets, dtype=np.int32),
        bounds=None,
    )
    array = OwnedGeometryArray(
        validity=_normalize_direct_vector(validity, dtype=np.bool_),
        tags=_normalize_direct_vector(tags, dtype=np.int8),
        family_row_offsets=_normalize_direct_vector(family_row_offsets, dtype=np.int32),
        families={family: buffer},
        residency=Residency.HOST,
        geoarrow_backed=True,
        shares_geoarrow_memory=shares_memory,
    )
    detail = (
        "created owned geometry array from shared GeoArrow-style buffers"
        if shares_memory
        else "created owned geometry array from normalized GeoArrow-style buffers"
    )
    array._record(DiagnosticKind.CREATED, detail, visible=True)
    return array


def _family_decode_state(family, array):
    from vibespatial.geometry.owned import GeoArrowBufferView

    validity = _arrow_validity_mask(array)
    row_count = int(validity.sum())
    tags = np.full(len(array), -1, dtype=np.int8)
    family_row_offsets = np.full(len(array), -1, dtype=np.int32)
    if row_count:
        tags[validity] = FAMILY_TAGS[family]
        family_row_offsets[validity] = np.arange(row_count, dtype=np.int32)
    return validity, tags, family_row_offsets, GeoArrowBufferView

def _decode_geoarrow_point(array, family):
    validity, tags, family_row_offsets, GeoArrowBufferView = _family_decode_state(family, array)
    row_count = int(validity.sum())
    if array.null_count == 0:
        try:
            x_all, y_all = _extract_point_xy_zero_copy(array)
        except Exception:
            pass
        else:
            empty_mask = np.isnan(x_all) & np.isnan(y_all)
            if not bool(empty_mask.any()):
                geometry_offsets = np.arange(len(array) + 1, dtype=np.int32)
                view = GeoArrowBufferView(
                    family=family,
                    x=x_all,
                    y=y_all,
                    geometry_offsets=geometry_offsets,
                    empty_mask=empty_mask,
                    shares_memory=True,
                )
                return _build_single_family_owned_direct(
                    family=family,
                    validity=validity,
                    tags=tags,
                    family_row_offsets=family_row_offsets,
                    view=view,
                    shares_memory=False,
                )
    x_all, y_all = _extract_point_xy(array)
    x_valid = x_all[validity]
    y_valid = y_all[validity]
    empty_mask = np.isnan(x_valid) & np.isnan(y_valid)
    non_empty_mask = ~empty_mask
    geometry_offsets = np.empty(row_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    if row_count:
        geometry_offsets[1:] = np.cumsum(non_empty_mask.astype(np.int32), dtype=np.int32)
    view = GeoArrowBufferView(
        family=family,
        x=x_valid[non_empty_mask],
        y=y_valid[non_empty_mask],
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
    )
    return _build_single_family_owned(
        family=family,
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        view=view,
    )


def _decode_geoarrow_linestring_like(array, family):
    validity, tags, family_row_offsets, GeoArrowBufferView = _family_decode_state(family, array)
    if array.null_count == 0:
        try:
            geometry_offsets = _arrow_primitive_numpy(array.offsets, dtype=np.int32, zero_copy=True)
            x_all, y_all = _extract_point_xy_zero_copy(array.values)
        except Exception:
            pass
        else:
            view = GeoArrowBufferView(
                family=family,
                x=x_all,
                y=y_all,
                geometry_offsets=geometry_offsets,
                empty_mask=np.diff(geometry_offsets) == 0,
                shares_memory=True,
            )
            return _build_single_family_owned_direct(
                family=family,
                validity=validity,
                tags=tags,
                family_row_offsets=family_row_offsets,
                view=view,
                shares_memory=False,
            )
    top_offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    geometry_offsets, coord_mask = _child_selection_mask(top_offsets, validity)
    x_all, y_all = _extract_point_xy(array.values)
    view = GeoArrowBufferView(
        family=family,
        x=x_all[coord_mask],
        y=y_all[coord_mask],
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
    )
    return _build_single_family_owned(
        family=family,
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        view=view,
    )


def _decode_geoarrow_polygon(array, family):
    validity, tags, family_row_offsets, GeoArrowBufferView = _family_decode_state(family, array)
    if array.null_count == 0:
        try:
            geometry_offsets = _arrow_primitive_numpy(array.offsets, dtype=np.int32, zero_copy=True)
            child = array.values
            ring_offsets = _arrow_primitive_numpy(child.offsets, dtype=np.int32, zero_copy=True)
            x_all, y_all = _extract_point_xy_zero_copy(child.values)
        except Exception:
            pass
        else:
            view = GeoArrowBufferView(
                family=family,
                x=x_all,
                y=y_all,
                geometry_offsets=geometry_offsets,
                empty_mask=np.diff(geometry_offsets) == 0,
                ring_offsets=ring_offsets,
                shares_memory=True,
            )
            return _build_single_family_owned_direct(
                family=family,
                validity=validity,
                tags=tags,
                family_row_offsets=family_row_offsets,
                view=view,
                shares_memory=False,
            )
    top_offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    geometry_offsets, level_mask = _child_selection_mask(top_offsets, validity)
    child = array.values
    child_offsets = np.asarray(child.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_offsets, coord_mask = _child_selection_mask(child_offsets, level_mask)
    x_all, y_all = _extract_point_xy(child.values)
    view = GeoArrowBufferView(
        family=family,
        x=x_all[coord_mask],
        y=y_all[coord_mask],
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        ring_offsets=ring_offsets,
    )
    return _build_single_family_owned(
        family=family,
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        view=view,
    )


def _decode_geoarrow_multilinestring(array, family):
    validity, tags, family_row_offsets, GeoArrowBufferView = _family_decode_state(family, array)
    if array.null_count == 0:
        try:
            geometry_offsets = _arrow_primitive_numpy(array.offsets, dtype=np.int32, zero_copy=True)
            child = array.values
            part_offsets = _arrow_primitive_numpy(child.offsets, dtype=np.int32, zero_copy=True)
            x_all, y_all = _extract_point_xy_zero_copy(child.values)
        except Exception:
            pass
        else:
            view = GeoArrowBufferView(
                family=family,
                x=x_all,
                y=y_all,
                geometry_offsets=geometry_offsets,
                empty_mask=np.diff(geometry_offsets) == 0,
                part_offsets=part_offsets,
                shares_memory=True,
            )
            return _build_single_family_owned_direct(
                family=family,
                validity=validity,
                tags=tags,
                family_row_offsets=family_row_offsets,
                view=view,
                shares_memory=False,
            )
    top_offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    geometry_offsets, level_mask = _child_selection_mask(top_offsets, validity)
    child = array.values
    child_offsets = np.asarray(child.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    part_offsets, coord_mask = _child_selection_mask(child_offsets, level_mask)
    x_all, y_all = _extract_point_xy(child.values)
    view = GeoArrowBufferView(
        family=family,
        x=x_all[coord_mask],
        y=y_all[coord_mask],
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
    )
    return _build_single_family_owned(
        family=family,
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        view=view,
    )


def _decode_geoarrow_multipolygon(array, family):
    validity, tags, family_row_offsets, GeoArrowBufferView = _family_decode_state(family, array)
    if array.null_count == 0:
        try:
            geometry_offsets = _arrow_primitive_numpy(array.offsets, dtype=np.int32, zero_copy=True)
            polygon_array = array.values
            part_offsets = _arrow_primitive_numpy(polygon_array.offsets, dtype=np.int32, zero_copy=True)
            ring_array = polygon_array.values
            ring_offsets = _arrow_primitive_numpy(ring_array.offsets, dtype=np.int32, zero_copy=True)
            x_all, y_all = _extract_point_xy_zero_copy(ring_array.values)
        except Exception:
            pass
        else:
            view = GeoArrowBufferView(
                family=family,
                x=x_all,
                y=y_all,
                geometry_offsets=geometry_offsets,
                empty_mask=np.diff(geometry_offsets) == 0,
                part_offsets=part_offsets,
                ring_offsets=ring_offsets,
                shares_memory=True,
            )
            return _build_single_family_owned_direct(
                family=family,
                validity=validity,
                tags=tags,
                family_row_offsets=family_row_offsets,
                view=view,
                shares_memory=False,
            )
    top_offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    geometry_offsets, polygon_mask = _child_selection_mask(top_offsets, validity)
    polygon_array = array.values
    polygon_offsets = np.asarray(polygon_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    part_offsets, ring_mask = _child_selection_mask(polygon_offsets, polygon_mask)
    ring_array = polygon_array.values
    ring_offsets_src = np.asarray(ring_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int32)
    ring_offsets, coord_mask = _child_selection_mask(ring_offsets_src, ring_mask)
    x_all, y_all = _extract_point_xy(ring_array.values)
    view = GeoArrowBufferView(
        family=family,
        x=x_all[coord_mask],
        y=y_all[coord_mask],
        geometry_offsets=geometry_offsets,
        empty_mask=np.diff(geometry_offsets) == 0,
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )
    return _build_single_family_owned(
        family=family,
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        view=view,
    )

def _decode_geoarrow_array_to_owned(
    field,
    array,
    *,
    encoding: str | None = None,
    allow_wkb_fallback: bool = True,
) -> OwnedGeometryArray:

    metadata = field.metadata or {}
    ext_name = metadata.get(b"ARROW:extension:name", b"").decode()
    if not ext_name and encoding is not None:
        normalized = encoding.lower()
        ext_name = normalized if normalized.startswith("geoarrow.") else f"geoarrow.{normalized}"
    family_map = {
        "geoarrow.point": GeometryFamily.POINT,
        "geoarrow.linestring": GeometryFamily.LINESTRING,
        "geoarrow.polygon": GeometryFamily.POLYGON,
        "geoarrow.multipoint": GeometryFamily.MULTIPOINT,
        "geoarrow.multilinestring": GeometryFamily.MULTILINESTRING,
        "geoarrow.multipolygon": GeometryFamily.MULTIPOLYGON,
    }
    if ext_name == "geoarrow.wkb":
        try:
            return decode_wkb_arrow_array_owned(
                array,
                allow_fallback=allow_wkb_fallback,
            )
        except NotImplementedError as exc:
            raise _GeoArrowNativeCompatibilityRoute(str(exc)) from exc
    if ext_name not in family_map:
        raise ValueError(f"Unsupported GeoArrow extension type: {ext_name}")
    family = family_map[ext_name]

    if family is GeometryFamily.POINT:
        return _decode_geoarrow_point(array, family)

    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return _decode_geoarrow_linestring_like(array, family)

    if family is GeometryFamily.POLYGON:
        return _decode_geoarrow_polygon(array, family)

    if family is GeometryFamily.MULTILINESTRING:
        return _decode_geoarrow_multilinestring(array, family)

    if family is GeometryFamily.MULTIPOLYGON:
        return _decode_geoarrow_multipolygon(array, family)

    raise AssertionError(f"Unhandled family: {family}")


def _geoarrow_leaf_point_type(storage_type):
    import pyarrow as pa

    while pa.types.is_list(storage_type) or pa.types.is_large_list(storage_type):
        storage_type = storage_type.value_type
    return storage_type


def _geoarrow_storage_type_supports_native_2d_import(storage_type) -> bool:
    import pyarrow as pa

    point_type = _geoarrow_leaf_point_type(storage_type)
    if pa.types.is_fixed_size_list(point_type):
        return point_type.list_size == 2
    if pa.types.is_struct(point_type):
        field_names = [field.name for field in point_type]
        return field_names == ["x", "y"]
    return False


def _geoarrow_native_import_support(field, array) -> tuple[bool, str]:
    import pyarrow as pa

    metadata = field.metadata or {}
    ext_name = metadata.get(b"ARROW:extension:name", b"").decode()

    if not ext_name:
        return False, "Arrow import requires a GeoArrow extension field."

    if ext_name == "geoarrow.wkb":
        wkb_array = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
        if not pa.types.is_binary(wkb_array.type) and not pa.types.is_large_binary(wkb_array.type):
            return (
                False,
                "Non-binary WKB imports route through the explicit compatibility bridge.",
            )
        return True, ""

    family_exts = {
        "geoarrow.point",
        "geoarrow.linestring",
        "geoarrow.polygon",
        "geoarrow.multipoint",
        "geoarrow.multilinestring",
        "geoarrow.multipolygon",
    }
    if ext_name not in family_exts:
        return False, f"Unsupported GeoArrow extension type {ext_name!r} routes through the explicit compatibility bridge."

    if not _geoarrow_storage_type_supports_native_2d_import(field.type):
        return False, "Z-enabled GeoArrow import currently routes through the explicit compatibility bridge."

    return True, ""

def _sample_owned_for_geoarrow_benchmark(
    *,
    geometry_type: str,
    rows: int,
    seed: int = 0,
) -> OwnedGeometryArray:
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons

    if geometry_type == "point":
        dataset = generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed))
    elif geometry_type == "polygon":
        dataset = generate_polygons(
            SyntheticSpec(
                "polygon",
                "regular-grid",
                count=rows,
                seed=seed,
                vertices=6,
                hole_probability=0.0,
            )
        )
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}")
    return from_shapely_geometries(list(dataset.geometries))


def _benchmark_measurement_repeat(*, rows: int, repeat: int, min_repeat: int = 20) -> int:
    if rows < 10_000:
        return repeat
    return max(repeat, min_repeat)

def benchmark_geoarrow_bridge(
    *,
    operation: str,
    geometry_type: str = "point",
    rows: int = 100_000,
    repeat: int = 20,
    seed: int = 0,
) -> list[GeoArrowBridgeBenchmark]:
    owned = _sample_owned_for_geoarrow_benchmark(geometry_type=geometry_type, rows=rows, seed=seed)
    export_modes = (BufferSharingMode.COPY, BufferSharingMode.SHARE)
    import_modes = (BufferSharingMode.COPY, BufferSharingMode.AUTO, BufferSharingMode.SHARE)
    results: list[GeoArrowBridgeBenchmark] = []
    warmup_iters = 5
    measured_repeat = _benchmark_measurement_repeat(rows=rows, repeat=repeat)
    if operation == "encode":
        for sharing in export_modes:
            for _ in range(warmup_iters):
                view = owned.to_geoarrow(sharing=sharing)
            start = perf_counter()
            for _ in range(measured_repeat):
                view = owned.to_geoarrow(sharing=sharing)
            elapsed = (perf_counter() - start) / measured_repeat
            results.append(
                GeoArrowBridgeBenchmark(
                    operation=operation,
                    sharing=sharing.value,
                    geometry_type=geometry_type,
                    rows=rows,
                    elapsed_seconds=elapsed,
                    shares_memory=np.shares_memory(view.validity, owned.validity),
                )
            )
        return results

    if operation != "decode":
        raise ValueError(f"Unsupported GeoArrow bridge operation: {operation}")

    aligned_view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    for sharing in import_modes:
        for _ in range(warmup_iters):
            adopted = from_geoarrow(aligned_view, sharing=sharing)
        start = perf_counter()
        for _ in range(measured_repeat):
            adopted = from_geoarrow(aligned_view, sharing=sharing)
        elapsed = (perf_counter() - start) / measured_repeat
        results.append(
            GeoArrowBridgeBenchmark(
                operation=operation,
                    sharing=sharing.value,
                    geometry_type=geometry_type,
                    rows=rows,
                    elapsed_seconds=elapsed,
                    shares_memory=np.shares_memory(adopted.validity, aligned_view.validity),
                )
            )
    return results

def benchmark_native_geometry_codec(
    *,
    operation: str,
    geometry_type: str = "point",
    rows: int = 100_000,
    repeat: int = 5,
    seed: int = 0,
) -> list[NativeGeometryBenchmark]:
    from vibespatial.api.io._geoarrow import construct_geometry_array

    owned = _sample_owned_for_geoarrow_benchmark(geometry_type=geometry_type, rows=rows, seed=seed)
    shapely_values = np.asarray(owned.to_shapely(), dtype=object)
    results: list[NativeGeometryBenchmark] = []
    measured_repeat = _benchmark_measurement_repeat(rows=rows, repeat=repeat)

    if operation == "encode":
        start = perf_counter()
        for _ in range(measured_repeat):
            construct_geometry_array(shapely_values, field_name="geometry")
        host_elapsed = (perf_counter() - start) / measured_repeat
        results.append(
            NativeGeometryBenchmark(
                operation=operation,
                geometry_type=geometry_type,
                implementation="host_bridge",
                rows=rows,
                elapsed_seconds=host_elapsed,
                rows_per_second=rows / host_elapsed if host_elapsed else float("inf"),
            )
        )
        start = perf_counter()
        for _ in range(measured_repeat):
            encode_owned_geoarrow_array(owned, field_name="geometry")
        native_elapsed = (perf_counter() - start) / measured_repeat
        results.append(
            NativeGeometryBenchmark(
                operation=operation,
                geometry_type=geometry_type,
                implementation="native_owned",
                rows=rows,
                elapsed_seconds=native_elapsed,
                rows_per_second=rows / native_elapsed if native_elapsed else float("inf"),
            )
        )
        return results

    if operation != "decode":
        raise ValueError(f"Unsupported native geometry benchmark operation: {operation}")

    field, geom_arr = encode_owned_geoarrow_array(owned, field_name="geometry")
    _decode_geoarrow_array_to_owned(field, geom_arr)
    start = perf_counter()
    for _ in range(measured_repeat):
        _decode_geoarrow_array_to_owned(field, geom_arr)
    native_elapsed = (perf_counter() - start) / measured_repeat
    results.append(
        NativeGeometryBenchmark(
            operation=operation,
            geometry_type=geometry_type,
            implementation="native_owned",
            rows=rows,
            elapsed_seconds=native_elapsed,
            rows_per_second=rows / native_elapsed if native_elapsed else float("inf"),
        )
    )
    from vibespatial.api.io._geoarrow import construct_shapely_array

    construct_shapely_array(geom_arr, field.metadata[b"ARROW:extension:name"].decode())
    start = perf_counter()
    for _ in range(measured_repeat):
        construct_shapely_array(geom_arr, field.metadata[b"ARROW:extension:name"].decode())
    host_elapsed = (perf_counter() - start) / measured_repeat
    results.append(
        NativeGeometryBenchmark(
            operation=operation,
            geometry_type=geometry_type,
            implementation="host_bridge",
            rows=rows,
            elapsed_seconds=host_elapsed,
            rows_per_second=rows / host_elapsed if host_elapsed else float("inf"),
        )
    )
    return results

def benchmark_wkb_bridge(
    *,
    operation: str,
    geometry_type: str = "point",
    rows: int = 100_000,
    repeat: int = 5,
    seed: int = 0,
) -> list[WKBBridgeBenchmark]:
    import pyarrow as pa
    import shapely

    owned = _sample_owned_for_geoarrow_benchmark(geometry_type=geometry_type, rows=rows, seed=seed)
    results: list[WKBBridgeBenchmark] = []
    warmup_iters = 5
    measured_repeat = _benchmark_measurement_repeat(rows=rows, repeat=repeat, min_repeat=50)

    if operation == "encode":
        for _ in range(warmup_iters):
            pa.array(shapely.to_wkb(owned.to_shapely()), type=pa.binary())
        start = perf_counter()
        for _ in range(measured_repeat):
            pa.array(shapely.to_wkb(owned.to_shapely()), type=pa.binary())
        host_elapsed = (perf_counter() - start) / measured_repeat
        results.append(
            WKBBridgeBenchmark(
                operation=operation,
                geometry_type=geometry_type,
                implementation="host_bridge",
                rows=rows,
                fallback_rows=0,
                elapsed_seconds=host_elapsed,
                rows_per_second=rows / host_elapsed if host_elapsed else float("inf"),
            )
        )
        native_values, native_plan = _encode_native_wkb(owned)
        fallback_rows = native_plan.fallback_rows
        if not native_values:
            fallback_rows = rows
        for _ in range(warmup_iters):
            _encode_owned_wkb_array(owned)
        start = perf_counter()
        for _ in range(measured_repeat):
            _encode_owned_wkb_array(owned)
        native_elapsed = (perf_counter() - start) / measured_repeat
        results.append(
            WKBBridgeBenchmark(
                operation=operation,
                geometry_type=geometry_type,
                implementation="native_owned",
                rows=rows,
                fallback_rows=fallback_rows,
                elapsed_seconds=native_elapsed,
                rows_per_second=rows / native_elapsed if native_elapsed else float("inf"),
            )
        )
        return results

    if operation != "decode":
        raise ValueError(f"Unsupported WKB bridge benchmark operation: {operation}")

    wkb_values = owned.to_wkb()
    for _ in range(warmup_iters):
        from_wkb(wkb_values)
    start = perf_counter()
    for _ in range(measured_repeat):
        from_wkb(wkb_values)
    host_elapsed = (perf_counter() - start) / measured_repeat
    results.append(
        WKBBridgeBenchmark(
            operation=operation,
            geometry_type=geometry_type,
            implementation="host_bridge",
            rows=rows,
            fallback_rows=0,
            elapsed_seconds=host_elapsed,
            rows_per_second=rows / host_elapsed if host_elapsed else float("inf"),
        )
    )
    fallback_rows = plan_wkb_partition(wkb_values).fallback_rows
    for _ in range(warmup_iters):
        decode_wkb_owned(wkb_values)
    start = perf_counter()
    for _ in range(measured_repeat):
        decode_wkb_owned(wkb_values)
    native_elapsed = (perf_counter() - start) / measured_repeat
    results.append(
        WKBBridgeBenchmark(
            operation=operation,
            geometry_type=geometry_type,
            implementation="native_owned",
            rows=rows,
            fallback_rows=fallback_rows,
            elapsed_seconds=native_elapsed,
            rows_per_second=rows / native_elapsed if native_elapsed else float("inf"),
        )
    )
    return results

def geodataframe_to_arrow(
    df,
    *,
    index: bool | None = None,
    geometry_encoding: str = "WKB",
    interleaved: bool = True,
    include_z: bool | None = None,
):
    import pandas as pd
    import pyarrow as pa

    from vibespatial.api.io._geoarrow import ArrowTable, construct_wkb_array

    geometry_mask = df.dtypes.map(
        lambda dtype: getattr(dtype, "name", None) in ("geometry", "device_geometry")
    )
    geometry_columns = df.columns[geometry_mask]
    geometry_indices = np.asarray(geometry_mask).nonzero()[0]
    df_attr = pd.DataFrame(df.copy(deep=False))
    for col in geometry_columns:
        df_attr[col] = None
    table = pa.Table.from_pandas(df_attr, preserve_index=index)
    geometry_modes: list[ExecutionMode] = []
    normalized_encoding = geometry_encoding.lower()
    for column_index, column_name in zip(geometry_indices, geometry_columns):
        series = df[column_name]
        if normalized_encoding == "geoarrow":
            field, geom_arr, selected = _construct_geoarrow_array_with_explicit_fallback(
                series,
                field_name=column_name,
                interleaved=interleaved,
                include_z=include_z,
                surface="geopandas.geodataframe.to_arrow",
                fallback_to_wkb_on_error=False,
                return_mode=True,
            )
        elif normalized_encoding == "wkb":
            owned = _series_owned_geometry(series)
            if owned is not None:
                field, geom_arr, selected = _encode_owned_wkb_array(
                    owned,
                    field_name=column_name,
                    crs=series.crs,
                    return_mode=True,
                )
            else:
                field, geom_arr = construct_wkb_array(
                    np.asarray(series.array),
                    field_name=column_name,
                    crs=series.crs,
                )
                selected = ExecutionMode.CPU
        else:
            raise ValueError(
                f"Expected geometry encoding 'WKB' or 'geoarrow' got {geometry_encoding}"
            )
        geometry_modes.append(selected)
        table = table.set_column(column_index, field, geom_arr)
    selected = (
        ExecutionMode.GPU
        if geometry_modes and all(mode is ExecutionMode.GPU for mode in geometry_modes)
        else ExecutionMode.CPU
    )
    record_dispatch_event(
        surface="geopandas.geodataframe.to_arrow",
        operation="to_arrow",
        implementation=(
            "native_geoarrow_device_export"
            if selected is ExecutionMode.GPU
            else "repo_owned_geoarrow_adapter"
        ),
        reason=(
            "GeoArrow export encoded every geometry column through the repo-owned device path."
            if selected is ExecutionMode.GPU
            else "GeoArrow export routes through the repo-owned adapter and owned buffer policy."
        ),
        selected=selected,
    )
    return ArrowTable(table)


def native_tabular_to_arrow(
    payload,
    *,
    index: bool | None = None,
    geometry_encoding: str = "WKB",
    interleaved: bool = True,
    include_z: bool | None = None,
    force_device_geometry_encode: bool = False,
):

    from vibespatial.api._native_results import NativeTabularResult
    from vibespatial.api.io._geoarrow import construct_wkb_array
    from vibespatial.io.wkb import _encode_owned_wkb_array

    if not isinstance(payload, NativeTabularResult):
        raise TypeError("native_tabular_to_arrow expects a NativeTabularResult")

    resolved_column_order = list(payload.resolved_column_order)
    geometry_columns = sorted(
        payload.geometry_columns,
        key=lambda column: resolved_column_order.index(column.name),
    )
    geometry_names = {column.name for column in geometry_columns}
    attr_columns = [column for column in resolved_column_order if column not in geometry_names]
    if attr_columns:
        table = payload.attributes.to_arrow(
            index=index,
            columns=attr_columns,
        )
    else:
        table = payload.attributes.to_arrow(
            index=index,
            columns=[],
        )

    geometry_encoding_dict: dict[str, str] = {}
    geometry_modes: list[ExecutionMode] = []
    for geometry_column in geometry_columns:
        geometry_column_index = resolved_column_order.index(geometry_column.name)
        if geometry_encoding.lower() == "geoarrow":
            field, geom_arr, selected = _construct_geoarrow_array_with_explicit_fallback(
                geometry_column.geometry.to_geoseries(
                    index=payload.attributes.index,
                    name=geometry_column.name,
                ),
                field_name=geometry_column.name,
                interleaved=interleaved,
                include_z=include_z,
                surface="vibespatial.native_tabular.to_arrow",
                fallback_to_wkb_on_error=False,
                return_mode=True,
            )
            geometry_encoding_dict[geometry_column.name] = (
                field.metadata[b"ARROW:extension:name"]
                .decode()
                .removeprefix("geoarrow.")
            )
        elif geometry_encoding.lower() == "wkb":
            owned = geometry_column.geometry.owned
            if owned is not None:
                field, geom_arr, selected = _encode_owned_wkb_array(
                    owned,
                    field_name=geometry_column.name,
                    crs=geometry_column.geometry.crs,
                    return_mode=True,
                    force_device=force_device_geometry_encode,
                )
            else:
                geometry_series = geometry_column.geometry.to_geoseries(
                    index=payload.attributes.index,
                    name=geometry_column.name,
                )
                field, geom_arr = construct_wkb_array(
                    np.asarray(geometry_series.array),
                    field_name=geometry_column.name,
                    crs=geometry_series.crs,
                )
                selected = ExecutionMode.CPU
            geometry_encoding_dict[geometry_column.name] = "WKB"
        else:
            raise ValueError(
                f"Expected geometry encoding 'WKB' or 'geoarrow' got {geometry_encoding}"
            )

        geometry_modes.append(selected)
        table = table.add_column(geometry_column_index, field, geom_arr)
    selected = (
        ExecutionMode.GPU
        if geometry_modes and all(mode is ExecutionMode.GPU for mode in geometry_modes)
        else ExecutionMode.CPU
    )
    record_dispatch_event(
        surface="vibespatial.native_tabular.to_arrow",
        operation="to_arrow",
        implementation=(
            "native_tabular_device_arrow_export"
            if selected is ExecutionMode.GPU
            else "repo_owned_geoarrow_adapter"
        ),
        reason=(
            "Native tabular Arrow export encoded every geometry column through the device-backed path."
            if selected is ExecutionMode.GPU
            else "Native tabular export lowers directly to Arrow without rebuilding a GeoDataFrame."
        ),
        selected=selected,
    )
    return table, geometry_encoding_dict

def geoseries_to_arrow(
    series,
    *,
    geometry_encoding: str = "WKB",
    interleaved: bool = True,
    include_z: bool | None = None,
):
    from vibespatial.api.io._geoarrow import GeoArrowArray, construct_wkb_array

    field_name = series.name if series.name is not None else ""
    normalized_encoding = geometry_encoding.lower()
    if normalized_encoding == "geoarrow":
        field, geom_arr, selected = _construct_geoarrow_array_with_explicit_fallback(
            series,
            field_name=field_name,
            interleaved=interleaved,
            include_z=include_z,
            surface="geopandas.geoseries.to_arrow",
            fallback_to_wkb_on_error=True,
            return_mode=True,
        )
    elif normalized_encoding == "wkb":
        owned = _series_owned_geometry(series)
        if owned is not None:
            field, geom_arr, selected = _encode_owned_wkb_array(
                owned,
                field_name=field_name,
                crs=series.crs,
                return_mode=True,
            )
        else:
            field, geom_arr = construct_wkb_array(
                series.array.to_numpy(),
                field_name=field_name,
                crs=series.crs,
            )
            selected = ExecutionMode.CPU
    else:
        raise ValueError(
            "Expected geometry encoding 'WKB' or 'geoarrow' "
            f"got {geometry_encoding}"
        )
    record_dispatch_event(
        surface="geopandas.geoseries.to_arrow",
        operation="to_arrow",
        implementation=(
            "native_series_device_arrow_export"
            if selected is ExecutionMode.GPU
            else "repo_owned_geoarrow_adapter"
        ),
        reason=(
            "GeoSeries Arrow export encoded geometry through the device-backed path."
            if selected is ExecutionMode.GPU
            else "GeoArrow export routes through the repo-owned adapter and owned buffer policy."
        ),
        selected=selected,
    )
    return GeoArrowArray(field, geom_arr)

def geodataframe_from_arrow(table, *, geometry: str | None = None, to_pandas_kwargs: dict | None = None):
    import pyarrow as pa

    from vibespatial.api import GeoDataFrame
    from vibespatial.api.io._geoarrow import _get_arrow_geometry_field, arrow_to_geopandas

    if not isinstance(table, pa.Table):
        table = pa.table(table)

    geom_fields = []
    for index, field in enumerate(table.schema):
        geom = _get_arrow_geometry_field(field)
        if geom is not None:
            geom_fields.append((index, field.name, *geom))

    if not geom_fields:
        raise ValueError("No geometry column found in the Arrow table.")

    table_attr = table.drop([field_name for _, field_name, *_rest in geom_fields])
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    df = table_attr.to_pandas(**to_pandas_kwargs)

    native_supported = True
    native_reason = ""
    for index, _column_name, _ext_name, _ext_meta in geom_fields:
        geometry_array = table.column(index)
        native_supported, native_reason = _geoarrow_native_import_support(
            table.schema.field(index),
            geometry_array,
        )
        if not native_supported:
            break

    if not native_supported:
        result = arrow_to_geopandas(table, geometry=geometry, to_pandas_kwargs=to_pandas_kwargs)
        record_dispatch_event(
            surface="geopandas.geodataframe.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason=native_reason,
            selected=ExecutionMode.CPU,
        )
        return result

    try:
        selected = ExecutionMode.GPU
        for index, column_name, ext_name, ext_meta in geom_fields:
            crs = None
            if ext_meta is not None and "crs" in ext_meta:
                crs = ext_meta["crs"]
            array = table[column_name]
            if isinstance(array, pa.ChunkedArray):
                array = array.combine_chunks()
            owned = _decode_geoarrow_array_to_owned(
                table.schema.field(index),
                array,
                allow_wkb_fallback=False,
            )
            series = geoseries_from_owned(owned, name=column_name, crs=crs)
            if not isinstance(series.values, DeviceGeometryArray):
                selected = ExecutionMode.CPU
            df.insert(index, column_name, series)
    except _GeoArrowNativeCompatibilityRoute as exc:
        result = arrow_to_geopandas(table, geometry=geometry, to_pandas_kwargs=to_pandas_kwargs)
        record_dispatch_event(
            surface="geopandas.geodataframe.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason=str(exc),
            selected=ExecutionMode.CPU,
        )
        return result
    except Exception as exc:
        record_fallback_event(
            surface="geopandas.geodataframe.from_arrow",
            reason="Native GeoArrow import failed; falling back to the host adapter",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/from_arrow",
            d2h_transfer=False,
        )
        result = arrow_to_geopandas(table, geometry=geometry, to_pandas_kwargs=to_pandas_kwargs)
        record_dispatch_event(
            surface="geopandas.geodataframe.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason="GeoArrow import routes through the repo-owned host adapter when native decode fails.",
            selected=ExecutionMode.CPU,
        )
        return result

    result = GeoDataFrame(df, geometry=geometry or geom_fields[0][1])
    record_dispatch_event(
        surface="geopandas.geodataframe.from_arrow",
        operation="from_arrow",
        implementation="native_owned_geoarrow_import",
        reason="GeoArrow import decodes the Arrow geometry columns into owned buffers before public export.",
        selected=selected,
    )
    return result

def geoseries_from_arrow(arr, **kwargs):
    import pyarrow as pa

    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.api.io._geoarrow import _get_arrow_geometry_field, arrow_to_geometry_array

    schema_capsule, array_capsule = arr.__arrow_c_array__()
    field = pa.Field._import_from_c_capsule(schema_capsule)
    pa_arr = pa.Array._import_from_c_capsule(field.__arrow_c_schema__(), array_capsule)

    geom_info = _get_arrow_geometry_field(field)
    if geom_info is None:
        raise ValueError("No GeoArrow geometry field found.")
    ext_name, ext_meta = geom_info
    crs = None
    if ext_meta is not None and "crs" in ext_meta:
        crs = ext_meta["crs"]

    native_supported, native_reason = _geoarrow_native_import_support(field, pa_arr)
    if not native_supported:
        series = GeoSeries(arrow_to_geometry_array(arr), **kwargs)
        record_dispatch_event(
            surface="geopandas.geoseries.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason=native_reason,
            selected=ExecutionMode.CPU,
        )
        return series

    try:
        owned = _decode_geoarrow_array_to_owned(
            field,
            pa_arr,
            allow_wkb_fallback=False,
        )
        native_kwargs = dict(kwargs)
        series_crs = native_kwargs.pop("crs", crs)
        series = geoseries_from_owned(owned, crs=series_crs, **native_kwargs)
        selected = ExecutionMode.GPU if isinstance(series.values, DeviceGeometryArray) else ExecutionMode.CPU
    except _GeoArrowNativeCompatibilityRoute as exc:
        series = GeoSeries(arrow_to_geometry_array(arr), **kwargs)
        selected = ExecutionMode.CPU
        record_dispatch_event(
            surface="geopandas.geoseries.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason=str(exc),
            selected=selected,
        )
        return series
    except Exception as exc:
        record_fallback_event(
            surface="geopandas.geoseries.from_arrow",
            reason="Native GeoArrow series import failed; falling back to the host adapter",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/from_arrow",
            d2h_transfer=False,
        )
        series = GeoSeries(arrow_to_geometry_array(arr), **kwargs)
        selected = ExecutionMode.CPU
        record_dispatch_event(
            surface="geopandas.geoseries.from_arrow",
            operation="from_arrow",
            implementation="repo_owned_geoarrow_adapter",
            reason="GeoArrow series import routes through the repo-owned host adapter when native decode fails.",
            selected=selected,
        )
        return series

    record_dispatch_event(
        surface="geopandas.geoseries.from_arrow",
        operation="from_arrow",
        implementation="native_owned_geoarrow_import",
        reason="GeoArrow series import decodes the Arrow geometry column into owned buffers before public export.",
        selected=selected,
    )
    return series

def geoseries_from_owned(
    array: OwnedGeometryArray,
    *,
    name: str = "geometry",
    crs: Any | None = None,
    interleaved: bool = True,
    use_device_array: bool = True,
    **kwargs,
):
    if crs is not None and not hasattr(crs, "to_json_dict"):
        from pyproj import CRS

        crs = CRS.from_user_input(crs)

    # Respect session-wide execution mode: when CPU is requested, skip the
    # DeviceGeometryArray fast path so downstream operations stay on host.
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        use_device_array = False

    # Fast path: wrap in DeviceGeometryArray to avoid D->H->Shapely roundtrip.
    # The OwnedGeometryArray stays as source of truth; Shapely objects are
    # only materialized lazily when downstream code actually needs them.
    if use_device_array:
        from vibespatial.api.geoseries import GeoSeries

        dga = DeviceGeometryArray._from_owned(array, crs=crs)
        series = GeoSeries(dga, **kwargs)
        series.name = name
        return series

    # Legacy path: materialise through the GeoArrow bridge.
    if array.residency is Residency.DEVICE:
        array.move_to(
            Residency.HOST,
            trigger=TransferTrigger.USER_MATERIALIZATION,
            reason="materialized GeoSeries via GeoArrow bridge",
        )
    else:
        array._ensure_host_state()
    array._record(DiagnosticKind.MATERIALIZATION, "materialized GeoSeries via GeoArrow bridge", visible=True)

    from vibespatial.api.io._geoarrow import GeoArrowArray

    field, geom_arr = encode_owned_geoarrow_array(
        array,
        field_name=name,
        crs=crs,
        interleaved=interleaved,
    )
    series = geoseries_from_arrow(GeoArrowArray(field, geom_arr), crs=crs, **kwargs)
    series.name = name
    return series
