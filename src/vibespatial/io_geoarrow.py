from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from vibespatial.device_geometry_array import DeviceGeometryArray
from vibespatial.dispatch import record_dispatch_event
from vibespatial.fallbacks import record_fallback_event
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.io_support import IOFormat, IOOperation, IOPathKind, plan_io_support
from vibespatial.io_wkb import (
    _decode_native_wkb,
    _encode_native_wkb,
    _homogeneous_family,
    decode_wkb_arrow_array_owned,
)
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    BufferSharingMode,
    DiagnosticKind,
    FamilyGeometryBuffer,
    MixedGeoArrowView,
    OwnedGeometryArray,
    from_geoarrow,
    from_shapely_geometries,
    from_wkb,
)
from vibespatial.residency import Residency, TransferTrigger
from vibespatial.runtime import ExecutionMode


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
    plan = plan_geoarrow_codec(IOOperation.ENCODE)
    record_dispatch_event(
        surface="vibespatial.io.geoarrow",
        operation="encode",
        implementation="owned_geoarrow_bridge",
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    record_fallback_event(
        surface="vibespatial.io.geoarrow",
        reason="explicit CPU fallback until the device-side GeoArrow encoder lands",
        detail="owned buffers were exposed as a shared host-visible Arrow-compatible view",
        selected=ExecutionMode.CPU,
        pipeline="io/geoarrow_encode",
        d2h_transfer=True,
    )
    return array.to_geoarrow(sharing=BufferSharingMode.SHARE)


def decode_owned_geoarrow(view: MixedGeoArrowView) -> OwnedGeometryArray:
    plan = plan_geoarrow_codec(IOOperation.DECODE)
    record_dispatch_event(
        surface="vibespatial.io.geoarrow",
        operation="decode",
        implementation="owned_geoarrow_bridge",
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    record_fallback_event(
        surface="vibespatial.io.geoarrow",
        reason="explicit CPU fallback until the device-side GeoArrow decoder lands",
        detail="aligned Arrow-compatible geometry buffers were adopted zero-copy on host when possible",
        selected=ExecutionMode.CPU,
        pipeline="io/geoarrow_decode",
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

    if bool(buffer.empty_mask.any()):
        raise ValueError("Native point GeoArrow fast path does not support empty Point rows")
    x_full = np.zeros(array.row_count, dtype=np.float64)
    y_full = np.zeros(array.row_count, dtype=np.float64)
    valid_rows = np.flatnonzero(array.validity)
    locals_ = array.family_row_offsets[valid_rows]
    x_full[valid_rows] = buffer.x[locals_]
    y_full[valid_rows] = buffer.y[locals_]
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
        point_values = pa.FixedSizeListArray.from_arrays(pa.array(np.column_stack([buffer.x, buffer.y]).ravel(), type=pa.float64()), 2)
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
    family = _homogeneous_family(array)
    buffer = array.families[family]
    if family.value == "point":
        return _encode_point_family(buffer, array, field_name=field_name, crs=crs, interleaved=interleaved)
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
        array=array,
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
        owned = arr.to_owned()
        if owned.row_count == 0:
            return "empty geometry column requires upstream GeoArrow constructor semantics"
        if not bool(owned.validity.any()):
            return "all-missing geometry column requires upstream GeoArrow constructor semantics"
        for buf in owned.families.values():
            if bool(buf.empty_mask.any()):
                return "empty geometry rows require upstream GeoArrow constructor semantics"
        try:
            _homogeneous_family(owned)
        except ValueError as exc:
            return str(exc)
        return None

    values = np.asarray(arr)
    if len(values) == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    missing = shapely.is_missing(values)
    if bool(missing.all()):
        return "all-missing geometry column requires upstream GeoArrow constructor semantics"
    present = values[~missing]
    if bool(shapely.is_empty(present).any()):
        return "empty geometry rows require upstream GeoArrow constructor semantics"
    if bool(shapely.has_z(present).any()):
        return "3D geometry rows require upstream GeoArrow constructor semantics"
    try:
        _homogeneous_family(arr.to_owned())
    except ValueError as exc:
        return str(exc)
    return None

def _construct_geoarrow_array_with_explicit_fallback(
    series,
    *,
    field_name: str,
    interleaved: bool,
    include_z: bool | None,
    surface: str,
    fallback_to_wkb_on_error: bool,
):
    from vibespatial.api.io._geoarrow import construct_geometry_array, construct_wkb_array

    fast_path_reason = _owned_geoarrow_fast_path_reason(series, include_z=include_z)
    if fast_path_reason is None:
        try:
            return encode_owned_geoarrow_array(
                series.array.to_owned(),
                field_name=field_name,
                crs=series.crs,
                interleaved=interleaved,
            )
        except Exception as exc:
            fast_path_reason = str(exc)
    values = np.asarray(series.array)
    try:
        return construct_geometry_array(
            values,
            include_z=include_z,
            field_name=field_name,
            crs=series.crs,
            interleaved=interleaved,
        )
    except Exception as exc:
        if not fallback_to_wkb_on_error:
            raise
        record_fallback_event(
            surface=surface,
            reason="explicit CPU fallback to WKB until native GeoArrow encoder covers this geometry mix",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/geoarrow_encode",
            d2h_transfer=True,
        )
        return construct_wkb_array(
            values,
            field_name=field_name,
            crs=series.crs,
        )

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

def _build_single_family_owned(*, family, validity, tags, family_row_offsets, view) -> OwnedGeometryArray:
    from vibespatial.owned_geometry import MixedGeoArrowView

    mixed = MixedGeoArrowView(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={family: view},
        shares_memory=False,
    )
    return from_geoarrow(mixed, sharing=BufferSharingMode.AUTO)


def _family_decode_state(family, array):
    from vibespatial.owned_geometry import GeoArrowBufferView

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
    x_all, y_all = _extract_point_xy(array)
    view = GeoArrowBufferView(
        family=family,
        x=x_all[validity],
        y=y_all[validity],
        geometry_offsets=np.arange(row_count + 1, dtype=np.int32),
        empty_mask=np.zeros(row_count, dtype=bool),
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

def _decode_geoarrow_array_to_owned(field, array, *, encoding: str | None = None) -> OwnedGeometryArray:

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
        return decode_wkb_arrow_array_owned(array)
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
    if operation == "encode":
        for sharing in export_modes:
            start = perf_counter()
            for _ in range(repeat):
                view = owned.to_geoarrow(sharing=sharing)
            elapsed = (perf_counter() - start) / repeat
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
        start = perf_counter()
        for _ in range(repeat):
            adopted = from_geoarrow(aligned_view, sharing=sharing)
        elapsed = (perf_counter() - start) / repeat
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

    if operation == "encode":
        start = perf_counter()
        for _ in range(repeat):
            construct_geometry_array(shapely_values, field_name="geometry")
        host_elapsed = (perf_counter() - start) / repeat
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
        for _ in range(repeat):
            encode_owned_geoarrow_array(owned, field_name="geometry")
        native_elapsed = (perf_counter() - start) / repeat
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
    start = perf_counter()
    for _ in range(repeat):
        _decode_geoarrow_array_to_owned(field, geom_arr)
    native_elapsed = (perf_counter() - start) / repeat
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

    start = perf_counter()
    for _ in range(repeat):
        construct_shapely_array(geom_arr, field.metadata[b"ARROW:extension:name"].decode())
    host_elapsed = (perf_counter() - start) / repeat
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
    owned = _sample_owned_for_geoarrow_benchmark(geometry_type=geometry_type, rows=rows, seed=seed)
    wkb_values = owned.to_wkb()
    results: list[WKBBridgeBenchmark] = []

    if operation == "encode":
        start = perf_counter()
        for _ in range(repeat):
            owned.to_wkb()
        host_elapsed = (perf_counter() - start) / repeat
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
        start = perf_counter()
        for _ in range(repeat):
            native, plan = _encode_native_wkb(owned)
        native_elapsed = (perf_counter() - start) / repeat
        results.append(
            WKBBridgeBenchmark(
                operation=operation,
                geometry_type=geometry_type,
                implementation="native_owned",
                rows=rows,
                fallback_rows=plan.fallback_rows,
                elapsed_seconds=native_elapsed,
                rows_per_second=rows / native_elapsed if native_elapsed else float("inf"),
            )
        )
        return results

    if operation != "decode":
        raise ValueError(f"Unsupported WKB bridge benchmark operation: {operation}")

    start = perf_counter()
    for _ in range(repeat):
        from_wkb(wkb_values)
    host_elapsed = (perf_counter() - start) / repeat
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
    start = perf_counter()
    for _ in range(repeat):
        native, plan = _decode_native_wkb(wkb_values)
    native_elapsed = (perf_counter() - start) / repeat
    results.append(
        WKBBridgeBenchmark(
            operation=operation,
            geometry_type=geometry_type,
            implementation="native_owned",
            rows=rows,
            fallback_rows=plan.fallback_rows,
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

    from vibespatial.api.io._geoarrow import ArrowTable

    record_dispatch_event(
        surface="geopandas.geodataframe.to_arrow",
        operation="to_arrow",
        implementation="repo_owned_geoarrow_adapter",
        reason="GeoArrow export routes through the repo-owned adapter and owned buffer policy.",
        selected=ExecutionMode.CPU,
    )
    if geometry_encoding.lower() == "geoarrow":
        geometry_mask = df.dtypes == "geometry"
        geometry_columns = df.columns[geometry_mask]
        geometry_indices = np.asarray(geometry_mask).nonzero()[0]
        df_attr = pd.DataFrame(df.copy(deep=False))
        for col in geometry_columns:
            df_attr[col] = None
        table = pa.Table.from_pandas(df_attr, preserve_index=index)
        for column_index, column_name in zip(geometry_indices, geometry_columns):
            field, geom_arr = _construct_geoarrow_array_with_explicit_fallback(
                df[column_name],
                field_name=column_name,
                interleaved=interleaved,
                include_z=include_z,
                surface="geopandas.geodataframe.to_arrow",
                fallback_to_wkb_on_error=False,
            )
            table = table.set_column(column_index, field, geom_arr)
        return ArrowTable(table)

    from vibespatial.api.io._geoarrow import geopandas_to_arrow

    table, _ = geopandas_to_arrow(
        df,
        index=index,
        geometry_encoding=geometry_encoding,
        interleaved=interleaved,
        include_z=include_z,
    )
    return ArrowTable(table)

def geoseries_to_arrow(
    series,
    *,
    geometry_encoding: str = "WKB",
    interleaved: bool = True,
    include_z: bool | None = None,
):
    from vibespatial.api.io._geoarrow import GeoArrowArray, construct_wkb_array

    record_dispatch_event(
        surface="geopandas.geoseries.to_arrow",
        operation="to_arrow",
        implementation="repo_owned_geoarrow_adapter",
        reason="GeoArrow export routes through the repo-owned adapter and owned buffer policy.",
        selected=ExecutionMode.CPU,
    )
    field_name = series.name if series.name is not None else ""
    if geometry_encoding.lower() == "geoarrow":
        field, geom_arr = _construct_geoarrow_array_with_explicit_fallback(
            series,
            field_name=field_name,
            interleaved=interleaved,
            include_z=include_z,
            surface="geopandas.geoseries.to_arrow",
            fallback_to_wkb_on_error=True,
        )
    elif geometry_encoding.lower() == "wkb":
        field, geom_arr = construct_wkb_array(
            series.array.to_numpy(),
            field_name=field_name,
            crs=series.crs,
        )
    else:
        raise ValueError(
            "Expected geometry encoding 'WKB' or 'geoarrow' "
            f"got {geometry_encoding}"
        )
    return GeoArrowArray(field, geom_arr)

def geodataframe_from_arrow(table, *, geometry: str | None = None, to_pandas_kwargs: dict | None = None):
    record_dispatch_event(
        surface="geopandas.geodataframe.from_arrow",
        operation="from_arrow",
        implementation="repo_owned_geoarrow_adapter",
        reason="GeoArrow import routes through the repo-owned adapter and owned buffer policy.",
        selected=ExecutionMode.CPU,
    )
    from vibespatial.api.io._geoarrow import arrow_to_geopandas

    return arrow_to_geopandas(table, geometry=geometry, to_pandas_kwargs=to_pandas_kwargs)

def geoseries_from_arrow(arr, **kwargs):
    record_dispatch_event(
        surface="geopandas.geoseries.from_arrow",
        operation="from_arrow",
        implementation="repo_owned_geoarrow_adapter",
        reason="GeoArrow import routes through the repo-owned adapter and owned buffer policy.",
        selected=ExecutionMode.CPU,
    )
    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.api.io._geoarrow import arrow_to_geometry_array

    return GeoSeries(arrow_to_geometry_array(arr), **kwargs)

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
