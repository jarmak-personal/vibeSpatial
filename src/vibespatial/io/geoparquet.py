from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeReadProvenance,
    NativeTabularResult,
    _concat_native_tabular_results,
    native_attribute_table_from_arrow_table,
    to_native_tabular_result,
)
from vibespatial.api._native_state import attach_native_state_from_native_tabular_result
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    concatenate_owned_arrays,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency, TransferTrigger

from .geoarrow import (
    _authoritative_geoarrow_host_view,
    _decode_geoarrow_array_to_owned,
    _GeoArrowNativeCompatibilityRoute,
    _owned_geoarrow_fast_path_reason,
    encode_owned_geoarrow_array,
    native_tabular_to_arrow,
)
from .geoparquet_planner import (
    GeoParquetMetadataSummary,
    build_geoparquet_metadata_summary,
    select_row_groups,
)
from .pylibcudf import (
    _decode_pylibcudf_geoparquet_column_to_owned,
    _is_pylibcudf_table,
)
from .support import IOFormat, IOOperation, IOPathKind, plan_io_support
from .wkb import (
    _encode_owned_wkb_array,
    _write_geoparquet_native_device,
    _write_geoparquet_native_device_payload,
    has_pyarrow_support,
    has_pylibcudf_support,
)

_SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS = 2_048
_SMALL_TERMINAL_ARROW_EXPORT_FAMILIES = frozenset({
    GeometryFamily.POLYGON,
    GeometryFamily.MULTIPOLYGON,
})


def _record_public_geoparquet_dispatch(
    *,
    selected: ExecutionMode,
    implementation: str,
    reason: str,
    row_count: int,
    detail: str | None = None,
) -> None:
    row_detail = f"rows={row_count}"
    if detail:
        row_detail = f"{row_detail}, {detail}"
    record_dispatch_event(
        surface="geopandas.geodataframe.to_parquet",
        operation="to_parquet",
        implementation=implementation,
        reason=reason,
        detail=row_detail,
        selected=selected,
    )


def _payload_geometry_series(payload: NativeTabularResult):
    return _authoritative_geometry_series(
        payload.geometry.to_geoseries(
            index=payload.attributes.index,
            name=payload.geometry_name,
        )
    )


def _authoritative_owned_geometry_array(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    if owned.device_state is None:
        return owned

    validity, tags, family_row_offsets, families = _authoritative_geoarrow_host_view(owned)
    authoritative = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=owned.residency,
        diagnostics=list(owned.diagnostics),
        runtime_history=list(owned.runtime_history),
        geoarrow_backed=owned.geoarrow_backed,
        shares_geoarrow_memory=owned.shares_geoarrow_memory,
        device_adopted=owned.device_adopted,
        device_state=owned.device_state,
        _row_count=owned.row_count,
    )
    if owned._base is not None:
        authoritative._base = owned._base
    if owned._index_map is not None:
        authoritative._index_map = owned._index_map
    return authoritative


def _authoritative_geometry_series(series):
    arr = series.array
    if isinstance(arr, DeviceGeometryArray) and arr.owned.device_state is not None:
        from vibespatial.api.geoseries import GeoSeries

        authoritative_owned = _authoritative_owned_geometry_array(arr.owned)
        authoritative_array = DeviceGeometryArray._from_owned(
            authoritative_owned,
            crs=series.crs,
        )
        return GeoSeries(authoritative_array, index=series.index, name=series.name, crs=series.crs)
    return series


def _authoritative_native_tabular_result(
    payload: NativeTabularResult,
) -> NativeTabularResult:
    def authoritative_geometry_result(geometry: GeometryNativeResult) -> GeometryNativeResult:
        owned = geometry.owned
        if owned is None or owned.device_state is None:
            return geometry
        return GeometryNativeResult.from_owned(
            _authoritative_owned_geometry_array(owned),
            crs=geometry.crs,
        )

    authoritative_geometry = authoritative_geometry_result(payload.geometry)
    authoritative_secondary = tuple(
        NativeGeometryColumn(
            column.name,
            authoritative_geometry_result(column.geometry),
        )
        for column in payload.secondary_geometry
    )
    if authoritative_geometry is payload.geometry and authoritative_secondary == payload.secondary_geometry:
        return payload
    return NativeTabularResult(
        attributes=payload.attributes,
        geometry=authoritative_geometry,
        geometry_name=payload.geometry_name,
        column_order=payload.column_order,
        attrs=payload.attrs,
        secondary_geometry=authoritative_secondary,
        provenance=payload.provenance,
    )


def _record_terminal_geoparquet_compatibility_export(
    *,
    detail: str,
    implementation: str,
    row_count: int,
) -> None:
    _record_public_geoparquet_dispatch(
        selected=ExecutionMode.CPU,
        implementation=implementation,
        reason=(
            "terminal GeoParquet export used the explicit Arrow compatibility writer "
            "after the native device writer declined a sink feature"
        ),
        row_count=row_count,
        detail=detail,
    )


def _terminal_arrow_export_selected_mode(owned: OwnedGeometryArray | None) -> ExecutionMode:
    if owned is not None and (owned.device_state is not None or owned.residency is Residency.DEVICE):
        return ExecutionMode.GPU
    return ExecutionMode.CPU


def _record_terminal_geoparquet_native_arrow_export(
    *,
    detail: str,
    implementation: str,
    row_count: int,
    owned: OwnedGeometryArray | None,
) -> None:
    selected = _terminal_arrow_export_selected_mode(owned)
    _record_public_geoparquet_dispatch(
        selected=selected,
        implementation=implementation,
        reason=(
            "terminal GeoParquet export used the shared native Arrow sink after "
            "owned geometry encoding"
        ),
        row_count=row_count,
        detail=detail,
    )


def _owned_prefers_small_terminal_arrow_export(owned: OwnedGeometryArray | None) -> bool:
    if owned is None:
        return False
    families = frozenset(owned.families)
    return bool(families) and families.issubset(_SMALL_TERMINAL_ARROW_EXPORT_FAMILIES)


def _small_terminal_arrow_export_detail(
    *,
    row_count: int,
    polygonal_terminal_candidate: bool,
) -> str | None:
    if not polygonal_terminal_candidate or row_count > _SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS:
        return None
    return (
        "small terminal GeoParquet write prefers the shared native Arrow sink; "
        "polygonal outputs are faster through the Arrow sink at this size while "
        "geometry encoding stays owned/native; "
        f"row_count={row_count} <= {_SMALL_TERMINAL_ARROW_EXPORT_MAX_ROWS}"
    )


def _try_promote_geoparquet_geometry_columns_to_device(
    df,
    geometry_columns,
) -> tuple[bool, list]:
    """Build device-owned geometry columns for public GeoParquet writes."""
    if not has_gpu_runtime() or geometry_columns.size == 0:
        return False, []
    if not df.columns.is_unique:
        return False, []

    candidates = []
    snapshots = []
    try:
        for column_name in geometry_columns:
            array = df[column_name].array
            if isinstance(array, DeviceGeometryArray):
                owned = array.owned
            else:
                original_owned = getattr(array, "_owned", None)
                original_residency = (
                    original_owned.residency
                    if original_owned is not None
                    else None
                )
                snapshots.append((array, original_owned, original_residency))
                has_z = getattr(array, "has_z", None)
                if has_z is None:
                    raise ValueError("geometry array does not expose has_z")
                if callable(has_z):
                    has_z = has_z()
                if bool(np.any(np.asarray(has_z, dtype=bool))):
                    raise ValueError("3D geometry is not supported by owned GeoParquet promotion")
                to_owned = getattr(array, "to_owned", None)
                if not callable(to_owned):
                    raise ValueError("geometry array does not expose to_owned")
                owned = to_owned()
            if owned is None:
                raise ValueError("geometry column did not produce owned buffers")
            candidates.append((owned, owned.residency))
        for owned, _original_residency in candidates:
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason=(
                    "GeoParquet public write promotes supported host geometry "
                    "to device-owned buffers"
                ),
            )
    except (NotImplementedError, TypeError, ValueError):
        _restore_geoparquet_promoted_geometry_columns(snapshots)
        return False, []
    return True, snapshots


def _restore_geoparquet_promoted_geometry_columns(snapshots) -> None:
    for array, original_owned, original_residency in reversed(snapshots):
        if original_owned is not None and original_residency is not None:
            if original_owned.residency is not original_residency:
                original_owned.move_to(
                    original_residency,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason=(
                        "GeoParquet public write restores caller geometry "
                        "residency after temporary device-owned promotion"
                    ),
                )
        array._owned = original_owned


def _geometry_columns_are_device_owned(df, geometry_columns) -> bool:
    if geometry_columns.size == 0:
        return False
    for column_name in geometry_columns:
        array = df[column_name].array
        owned = array.owned if isinstance(array, DeviceGeometryArray) else getattr(array, "_owned", None)
        if owned is None:
            return False
        if owned.device_state is None and owned.residency is not Residency.DEVICE:
            return False
    return True


def _write_native_tabular_result_with_arrow(
    payload: NativeTabularResult,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    **kwargs,
) -> None:
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _native_tabular_to_arrow

    table = _native_tabular_to_arrow(
        payload,
        index=index,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
    )
    pq.write_table(table, path, compression=compression, **kwargs)


def _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
    table,
    *,
    column_name: str,
    column_index: int,
    encoding: str | None,
    schema=None,
) -> OwnedGeometryArray:
    import pyarrow as pa

    try:
        return _decode_pylibcudf_geoparquet_column_to_owned(
            table.columns()[column_index],
            encoding,
        )
    except NotImplementedError as exc:
        record_fallback_event(
            surface="vibespatial.io.geoparquet",
            reason="explicit CPU fallback after GPU GeoParquet geometry decode could not complete",
            detail=(
                f"column={column_name}, encoding={encoding!r}, "
                f"detail={type(exc).__name__}: {exc}"
            ),
            selected=ExecutionMode.CPU,
            pipeline="io/read_parquet",
            d2h_transfer=True,
        )
        arrow_table = table.to_arrow()
        arrow_column_index = arrow_table.schema.get_field_index(column_name)
        if arrow_column_index == -1:
            arrow_column_index = int(column_index)
        if schema is not None and column_name in schema.names:
            field = schema.field(column_name)
        else:
            field = arrow_table.schema.field(arrow_column_index)
        array = arrow_table.column(arrow_column_index).combine_chunks()
        normalized_encoding = None if encoding is None else str(encoding).lower()
        if normalized_encoding == "wkb":
            if pa.types.is_string(array.type):
                array = pa.Array.from_buffers(
                    pa.binary(),
                    len(array),
                    array.buffers(),
                    null_count=array.null_count,
                )
                field = pa.field(
                    field.name,
                    pa.binary(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            elif pa.types.is_large_string(array.type):
                array = pa.Array.from_buffers(
                    pa.large_binary(),
                    len(array),
                    array.buffers(),
                    null_count=array.null_count,
                )
                field = pa.field(
                    field.name,
                    pa.large_binary(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
        try:
            return _decode_geoarrow_array_to_owned(field, array, encoding=encoding)
        except _GeoArrowNativeCompatibilityRoute as geoarrow_exc:
            raise NotImplementedError(str(geoarrow_exc)) from geoarrow_exc


def _decode_arrow_geoparquet_column_to_host_geoseries(
    table,
    *,
    column_name: str,
    column_index: int,
    encoding: str | None,
    crs,
    index,
):
    import pyarrow as pa

    from vibespatial.api.geometry_array import from_wkb
    from vibespatial.api.geoseries import GeoSeries

    arrow_table = table.to_arrow() if _is_pylibcudf_table(table) else table
    arrow_column_index = arrow_table.schema.get_field_index(column_name)
    if arrow_column_index == -1:
        arrow_column_index = int(column_index)
    array = arrow_table.column(arrow_column_index).combine_chunks()
    normalized_encoding = None if encoding is None else str(encoding).lower()
    if normalized_encoding == "wkb":
        if pa.types.is_string(array.type):
            array = pa.Array.from_buffers(
                pa.binary(),
                len(array),
                array.buffers(),
                null_count=array.null_count,
            )
        elif pa.types.is_large_string(array.type):
            array = pa.Array.from_buffers(
                pa.large_binary(),
                len(array),
                array.buffers(),
                null_count=array.null_count,
            )
        values = np.asarray(array.to_pylist(), dtype=object)
        return GeoSeries(from_wkb(values, crs=crs), index=index, crs=crs, name=column_name)
    raise NotImplementedError(
        "host GeoSeries GeoParquet fallback currently supports WKB-encoded columns only"
    )


def _write_geoparquet_native_tabular_result(
    payload: NativeTabularResult,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    **kwargs,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _create_geometry_metadata, _replace_table_schema_metadata

    small_write_detail = _small_terminal_arrow_export_detail(
        row_count=payload.geometry.row_count,
        polygonal_terminal_candidate=_owned_prefers_small_terminal_arrow_export(
            payload.geometry.owned
        ),
    )
    if small_write_detail is not None:
        _record_terminal_geoparquet_native_arrow_export(
            detail=small_write_detail,
            implementation="native_payload_arrow_terminal_export",
            row_count=payload.geometry.row_count,
            owned=payload.geometry.owned,
        )
        _write_native_tabular_result_with_arrow(
            payload,
            path,
            index=index,
            compression=compression,
            geometry_encoding=geometry_encoding,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            **kwargs,
        )
        return

    device_write = _write_geoparquet_native_device_payload(
        payload.attributes,
        payload.geometry.owned,
        path,
        geometry_name=payload.geometry_name,
        geometry_crs=payload.geometry.crs,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        column_order=payload.resolved_column_order,
        frame_attrs=payload.attrs,
        **kwargs,
    )
    if device_write.written:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.GPU,
            implementation="native_payload_device_export",
            reason=(
                "GeoParquet export stayed on the native device payload writer and did not "
                "materialize a public GeoDataFrame-shaped Arrow export"
            ),
            row_count=payload.geometry.row_count,
        )
        return
    if device_write.fallback_detail is not None:
        record_fallback_event(
            surface="geopandas.geodataframe.to_parquet",
            reason="explicit CPU fallback from the native device GeoParquet payload writer to the Arrow writer",
            detail=device_write.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/to_parquet",
            d2h_transfer=True,
        )
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_payload_arrow_fallback_export",
            reason=(
                "GeoParquet export fell back to the explicit Arrow writer after the "
                "native device payload writer declined the sink"
            ),
            row_count=payload.geometry.row_count,
            detail=device_write.fallback_detail,
        )
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_payload_arrow_compatibility_export",
            row_count=payload.geometry.row_count,
        )
    else:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_payload_arrow_export",
            reason=(
                "GeoParquet export used the explicit Arrow writer because the native "
                "device payload writer was unavailable for this payload"
            ),
            row_count=payload.geometry.row_count,
        )

    payload = _authoritative_native_tabular_result(payload)
    geometry_series = _payload_geometry_series(payload)

    table, geometry_encoding_dict = native_tabular_to_arrow(
        payload,
        index=index,
        geometry_encoding=geometry_encoding,
        interleaved=False,
        include_z=None,
    )

    geo_metadata = _create_geometry_metadata(
        {payload.geometry_name: geometry_series},
        primary_column=payload.geometry_name,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )

    if write_covering_bbox:
        bounds = geometry_series.bounds
        bbox_array = pa.StructArray.from_arrays(
            [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
            names=["xmin", "ymin", "xmax", "ymax"],
        )
        table = table.append_column("bbox", bbox_array)

    table = _replace_table_schema_metadata(
        table,
        geo_metadata=geo_metadata,
        attrs=payload.attrs,
    )

    pq.write_table(table, path, compression=compression, **kwargs)

@dataclass(frozen=True)
class GeoParquetScanPlan:
    selected_path: IOPathKind
    canonical_gpu: bool
    uses_pylibcudf: bool
    bbox_requested: bool
    metadata_summary_available: bool
    metadata_source: str | None
    uses_covering_bbox: bool
    uses_point_encoding_pushdown: bool
    row_group_pushdown: bool
    planner_strategy: str
    available_row_groups: int | None
    selected_row_groups: tuple[int, ...] | None
    decoded_row_fraction_estimate: float | None
    pruned_row_group_fraction: float | None
    reason: str


@dataclass(frozen=True)
class GeoParquetChunkPlan:
    chunk_index: int
    row_groups: tuple[int, ...] | None
    estimated_rows: int


@dataclass(frozen=True)
class GeoParquetEnginePlan:
    selected_path: IOPathKind
    backend: str
    geometry_encoding: str | None
    chunk_count: int
    target_chunk_rows: int | None
    uses_row_group_pruning: bool
    reason: str


@dataclass(frozen=True)
class GeoParquetReadBackendPlan:
    requested_backend: str
    selected_backend: str
    selected_mode: ExecutionMode
    can_use_pylibcudf: bool
    gpu_rejection_reason: str | None
    reason: str


@dataclass(frozen=True)
class GeoParquetEngineBenchmark:
    backend: str
    geometry_encoding: str
    rows: int
    chunk_rows: int | None
    chunk_count: int
    elapsed_seconds: float
    rows_per_second: float
    planning_elapsed_seconds: float = 0.0
    scan_elapsed_seconds: float = 0.0
    decode_elapsed_seconds: float = 0.0
    concat_elapsed_seconds: float = 0.0


_PYLIBCUDF_GEOPARQUET_ENCODINGS = frozenset({
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
    "wkb",
})
_DEFAULT_GPU_GEOPARQUET_CHUNK_ROWS = 250_000


def _unsupported_pylibcudf_geoparquet_encoding(
    geo_metadata: dict[str, Any] | None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str | None] | None:
    if geo_metadata is None:
        return None
    requested = set(columns or ())
    geometry_columns = [
        name for name in geo_metadata["columns"]
        if not requested or name in requested
    ]
    for column_name in geometry_columns:
        encoding = geo_metadata["columns"][column_name].get("encoding")
        normalized = None if encoding is None else str(encoding).lower()
        if normalized not in _PYLIBCUDF_GEOPARQUET_ENCODINGS:
            return column_name, normalized
    return None

def _rebuild_arrow_array_with_schema_type(array, expected_type):
    import pyarrow as pa

    if array.type == expected_type:
        return array
    if (
        pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
        or pa.types.is_binary(array.type)
        or pa.types.is_large_binary(array.type)
    ) and (
        pa.types.is_string(expected_type)
        or pa.types.is_large_string(expected_type)
        or pa.types.is_binary(expected_type)
        or pa.types.is_large_binary(expected_type)
    ):
        return array.cast(expected_type)

    children = None
    if pa.types.is_struct(expected_type):
        children = [
            _rebuild_arrow_array_with_schema_type(array.field(index), expected_type[index].type)
            for index in range(expected_type.num_fields)
        ]
    elif (
        pa.types.is_list(expected_type)
        or pa.types.is_large_list(expected_type)
        or pa.types.is_fixed_size_list(expected_type)
    ):
        children = [
            _rebuild_arrow_array_with_schema_type(array.values, expected_type.value_type)
        ]

    return pa.Array.from_buffers(
        expected_type,
        len(array),
        array.buffers()[: expected_type.num_buffers],
        null_count=array.null_count,
        offset=array.offset,
        children=children,
    )


def _rebuild_arrow_table_with_schema(table, schema):
    import pyarrow as pa

    columns = []
    for index, field in enumerate(schema):
        source_column = table.column(index)
        columns.append(
            pa.chunked_array(
                [
                    _rebuild_arrow_array_with_schema_type(chunk, field.type)
                    for chunk in source_column.chunks
                ],
                type=field.type,
            )
        )
    return pa.Table.from_arrays(columns, schema=schema)


def _project_arrow_schema(schema, columns):
    if columns is None:
        return schema

    import pyarrow as pa

    selected_fields = [schema.field(name) for name in columns if name in schema.names]
    return pa.schema(selected_fields, metadata=schema.metadata)


def _is_local_arrow_filesystem(filesystem) -> bool:
    if filesystem is None:
        return True
    try:
        import pyarrow.fs as pafs
    except ImportError:  # pragma: no cover - pyarrow present in normal envs
        return False
    if isinstance(filesystem, pafs.LocalFileSystem):
        return True
    base_fs = getattr(filesystem, "base_fs", None)
    if base_fs is not None and base_fs is not filesystem:
        return _is_local_arrow_filesystem(base_fs)
    return False


def _is_local_geoparquet_file(path, *, filesystem=None) -> bool:
    if filesystem is not None and not _is_local_arrow_filesystem(filesystem):
        return False
    if filesystem is not None and _is_local_arrow_filesystem(filesystem):
        try:
            import pyarrow.fs as pafs

            info = filesystem.get_file_info(path)
            return info.type == pafs.FileType.File
        except Exception:
            return False
    candidate = path
    if isinstance(candidate, PathLike):
        candidate = candidate.__fspath__()
    if not isinstance(candidate, str):
        candidate = str(candidate)
    if "://" in candidate:
        return False
    return Path(candidate).is_file()


def _is_local_geoparquet_source(path, *, filesystem=None) -> bool:
    if isinstance(path, (bytes, io.BytesIO)):
        return True
    if filesystem is not None and _is_local_arrow_filesystem(filesystem):
        try:
            import pyarrow.fs as pafs

            info = filesystem.get_file_info(path)
            if info.type in {pafs.FileType.File, pafs.FileType.Directory}:
                return True
        except Exception:
            pass
    try:
        is_file = _is_local_geoparquet_file(path, filesystem=filesystem)
    except TypeError as exc:
        if "unexpected keyword argument 'filesystem'" not in str(exc):
            raise
        is_file = _is_local_geoparquet_file(path)
    if is_file:
        return True
    candidate = path
    if isinstance(candidate, PathLike):
        candidate = candidate.__fspath__()
    if not isinstance(candidate, str):
        candidate = str(candidate)
    if "://" in candidate:
        return False
    return Path(candidate).is_dir()


def _validate_geoparquet_bbox_support(
    geo_metadata: dict[str, Any] | None,
    bbox,
) -> None:
    if bbox is None or geo_metadata is None:
        return
    from vibespatial.api.io.arrow import _get_parquet_bbox_filter

    _get_parquet_bbox_filter(geo_metadata, bbox)


def _normalize_parquet_filters(filters):
    if filters is None:
        return None
    import pyarrow.compute as pc
    import pyarrow.parquet as parquet

    if isinstance(filters, pc.Expression):
        return filters
    return parquet.filters_to_expression(filters)


def _hidden_index_fields_from_schema_metadata(
    schema_metadata: dict[bytes, bytes] | None,
) -> list[str]:
    if schema_metadata is None:
        return []
    pandas_metadata_raw = schema_metadata.get(b"pandas")
    if pandas_metadata_raw is None:
        return []
    pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
    return [
        str(index_column)
        for index_column in (pandas_metadata.get("index_columns") or [])
        if not isinstance(index_column, dict)
    ]


def _merge_column_projection(*column_groups) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for group in column_groups:
        if group is None:
            continue
        for name in group:
            text = str(name)
            if text not in seen:
                seen.add(text)
                merged.append(text)
    return merged or None


def _parquet_filter_column_names(
    filters,
    *,
    available_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    if filters is None:
        return ()

    ordered: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        text = str(name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)

    def walk_filter_tree(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk_filter_tree(item)
            return
        if isinstance(node, tuple):
            if len(node) == 3 and isinstance(node[0], str):
                add_name(node[0])
                return
            for item in node:
                walk_filter_tree(item)

    walk_filter_tree(filters)
    if available_columns is None:
        return tuple(ordered)

    expr_text = str(_normalize_parquet_filters(filters))
    resolved: list[str] = []
    for name in available_columns:
        text = str(name)
        if text in seen or re.search(
            rf"(?<![0-9A-Za-z_]){re.escape(text)}(?![0-9A-Za-z_])",
            expr_text,
        ):
            resolved.append(text)
    return tuple(resolved)


def _compile_pylibcudf_parquet_filter(
    filters,
    *,
    available_columns: tuple[str, ...] | list[str],
):
    if filters is None:
        return None
    import pylibcudf as plc

    normalized = _normalize_parquet_filters(filters)
    return plc.expressions.to_expression(str(normalized), tuple(str(name) for name in available_columns))


def _supports_pylibcudf_geoparquet_read(
    path,
    *,
    bbox,
    columns,
    storage_options,
    filesystem,
    filters,
    to_pandas_kwargs,
    geo_metadata,
    available_columns=None,
) -> tuple[bool, str]:
    if not has_pylibcudf_support():
        return False, "pylibcudf is not installed for the GPU GeoParquet reader"
    if not has_gpu_runtime():
        return False, "GPU GeoParquet reader requires an available CUDA runtime"
    if storage_options is not None:
        return False, "filesystem-backed GeoParquet reads still route through host pyarrow"
    if filesystem is not None and not _is_local_arrow_filesystem(filesystem):
        return False, "filesystem-backed GeoParquet reads still route through host pyarrow"
    if geo_metadata is not None:
        primary = geo_metadata["primary_column"]
        if primary not in geo_metadata["columns"]:
            return False, "GeoParquet metadata without a readable primary geometry routes through host pyarrow"
        if bbox is not None:
            try:
                _validate_geoparquet_bbox_support(geo_metadata, bbox)
            except ValueError as exc:
                return False, str(exc)
        unsupported = _unsupported_pylibcudf_geoparquet_encoding(geo_metadata, columns)
        if unsupported is not None:
            column_name, encoding = unsupported
            return (
                False,
                f"geometry column {column_name!r} with GeoParquet encoding {encoding!r} still routes through host pyarrow",
            )
    if filters is not None and available_columns is not None:
        try:
            _compile_pylibcudf_parquet_filter(filters, available_columns=available_columns)
        except Exception as exc:
            return False, f"predicate filter could not be compiled for the pylibcudf scan backend: {exc}"
    if not _is_local_geoparquet_source(path, filesystem=filesystem):
        return False, "dataset and non-local GeoParquet paths still route through host pyarrow"
    return True, "local GeoParquet scan can use the pylibcudf reader"


def plan_geoparquet_read_backend(
    path,
    *,
    backend: str,
    bbox,
    columns,
    storage_options,
    filesystem,
    filters,
    to_pandas_kwargs,
    geo_metadata,
    available_columns=None,
) -> GeoParquetReadBackendPlan:
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported GeoParquet backend request: {backend!r}")

    can_use_pylibcudf, gpu_reason = _supports_pylibcudf_geoparquet_read(
        path,
        bbox=bbox,
        columns=columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=filters,
        to_pandas_kwargs=to_pandas_kwargs,
        geo_metadata=geo_metadata,
        available_columns=available_columns,
    )
    if backend == "cpu":
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pyarrow",
            selected_mode=ExecutionMode.CPU,
            can_use_pylibcudf=can_use_pylibcudf,
            gpu_rejection_reason=None,
            reason="explicit CPU backend requested for the GeoParquet read path",
        )
    if can_use_pylibcudf:
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pylibcudf",
            selected_mode=ExecutionMode.GPU,
            can_use_pylibcudf=True,
            gpu_rejection_reason=None,
            reason=(
                "explicit GPU backend requested for the GeoParquet read path"
                if backend == "gpu"
                else "auto selected the GPU GeoParquet scan backend"
            ),
        )

    reason = gpu_reason or "GPU GeoParquet scan backend is unavailable"
    if backend == "gpu":
        return GeoParquetReadBackendPlan(
            requested_backend=backend,
            selected_backend="pyarrow",
            selected_mode=ExecutionMode.CPU,
            can_use_pylibcudf=False,
            gpu_rejection_reason=reason,
            reason=f"explicit GPU backend requested but {reason}",
        )
    return GeoParquetReadBackendPlan(
        requested_backend=backend,
        selected_backend="pyarrow",
        selected_mode=ExecutionMode.CPU,
        can_use_pylibcudf=False,
        gpu_rejection_reason=reason,
        reason=f"auto selected the host GeoParquet scan backend because {reason}",
    )


def _record_geoparquet_scan_backend_fallback(*, surface: str, detail: str) -> None:
    record_fallback_event(
        surface=surface,
        reason="explicit CPU fallback for GeoParquet scan backend selection",
        detail=detail,
        selected=ExecutionMode.CPU,
        pipeline="io/read_parquet",
        d2h_transfer=False,
    )


def _is_geoparquet_scan_ineligible_for_gpu_fallback(geo_metadata) -> bool:
    if geo_metadata is None:
        return False
    primary = geo_metadata.get("primary_column")
    columns = geo_metadata.get("columns", {})
    return primary not in columns


def plan_geoparquet_scan(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    geo_metadata: dict[str, Any] | None = None,
    metadata_summary: GeoParquetMetadataSummary | None = None,
    planner_strategy: str = "auto",
) -> GeoParquetScanPlan:
    plan = plan_io_support(IOFormat.GEOPARQUET, IOOperation.SCAN)
    uses_covering_bbox = False
    uses_point_encoding_pushdown = False
    if geo_metadata is not None and bbox is not None:
        primary = geo_metadata["primary_column"]
        column_meta = geo_metadata["columns"][primary]
        uses_covering_bbox = "covering" in column_meta
        uses_point_encoding_pushdown = column_meta.get("encoding") == "point"
    prune_result = None
    if metadata_summary is not None and bbox is not None:
        prune_result = select_row_groups(metadata_summary, bbox, strategy=planner_strategy)
    return GeoParquetScanPlan(
        selected_path=plan.selected_path,
        canonical_gpu=plan.canonical_gpu,
        uses_pylibcudf=has_pylibcudf_support() and has_gpu_runtime(),
        bbox_requested=bbox is not None,
        metadata_summary_available=metadata_summary is not None,
        metadata_source=metadata_summary.source if metadata_summary is not None else None,
        uses_covering_bbox=uses_covering_bbox,
        uses_point_encoding_pushdown=uses_point_encoding_pushdown,
        row_group_pushdown=(
            bbox is not None
            and (prune_result is not None or uses_covering_bbox or uses_point_encoding_pushdown)
        ),
        planner_strategy=prune_result.strategy if prune_result is not None else planner_strategy,
        available_row_groups=metadata_summary.row_group_count if metadata_summary is not None else None,
        selected_row_groups=prune_result.selected_row_groups if prune_result is not None else None,
        decoded_row_fraction_estimate=prune_result.decoded_row_fraction if prune_result is not None else None,
        pruned_row_group_fraction=prune_result.pruned_row_group_fraction if prune_result is not None else None,
        reason=(
            "GeoParquet reads should prefer a GPU scanner plus metadata-first pushdown; "
            "without pylibcudf the current path falls back to host pyarrow scanning, "
            "but row groups can still be pruned before full geometry decode."
        ),
    )

def plan_geoparquet_engine(
    *,
    geo_metadata: dict[str, Any] | None,
    scan_plan: GeoParquetScanPlan,
    chunk_plans: tuple[GeoParquetChunkPlan, ...],
    target_chunk_rows: int | None,
    read_plan: GeoParquetReadBackendPlan,
) -> GeoParquetEnginePlan:
    primary_column = None if geo_metadata is None else geo_metadata["primary_column"]
    geometry_encoding = None
    if geo_metadata is not None and primary_column is not None and primary_column in geo_metadata["columns"]:
        geometry_encoding = geo_metadata["columns"][primary_column].get("encoding")
    return GeoParquetEnginePlan(
        selected_path=scan_plan.selected_path,
        backend=read_plan.selected_backend,
        geometry_encoding=geometry_encoding,
        chunk_count=len(chunk_plans),
        target_chunk_rows=target_chunk_rows,
        uses_row_group_pruning=scan_plan.row_group_pushdown,
        reason=(
            f"{read_plan.reason}; keep row-group pruning from the metadata planner and "
            "decode supported geometry encodings directly into owned buffers after scan."
        ),
    )

def _plan_geoparquet_chunks(
    *,
    metadata_summary: GeoParquetMetadataSummary | None,
    selected_row_groups: tuple[int, ...] | list[int] | None,
    target_chunk_rows: int | None,
) -> tuple[GeoParquetChunkPlan, ...]:
    if selected_row_groups is None:
        estimated_rows = metadata_summary.total_rows if metadata_summary is not None else 0
        return (GeoParquetChunkPlan(chunk_index=0, row_groups=None, estimated_rows=estimated_rows),)
    if len(selected_row_groups) == 0:
        return (GeoParquetChunkPlan(chunk_index=0, row_groups=tuple(), estimated_rows=0),)
    row_groups = tuple(selected_row_groups)
    if metadata_summary is None or target_chunk_rows is None or target_chunk_rows <= 0:
        estimated_rows = (
            int(sum(metadata_summary.row_group_rows[list(row_groups)]))
            if metadata_summary is not None
            else 0
        )
        return (GeoParquetChunkPlan(chunk_index=0, row_groups=row_groups, estimated_rows=estimated_rows),)
    chunks: list[GeoParquetChunkPlan] = []
    current: list[int] = []
    current_rows = 0
    for row_group in row_groups:
        group_rows = int(metadata_summary.row_group_rows[row_group])
        if current and current_rows + group_rows > target_chunk_rows:
            chunks.append(
                GeoParquetChunkPlan(
                    chunk_index=len(chunks),
                    row_groups=tuple(current),
                    estimated_rows=current_rows,
                )
            )
            current = []
            current_rows = 0
        current.append(row_group)
        current_rows += group_rows
    if current:
        chunks.append(
            GeoParquetChunkPlan(
                chunk_index=len(chunks),
                row_groups=tuple(current),
                estimated_rows=current_rows,
            )
        )
    return tuple(chunks)


def _effective_geoparquet_chunk_rows(
    chunk_rows: int | None,
    *,
    selected_backend: str,
) -> int | None:
    if chunk_rows is not None:
        return chunk_rows
    if selected_backend == "pylibcudf":
        return _DEFAULT_GPU_GEOPARQUET_CHUNK_ROWS
    return None


def _geoparquet_scan_sources(path, *, metadata_summary: GeoParquetMetadataSummary | None):
    if metadata_summary is not None and metadata_summary.source_paths is not None:
        return list(metadata_summary.source_paths)
    if isinstance(path, (bytes, io.BytesIO)):
        return [path]
    if isinstance(path, (str, PathLike)) and Path(path).is_dir():
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        return [str(fragment.path) for fragment in dataset.get_fragments()]
    return [path]


def _geoparquet_scan_row_groups(
    *,
    metadata_summary: GeoParquetMetadataSummary | None,
    selected_row_groups: tuple[int, ...] | list[int] | None,
):
    if selected_row_groups is None:
        return None
    if metadata_summary is not None and metadata_summary.source_paths is not None:
        source_count = len(metadata_summary.source_paths)
        grouped = [[] for _ in range(source_count)]
        source_indices = metadata_summary.row_group_source_indices
        source_row_groups = metadata_summary.row_group_source_row_groups
        if source_indices is None or source_row_groups is None:
            raise ValueError("dataset row-group selection requires per-source row-group metadata")
        for row_group in selected_row_groups:
            source_index = int(source_indices[row_group])
            grouped[source_index].append(int(source_row_groups[row_group]))
        return grouped
    return [list(selected_row_groups)]


def _native_bbox_row_positions(payload: NativeTabularResult, bbox):
    if bbox is None:
        return None
    geometry = payload.geometry
    if geometry.owned is not None:
        import cupy as cp

        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        d_bounds = compute_geometry_bounds_device(geometry.owned)
        d_keep = ~(
            (d_bounds[:, 0] > bbox[2])
            | (d_bounds[:, 1] > bbox[3])
            | (d_bounds[:, 2] < bbox[0])
            | (d_bounds[:, 3] < bbox[1])
        )
        return cp.flatnonzero(d_keep).astype(cp.int64, copy=False)

    bounds = np.asarray(geometry.series.bounds, dtype=np.float64)
    keep = ~(
        (bounds[:, 0] > bbox[2])
        | (bounds[:, 1] > bbox[3])
        | (bounds[:, 2] < bbox[0])
        | (bounds[:, 3] < bbox[1])
    )
    return np.flatnonzero(keep).astype(np.int64, copy=False)


def _apply_native_bbox_filter(payload: NativeTabularResult, bbox) -> NativeTabularResult:
    if bbox is None:
        return payload
    return payload.take(_native_bbox_row_positions(payload, bbox))


def _apply_owned_bbox_filter(owned: OwnedGeometryArray, bbox) -> OwnedGeometryArray:
    if bbox is None:
        return owned
    import cupy as cp

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_bounds = compute_geometry_bounds_device(owned)
    d_keep = ~(
        (d_bounds[:, 0] > bbox[2])
        | (d_bounds[:, 1] > bbox[3])
        | (d_bounds[:, 2] < bbox[0])
        | (d_bounds[:, 3] < bbox[1])
    )
    return owned.take(cp.flatnonzero(d_keep).astype(cp.int64, copy=False))


def _table_row_count(table) -> int:
    num_rows = getattr(table, "num_rows", None)
    if num_rows is None:
        raise ValueError("table does not expose row count")
    if callable(num_rows):
        return int(num_rows())
    return int(num_rows)


def _pylibcudf_table_to_geopandas(
    table,
    *,
    path,
    row_groups=None,
    filesystem=None,
    geo_metadata: dict[str, Any] | None,
    schema=None,
    table_column_names=None,
    to_pandas_kwargs=None,
    df_attrs=None,
):
    payload = _geoparquet_table_to_native_tabular_result(
        table,
        path=path,
        row_groups=row_groups,
        filesystem=filesystem,
        geo_metadata=geo_metadata,
        schema=schema,
        table_column_names=table_column_names,
        to_pandas_kwargs=to_pandas_kwargs,
        df_attrs=df_attrs,
    )
    frame = payload.to_geodataframe()
    attach_native_state_from_native_tabular_result(frame, payload)
    return frame


def _geoparquet_table_to_native_tabular_result(
    table,
    *,
    path,
    row_groups=None,
    filesystem=None,
    geo_metadata: dict[str, Any] | None,
    schema=None,
    table_column_names=None,
    requested_columns=None,
    to_pandas_kwargs=None,
    df_attrs=None,
    attrs_arrow=None,
    provenance: NativeReadProvenance | None = None,
    scanned_with_pylibcudf: bool | None = None,
    filters=None,
    sources=None,
) -> NativeTabularResult:
    import warnings

    import pandas as pd

    if geo_metadata is None:
        raise ValueError("GeoParquet metadata is required for native tabular decode")
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    if scanned_with_pylibcudf is None:
        scanned_with_pylibcudf = _is_pylibcudf_table(table)

    if schema is not None:
        result_column_names = list(schema.names)
    elif hasattr(table, "column_names"):
        result_column_names = list(table.column_names)
    else:
        raise ValueError("GeoParquet native decode requires schema metadata")
    schema_metadata = None
    if schema is not None:
        schema_metadata = schema.metadata
    elif hasattr(table, "schema"):
        schema_metadata = table.schema.metadata
    hidden_index_fields = _hidden_index_fields_from_schema_metadata(schema_metadata)
    if hidden_index_fields:
        hidden_index_field_set = set(hidden_index_fields)
        result_column_names = [
            column_name
            for column_name in result_column_names
            if column_name not in hidden_index_field_set
        ]
    if requested_columns is None:
        from vibespatial.api.io.arrow import (
            _check_if_covering_in_geo_metadata,
            _get_bbox_encoding_column_name,
        )

        if geo_metadata is not None and _check_if_covering_in_geo_metadata(geo_metadata):
            bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
            result_column_names = [
                column_name for column_name in result_column_names if column_name != bbox_column_name
            ]

    geometry_columns = [col for col in geo_metadata["columns"] if col in result_column_names]
    geometry_columns.sort(key=result_column_names.index)
    if not geometry_columns:
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry_name = geo_metadata["primary_column"]
    if geometry_name not in geometry_columns:
        geometry_name = geometry_columns[0]
        if len(geometry_columns) > 1:
            warnings.warn(
                "Multiple non-primary geometry columns read from Parquet/Feather "
                "file. The first column read was promoted to the primary geometry.",
                stacklevel=3,
            )

    non_geometry_columns = [col for col in result_column_names if col not in geometry_columns]
    if table_column_names is None:
        if hasattr(table, "column_names"):
            table_column_names = list(table.column_names)
        else:
            table_column_names = list(geometry_columns)
    else:
        table_column_names = list(table_column_names)

    if attrs_arrow is None and not non_geometry_columns and not hidden_index_fields:
        attributes = NativeAttributeTable(
            dataframe=pd.DataFrame(index=pd.RangeIndex(_table_row_count(table))),
            to_pandas_kwargs=to_pandas_kwargs,
        )
    else:
        if attrs_arrow is None:
            if scanned_with_pylibcudf:
                attrs_arrow = _read_non_geometry_geoparquet_columns_as_arrow(
                    path,
                    columns=_merge_column_projection(non_geometry_columns, hidden_index_fields) or [],
                    row_groups=row_groups,
                    filesystem=filesystem,
                    filters=filters,
                    sources=sources,
                )
            else:
                attrs_arrow = table.drop(geometry_columns)

        attributes = native_attribute_table_from_arrow_table(
            attrs_arrow,
            to_pandas_kwargs=to_pandas_kwargs,
        )

    decoded_geometry: dict[str, GeometryNativeResult] = {}
    row_count = None
    for column_name in geometry_columns:
        column_meta = geo_metadata["columns"][column_name]
        crs = _geoparquet_geometry_column_crs(column_meta)
        scan_column_index = table_column_names.index(column_name)
        try:
            if scanned_with_pylibcudf:
                owned = _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
                    table,
                    column_name=column_name,
                    column_index=scan_column_index,
                    encoding=column_meta.get("encoding"),
                    schema=schema,
                )
            else:
                owned = _decode_arrow_geoparquet_table_to_owned(
                    table,
                    geo_metadata,
                    column_index=scan_column_index,
                )
            if row_count is None:
                row_count = owned.row_count
            decoded_geometry[column_name] = GeometryNativeResult.from_owned(owned, crs=crs)
        except NotImplementedError as exc:
            record_fallback_event(
                surface="vibespatial.io.geoparquet",
                reason=(
                    "explicit CPU compatibility fallback after GeoParquet geometry decode "
                    "produced families outside the owned native result model"
                ),
                detail=(
                    f"column={column_name}, encoding={column_meta.get('encoding')!r}, "
                    f"detail={type(exc).__name__}: {exc}"
                ),
                selected=ExecutionMode.CPU,
                pipeline="io/read_parquet",
                d2h_transfer=True,
            )
            host_decode_table = table.to_arrow() if scanned_with_pylibcudf else table
            host_series = _decode_arrow_geoparquet_column_to_host_geoseries(
                host_decode_table,
                column_name=column_name,
                column_index=scan_column_index,
                encoding=column_meta.get("encoding"),
                crs=crs,
                index=attributes.index if len(attributes.index) else None,
            )
            if row_count is None:
                row_count = len(host_series)
            decoded_geometry[column_name] = GeometryNativeResult.from_geoseries(host_series)

    if row_count is not None and len(attributes.index) != row_count and not len(attributes.columns):
        attributes = NativeAttributeTable(
            dataframe=pd.DataFrame(index=pd.RangeIndex(row_count)),
            to_pandas_kwargs=to_pandas_kwargs,
        )

    secondary_geometry = tuple(
        NativeGeometryColumn(column_name, decoded_geometry[column_name])
        for column_name in geometry_columns
        if column_name != geometry_name
    )
    attrs = None if df_attrs is None else json.loads(df_attrs)
    return NativeTabularResult(
        attributes=attributes,
        geometry=decoded_geometry[geometry_name],
        geometry_name=geometry_name,
        column_order=tuple(result_column_names),
        attrs=attrs,
        secondary_geometry=secondary_geometry,
        provenance=provenance,
    )


def _read_geoparquet_with_pylibcudf(
    path,
    *,
    columns=None,
    row_groups=None,
    filesystem=None,
    geo_metadata=None,
    to_pandas_kwargs=None,
):
    geometry_scan_columns = None
    if geo_metadata is not None:
        if columns is None:
            geometry_scan_columns = list(geo_metadata["columns"])
        else:
            requested_columns = set(columns)
            geometry_scan_columns = [
                name for name in geo_metadata["columns"]
                if name in requested_columns
            ]
        if not geometry_scan_columns:
            geometry_scan_columns = None
    gpu_table = _read_geoparquet_table_with_pylibcudf(
        path,
        columns=geometry_scan_columns or columns,
        row_groups=row_groups,
        filesystem=filesystem,
    )
    schema = None
    df_attrs = None
    if has_pyarrow_support():
        import pyarrow.parquet as pq

        schema = _project_arrow_schema(
            pq.read_schema(path, filesystem=filesystem),
            columns,
        )
        metadata = schema.metadata
        df_attrs = None if metadata is None else metadata.get(b"PANDAS_ATTRS")
    return _pylibcudf_table_to_geopandas(
        gpu_table,
        path=path,
        row_groups=row_groups,
        filesystem=filesystem,
        geo_metadata=geo_metadata,
        schema=schema,
        table_column_names=geometry_scan_columns,
        to_pandas_kwargs=to_pandas_kwargs,
        df_attrs=df_attrs,
    )

def _read_non_geometry_geoparquet_columns_as_arrow(
    path,
    *,
    columns,
    row_groups=None,
    filesystem=None,
    filters=None,
    sources=None,
):
    """ADR-0042: Read non-geometry columns as an Arrow table.

    Returns a PyArrow Table instead of a pandas DataFrame so that the
    caller can defer ``.to_pandas()`` to the GeoDataFrame construction
    boundary.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _coerce_pyarrow_parquet_source

    scan_sources = [_coerce_pyarrow_parquet_source(source) for source in (sources or [path])]
    schema = pq.read_schema(scan_sources[0], filesystem=filesystem)
    requested_columns = _merge_column_projection(
        list(columns),
        _hidden_index_fields_from_schema_metadata(schema.metadata),
    ) or []
    filter_expression = _normalize_parquet_filters(filters)
    filter_columns = _parquet_filter_column_names(filters)
    scan_columns = _merge_column_projection(requested_columns, filter_columns)

    if row_groups is None:
        if len(scan_sources) == 1:
            table = pq.read_table(
                scan_sources[0],
                columns=scan_columns,
                filesystem=filesystem,
                filters=filter_expression,
                use_pandas_metadata=True,
            )
            return table.select(requested_columns)
        tables = [
            pq.read_table(
                source,
                columns=scan_columns,
                filesystem=filesystem,
                filters=filter_expression,
                use_pandas_metadata=True,
            ).select(requested_columns)
            for source in scan_sources
        ]
    else:
        if len(scan_sources) == 1 and (not row_groups or isinstance(row_groups[0], int)):
            grouped_row_groups = [list(row_groups)]
        else:
            grouped_row_groups = [list(groups) for groups in row_groups]
        tables = []
        for source, source_row_groups in zip(scan_sources, grouped_row_groups, strict=True):
            parquet_file = pq.ParquetFile(source, filesystem=filesystem)
            table = parquet_file.read_row_groups(
                source_row_groups,
                columns=scan_columns,
                use_threads=True,
                use_pandas_metadata=True,
            )
            if filter_expression is not None:
                if not isinstance(filter_expression, pc.Expression):
                    filter_expression = pq.filters_to_expression(filter_expression)
                table = table.filter(filter_expression)
            tables.append(table.select(requested_columns))

    if not tables:
        return pa.table({name: pa.array([], type=pa.null()) for name in requested_columns})
    if len(tables) == 1:
        return _normalize_arrow_pandas_range_metadata(tables[0])
    return _normalize_arrow_pandas_range_metadata(pa.concat_tables(tables))


def _read_non_geometry_geoparquet_columns_with_pyarrow(
    path,
    *,
    columns,
    row_groups=None,
    filesystem=None,
    to_pandas_kwargs=None,
):
    """Legacy wrapper that returns pandas DataFrame directly.

    Kept for callers that need immediate pandas conversion.
    """
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    arrow_table = _read_non_geometry_geoparquet_columns_as_arrow(
        path, columns=columns, row_groups=row_groups, filesystem=filesystem,
    )
    return arrow_table.to_pandas(**to_pandas_kwargs)

@lru_cache(maxsize=32)
def _cached_geoparquet_crs_from_user_input(crs_value: str) -> Any:
    from pyproj import CRS

    return CRS.from_user_input(crs_value)


@lru_cache(maxsize=32)
def _cached_geoparquet_crs_from_json(crs_json: str) -> Any:
    from pyproj import CRS

    from vibespatial.api.io.arrow import _remove_id_from_member_of_ensembles

    crs = json.loads(crs_json)
    _remove_id_from_member_of_ensembles(crs)
    return CRS.from_user_input(crs)


def _geoparquet_geometry_column_crs(column_metadata: dict[str, Any]) -> Any:
    if "crs" in column_metadata:
        crs = column_metadata["crs"]
        if isinstance(crs, dict):
            try:
                return _cached_geoparquet_crs_from_json(
                    json.dumps(crs, sort_keys=True, separators=(",", ":"))
                )
            except Exception:
                return crs
        try:
            return _cached_geoparquet_crs_from_user_input(str(crs))
        except Exception:
            return crs
    try:
        return _cached_geoparquet_crs_from_user_input("OGC:CRS84")
    except Exception:
        return "OGC:CRS84"

def _read_geoparquet_table_with_pylibcudf(
    path,
    *,
    columns=None,
    row_groups=None,
    filesystem=None,
    filters=None,
    sources=None,
    available_columns=None,
):
    import pylibcudf as plc

    scan_sources = list(sources or [path])
    source = plc.io.types.SourceInfo(scan_sources)
    builder = plc.io.parquet.ParquetReaderOptions.builder(source)
    # GeoParquet metadata is loaded separately, and host bridge paths rebuild the
    # projected Arrow schema explicitly. Leaving these defaults enabled adds a
    # large one-time scan penalty in pylibcudf without helping the owned decode.
    builder.use_arrow_schema(False)
    builder.use_pandas_metadata(False)
    options = builder.build()
    if columns is not None:
        options.set_columns(list(columns))
    if row_groups is not None:
        if len(scan_sources) == 1 and (not row_groups or isinstance(row_groups[0], int)):
            grouped_row_groups = [list(row_groups)]
        else:
            grouped_row_groups = [list(groups) for groups in row_groups]
        options.set_row_groups(grouped_row_groups)
    if filters is not None:
        options.set_filter(
            _compile_pylibcudf_parquet_filter(
                filters,
                available_columns=tuple(available_columns or ()),
            )
        )
    table_with_metadata = plc.io.parquet.read_parquet(options)
    return table_with_metadata.tbl

def _parquet_column_path(*components: str) -> str:
    return ".".join(components)


def _load_geoparquet_metadata(path, *, filesystem=None, storage_options=None):
    from vibespatial.api.io.arrow import (
        _get_filesystem_path,
        _read_parquet_schema_and_metadata,
        _validate_and_decode_metadata,
    )

    filesystem, normalized_path = _get_filesystem_path(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    _, metadata = _read_parquet_schema_and_metadata(normalized_path, filesystem)
    geo_metadata = None
    if metadata is not None and b"geo" in metadata:
        geo_metadata = _validate_and_decode_metadata(metadata)
    return filesystem, normalized_path, metadata, geo_metadata


def _build_geoparquet_metadata_summary_from_pyarrow(
    path,
    *,
    filesystem,
    geo_metadata: dict[str, Any],
) -> GeoParquetMetadataSummary | None:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    import pyarrow.parquet as parquet

    from vibespatial.api.io.arrow import _coerce_pyarrow_parquet_source

    primary = geo_metadata["primary_column"]
    if primary not in geo_metadata["columns"]:
        return None
    column_meta = geo_metadata["columns"][primary]

    if "covering" in column_meta:
        bbox_meta = column_meta["covering"]["bbox"]
        xmin_path = _parquet_column_path(*bbox_meta["xmin"])
        ymin_path = _parquet_column_path(*bbox_meta["ymin"])
        xmax_path = _parquet_column_path(*bbox_meta["xmax"])
        ymax_path = _parquet_column_path(*bbox_meta["ymax"])
        source = "covering_bbox"
    elif column_meta.get("encoding") == "point":
        xmin_path = _parquet_column_path(primary, "x")
        ymin_path = _parquet_column_path(primary, "y")
        xmax_path = xmin_path
        ymax_path = ymin_path
        source = "point_encoding"
    else:
        return None

    row_group_rows: list[int] = []
    xmin: list[float] = []
    ymin: list[float] = []
    xmax: list[float] = []
    ymax: list[float] = []
    source_paths: list[str] | None = None
    row_group_source_indices: list[int] | None = None
    row_group_source_row_groups: list[int] | None = None
    required = (xmin_path, ymin_path, xmax_path, ymax_path)

    def append_metadata(file_metadata, *, source_index: int | None = None) -> bool:
        for row_group_index in range(file_metadata.num_row_groups):
            group = file_metadata.row_group(row_group_index)
            stats_by_path: dict[str, tuple[float, float]] = {}
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.path_in_schema not in required:
                    continue
                stats = column.statistics
                if stats is None or not getattr(stats, "has_min_max", False):
                    continue
                stats_by_path[column.path_in_schema] = (float(stats.min), float(stats.max))
            if any(path_name not in stats_by_path for path_name in required):
                return False
            row_group_rows.append(int(group.num_rows))
            xmin.append(stats_by_path[xmin_path][0])
            ymin.append(stats_by_path[ymin_path][0])
            xmax.append(stats_by_path[xmax_path][1])
            ymax.append(stats_by_path[ymax_path][1])
            if source_index is not None:
                assert row_group_source_indices is not None
                assert row_group_source_row_groups is not None
                row_group_source_indices.append(int(source_index))
                row_group_source_row_groups.append(int(row_group_index))
        return True

    if filesystem is not None and hasattr(filesystem, "get_file_info"):
        info = filesystem.get_file_info(path)
        if info.type == pafs.FileType.Directory:
            dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
            fragments = list(dataset.get_fragments())
            source_paths = []
            row_group_source_indices = []
            row_group_source_row_groups = []
            for source_index, fragment in enumerate(fragments):
                file_metadata = getattr(fragment, "metadata", None)
                if file_metadata is None:
                    return None
                source_paths.append(str(fragment.path))
                if not append_metadata(file_metadata, source_index=source_index):
                    return None
        elif info.type == pafs.FileType.File:
            path = _coerce_pyarrow_parquet_source(path)
            if not append_metadata(parquet.ParquetFile(path, filesystem=filesystem).metadata):
                return None
        else:
            return None
    elif filesystem is None and isinstance(path, (str, PathLike)) and Path(path).is_dir():
        dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
        fragments = list(dataset.get_fragments())
        source_paths = []
        row_group_source_indices = []
        row_group_source_row_groups = []
        for source_index, fragment in enumerate(fragments):
            file_metadata = getattr(fragment, "metadata", None)
            if file_metadata is None:
                return None
            source_paths.append(str(fragment.path))
            if not append_metadata(file_metadata, source_index=source_index):
                return None
    else:
        path = _coerce_pyarrow_parquet_source(path)
        if not append_metadata(parquet.ParquetFile(path, filesystem=filesystem).metadata):
            return None

    return build_geoparquet_metadata_summary(
        source=source,
        row_group_rows=row_group_rows,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        source_paths=source_paths,
        row_group_source_indices=row_group_source_indices,
        row_group_source_row_groups=row_group_source_row_groups,
    )

def _read_geoparquet_with_pyarrow(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    row_groups: tuple[int, ...] | list[int] | None = None,
    **kwargs,
):
    table, geo_metadata, df_attrs = _read_geoparquet_table_with_pyarrow(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        row_groups=row_groups,
        **kwargs,
    )
    from vibespatial.api.io.arrow import _arrow_to_geopandas

    return _arrow_to_geopandas(
        table,
        geo_metadata,
        to_pandas_kwargs,
        df_attrs,
        fallback_surface="geopandas.read_parquet",
        fallback_pipeline="io/read_parquet",
    )


def _normalize_arrow_pandas_range_metadata(table):
    effective_metadata = table.schema.metadata
    if effective_metadata is not None:
        effective_metadata = dict(effective_metadata)
        pandas_metadata_raw = effective_metadata.get(b"pandas")
        if pandas_metadata_raw is not None:
            pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
            index_columns = pandas_metadata.get("index_columns") or []
            if len(index_columns) == 1 and isinstance(index_columns[0], dict):
                range_spec = dict(index_columns[0])
                if range_spec.get("kind") == "range":
                    range_start = int(range_spec.get("start", 0))
                    range_stop = int(range_spec.get("stop", table.num_rows))
                    range_step = int(range_spec.get("step", 1))
                    expected_rows = max(0, (range_stop - range_start + (range_step - 1)) // range_step)
                    if expected_rows != int(table.num_rows):
                        range_spec["start"] = 0
                        range_spec["stop"] = int(table.num_rows)
                        range_spec["step"] = 1
                        pandas_metadata["index_columns"] = [range_spec]
                        effective_metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
    if effective_metadata is not None:
        return table.replace_schema_metadata(effective_metadata)
    return table


def _read_geoparquet_table_with_pyarrow(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    row_groups: tuple[int, ...] | list[int] | None = None,
    **kwargs,
):
    import pyarrow.parquet as parquet

    from vibespatial.api.io.arrow import (
        _check_if_covering_in_geo_metadata,
        _coerce_pyarrow_parquet_source,
        _get_bbox_encoding_column_name,
        _get_filesystem_path,
        _get_non_bbox_columns,
        _get_parquet_bbox_filter,
        _read_parquet_schema_and_metadata,
        _splice_bbox_and_filters,
        _validate_and_decode_metadata,
    )
    from vibespatial.api.io.file import _expand_user

    filesystem = kwargs.pop("filesystem", None)
    filesystem, normalized_path = _get_filesystem_path(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    normalized_path = _expand_user(normalized_path)
    schema, metadata = _read_parquet_schema_and_metadata(normalized_path, filesystem)
    parquet_source = _coerce_pyarrow_parquet_source(normalized_path)
    geo_metadata = _validate_and_decode_metadata(metadata)
    if len(geo_metadata["columns"]) == 0:
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )
    bbox_filter = _get_parquet_bbox_filter(geo_metadata, bbox) if bbox is not None else None
    if not columns and _check_if_covering_in_geo_metadata(geo_metadata):
        columns = _get_non_bbox_columns(schema, geo_metadata)
    if "filters" in kwargs:
        filters = _splice_bbox_and_filters(kwargs.pop("filters"), bbox_filter)
    else:
        filters = bbox_filter
    kwargs["use_pandas_metadata"] = True
    is_directory_dataset = isinstance(normalized_path, (str, PathLike)) and Path(normalized_path).is_dir()
    added_bbox_column = None
    row_group_columns = columns
    if row_group_columns is not None and filters is not None:
        row_group_columns = _merge_column_projection(
            row_group_columns,
            _parquet_filter_column_names(filters, available_columns=tuple(schema.names)),
        )
    if (
        row_groups is not None
        and bbox_filter is not None
        and _check_if_covering_in_geo_metadata(geo_metadata)
    ):
        bbox_column_name = _get_bbox_encoding_column_name(geo_metadata)
        if row_group_columns is not None and bbox_column_name not in row_group_columns:
            row_group_columns = [*row_group_columns, bbox_column_name]
            added_bbox_column = bbox_column_name
    if row_groups is None or is_directory_dataset:
        table = parquet.read_table(
            parquet_source,
            columns=columns,
            filesystem=filesystem,
            filters=filters,
            **kwargs,
        )
    else:
        import pyarrow.compute as pc

        parquet_file = parquet.ParquetFile(parquet_source, filesystem=filesystem)
        table = parquet_file.read_row_groups(
            list(row_groups),
            columns=row_group_columns,
            use_threads=kwargs.get("use_threads", True),
            use_pandas_metadata=kwargs["use_pandas_metadata"],
        )
        if filters is not None:
            if not isinstance(filters, pc.Expression):
                filters = parquet.filters_to_expression(filters)
            table = table.filter(filters)
        if added_bbox_column is not None and added_bbox_column in table.column_names:
            table = table.drop([added_bbox_column])
        if columns is not None:
            table = table.select(columns)
    effective_metadata = metadata
    if effective_metadata is not None:
        # `pq.read_table(..., use_pandas_metadata=True)` does not reliably
        # carry file footer metadata onto the returned Table schema. Reattach
        # it here so downstream `.to_pandas()` can restore index columns.
        table = table.replace_schema_metadata(effective_metadata)
    table = _normalize_arrow_pandas_range_metadata(table)
    effective_metadata = table.schema.metadata
    if effective_metadata and b"PANDAS_ATTRS" in effective_metadata:
        df_attrs = effective_metadata[b"PANDAS_ATTRS"]
    else:
        df_attrs = None
    return table, geo_metadata, df_attrs

def _decode_arrow_geoparquet_table_to_owned(
    table,
    geo_metadata: dict[str, Any],
    *,
    column_index: int | None = None,
) -> OwnedGeometryArray:
    primary = geo_metadata["primary_column"]
    field_index = table.schema.get_field_index(primary) if column_index is None else int(column_index)
    if field_index == -1:
        field_index = 0
    field = table.schema.field(field_index)
    array = table.column(field_index).combine_chunks()
    column_name = field.name
    if column_name not in geo_metadata["columns"]:
        if column_index is None:
            column_name = primary
        elif len(geo_metadata["columns"]) == 1:
            column_name = next(iter(geo_metadata["columns"]))
        else:
            raise KeyError(column_name)
    encoding = geo_metadata["columns"][column_name].get("encoding")
    return _decode_geoarrow_array_to_owned(field, array, encoding=encoding)


def _decode_geoparquet_table_to_owned(
    table,
    geo_metadata: dict[str, Any],
    *,
    column_index: int | None = None,
) -> OwnedGeometryArray:
    if _is_pylibcudf_table(table):
        unsupported = _unsupported_pylibcudf_geoparquet_encoding(geo_metadata)
        if unsupported is not None:
            column_name, encoding = unsupported
            record_fallback_event(
                surface="vibespatial.io.geoparquet",
                reason="explicit CPU fallback until pylibcudf device decode covers the current GeoParquet encoding",
                detail=f"column={column_name}, encoding={encoding!r}",
                selected=ExecutionMode.CPU,
                pipeline="io/read_parquet",
                d2h_transfer=True,
            )
            table = table.to_arrow()
        else:
            primary = geo_metadata["primary_column"]
            decode_index = 0 if column_index is None else int(column_index)
            return _decode_pylibcudf_geoparquet_column_with_arrow_fallback(
                table,
                column_name=primary,
                column_index=decode_index,
                encoding=geo_metadata["columns"][primary].get("encoding"),
            )

    return _decode_arrow_geoparquet_table_to_owned(
        table,
        geo_metadata,
        column_index=column_index,
    )


def _geoparquet_native_provenance(
    *,
    surface: str,
    path,
    backend: str,
    selected_row_groups: tuple[int, ...] | list[int] | None,
    bbox: tuple[float, float, float, float] | None,
    metadata_source: str | None,
    planner_strategy: str | None,
    chunk_rows: int | None,
) -> NativeReadProvenance:
    source = None if path is None else str(path)
    groups = None if selected_row_groups is None else tuple(int(group) for group in selected_row_groups)
    return NativeReadProvenance(
        surface=surface,
        format_name="geoparquet",
        source=source,
        backend=backend,
        selected_row_groups=groups,
        bbox=bbox,
        metadata_source=metadata_source,
        planner_strategy=planner_strategy,
        chunk_rows=chunk_rows,
    )


def _read_geoparquet_native_impl(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    to_pandas_kwargs=None,
    surface: str,
    operation: str,
    **kwargs,
) -> NativeTabularResult:
    if not has_pyarrow_support():
        raise ImportError("pyarrow is required for native GeoParquet reads")

    metadata_summary = None
    geo_metadata = None
    filesystem = kwargs.get("filesystem")
    normalized_path = path
    filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
    )
    if geo_metadata is not None:
        metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
            normalized_path,
            filesystem=filesystem,
            geo_metadata=geo_metadata,
        )
    if geo_metadata is None:
        raise ValueError("Missing geo metadata in Parquet/Feather file")
    _validate_geoparquet_bbox_support(geo_metadata, bbox)

    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    row_groups = kwargs.pop("row_groups", None)
    read_kwargs = dict(kwargs)
    read_kwargs.pop("filesystem", None)
    selected_row_groups = (
        scan_plan.selected_row_groups
        if scan_plan.selected_row_groups is not None
        else row_groups
    )
    read_plan = plan_geoparquet_read_backend(
        normalized_path,
        backend=backend,
        bbox=bbox,
        columns=columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=kwargs.get("filters"),
        to_pandas_kwargs=to_pandas_kwargs,
        geo_metadata=geo_metadata,
        available_columns=None,
    )
    if backend == "gpu" and not read_plan.can_use_pylibcudf:
        raise RuntimeError(read_plan.reason)
    if selected_row_groups is None and metadata_summary is not None:
        selected_row_groups = tuple(range(metadata_summary.row_group_count))
    effective_chunk_rows = _effective_geoparquet_chunk_rows(
        chunk_rows,
        selected_backend=read_plan.selected_backend,
    )
    chunk_plans = _plan_geoparquet_chunks(
        metadata_summary=metadata_summary,
        selected_row_groups=selected_row_groups,
        target_chunk_rows=effective_chunk_rows,
    )

    engine_plan = plan_geoparquet_engine(
        geo_metadata=geo_metadata,
        scan_plan=scan_plan,
        chunk_plans=chunk_plans,
        target_chunk_rows=effective_chunk_rows,
        read_plan=read_plan,
    )
    record_dispatch_event(
        surface=surface,
        operation=operation,
        implementation="repo_owned_geoparquet_engine",
        reason=engine_plan.reason,
        selected=read_plan.selected_mode,
    )
    if (
        read_plan.requested_backend == "auto"
        and read_plan.gpu_rejection_reason is not None
        and not _is_geoparquet_scan_ineligible_for_gpu_fallback(geo_metadata)
    ):
        _record_geoparquet_scan_backend_fallback(
            surface=surface,
            detail=read_plan.gpu_rejection_reason,
        )
    if scan_plan.row_group_pushdown:
        record_dispatch_event(
            surface=surface,
            operation="row_group_pushdown",
            implementation="repo_owned_geoparquet_planner",
            reason=(
                f"{scan_plan.planner_strategy} planner selected "
                f"{len(scan_plan.selected_row_groups or ())}/{scan_plan.available_row_groups} row groups "
                f"from {scan_plan.metadata_source}"
            ),
            selected=ExecutionMode.CPU,
        )

    provenance = _geoparquet_native_provenance(
        surface=surface,
        path=normalized_path,
        backend=read_plan.selected_backend,
        selected_row_groups=selected_row_groups,
        bbox=bbox,
        metadata_source=scan_plan.metadata_source,
        planner_strategy=scan_plan.planner_strategy,
        chunk_rows=effective_chunk_rows,
    )

    geometry_scan_columns = None
    if columns is None:
        geometry_scan_columns = list(geo_metadata["columns"])
    else:
        requested_columns = set(columns)
        geometry_scan_columns = [
            name for name in geo_metadata["columns"]
            if name in requested_columns
        ]
    if not geometry_scan_columns:
        geometry_scan_columns = None

    projected_schema = None
    df_attrs = None
    available_columns = None
    filter_columns: tuple[str, ...] = ()
    scan_projection = geometry_scan_columns or columns
    scan_sources = _geoparquet_scan_sources(
        normalized_path,
        metadata_summary=metadata_summary,
    )
    if read_plan.selected_backend == "pylibcudf":
        from vibespatial.api.io.arrow import _read_parquet_schema_and_metadata

        schema, _ = _read_parquet_schema_and_metadata(normalized_path, filesystem)
        available_columns = tuple(schema.names)
        projected_schema = _project_arrow_schema(schema, columns)
        schema_metadata = projected_schema.metadata
        df_attrs = None if schema_metadata is None else schema_metadata.get(b"PANDAS_ATTRS")
        filter_columns = _parquet_filter_column_names(
            read_kwargs.get("filters"),
            available_columns=available_columns,
        )
        scan_projection = _merge_column_projection(geometry_scan_columns, filter_columns)

    native_results: list[NativeTabularResult] = []
    for chunk in chunk_plans:
        chunk_row_groups = chunk.row_groups
        chunk_scan_row_groups = _geoparquet_scan_row_groups(
            metadata_summary=metadata_summary,
            selected_row_groups=chunk_row_groups,
        )
        if read_plan.selected_backend == "pylibcudf":
            table = _read_geoparquet_table_with_pylibcudf(
                normalized_path,
                columns=scan_projection,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                filters=read_kwargs.get("filters"),
                sources=scan_sources,
                available_columns=scan_projection or available_columns,
            )
            payload = _geoparquet_table_to_native_tabular_result(
                table,
                path=normalized_path,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
                schema=projected_schema,
                table_column_names=scan_projection,
                requested_columns=columns,
                to_pandas_kwargs=to_pandas_kwargs,
                df_attrs=df_attrs,
                scanned_with_pylibcudf=True,
                filters=read_kwargs.get("filters"),
                sources=scan_sources,
            )
            native_results.append(_apply_native_bbox_filter(payload, bbox))
            continue

        table, table_geo_metadata, table_df_attrs = _read_geoparquet_table_with_pyarrow(
            normalized_path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            row_groups=chunk_row_groups,
            filesystem=filesystem,
            **read_kwargs,
        )
        native_results.append(
            _geoparquet_table_to_native_tabular_result(
                table,
                path=normalized_path,
                row_groups=chunk_row_groups,
                filesystem=filesystem,
                geo_metadata=table_geo_metadata,
                schema=table.schema,
                requested_columns=columns,
                to_pandas_kwargs=to_pandas_kwargs,
                df_attrs=table_df_attrs,
                scanned_with_pylibcudf=False,
            )
        )

    if len(native_results) == 1:
        payload = native_results[0]
        # Native reads must not eagerly mirror device geometry buffers to host.
        # Public GeoDataFrame compatibility export can wrap the device state
        # directly; host authoritative views belong to explicit export paths.
        return NativeTabularResult(
            attributes=payload.attributes,
            geometry=payload.geometry,
            geometry_name=payload.geometry_name,
            column_order=payload.column_order,
            attrs=payload.attrs,
            secondary_geometry=payload.secondary_geometry,
            provenance=provenance,
        )

    return _concat_native_tabular_results(
        native_results,
        geometry_name=native_results[0].geometry_name,
        crs=native_results[0].geometry.crs,
        provenance=provenance,
    )


def read_geoparquet_native(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    to_pandas_kwargs=None,
    **kwargs,
) -> NativeTabularResult:
    """Read a GeoParquet file into the shared native tabular result boundary."""
    return _read_geoparquet_native_impl(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        chunk_rows=chunk_rows,
        backend=backend,
        to_pandas_kwargs=to_pandas_kwargs,
        surface="vibespatial.read_geoparquet_native",
        operation="read_native",
        **kwargs,
    )

def read_geoparquet_owned(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    chunk_rows: int | None = None,
    backend: str = "auto",
    **kwargs,
) -> OwnedGeometryArray:
    metadata_summary = None
    geo_metadata = None
    filesystem = kwargs.get("filesystem")
    normalized_path = path
    if has_pyarrow_support():
        filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(
            path,
            filesystem=filesystem,
            storage_options=storage_options,
        )
        if geo_metadata is not None:
            metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
                normalized_path,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
            )
    if geo_metadata is None:
        raise ValueError("GeoParquet metadata is required for owned-buffer scan")
    _validate_geoparquet_bbox_support(geo_metadata, bbox)
    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    row_groups = kwargs.pop("row_groups", None)
    selected_row_groups = (
        scan_plan.selected_row_groups
        if scan_plan.selected_row_groups is not None
        else row_groups
    )
    primary_column = geo_metadata["primary_column"]
    scan_columns = [primary_column] if columns is None else list(dict.fromkeys([*columns, primary_column]))
    decode_column_index = scan_columns.index(primary_column)
    read_plan = plan_geoparquet_read_backend(
        normalized_path,
        backend=backend,
        bbox=bbox,
        columns=scan_columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=kwargs.get("filters"),
        to_pandas_kwargs=None,
        geo_metadata=geo_metadata,
        available_columns=None,
    )
    if backend == "gpu" and not read_plan.can_use_pylibcudf:
        raise RuntimeError(read_plan.reason)
    if selected_row_groups is None and metadata_summary is not None:
        selected_row_groups = tuple(range(metadata_summary.row_group_count))
    effective_chunk_rows = _effective_geoparquet_chunk_rows(
        chunk_rows,
        selected_backend=read_plan.selected_backend,
    )
    chunk_plans = _plan_geoparquet_chunks(
        metadata_summary=metadata_summary,
        selected_row_groups=selected_row_groups,
        target_chunk_rows=effective_chunk_rows,
    )
    engine_plan = plan_geoparquet_engine(
        geo_metadata=geo_metadata,
        scan_plan=scan_plan,
        chunk_plans=chunk_plans,
        target_chunk_rows=effective_chunk_rows,
        read_plan=read_plan,
    )
    use_pylibcudf = read_plan.selected_backend == "pylibcudf"
    available_columns = None
    filter_columns: tuple[str, ...] = ()
    scan_sources = _geoparquet_scan_sources(
        normalized_path,
        metadata_summary=metadata_summary,
    )
    if use_pylibcudf and kwargs.get("filters") is not None:
        from vibespatial.api.io.arrow import _read_parquet_schema_and_metadata

        schema, _ = _read_parquet_schema_and_metadata(normalized_path, filesystem)
        available_columns = tuple(schema.names)
        filter_columns = _parquet_filter_column_names(
            kwargs.get("filters"),
            available_columns=available_columns,
        )
        scan_columns = _merge_column_projection(scan_columns, filter_columns) or scan_columns
        decode_column_index = scan_columns.index(primary_column)

    record_dispatch_event(
        surface="vibespatial.io.geoparquet",
        operation="read_owned",
        implementation="repo_owned_geoparquet_engine",
        reason=engine_plan.reason,
        selected=read_plan.selected_mode,
    )
    if read_plan.requested_backend == "auto" and read_plan.gpu_rejection_reason is not None:
        _record_geoparquet_scan_backend_fallback(
            surface="vibespatial.io.geoparquet",
            detail=read_plan.gpu_rejection_reason,
        )

    chunks: list[OwnedGeometryArray] = []
    for chunk in chunk_plans:
        chunk_scan_row_groups = _geoparquet_scan_row_groups(
            metadata_summary=metadata_summary,
            selected_row_groups=chunk.row_groups,
        )
        if use_pylibcudf:
            table = _read_geoparquet_table_with_pylibcudf(
                normalized_path,
                columns=scan_columns,
                row_groups=chunk_scan_row_groups,
                filesystem=filesystem,
                filters=kwargs.get("filters"),
                sources=scan_sources,
                available_columns=scan_columns or available_columns,
            )
        else:
            table, _, _ = _read_geoparquet_table_with_pyarrow(
                path,
                columns=scan_columns,
                storage_options=storage_options,
                bbox=bbox,
                row_groups=chunk.row_groups,
                **kwargs,
            )
        owned = _decode_geoparquet_table_to_owned(
            table,
            geo_metadata,
            column_index=decode_column_index,
        )
        if use_pylibcudf and bbox is not None:
            owned = _apply_owned_bbox_filter(owned, bbox)
        chunks.append(owned)
    return concatenate_owned_arrays(chunks)

def write_geoparquet(
    df,
    path,
    *,
    index: bool | None = None,
    compression: str | None = "snappy",
    geometry_encoding: str = "WKB",
    schema_version: str | None = None,
    write_covering_bbox: bool = False,
    **kwargs,
) -> None:
    payload = to_native_tabular_result(df)
    if payload is not None:
        if write_covering_bbox and "bbox" in payload.attributes.columns:
            raise ValueError(
                "An existing column 'bbox' already exists in the dataframe. "
                "Please rename to write covering bbox."
            )
        _write_geoparquet_native_tabular_result(
            payload,
            path,
            index=index,
            compression=compression,
            geometry_encoding=geometry_encoding,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            **kwargs,
        )
        return

    if write_covering_bbox and "bbox" in df.columns:
        raise ValueError(
            "An existing column 'bbox' already exists in the dataframe. "
            "Please rename to write covering bbox."
        )

    # Check if every geometry column can use owned backing (either already
    # device-backed or promotable from supported host geometry). When the full
    # geometry surface is native, use the owned-buffer encoder and avoid the
    # D→H→Shapely roundtrip.
    geometry_mask = df.dtypes.map(lambda d: d.name in ("geometry", "device_geometry"))
    geometry_columns = df.columns[geometry_mask]
    all_geometry_columns_owned = geometry_columns.size > 0 and all(
        isinstance(df[col].array, DeviceGeometryArray)
        or getattr(df[col].array, "_owned", None) is not None
        for col in geometry_columns
    )
    promotion_snapshots = []
    if not all_geometry_columns_owned:
        (
            all_geometry_columns_owned,
            promotion_snapshots,
        ) = _try_promote_geoparquet_geometry_columns_to_device(
            df,
            geometry_columns,
        )

    if all_geometry_columns_owned:
        try:
            small_write_detail = None
            terminal_owned = None
            if geometry_columns.size == 1:
                terminal_owned = df[geometry_columns[0]].array.to_owned()
                small_write_detail = _small_terminal_arrow_export_detail(
                    row_count=len(df),
                    polygonal_terminal_candidate=_owned_prefers_small_terminal_arrow_export(terminal_owned),
                )
            if small_write_detail is not None:
                _record_terminal_geoparquet_native_arrow_export(
                    detail=small_write_detail,
                    implementation="native_geodataframe_arrow_terminal_export",
                    row_count=len(df),
                    owned=terminal_owned,
                )
                from vibespatial.api._native_results import _spatial_to_native_tabular_result

                payload = _spatial_to_native_tabular_result(df)
                payload = _authoritative_native_tabular_result(payload)
                _write_native_tabular_result_with_arrow(
                    payload,
                    path,
                    index=index,
                    compression=compression,
                    geometry_encoding=geometry_encoding,
                    schema_version=schema_version,
                    write_covering_bbox=write_covering_bbox,
                    **kwargs,
                )
                return
            _write_geoparquet_native(
                df,
                path,
                index=index,
                compression=compression,
                geometry_encoding=geometry_encoding,
                schema_version=schema_version,
                write_covering_bbox=write_covering_bbox,
                geometry_columns=geometry_columns,
                **kwargs,
            )
            return
        finally:
            if promotion_snapshots:
                _restore_geoparquet_promoted_geometry_columns(promotion_snapshots)

    from vibespatial.api.io.arrow import _to_parquet

    _to_parquet(
        df,
        path,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        **kwargs,
    )
    _record_public_geoparquet_dispatch(
        selected=ExecutionMode.CPU,
        implementation="repo_owned_geoparquet_arrow_export",
        reason=(
            "GeoParquet export used the explicit Arrow compatibility writer because the "
            "public frame was not fully backed by native owned geometry buffers"
        ),
        row_count=len(df),
    )

def _write_geoparquet_native(
    df,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    geometry_columns,
    **kwargs,
) -> None:
    """Write GeoParquet using owned-buffer GeoArrow encoding — no Shapely materialization."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vibespatial.api.io.arrow import _create_metadata, _replace_table_schema_metadata

    geometry_indices = np.asarray(
        df.dtypes.map(lambda d: d.name in ("geometry", "device_geometry"))
    ).nonzero()[0]

    device_write = _write_geoparquet_native_device(
        df,
        path,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        geometry_columns=geometry_columns,
        **kwargs,
    )
    if device_write.written:
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.GPU,
            implementation="native_geodataframe_device_export",
            reason=(
                "GeoParquet export stayed on the native device writer for a device-backed "
                "public GeoDataFrame"
            ),
            row_count=len(df),
        )
        return
    if device_write.fallback_detail is not None:
        record_fallback_event(
            surface="geopandas.geodataframe.to_parquet",
            reason="explicit CPU fallback from the native device GeoParquet writer to the Arrow writer",
            detail=device_write.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/to_parquet",
            d2h_transfer=True,
        )
        _record_public_geoparquet_dispatch(
            selected=ExecutionMode.CPU,
            implementation="native_geodataframe_arrow_fallback_export",
            reason=(
                "GeoParquet export fell back to the explicit Arrow writer after the "
                "native device writer declined the sink"
            ),
            row_count=len(df),
            detail=device_write.fallback_detail,
        )
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_geodataframe_arrow_compatibility_export",
            row_count=len(df),
        )
    else:
        selected = (
            ExecutionMode.GPU
            if _geometry_columns_are_device_owned(df, geometry_columns)
            else ExecutionMode.CPU
        )
        _record_public_geoparquet_dispatch(
            selected=selected,
            implementation=(
                "native_geodataframe_arrow_device_encoded_export"
                if selected is ExecutionMode.GPU
                else "native_geodataframe_arrow_export"
            ),
            reason=(
                "GeoParquet export used the Arrow sink after encoding geometry "
                "from device-owned buffers because the native device writer was "
                "unavailable for this public GeoDataFrame"
                if selected is ExecutionMode.GPU
                else "GeoParquet export used the explicit Arrow writer because the native "
                "device writer was unavailable for this public GeoDataFrame"
            ),
            row_count=len(df),
        )

    # Build a table from non-geometry columns
    geometry_columns_set = set(geometry_columns)
    df_attr = pd.DataFrame(
        {
            col: (None if col in geometry_columns_set else df[col])
            for col in df.columns
        },
        index=df.index,
    )
    table = pa.Table.from_pandas(df_attr, preserve_index=index)

    geometry_encoding_dict = {}
    use_geoarrow = geometry_encoding.lower() == "geoarrow"

    for col_idx, col_name in zip(geometry_indices, geometry_columns):
        series = _authoritative_geometry_series(df[col_name])
        arr = series.array
        owned = arr.to_owned()

        if use_geoarrow:
            # Try native GeoArrow encoding from owned buffers
            fast_path_reason = _owned_geoarrow_fast_path_reason(series, include_z=None)
            if fast_path_reason is None:
                try:
                    field, geom_arr = encode_owned_geoarrow_array(
                        owned,
                        field_name=col_name,
                        crs=series.crs,
                        interleaved=False,
                    )
                    table = table.set_column(col_idx, field, geom_arr)
                    encoding_name = (
                        field.metadata[b"ARROW:extension:name"]
                        .decode()
                        .removeprefix("geoarrow.")
                    )
                    geometry_encoding_dict[col_name] = encoding_name
                    continue
                except Exception:
                    pass
            # Fallback: WKB for mixed/empty/3D geometries.
            # The WKB encoder (_encode_owned_wkb_array) records its own
            # dispatch/fallback events including d2h_transfer status, so we
            # only record the GeoArrow->WKB encoding decision here.
            record_fallback_event(
                surface="geopandas.geodataframe.to_parquet",
                reason=f"GeoArrow fast path unavailable for column {col_name}: {fast_path_reason}; falling back to WKB",
                detail=fast_path_reason or "encode error",
                selected=ExecutionMode.CPU,
                pipeline="io/to_parquet",
                d2h_transfer=False,
            )

        # WKB encoding — use owned-buffer WKB encoder when available
        field, wkb_arr = _encode_owned_wkb_array(
            owned, field_name=col_name, crs=series.crs
        )
        table = table.set_column(col_idx, field, wkb_arr)
        geometry_encoding_dict[col_name] = "WKB"

    # Build GeoParquet metadata
    geo_metadata = _create_metadata(
        df,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )

    if write_covering_bbox:
        bounds = df.bounds
        bbox_array = pa.StructArray.from_arrays(
            [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]],
            names=["xmin", "ymin", "xmax", "ymax"],
        )
        table = table.append_column("bbox", bbox_array)

    table = _replace_table_schema_metadata(
        table,
        geo_metadata=geo_metadata,
        attrs=df.attrs or None,
    )

    pq.write_table(table, path, compression=compression, **kwargs)

def read_geoparquet(
    path,
    *,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    **kwargs,
):
    """Read a GeoParquet file into a GeoDataFrame.

    When PyArrow is available the reader plans row-group selection from
    spatial metadata, keeps the table columnar through scan/decode, and
    only materializes a ``GeoDataFrame`` at the terminal public read
    boundary.

    Aliased as ``vibespatial.read_parquet()``.

    Parameters
    ----------
    path : str or Path
        Path to the GeoParquet file.
    columns : list of str, optional
        Subset of columns to read.
    storage_options : dict, optional
        Storage options for fsspec-compatible filesystems.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial filter bounding box for row-group pruning.
    to_pandas_kwargs : dict, optional
        Extra keyword arguments passed to ``pyarrow.Table.to_pandas()``.
    **kwargs
        Passed through to the underlying Parquet reader.

    Returns
    -------
    GeoDataFrame
    """
    if has_pyarrow_support():
        payload = _read_geoparquet_native_impl(
            path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            chunk_rows=None,
            backend="auto",
            to_pandas_kwargs=to_pandas_kwargs,
            surface="geopandas.read_parquet",
            operation="read_parquet",
            **kwargs,
        )
        frame = payload.to_geodataframe()
        attach_native_state_from_native_tabular_result(frame, payload)
        return frame
    from vibespatial.api.io.arrow import _read_parquet

    return _read_parquet(
        path,
        columns=columns,
        storage_options=storage_options,
        bbox=bbox,
        to_pandas_kwargs=to_pandas_kwargs,
        **kwargs,
    )

def benchmark_geoparquet_scan_engine(
    *,
    geometry_type: str = "point",
    rows: int = 100_000,
    geometry_encoding: str = "geoarrow",
    chunk_rows: int | None = None,
    compression: str | None = None,
    backend: str = "cpu",
    repeat: int = 5,
    seed: int = 0,
) -> GeoParquetEngineBenchmark:
    import tempfile

    if geometry_type == "point":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_points

        dataset = generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed))
    elif geometry_type == "polygon":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

        dataset = generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=rows, seed=seed, vertices=6)
        )
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}")

    gdf = dataset.to_geodataframe()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.parquet"
        gdf.to_parquet(path, geometry_encoding=geometry_encoding, compression=compression)
        planning_elapsed = 0.0
        scan_elapsed = 0.0
        decode_elapsed = 0.0
        concat_elapsed = 0.0
        total_elapsed = 0.0
        for _ in range(2):
            read_geoparquet_owned(path, backend=backend, chunk_rows=chunk_rows)
        for _ in range(repeat):
            iteration_start = perf_counter()
            metadata_summary = None
            filesystem = None
            normalized_path = path
            planning_start = perf_counter()
            filesystem, normalized_path, _, geo_metadata = _load_geoparquet_metadata(path)
            if geo_metadata is None:
                raise ValueError("GeoParquet metadata is required for owned-buffer benchmark")
            metadata_summary = _build_geoparquet_metadata_summary_from_pyarrow(
                normalized_path,
                filesystem=filesystem,
                geo_metadata=geo_metadata,
            )
            scan_plan = plan_geoparquet_scan(
                bbox=None,
                geo_metadata=geo_metadata,
                metadata_summary=metadata_summary,
            )
            selected_row_groups = scan_plan.selected_row_groups
            if selected_row_groups is None and metadata_summary is not None:
                selected_row_groups = tuple(range(metadata_summary.row_group_count))
            primary_column = geo_metadata["primary_column"]
            scan_columns = [primary_column]
            decode_column_index = 0
            read_plan = plan_geoparquet_read_backend(
                normalized_path,
                backend=backend,
                bbox=None,
                columns=scan_columns,
                storage_options=None,
                filesystem=filesystem,
                filters=None,
                to_pandas_kwargs=None,
                geo_metadata=geo_metadata,
            )
            if backend == "gpu" and not read_plan.can_use_pylibcudf:
                raise RuntimeError(read_plan.reason)
            effective_chunk_rows = _effective_geoparquet_chunk_rows(
                chunk_rows,
                selected_backend=read_plan.selected_backend,
            )
            use_pylibcudf = read_plan.selected_backend == "pylibcudf"
            chunk_plans = _plan_geoparquet_chunks(
                metadata_summary=metadata_summary,
                selected_row_groups=selected_row_groups,
                target_chunk_rows=effective_chunk_rows,
            )
            planning_elapsed += perf_counter() - planning_start

            chunks: list[OwnedGeometryArray] = []
            for chunk in chunk_plans:
                scan_start = perf_counter()
                if use_pylibcudf:
                    table = _read_geoparquet_table_with_pylibcudf(
                        normalized_path,
                        columns=scan_columns,
                        row_groups=chunk.row_groups or None,
                        filesystem=filesystem,
                    )
                else:
                    table, _, _ = _read_geoparquet_table_with_pyarrow(
                        path,
                        columns=scan_columns,
                        bbox=None,
                        row_groups=chunk.row_groups or None,
                    )
                scan_elapsed += perf_counter() - scan_start

                decode_start = perf_counter()
                chunks.append(
                    _decode_geoparquet_table_to_owned(
                        table,
                        geo_metadata,
                        column_index=decode_column_index,
                    )
                )
                decode_elapsed += perf_counter() - decode_start

            concat_start = perf_counter()
            concatenate_owned_arrays(chunks)
            concat_elapsed += perf_counter() - concat_start
            total_elapsed += perf_counter() - iteration_start
        elapsed = total_elapsed / repeat
    return GeoParquetEngineBenchmark(
        backend=read_plan.selected_backend,
        geometry_encoding=geometry_encoding,
        rows=rows,
        chunk_rows=chunk_rows,
        chunk_count=1 if chunk_rows is None else max(1, int(np.ceil(rows / chunk_rows))),
        elapsed_seconds=elapsed,
        rows_per_second=rows / elapsed if elapsed else float("inf"),
        planning_elapsed_seconds=planning_elapsed / repeat,
        scan_elapsed_seconds=scan_elapsed / repeat,
        decode_elapsed_seconds=decode_elapsed / repeat,
        concat_elapsed_seconds=concat_elapsed / repeat,
    )
