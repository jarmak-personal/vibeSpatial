from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from vibespatial.api._native_results import NativeTabularResult, to_native_tabular_result
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    concatenate_owned_arrays,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event

from .geoarrow import (
    _decode_geoarrow_array_to_owned,
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
    _decode_pylibcudf_geoparquet_table_to_owned,
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


def _payload_geometry_series(payload: NativeTabularResult):
    return payload.geometry.to_geoseries(
        index=payload.attributes.index,
        name=payload.geometry_name,
    )


def _record_terminal_geoparquet_compatibility_export(*, detail: str, implementation: str) -> None:
    record_dispatch_event(
        surface="geopandas.geodataframe.to_parquet",
        operation="to_parquet",
        implementation=implementation,
        reason=(
            "terminal GeoParquet export used the explicit Arrow compatibility writer "
            "after the native device writer declined a sink feature"
        ),
        detail=detail,
        selected=ExecutionMode.CPU,
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

    geometry_series = _payload_geometry_series(payload)

    device_write = _write_geoparquet_native_device_payload(
        payload.attributes,
        geometry_series,
        path,
        index=index,
        compression=compression,
        geometry_encoding=geometry_encoding,
        schema_version=schema_version,
        write_covering_bbox=write_covering_bbox,
        column_order=payload.column_order,
        frame_attrs=payload.attrs,
        **kwargs,
    )
    if device_write.written:
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
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_payload_arrow_compatibility_export",
        )

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
    row_groups: tuple[int, ...]
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

def _is_local_geoparquet_file(path) -> bool:
    candidate = path
    if isinstance(candidate, PathLike):
        candidate = candidate.__fspath__()
    if not isinstance(candidate, str):
        candidate = str(candidate)
    if "://" in candidate:
        return False
    return Path(candidate).is_file()


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
) -> tuple[bool, str]:
    if not has_pylibcudf_support():
        return False, "pylibcudf is not installed for the GPU GeoParquet reader"
    if not has_gpu_runtime():
        return False, "GPU GeoParquet reader requires an available CUDA runtime"
    if bbox is not None:
        return False, "bbox pushdown remains on the host path until the GPU filter expression path lands"
    if filesystem is not None or storage_options is not None:
        return False, "filesystem-backed GeoParquet reads still route through host pyarrow"
    if filters is not None:
        return False, "predicate pushdown remains on the host pyarrow path"
    if to_pandas_kwargs not in (None, {}):
        return False, "custom pandas conversion kwargs still route through host pyarrow"
    if geo_metadata is not None:
        primary = geo_metadata["primary_column"]
        if primary not in geo_metadata["columns"]:
            return False, "GeoParquet metadata without a readable primary geometry routes through host pyarrow"
        if "covering" in geo_metadata["columns"][primary]:
            return False, "covering-bbox columns still route through host pyarrow to preserve default projection"
        unsupported = _unsupported_pylibcudf_geoparquet_encoding(geo_metadata, columns)
        if unsupported is not None:
            column_name, encoding = unsupported
            return (
                False,
                f"geometry column {column_name!r} with GeoParquet encoding {encoding!r} still routes through host pyarrow",
            )
    if not _is_local_geoparquet_file(path):
        return False, "dataset and non-local GeoParquet paths still route through host pyarrow"
    return True, "local single-file GeoParquet scan can use the pylibcudf reader"

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
) -> GeoParquetEnginePlan:
    primary_column = None if geo_metadata is None else geo_metadata["primary_column"]
    geometry_encoding = None
    if geo_metadata is not None and primary_column is not None:
        geometry_encoding = geo_metadata["columns"][primary_column].get("encoding")
    backend = "pylibcudf" if scan_plan.uses_pylibcudf else "pyarrow"
    return GeoParquetEnginePlan(
        selected_path=scan_plan.selected_path,
        backend=backend,
        geometry_encoding=geometry_encoding,
        chunk_count=len(chunk_plans),
        target_chunk_rows=target_chunk_rows,
        uses_row_group_pruning=scan_plan.row_group_pushdown,
        reason=(
            "Use the GPU parquet reader when available, keep row-group pruning from the "
            "metadata planner, and decode supported geometry encodings directly into "
            "owned buffers after scan."
        ),
    )

def _plan_geoparquet_chunks(
    *,
    metadata_summary: GeoParquetMetadataSummary | None,
    selected_row_groups: tuple[int, ...] | list[int] | None,
    target_chunk_rows: int | None,
) -> tuple[GeoParquetChunkPlan, ...]:
    if not selected_row_groups:
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
    import json
    import warnings

    import pandas as pd

    from vibespatial.api.geodataframe import GeoDataFrame

    if geo_metadata is None:
        raise ValueError("GeoParquet metadata is required for pylibcudf GeoDataFrame decode")
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}

    if schema is not None:
        result_column_names = list(schema.names)
    elif hasattr(table, "column_names"):
        result_column_names = list(table.column_names)
    else:
        raise ValueError("pylibcudf GeoDataFrame decode requires schema metadata")

    geometry_columns = [col for col in geo_metadata["columns"] if col in result_column_names]
    geometry_columns.sort(key=result_column_names.index)
    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry = geo_metadata["primary_column"]
    if len(geometry_columns) and geometry not in geometry_columns:
        geometry = geometry_columns[0]
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
    # ADR-0042: keep Arrow tables through geometry decode and defer host
    # materialization to the explicit GeoDataFrame construction boundary.
    attrs_arrow = _read_non_geometry_geoparquet_columns_as_arrow(
        path,
        columns=non_geometry_columns,
        row_groups=row_groups,
        filesystem=filesystem,
    )

    columns_by_index = table.columns()
    decoded_geometry_owned: dict[str, tuple[OwnedGeometryArray, Any]] = {}
    row_count = None
    for column_name in geometry_columns:
        column_index = table_column_names.index(column_name)
        column_meta = geo_metadata["columns"][column_name]
        owned = _decode_pylibcudf_geoparquet_column_to_owned(
            columns_by_index[column_index],
            column_meta.get("encoding"),
        )
        crs = _geoparquet_geometry_column_crs(column_meta)
        if row_count is None:
            row_count = owned.row_count
        decoded_geometry_owned[column_name] = (owned, crs)

    # ADR-0042 transitional boundary: host conversion deferred until GeoDataFrame construction.
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}
    data = attrs_arrow.to_pandas(**to_pandas_kwargs)

    if data.empty and not non_geometry_columns and row_count is not None:
        data = pd.DataFrame(index=pd.RangeIndex(row_count))

    decoded_geometry: dict[str, pd.Series] = {}
    for column_name, (owned, crs) in decoded_geometry_owned.items():
        decoded_geometry[column_name] = pd.Series(
            DeviceGeometryArray._from_owned(owned, crs=crs),
            index=data.index,
            copy=False,
            name=column_name,
        )

    data_columns: dict[str, Any] = {}
    for column_name in result_column_names:
        if column_name in decoded_geometry:
            data_columns[column_name] = decoded_geometry[column_name]
        elif column_name in data.columns:
            data_columns[column_name] = data[column_name]
        # else: pandas index column, already restored into data.index
    gdf = pd.DataFrame(data_columns, index=data.index, copy=False)
    gdf.__class__ = GeoDataFrame
    gdf._geometry_column_name = geometry
    if df_attrs:
        gdf.attrs = json.loads(df_attrs)
    return gdf


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
):
    """ADR-0042: Read non-geometry columns as an Arrow table.

    Returns a PyArrow Table instead of a pandas DataFrame so that the
    caller can defer ``.to_pandas()`` to the GeoDataFrame construction
    boundary.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not columns:
        return pa.table({})
    if row_groups is None:
        return pq.read_table(path, columns=list(columns), filesystem=filesystem)
    parquet_file = pq.ParquetFile(path, filesystem=filesystem)
    return parquet_file.read_row_groups(
        list(row_groups),
        columns=list(columns),
        use_threads=True,
    )


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

def _geoparquet_geometry_column_crs(column_metadata: dict[str, Any]) -> Any:
    if "crs" in column_metadata:
        crs = column_metadata["crs"]
        if isinstance(crs, dict):
            from pyproj import CRS

            from vibespatial.api.io.arrow import _remove_id_from_member_of_ensembles

            _remove_id_from_member_of_ensembles(crs)
            return CRS.from_user_input(crs)
        try:
            from pyproj import CRS

            return CRS.from_user_input(crs)
        except Exception:
            return crs
    try:
        from pyproj import CRS

        return CRS.from_user_input("OGC:CRS84")
    except Exception:
        return "OGC:CRS84"

def _read_geoparquet_table_with_pylibcudf(
    path,
    *,
    columns=None,
    row_groups=None,
    filesystem=None,
):
    import pylibcudf as plc

    source = plc.io.types.SourceInfo([path])
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
        options.set_row_groups([list(row_groups)])
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
    import pyarrow.fs as pafs
    import pyarrow.parquet as parquet

    if filesystem is not None:
        if hasattr(filesystem, "get_file_info"):
            info = filesystem.get_file_info(path)
            if info.type != pafs.FileType.File:
                return None
        elif hasattr(filesystem, "isfile") and not filesystem.isfile(path):
            return None
    elif isinstance(path, (str, PathLike)) and Path(path).is_dir():
        return None
    metadata = parquet.ParquetFile(path, filesystem=filesystem).metadata
    primary = geo_metadata["primary_column"]
    if primary not in geo_metadata["columns"]:
        return None
    column_meta = geo_metadata["columns"][primary]
    row_group_rows: list[int] = []
    xmin: list[float] = []
    ymin: list[float] = []
    xmax: list[float] = []
    ymax: list[float] = []

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

    for row_group_index in range(metadata.num_row_groups):
        group = metadata.row_group(row_group_index)
        stats_by_path: dict[str, tuple[float, float]] = {}
        required = (xmin_path, ymin_path, xmax_path, ymax_path)
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            if column.path_in_schema not in required:
                continue
            stats = column.statistics
            if stats is None or not getattr(stats, "has_min_max", False):
                continue
            stats_by_path[column.path_in_schema] = (float(stats.min), float(stats.max))
        if any(path_name not in stats_by_path for path_name in required):
            return None
        row_group_rows.append(int(group.num_rows))
        xmin.append(stats_by_path[xmin_path][0])
        ymin.append(stats_by_path[ymin_path][0])
        xmax.append(stats_by_path[xmax_path][1])
        ymax.append(stats_by_path[ymax_path][1])

    return build_geoparquet_metadata_summary(
        source=source,
        row_group_rows=row_group_rows,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
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

    return _arrow_to_geopandas(table, geo_metadata, to_pandas_kwargs, df_attrs)

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
            normalized_path,
            columns=columns,
            filesystem=filesystem,
            filters=filters,
            **kwargs,
        )
    else:
        parquet_file = parquet.ParquetFile(normalized_path, filesystem=filesystem)
        table = parquet_file.read_row_groups(
            list(row_groups),
            columns=row_group_columns,
            use_threads=kwargs.get("use_threads", True),
            use_pandas_metadata=kwargs["use_pandas_metadata"],
        )
        if filters is not None:
            table = table.filter(filters)
        if added_bbox_column is not None and added_bbox_column in table.column_names:
            table = table.drop([added_bbox_column])
    if metadata is not None:
        # `pq.read_table(..., use_pandas_metadata=True)` does not reliably
        # carry file footer metadata onto the returned Table schema. Reattach
        # it here so downstream `.to_pandas()` can restore index columns.
        table = table.replace_schema_metadata(metadata)
    if metadata and b"PANDAS_ATTRS" in metadata:
        df_attrs = metadata[b"PANDAS_ATTRS"]
    else:
        df_attrs = None
    return table, geo_metadata, df_attrs

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
            return _decode_pylibcudf_geoparquet_table_to_owned(
                table,
                geo_metadata,
                column_index=column_index,
            )

    primary = geo_metadata["primary_column"]
    field_index = table.schema.get_field_index(primary) if column_index is None else int(column_index)
    if field_index == -1:
        field_index = 0
    field = table.schema.field(field_index)
    array = table.column(field_index).combine_chunks()
    encoding = geo_metadata["columns"][primary].get("encoding")
    return _decode_geoarrow_array_to_owned(field, array, encoding=encoding)

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
    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    row_groups = kwargs.pop("row_groups", None)
    selected_row_groups = scan_plan.selected_row_groups or row_groups
    if selected_row_groups is None and metadata_summary is not None:
        selected_row_groups = tuple(range(metadata_summary.row_group_count))
    chunk_plans = _plan_geoparquet_chunks(
        metadata_summary=metadata_summary,
        selected_row_groups=selected_row_groups,
        target_chunk_rows=chunk_rows,
    )
    engine_plan = plan_geoparquet_engine(
        geo_metadata=geo_metadata,
        scan_plan=scan_plan,
        chunk_plans=chunk_plans,
        target_chunk_rows=chunk_rows,
    )
    primary_column = geo_metadata["primary_column"]
    scan_columns = [primary_column] if columns is None else list(dict.fromkeys([*columns, primary_column]))
    decode_column_index = scan_columns.index(primary_column)
    use_pylibcudf = backend == "gpu" or (backend == "auto" and scan_plan.uses_pylibcudf)
    if use_pylibcudf and not scan_plan.uses_pylibcudf:
        raise RuntimeError("pylibcudf backend requested but unavailable")

    record_dispatch_event(
        surface="vibespatial.io.geoparquet",
        operation="read_owned",
        implementation="repo_owned_geoparquet_engine",
        reason=engine_plan.reason,
        selected=ExecutionMode.GPU if use_pylibcudf else ExecutionMode.CPU,
    )
    if bbox is not None:
        record_fallback_event(
            surface="vibespatial.io.geoparquet",
            reason="explicit CPU fallback until GPU bbox expression pushdown lands",
            detail="owned scan used host-side filter path after row-group pruning",
            selected=ExecutionMode.CPU,
            pipeline="io/read_parquet",
        )

    chunks: list[OwnedGeometryArray] = []
    for chunk in chunk_plans:
        if use_pylibcudf and bbox is None:
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
                storage_options=storage_options,
                bbox=bbox,
                row_groups=chunk.row_groups or None,
                **kwargs,
            )
        chunks.append(
            _decode_geoparquet_table_to_owned(
                table,
                geo_metadata,
                column_index=decode_column_index,
            )
        )
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
        record_dispatch_event(
            surface="geopandas.geodataframe.to_parquet",
            operation="to_parquet",
            implementation="repo_owned_geoparquet_adapter",
            reason="GeoParquet export routes through the repo-owned adapter and preserves the GPU-first metadata contract.",
            selected=ExecutionMode.CPU,
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

    record_dispatch_event(
        surface="geopandas.geodataframe.to_parquet",
        operation="to_parquet",
        implementation="repo_owned_geoparquet_adapter",
        reason="GeoParquet export routes through the repo-owned adapter and preserves the GPU-first metadata contract.",
        selected=ExecutionMode.CPU,
    )

    if write_covering_bbox and "bbox" in df.columns:
        raise ValueError(
            "An existing column 'bbox' already exists in the dataframe. "
            "Please rename to write covering bbox."
        )

    # Check if any geometry column already has owned backing (either via
    # DeviceGeometryArray or a GeometryArray with _owned populated).  When
    # owned is available, use the native owned-buffer encoder which avoids
    # the D→H→Shapely roundtrip.  We intentionally do NOT trigger owned
    # construction here — if only Shapely data exists, the standard pyarrow
    # path with shapely.to_wkb is faster than building owned first.
    geometry_mask = df.dtypes.map(lambda d: d.name in ("geometry", "device_geometry"))
    geometry_columns = df.columns[geometry_mask]
    has_owned = any(
        isinstance(df[col].array, DeviceGeometryArray)
        or getattr(df[col].array, "_owned", None) is not None
        for col in geometry_columns
    )

    if has_owned and geometry_columns.size > 0:
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
    elif device_write.compatibility_detail is not None:
        _record_terminal_geoparquet_compatibility_export(
            detail=device_write.compatibility_detail,
            implementation="native_geodataframe_arrow_compatibility_export",
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
        series = df[col_name]
        arr = series.array
        owned = arr.to_owned()

        if use_geoarrow:
            # Try native GeoArrow encoding from owned buffers
            fast_path_reason = _owned_geoarrow_fast_path_reason(series, include_z=None)
            if fast_path_reason is None:
                try:
                    owned._ensure_host_state()  # GeoArrow reads host buffers
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
    spatial metadata, decodes WKB geometry on GPU when possible, and
    produces device-resident ``OwnedGeometryArray`` without host
    round-trips.

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
    scan_plan = plan_geoparquet_scan(
        bbox=bbox,
        geo_metadata=geo_metadata,
        metadata_summary=metadata_summary,
    )
    filters = kwargs.get("filters")
    use_pylibcudf, gpu_read_reason = _supports_pylibcudf_geoparquet_read(
        normalized_path,
        bbox=bbox,
        columns=columns,
        storage_options=storage_options,
        filesystem=filesystem,
        filters=filters,
        to_pandas_kwargs=to_pandas_kwargs,
        geo_metadata=geo_metadata,
    )
    record_dispatch_event(
        surface="geopandas.read_parquet",
        operation="read_parquet",
        implementation="repo_owned_geoparquet_adapter",
        reason=scan_plan.reason,
        selected=ExecutionMode.GPU if scan_plan.uses_pylibcudf and use_pylibcudf else ExecutionMode.CPU,
    )
    row_groups = kwargs.pop("row_groups", None)
    planned_row_groups = scan_plan.selected_row_groups or row_groups
    if scan_plan.uses_pylibcudf and use_pylibcudf:
        return _read_geoparquet_with_pylibcudf(
            normalized_path,
            columns=columns,
            row_groups=planned_row_groups,
            filesystem=filesystem,
            geo_metadata=geo_metadata,
            to_pandas_kwargs=to_pandas_kwargs,
        )
    if scan_plan.row_group_pushdown:
        record_dispatch_event(
            surface="geopandas.read_parquet",
            operation="row_group_pushdown",
            implementation="repo_owned_geoparquet_planner",
            reason=(
                f"{scan_plan.planner_strategy} planner selected "
                f"{len(scan_plan.selected_row_groups or ())}/{scan_plan.available_row_groups} row groups "
                f"from {scan_plan.metadata_source}"
            ),
            selected=ExecutionMode.CPU,
        )
    if has_pyarrow_support():
        return _read_geoparquet_with_pyarrow(
            path,
            columns=columns,
            storage_options=storage_options,
            bbox=bbox,
            to_pandas_kwargs=to_pandas_kwargs,
            row_groups=planned_row_groups,
            **kwargs,
        )
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
        gdf.to_parquet(path, geometry_encoding=geometry_encoding)
        planning_elapsed = 0.0
        scan_elapsed = 0.0
        decode_elapsed = 0.0
        concat_elapsed = 0.0
        total_elapsed = 0.0
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
            chunk_plans = _plan_geoparquet_chunks(
                metadata_summary=metadata_summary,
                selected_row_groups=selected_row_groups,
                target_chunk_rows=chunk_rows,
            )
            primary_column = geo_metadata["primary_column"]
            scan_columns = [primary_column]
            decode_column_index = 0
            use_pylibcudf = backend == "gpu" or (backend == "auto" and scan_plan.uses_pylibcudf)
            if use_pylibcudf and not scan_plan.uses_pylibcudf:
                raise RuntimeError("pylibcudf backend requested but unavailable")
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
        backend="pylibcudf" if backend == "gpu" else "pyarrow",
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
