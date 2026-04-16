from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from vibespatial.geometry.owned import (
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import get_fallback_events, record_fallback_event

# Re-exported from io_geojson for backward compatibility
from .geojson import (  # noqa: F401
    GeoJSONIngestBenchmark,
    GeoJSONIngestPlan,
    GeoJSONOwnedBatch,
    benchmark_geojson_ingest,
    plan_geojson_ingest,
    read_geojson_owned,
)
from .support import IOFormat, IOOperation, IOPathKind, plan_io_support


@dataclass(frozen=True)
class VectorFilePlan:
    format: IOFormat
    operation: IOOperation
    selected_path: IOPathKind
    driver: str
    implementation: str
    reason: str


@dataclass(frozen=True)
class ShapefileIngestPlan:
    implementation: str
    selected_strategy: str
    uses_pyogrio_container: bool
    uses_arrow_batch: bool
    uses_native_wkb_decode: bool
    reason: str


@dataclass
class ShapefileOwnedBatch:
    geometry: OwnedGeometryArray
    attributes_table: object
    metadata: dict[str, object]


@dataclass(frozen=True)
class ShapefileIngestBenchmark:
    implementation: str
    geometry_type: str
    rows: int
    elapsed_seconds: float
    rows_per_second: float


@dataclass(frozen=True)
class _CsvSpatialLayout:
    column_names: tuple[str, ...]
    spatial_columns: dict[str, int]
    geometry_format: str | None = None


@dataclass(frozen=True)
class _CsvGpuSelection:
    payload: object
    implementation: str
    reason: str


def _native_payload_geometry_type(payload) -> str:
    from vibespatial.geometry.buffers import GeometryFamily

    family_names = {
        GeometryFamily.POINT: "Point",
        GeometryFamily.LINESTRING: "LineString",
        GeometryFamily.POLYGON: "Polygon",
        GeometryFamily.MULTIPOINT: "MultiPoint",
        GeometryFamily.MULTILINESTRING: "MultiLineString",
        GeometryFamily.MULTIPOLYGON: "MultiPolygon",
    }

    owned = payload.geometry.owned
    if owned is not None:
        geometry_types = sorted(
            family_names[family]
            for family in owned.families
            if family in family_names
        )
        return geometry_types[0] if len(geometry_types) == 1 else "Unknown"

    series = payload.geometry.series
    if series is None:
        return "Unknown"
    geometry_types = sorted(series.geom_type.dropna().unique())
    return geometry_types[0] if len(geometry_types) == 1 else "Unknown"


def _native_payload_file_crs(payload) -> str | None:
    crs = payload.geometry.crs
    if crs is None:
        return None
    if hasattr(crs, "to_wkt"):
        return crs.to_wkt()
    try:
        from pyproj import CRS

        return CRS.from_user_input(crs).to_wkt()
    except Exception:
        return str(crs)


def _prepare_native_payload_for_file(payload, *, index: bool | None):
    from pandas.api.types import is_integer_dtype

    from vibespatial.api._native_results import NativeTabularResult

    include_index = index
    if include_index is None:
        include_index = list(payload.attributes.index.names) != [None] or not is_integer_dtype(
            payload.attributes.index.dtype
        )
    if not include_index:
        return payload

    attributes = payload.attributes.reset_index(drop=False)
    return NativeTabularResult(
        attributes=attributes,
        geometry=payload.geometry,
        geometry_name=payload.geometry_name,
        column_order=tuple([*attributes.columns, payload.geometry_name]),
        attrs=payload.attrs,
    )


def _native_attribute_table_from_file_attributes(attributes, *, row_count: int):
    import pandas as pd

    from vibespatial.api._native_results import NativeAttributeTable

    if isinstance(attributes, NativeAttributeTable):
        return attributes
    if attributes is None:
        return NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count)))
    if isinstance(attributes, dict):
        frame = pd.DataFrame(attributes)
        if not attributes:
            frame = pd.DataFrame(index=pd.RangeIndex(row_count))
        return NativeAttributeTable(dataframe=frame)
    if isinstance(attributes, pd.DataFrame):
        return NativeAttributeTable(dataframe=attributes)
    return NativeAttributeTable.from_value(attributes)


def _geojson_properties_to_frame(properties, *, row_count: int):
    import pandas as pd

    if properties:
        return pd.DataFrame(properties)
    return pd.DataFrame(index=pd.RangeIndex(row_count))


def _native_attribute_table_from_geojson_properties_loader(
    properties_loader,
    *,
    row_count: int,
):
    import pandas as pd

    from vibespatial.api._native_results import NativeAttributeTable

    def _load_frame():
        return _geojson_properties_to_frame(
            properties_loader(),
            row_count=row_count,
        )

    return NativeAttributeTable.from_loader(
        _load_frame,
        index_override=pd.RangeIndex(row_count),
        columns=(),
    )


_OSM_PYOGRIO_ARROW_LAYERS = frozenset(
    {"points", "lines", "multilinestrings", "multipolygons", "other_relations"}
)
_OSM_PYOGRIO_DEFAULT_LAYERS = (
    "points",
    "lines",
    "multilinestrings",
    "multipolygons",
    "other_relations",
)


def _normalize_osm_layer(layer) -> str:
    if layer is None:
        return "all"
    normalized = str(layer).strip().lower()
    aliases = {
        "all": "all",
        "points": "points",
        "nodes": "points",
        "lines": "lines",
        "ways": "ways",
        "multipolygons": "multipolygons",
        "polygons": "multipolygons",
        "relations": "relations",
        "multilinestrings": "multilinestrings",
        "other_relations": "other_relations",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported OSM PBF layer. Expected one of "
            "'points', 'lines', 'ways', 'multipolygons', 'relations', "
            "'multilinestrings', 'other_relations', or 'all'."
        )
    return aliases[normalized]


def _osm_uses_pyogrio_compat_layer(*, layer, tags) -> bool:
    normalized_layer = _normalize_osm_layer(layer)
    return normalized_layer in _OSM_PYOGRIO_ARROW_LAYERS


def _osm_uses_pyogrio_compat_default(*, layer, tags, geometry_only: bool) -> bool:
    return layer is None and (geometry_only or tags in ("ways", True, False))


def _prepare_osm_pyogrio_kwargs(
    *,
    layer,
    tags,
    columns,
    kwargs,
) -> tuple[object, dict[str, object]]:
    pyogrio_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in ("engine", "tags", "geometry_only")
    }
    layer_columns = _osm_pyogrio_columns(
        layer=layer,
        geometry_only=bool(kwargs.get("geometry_only", False)),
        tags=tags,
    )
    if layer_columns is not None:
        columns = layer_columns
    return columns, pyogrio_kwargs


def _osm_pyogrio_columns(
    *,
    layer: str,
    geometry_only: bool,
    tags,
) -> list[str] | None:
    if geometry_only:
        return []
    if tags is not False:
        return None

    normalized_layer = _normalize_osm_layer(layer)
    if normalized_layer == "multipolygons":
        return ["osm_id", "osm_way_id"]
    return ["osm_id"]


def _native_tabular_result_with_attribute_column(payload, name: str, values):
    from vibespatial.api._native_results import NativeTabularResult

    attributes = payload.attributes.with_column(name, values)
    return NativeTabularResult(
        attributes=attributes,
        geometry=payload.geometry,
        geometry_name=payload.geometry_name,
        column_order=tuple([*attributes.columns, payload.geometry_name]),
        attrs=payload.attrs,
        secondary_geometry=payload.secondary_geometry,
        provenance=payload.provenance,
    )


def _add_osm_element_column(payload, *, osm_element: str):
    import numpy as np

    row_count = int(payload.geometry.row_count)
    values = np.empty(row_count, dtype=object)
    values[:] = osm_element
    return _native_tabular_result_with_attribute_column(payload, "osm_element", values)


def _add_osm_element_column_for_multipolygons(payload):
    import numpy as np
    import pyarrow.compute as pc

    row_count = int(payload.geometry.row_count)
    values = np.empty(row_count, dtype=object)
    values[:] = "relation"
    if "osm_way_id" not in payload.attributes.columns:
        return _native_tabular_result_with_attribute_column(payload, "osm_element", values)
    attrs_arrow = payload.attributes.to_arrow(index=False, columns=["osm_way_id"])
    way_mask = pc.is_valid(attrs_arrow["osm_way_id"]).to_numpy(zero_copy_only=False)
    values[way_mask] = "way"
    return _native_tabular_result_with_attribute_column(payload, "osm_element", values)


def _read_osm_pbf_pyogrio_layer_public(
    filename,
    *,
    layer: str,
    target_crs: str | None = None,
    geometry_only: bool = False,
    tags: bool | str = "ways",
    include_element_column: bool = False,
):
    import pyogrio

    import geopandas as gpd

    kwargs = {"layer": layer}
    layer_columns = _osm_pyogrio_columns(
        layer=layer,
        geometry_only=geometry_only,
        tags=tags,
    )
    if layer_columns is not None:
        kwargs["columns"] = layer_columns
    frame = pyogrio.read_dataframe(filename, **kwargs)
    if geometry_only:
        frame = gpd.GeoDataFrame(geometry=frame.geometry, crs=frame.crs)
    elif not include_element_column:
        pass
    elif layer == "points":
        frame["osm_element"] = "node"
    elif layer == "lines":
        frame["osm_element"] = "way"
    elif layer in {"multilinestrings", "other_relations"}:
        frame["osm_element"] = "relation"
    else:
        frame["osm_element"] = "relation"
        if "osm_way_id" in frame.columns:
            way_mask = frame["osm_way_id"].notna()
            frame.loc[way_mask, "osm_element"] = "way"

    if target_crs is not None and frame.crs is not None:
        frame = frame.to_crs(target_crs)
    elif target_crs is not None and frame.crs is None:
        frame = frame.set_crs(target_crs)
    return frame


def _read_osm_pbf_pyogrio_layer_native(
    filename,
    *,
    layer: str,
    target_crs: str | None = None,
    geometry_only: bool = False,
    tags: bool | str = "ways",
    include_element_column: bool = False,
):
    import pyogrio

    from vibespatial.api._native_results import (
        NativeTabularResult,
        _empty_geometry_native_result,
        _spatial_to_native_tabular_result,
        native_attribute_table_from_arrow_table,
    )

    layer_columns = _osm_pyogrio_columns(
        layer=layer,
        geometry_only=geometry_only,
        tags=tags,
    )
    metadata, table = pyogrio.read_arrow(
        filename,
        layer=layer,
        columns=layer_columns,
    )
    geometry_name = str(metadata.get("geometry_name") or "geometry")
    if geometry_name != "geometry" and "geometry" not in table.column_names:
        geometry_name = "geometry"
    if table.num_rows == 0:
        geom_idx, _geom_field = _select_arrow_geometry_column(table, metadata)
        attrs_table = table.remove_column(geom_idx)
        attributes = native_attribute_table_from_arrow_table(attrs_table)
        effective_crs = target_crs or metadata.get("crs") or "EPSG:4326"
        return NativeTabularResult(
            attributes=attributes,
            geometry=_empty_geometry_native_result(
                geometry_name=geometry_name,
                crs=effective_crs,
            ),
            geometry_name=geometry_name,
            column_order=tuple([*attributes.columns, geometry_name]),
        )
    try:
        payload = _pyogrio_arrow_wkb_to_native_tabular_result(
            table,
            metadata,
            target_crs=target_crs,
        )
    except NotImplementedError as exc:
        if "unsupported geometry family" not in str(exc):
            raise
        record_fallback_event(
            surface="vibespatial.io.osm_pbf",
            reason=(
                "explicit CPU compatibility fallback for an OSM pyogrio layer after "
                "GPU WKB decode could not handle the layer geometry family"
            ),
            detail=f"layer={layer!r}, detail={type(exc).__name__}: {exc}",
            pipeline="io/read_osm_pbf_pyogrio_layer",
            d2h_transfer=False,
        )
        return _spatial_to_native_tabular_result(
            _read_osm_pbf_pyogrio_layer_public(
                filename,
                layer=layer,
                target_crs=target_crs,
                geometry_only=geometry_only,
                tags=tags,
                include_element_column=include_element_column,
            )
        )
    if geometry_only:
        return payload
    if not include_element_column:
        return payload
    if layer == "points":
        return _add_osm_element_column(payload, osm_element="node")
    if layer == "lines":
        return _add_osm_element_column(payload, osm_element="way")
    if layer == "multipolygons":
        return _add_osm_element_column_for_multipolygons(payload)
    if layer in {"multilinestrings", "other_relations"}:
        return _add_osm_element_column(payload, osm_element="relation")
    return payload


def _read_osm_pbf_supported_layers_native(
    filename,
    *,
    target_crs: str | None = None,
    geometry_only: bool = False,
    tags: bool | str = "ways",
):
    from concurrent.futures import ThreadPoolExecutor

    from vibespatial.api._native_results import _concat_native_tabular_results

    worker_count = min(4, len(_OSM_PYOGRIO_DEFAULT_LAYERS))
    if worker_count <= 1:
        results = [
            _read_osm_pbf_pyogrio_layer_native(
                filename,
                layer=layer_name,
                target_crs=target_crs,
                geometry_only=geometry_only,
                tags=tags,
                include_element_column=True,
            )
            for layer_name in _OSM_PYOGRIO_DEFAULT_LAYERS
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="osm-pbf-layer") as executor:
            futures = [
                executor.submit(
                    _read_osm_pbf_pyogrio_layer_native,
                    filename,
                    layer=layer_name,
                    target_crs=target_crs,
                    geometry_only=geometry_only,
                    tags=tags,
                    include_element_column=True,
                )
                for layer_name in _OSM_PYOGRIO_DEFAULT_LAYERS
            ]
            results = [future.result() for future in futures]
    return _concat_native_tabular_results(
        results,
        geometry_name="geometry",
        crs=target_crs or "EPSG:4326",
    )


def _read_osm_pbf_supported_layers_public(
    filename,
    *,
    target_crs: str | None = None,
    geometry_only: bool = False,
    tags: bool | str = "ways",
):
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    import geopandas as gpd

    parts: list[gpd.GeoDataFrame] = []
    worker_count = min(4, len(_OSM_PYOGRIO_DEFAULT_LAYERS))
    if worker_count <= 1:
        frames = [
            _read_osm_pbf_pyogrio_layer_public(
                filename,
                layer=layer_name,
                target_crs=None,
                geometry_only=geometry_only,
                tags=tags,
                include_element_column=True,
            )
            for layer_name in _OSM_PYOGRIO_DEFAULT_LAYERS
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="osm-pbf-layer") as executor:
            futures = [
                executor.submit(
                    _read_osm_pbf_pyogrio_layer_public,
                    filename,
                    layer=layer_name,
                    target_crs=None,
                    geometry_only=geometry_only,
                    tags=tags,
                    include_element_column=True,
                )
                for layer_name in _OSM_PYOGRIO_DEFAULT_LAYERS
            ]
            frames = [future.result() for future in futures]

    for frame in frames:
        if len(frame) > 0:
            parts.append(frame)

    if parts:
        combined = pd.concat(parts, ignore_index=True, sort=False)
        crs = next((part.crs for part in parts if part.crs is not None), "EPSG:4326")
        gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))

    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)
    elif target_crs is not None and gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    return gdf


def _native_file_result_from_owned(
    owned: OwnedGeometryArray,
    *,
    geometry_name: str = "geometry",
    crs=None,
    attributes=None,
    row_count: int | None = None,
):
    from vibespatial.api._native_results import GeometryNativeResult, NativeTabularResult

    if row_count is None:
        row_count = int(owned.row_count)
    attribute_table = _native_attribute_table_from_file_attributes(
        attributes,
        row_count=row_count,
    )
    return NativeTabularResult(
        attributes=attribute_table,
        geometry=GeometryNativeResult.from_owned(owned, crs=crs),
        geometry_name=geometry_name,
        column_order=tuple([*attribute_table.columns, geometry_name]),
    )


def _native_geojson_result_from_gpu_result(
    gpu_result,
    *,
    target_crs: str | None = None,
):
    effective_crs = _resolve_target_crs_for_owned(
        gpu_result.owned,
        source_crs="EPSG:4326",
        target_crs=target_crs,
    )
    row_count = int(gpu_result.n_features)
    return _native_file_result_from_owned(
        gpu_result.owned,
        crs=effective_crs,
        attributes=_native_attribute_table_from_geojson_properties_loader(
            gpu_result.properties_loader(),
            row_count=row_count,
        ),
        row_count=row_count,
    )


def _csv_detect_geometry_value_format(value: str) -> str:
    stripped = value.lstrip()
    if (
        len(stripped) >= 10
        and stripped[:2] in ("00", "01")
        and len(stripped) % 2 == 0
        and all(ch in "0123456789abcdefABCDEF" for ch in stripped)
    ):
        return "wkb"
    return "wkt"


def _sniff_csv_spatial_layout(path: Path) -> _CsvSpatialLayout | None:
    from .csv_gpu import _detect_spatial_columns

    try:
        current_limit = csv.field_size_limit()
        if current_limit < (2**31 - 1):
            csv.field_size_limit(2**31 - 1)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return None
        column_names = tuple(str(name) for name in header)
        spatial_columns = _detect_spatial_columns(list(column_names))
        geometry_format: str | None = None
        if "geom" in spatial_columns:
            geom_idx = spatial_columns["geom"]
            for row in reader:
                if geom_idx >= len(row):
                    continue
                value = row[geom_idx].strip()
                if not value:
                    continue
                geometry_format = _csv_detect_geometry_value_format(value)
                break
            if geometry_format is None:
                geometry_format = "wkt"
        return _CsvSpatialLayout(
            column_names=column_names,
            spatial_columns=spatial_columns,
            geometry_format=geometry_format,
        )


def _write_vector_file_native_pyogrio(
    payload,
    filename,
    *,
    driver=None,
    schema=None,
    index=None,
    mode="w",
    crs=None,
    metadata=None,
    **kwargs,
):
    import pyogrio

    from vibespatial.api.io.file import _check_metadata_supported, _detect_driver, _expand_user

    if schema is not None:
        raise ValueError(
            "The 'schema' argument is not supported with the 'pyogrio' engine."
        )
    if crs is not None:
        raise ValueError("Passing 'crs' is not supported with the 'pyogrio' engine.")
    append = kwargs.pop("append", None)
    if append is not None:
        mode = "a" if append else "w"
    if mode not in ("w", "a"):
        raise ValueError(f"'mode' should be one of 'w' or 'a', got '{mode}' instead")

    normalized = _prepare_native_payload_for_file(payload, index=index)
    if not normalized.attributes.columns.is_unique:
        raise ValueError("GeoDataFrame cannot contain duplicated column names.")

    filename = _expand_user(filename)
    driver_name = _detect_driver(filename) if driver is None else driver
    _check_metadata_supported(metadata, "pyogrio", driver_name)
    geometry_type = kwargs.pop("geometry_type", _native_payload_geometry_type(normalized))
    arrow_table = normalized.to_arrow(index=False, geometry_encoding="WKB")

    pyogrio.write_arrow(
        arrow_table,
        filename,
        driver=driver_name,
        geometry_name=normalized.geometry_name,
        geometry_type=geometry_type,
        crs=_native_payload_file_crs(normalized),
        metadata=metadata,
        append=(mode == "a"),
        **kwargs,
    )


def plan_shapefile_ingest(*, prefer: str = "auto") -> ShapefileIngestPlan:
    selected_strategy = "shp-direct-gpu" if prefer == "auto" else prefer
    implementation = {
        "shp-direct-gpu": "shapefile_shp_direct_owned",
        "arrow-wkb": "shapefile_arrow_wkb_owned",
    }.get(selected_strategy, "shapefile_shp_direct_owned")
    if selected_strategy == "arrow-wkb":
        return ShapefileIngestPlan(
            implementation=implementation,
            selected_strategy=selected_strategy,
            uses_pyogrio_container=True,
            uses_arrow_batch=True,
            uses_native_wkb_decode=True,
            reason=(
                "Use host-side pyogrio container parsing for bbox, column projection, "
                "row-windowed reads, and other compatibility-sensitive Shapefile cases, "
                "then land geometry through Arrow WKB batches into owned buffers without "
                "per-row Shapely construction."
            ),
        )
    return ShapefileIngestPlan(
        implementation=implementation,
        selected_strategy=selected_strategy,
        uses_pyogrio_container=False,
        uses_arrow_batch=False,
        uses_native_wkb_decode=False,
        reason=(
            "Prefer direct SHP binary decode on GPU with the GPU DBF parser for plain "
            "local-file Shapefile reads. Fall back to the Arrow WKB compatibility path "
            "only when the request needs pyogrio container features."
        ),
    )


def _supports_direct_shapefile_geometry_decode(source: str | Path) -> bool:
    """True when the direct SHP decoder can preserve the source geometry model.

    The current direct SHP path is correct for Point, MultiPoint, and PolyLine
    records. Polygon records still need the Arrow/WKB boundary because SHP
    polygons can encode multipart exterior rings within a single record, and the
    direct decoder does not yet rebuild those as MultiPolygon geometries.
    """
    try:
        from .shp_gpu import (
            SHP_MULTIPOINT,
            SHP_POINT,
            SHP_POLYLINE,
            _read_shx_index,
        )
    except Exception:
        return False

    source_path = Path(source)
    if source_path.suffix.lower() != ".shp":
        return False

    try:
        header, _, _ = _read_shx_index(source_path.with_suffix(".shx"))
    except Exception:
        return False

    return header.shape_type in {SHP_POINT, SHP_MULTIPOINT, SHP_POLYLINE}


def _normalize_feature_window(rows) -> tuple[int, int | None]:
    if rows is None:
        return 0, None
    if isinstance(rows, int):
        if rows < 0:
            raise ValueError("negative row counts are not supported")
        return 0, rows
    if isinstance(rows, slice):
        if rows.step not in (None, 1):
            raise ValueError("slice steps other than 1 are not supported")
        start = 0 if rows.start is None else rows.start
        stop = rows.stop
        if start < 0 or (stop is not None and stop < 0):
            raise ValueError("negative row windows are not supported")
        max_features = None if stop is None else max(0, stop - start)
        return start, max_features
    raise TypeError("rows must be an int, slice, or None")


def _select_arrow_geometry_column(table, metadata: dict[str, object]) -> tuple[int, object]:
    for index, field in enumerate(table.schema):
        field_metadata = field.metadata or {}
        if field_metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb":
            return index, field
    geometry_name = metadata.get("geometry_name")
    if geometry_name and geometry_name in table.column_names:
        field_index = table.schema.get_field_index(str(geometry_name))
        if field_index != -1:
            return field_index, table.schema.field(field_index)
    raise ValueError("Shapefile Arrow table did not expose a GeoArrow WKB geometry column")


def _pyogrio_arrow_wkb_to_native_tabular_result(
    table,
    metadata: dict[str, object],
    *,
    target_crs: str | None = None,
):
    from vibespatial.api._native_results import (
        GeometryNativeResult,
        NativeTabularResult,
        native_attribute_table_from_arrow_table,
    )

    from .arrow import decode_wkb_arrow_array_owned

    geom_idx, _geom_field = _select_arrow_geometry_column(table, metadata)
    geom_column = table.column(geom_idx).combine_chunks()
    owned = decode_wkb_arrow_array_owned(geom_column)
    geometry_name = str(metadata.get("geometry_name") or "geometry")
    # Match GeoPandas read_file semantics for container-backed file reads:
    # expose the active geometry column as "geometry" even when the on-disk
    # container uses an internal name like "geom".
    if geometry_name != "geometry" and "geometry" not in table.column_names:
        geometry_name = "geometry"

    source_crs = metadata.get("crs")
    effective_crs = _resolve_target_crs_for_owned(
        owned,
        source_crs=source_crs,
        target_crs=target_crs,
    )

    attrs_table = table.remove_column(geom_idx)
    attributes = native_attribute_table_from_arrow_table(attrs_table)
    return NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(owned, crs=effective_crs),
        geometry_name=geometry_name,
        column_order=tuple([*attributes.columns, geometry_name]),
    )


def _materialize_native_file_read_result(payload):
    import datetime

    frame = payload.to_geodataframe()
    for col in frame.columns:
        if col == frame.geometry.name:
            continue
        series = frame[col]
        if hasattr(series, "dt") and series.dt.tz is not None:
            try:
                frame[col] = series.dt.tz_convert(datetime.UTC)
            except Exception:
                pass
    return frame


def read_geojson_native(
    source: str | Path,
    *,
    prefer: str = "auto",
    objective: str = "pipeline",
    track_properties: bool = True,
    target_crs: str | None = None,
):
    batch = read_geojson_owned(
        source,
        prefer=prefer,
        objective=objective,
        track_properties=track_properties,
    )
    effective_crs = _resolve_target_crs_for_owned(
        batch.geometry,
        source_crs="EPSG:4326",
        target_crs=target_crs,
    )
    attributes = None
    if track_properties:
        row_count = int(batch.geometry.row_count)
        if batch._properties is None:
            attributes = _native_attribute_table_from_geojson_properties_loader(
                lambda: batch.properties,
                row_count=row_count,
            )
        else:
            attributes = _geojson_properties_to_frame(
                batch.properties,
                row_count=row_count,
            )
    return _native_file_result_from_owned(
        batch.geometry,
        crs=effective_crs,
        attributes=attributes,
        row_count=int(batch.geometry.row_count),
    )


def read_shapefile_owned(
    source: str | Path,
    *,
    bbox=None,
    columns=None,
    rows=None,
    **kwargs,
) -> ShapefileOwnedBatch:
    plan = plan_shapefile_ingest()
    direct_eligible = (
        bbox is None
        and columns is None
        and rows is None
        and not kwargs
        and isinstance(source, str | Path)
        and "://" not in str(source)
    )
    direct_supported = direct_eligible and _supports_direct_shapefile_geometry_decode(source)
    if direct_supported:
        payload = _try_shapefile_shp_direct_gpu_read_native(source)
        if payload is not None:
            geometry = payload.geometry.owned
            if geometry is None:
                raise ValueError("direct Shapefile native read did not produce owned geometry")
            attributes_table = payload.attributes.to_arrow(index=False)
            metadata = {
                "geometry_name": payload.geometry_name,
                "crs": payload.geometry.crs,
            }
            record_dispatch_event(
                surface="vibespatial.io.shapefile",
                operation="read_owned",
                implementation=plan.implementation,
                reason=plan.reason,
                selected=ExecutionMode.GPU,
            )
            return ShapefileOwnedBatch(
                geometry=geometry,
                attributes_table=attributes_table,
                metadata=metadata,
            )

    # Falling through here is still the supported native path for Shapefile
    # reads. The Arrow/WKB route uses CPU container parsing plus native GPU WKB
    # decode, so it should be surfaced as a dispatch choice, not as a strict-
    # mode fallback event.

    fallback_plan = plan_shapefile_ingest(prefer="arrow-wkb")
    import pyogrio

    from .arrow import decode_wkb_arrow_array_owned

    skip_features, max_features = _normalize_feature_window(rows)
    record_dispatch_event(
        surface="vibespatial.io.shapefile",
        operation="read_owned",
        implementation=fallback_plan.implementation,
        reason=fallback_plan.reason,
        selected=ExecutionMode.CPU,
    )
    metadata, table = pyogrio.read_arrow(
        source,
        bbox=bbox,
        columns=columns,
        skip_features=skip_features,
        max_features=max_features,
        **kwargs,
    )
    geometry_index, _ = _select_arrow_geometry_column(table, metadata)
    geometry_column = table.column(geometry_index).combine_chunks()
    geometry = decode_wkb_arrow_array_owned(geometry_column)
    attributes_table = table.remove_column(geometry_index)
    return ShapefileOwnedBatch(
        geometry=geometry,
        attributes_table=attributes_table,
        metadata=metadata,
    )


def read_shapefile_native(
    source: str | Path,
    *,
    bbox=None,
    columns=None,
    rows=None,
    target_crs: str | None = None,
    **kwargs,
):
    from vibespatial.api._native_results import (
        GeometryNativeResult,
        NativeTabularResult,
        native_attribute_table_from_arrow_table,
    )

    batch = read_shapefile_owned(
        source,
        bbox=bbox,
        columns=columns,
        rows=rows,
        **kwargs,
    )
    effective_crs = _resolve_target_crs_for_owned(
        batch.geometry,
        source_crs=batch.metadata.get("crs"),
        target_crs=target_crs,
    )
    geometry_name = batch.metadata.get("geometry_name") or "geometry"
    attributes = native_attribute_table_from_arrow_table(batch.attributes_table)
    return NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(batch.geometry, crs=effective_crs),
        geometry_name=geometry_name,
        column_order=tuple([*attributes.columns, geometry_name]),
    )


def _normalize_driver(filename, driver: str | None = None) -> str:
    if driver is not None:
        return driver
    if not isinstance(filename, str | Path):
        return "GDAL-legacy"
    # Handle compound extensions like .shp.zip before Path.suffix
    fname_lower = str(filename).lower()
    if fname_lower.endswith(".shp.zip"):
        return "ESRI Shapefile"
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".json": "GeoJSON",
        ".geojson": "GeoJSON",
        ".shp": "ESRI Shapefile",
        ".dbf": "ESRI Shapefile",
        ".wkt": "WKT",
        ".csv": "CSV",
        ".tsv": "CSV",
        ".kml": "KML",
        ".pbf": "OSM-PBF",
        ".gpkg": "GPKG",
        ".gdb": "OpenFileGDB",
        ".fgb": "FlatGeobuf",
        ".gml": "GML",
        ".gpx": "GPX",
        ".topojson": "TopoJSON",
        ".geojsonl": "GeoJSONSeq",
        ".geojsonseq": "GeoJSONSeq",
        ".parquet": "GeoParquet",
        ".geoparquet": "GeoParquet",
        ".feather": "Feather",
        ".arrow": "Arrow",
        ".ipc": "Arrow",
    }
    return mapping.get(suffix, "GDAL-legacy")


def plan_vector_file_io(
    filename,
    *,
    operation: IOOperation | str,
    driver: str | None = None,
) -> VectorFilePlan:
    normalized_operation = operation if isinstance(operation, IOOperation) else IOOperation(operation)
    normalized_driver = _normalize_driver(filename, driver)
    if normalized_driver == "GeoJSON":
        io_format = IOFormat.GEOJSON
        implementation = "geojson_hybrid_adapter"
        reason = (
            "GeoJSON stays a staged hybrid path with pipeline-oriented auto routing "
            "that prefers the repo-owned GPU ingest path for unfiltered reads."
        )
    elif normalized_driver == "ESRI Shapefile":
        io_format = IOFormat.SHAPEFILE
        implementation = "shapefile_hybrid_adapter"
        reason = (
            "Shapefile stays an explicit hybrid path with pipeline-oriented auto routing "
            "that prefers the repo-owned native read plan on eligible public reads."
        )
    elif normalized_driver == "WKT":
        io_format = IOFormat.WKT
        implementation = "wkt_gpu_hybrid_adapter"
        reason = "WKT uses GPU byte-classification for geometry parsing with host fallback."
    elif normalized_driver == "CSV":
        io_format = IOFormat.CSV
        implementation = "csv_gpu_hybrid_adapter"
        reason = (
            "CSV uses libcudf table parse for large geometry-column files and "
            "GPU byte-classification for the remaining spatial layouts, with "
            "explicit host fallback when neither path applies."
        )
    elif normalized_driver == "KML":
        io_format = IOFormat.KML
        implementation = "kml_gpu_hybrid_adapter"
        reason = "KML uses GPU byte-classification for structural analysis and coordinate extraction with host fallback."
    elif normalized_driver == "OSM-PBF":
        io_format = IOFormat.OSM_PBF
        implementation = "osm_pbf_gpu_hybrid_adapter"
        reason = (
            "OSM PBF uses CPU protobuf parsing with GPU varint decoding and "
            "coordinate assembly, then projects tags through promoted columns "
            "plus lossless other_tags at the public boundary."
        )
    elif normalized_driver == "GPKG":
        io_format = IOFormat.GEOPACKAGE
        implementation = "gpkg_pyogrio_arrow_gpu_wkb"
        reason = "GeoPackage uses pyogrio Arrow container parse with GPU WKB geometry decode."
    elif normalized_driver == "OpenFileGDB":
        io_format = IOFormat.FILE_GEODATABASE
        implementation = "fgdb_pyogrio_arrow_gpu_wkb"
        reason = "File Geodatabase uses pyogrio Arrow container parse with GPU WKB geometry decode."
    elif normalized_driver == "FlatGeobuf":
        io_format = IOFormat.FLATGEOBUF
        implementation = "flatgeobuf_gpu_hybrid_adapter"
        reason = (
            "FlatGeobuf stays hybrid: eligible local unfiltered reads prefer the "
            "repo-owned direct FlatBuffer GPU decoder, while explicit pyogrio and "
            "container-shaped requests stay on the shared Arrow/WKB native boundary."
        )
    elif normalized_driver == "GML":
        io_format = IOFormat.GML
        implementation = "gml_pyogrio_arrow_gpu_wkb"
        reason = "GML uses pyogrio Arrow container parse with GPU WKB geometry decode."
    elif normalized_driver == "GPX":
        io_format = IOFormat.GPX
        implementation = "gpx_pyogrio_arrow_gpu_wkb"
        reason = "GPX uses pyogrio Arrow container parse with GPU WKB geometry decode."
    elif normalized_driver == "TopoJSON":
        io_format = IOFormat.TOPOJSON
        implementation = "topojson_pyogrio_arrow_gpu_wkb"
        reason = "TopoJSON uses pyogrio Arrow container parse with GPU WKB geometry decode."
    elif normalized_driver == "GeoJSONSeq":
        io_format = IOFormat.GEOJSONSEQ
        implementation = "geojsonseq_pyogrio_arrow_gpu_wkb"
        reason = "GeoJSON-Seq uses pyogrio Arrow container parse with GPU WKB geometry decode."
    else:
        io_format = IOFormat.GDAL_LEGACY
        implementation = "legacy_gdal_adapter"
        reason = "Legacy GDAL-backed vector formats route through host file-engine adapters for parity."
    support_plan = plan_io_support(io_format, normalized_operation)
    return VectorFilePlan(
        format=io_format,
        operation=normalized_operation,
        selected_path=support_plan.selected_path,
        driver=normalized_driver,
        implementation=implementation,
        reason=reason,
    )


def _reproject_owned_inplace(
    owned: OwnedGeometryArray,
    *,
    src_crs: str,
    dst_crs: str,
) -> None:
    """Transform all coordinate arrays in an OwnedGeometryArray in-place on device.

    Iterates over each geometry family's device coordinate buffers and
    applies ``transform_coordinates_inplace`` from the GPU transform stage.
    No host round-trip occurs.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Geometry array with device-resident coordinate buffers.
    src_crs : str
        Source CRS identifier.
    dst_crs : str
        Destination CRS identifier.
    """
    if src_crs == dst_crs:
        return

    from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

    device_state = owned._ensure_device_state()
    for dbuf in device_state.families.values():
        if dbuf.x.size == 0:
            continue
        transform_coordinates_inplace(dbuf.x, dbuf.y, src_crs=src_crs, dst_crs=dst_crs)
        # Invalidate cached bounds since coordinates changed.
        dbuf.bounds = None


def _resolve_target_crs_for_owned(
    owned: OwnedGeometryArray,
    *,
    source_crs: str | None,
    target_crs: str | None,
) -> str | None:
    if target_crs is None:
        return source_crs
    if source_crs is None:
        return target_crs
    _reproject_owned_inplace(owned, src_crs=source_crs, dst_crs=target_crs)
    return target_crs


def _reproject_gdf_gpu(gdf, target_crs: str) -> None:
    """Reproject a GPU-backed GeoDataFrame in-place via to_crs.

    This is used for GPU paths (e.g. Shapefile DBF) where the
    OwnedGeometryArray is already wrapped in a GeoDataFrame and the
    source CRS is known.
    """
    if gdf.crs is not None:
        gdf_new = gdf.to_crs(target_crs)
        # Replace geometry in-place by updating the backing column.
        gdf[gdf.geometry.name] = gdf_new.geometry
        gdf.crs = gdf_new.crs


def _attach_gpu_spatial_index(gdf) -> None:
    """Build a GPU spatial index and attach it to the GeoDataFrame.

    The index is stored as a ``_gpu_spatial_index`` attribute on the
    GeoDataFrame.  This is a ``GpuSpatialIndex`` (packed Hilbert R-tree)
    built entirely on the GPU from the geometry column's device-resident
    coordinate buffers.

    If the geometry column does not have an OwnedGeometryArray backing
    (e.g. CPU fallback), this is a no-op.
    """
    try:
        import cupy as _cp

        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.io.gpu_parse.indexing import build_spatial_index

        geom_values = gdf.geometry.values
        # Unwrap DeviceGeometryArray to get the OwnedGeometryArray.
        if isinstance(geom_values, DeviceGeometryArray):
            owned = geom_values._owned
        elif hasattr(geom_values, "_owned") and geom_values._owned is not None:
            owned = geom_values._owned
        else:
            return  # No owned backing -- cannot build GPU index.

        device_state = owned._ensure_device_state()

        # Collect all coordinate arrays across families and build a
        # unified geometry_offsets array for the spatial index.
        all_x = []
        all_y = []
        all_offsets = []
        coord_offset = 0
        for family in sorted(device_state.families.keys(), key=lambda f: f.value):
            dbuf = device_state.families[family]
            n_coords = int(dbuf.x.size)
            if n_coords == 0:
                continue
            all_x.append(dbuf.x)
            all_y.append(dbuf.y)
            # geometry_offsets for this family, shifted by cumulative coord count.
            offsets = dbuf.geometry_offsets
            if coord_offset > 0:
                offsets = offsets + coord_offset
            all_offsets.append(offsets)
            coord_offset += n_coords

        if not all_x:
            return  # No coordinates -- nothing to index.

        d_x = _cp.concatenate(all_x) if len(all_x) > 1 else all_x[0]
        d_y = _cp.concatenate(all_y) if len(all_y) > 1 else all_y[0]

        # Build a single merged offset array.  Each family's offsets end
        # at the start of the next family, so we take the first element of
        # each offset array except the first, then append the final element.
        if len(all_offsets) == 1:
            geometry_offsets = all_offsets[0]
        else:
            parts = [all_offsets[0]]
            for off in all_offsets[1:]:
                # Skip the leading 0 of subsequent offset arrays (already
                # covered by the previous array's last element).
                parts.append(off[1:])
            geometry_offsets = _cp.concatenate(parts)

        gpu_index = build_spatial_index(d_x, d_y, geometry_offsets)
        gdf._gpu_spatial_index = gpu_index
    except Exception:
        pass  # Best-effort; do not fail the read.


def _try_wkt_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a WKT file using the GPU byte-classification pipeline.

    WKT files contain one geometry per line with no attribute columns.
    Returns a GeoDataFrame with a geometry column only, or None on failure.
    """
    try:
        payload = _try_wkt_gpu_read_native(filename, target_crs=target_crs)
        gdf = _materialize_native_file_read_result(payload)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="wkt_gpu_byte_classify_adapter",
            reason=(
                "GPU byte-classification: direct GPU parsing of WKT with "
                "NVRTC kernels, one geometry per line."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_wkt_gpu_read_native(filename, *, target_crs: str | None = None):
    from .kvikio_reader import read_file_to_device
    from .wkt_gpu import read_wkt_gpu

    file_path = Path(filename)
    file_size = file_path.stat().st_size
    result = read_file_to_device(file_path, file_size)
    d_bytes = result.device_bytes
    owned = read_wkt_gpu(d_bytes)

    crs = target_crs if target_crs is not None else None
    return _native_file_result_from_owned(owned, crs=crs)


def _try_csv_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a CSV file using the best available GPU path."""
    try:
        selection = _select_csv_gpu_read_native(filename, target_crs=target_crs)
        gdf = _materialize_native_file_read_result(selection.payload)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation=selection.implementation,
            reason=selection.reason,
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_csv_byte_classify_read_native(filename, *, target_crs: str | None = None):
    from .csv_gpu import read_csv_gpu
    from .kvikio_reader import read_file_to_device

    file_path = Path(filename)
    file_size = file_path.stat().st_size
    result = read_file_to_device(file_path, file_size)
    d_bytes = result.device_bytes

    csv_result = read_csv_gpu(d_bytes)
    crs = target_crs if target_crs is not None else None
    return _native_file_result_from_owned(
        csv_result.geometry,
        crs=crs,
        attributes=csv_result.attributes,
        row_count=csv_result.n_rows,
    )


def _try_csv_pylibcudf_read_native(filename, *, target_crs: str | None = None):
    import cupy as cp
    import pyarrow as pa
    import pylibcudf as plc
    from pylibcudf.scalar import Scalar

    from .csv_gpu import _decode_hex_string_column_to_owned
    from .wkt_gpu import read_wkt_gpu

    file_path = Path(filename)
    layout = _sniff_csv_spatial_layout(file_path)
    if layout is None or "geom" not in layout.spatial_columns:
        return None

    geom_idx = layout.spatial_columns["geom"]
    if layout.geometry_format not in {"wkt", "wkb"}:
        return None

    options = plc.io.csv.CsvReaderOptions.builder(
        plc.io.types.SourceInfo([str(file_path)])
    ).build()
    options.set_header(0)
    csv_table = plc.io.csv.read_csv(options)
    column_names = csv_table.column_names()
    if geom_idx >= len(column_names):
        return None

    geom_column = csv_table.columns[geom_idx]
    if geom_column.null_count() != 0:
        return None

    if layout.geometry_format == "wkt":
        newline = Scalar.from_arrow(pa.scalar("\n"))
        empty = Scalar.from_arrow(pa.scalar(""))
        joined = plc.strings.combine.join_strings(geom_column, newline, empty)
        offsets = cp.asarray(joined.child(0).data()).view(cp.int32)
        total_bytes = int(offsets[1]) if int(joined.size()) else 0
        payload = cp.asarray(joined.data()).view(cp.uint8)[:total_bytes]
        owned = read_wkt_gpu(payload)
    else:
        owned = _decode_hex_string_column_to_owned(geom_column)

    attribute_names = [
        name for idx, name in enumerate(column_names) if idx != geom_idx
    ]
    attributes = None
    if attribute_names:
        attributes = csv_table.tbl.to_arrow(metadata=column_names).select(attribute_names)

    crs = target_crs if target_crs is not None else None
    return _native_file_result_from_owned(
        owned,
        crs=crs,
        attributes=attributes,
        row_count=int(csv_table.tbl.num_rows()),
    )


def _select_csv_gpu_read_native(
    filename,
    *,
    target_crs: str | None = None,
) -> _CsvGpuSelection:
    file_path = Path(filename)
    file_size = file_path.stat().st_size

    if file_size > _GPU_MIN_FILE_SIZE:
        payload = _try_csv_pylibcudf_read_native(filename, target_crs=target_crs)
        if payload is not None:
            return _CsvGpuSelection(
                payload=payload,
                implementation="csv_pylibcudf_table_adapter",
                reason=(
                    "GPU CSV table parse via pylibcudf/libcudf for large geometry-column "
                    "CSV, followed by native GPU WKT/WKB geometry decode."
                ),
            )

    payload = _try_csv_byte_classify_read_native(filename, target_crs=target_crs)
    return _CsvGpuSelection(
        payload=payload,
        implementation="csv_gpu_byte_classify_adapter",
        reason=(
            "GPU byte-classification: GPU structural analysis and coordinate "
            "extraction for CSV spatial data."
        ),
    )


def _try_csv_gpu_read_native(filename, *, target_crs: str | None = None):
    return _select_csv_gpu_read_native(
        filename,
        target_crs=target_crs,
    ).payload


def _try_shapefile_shp_direct_gpu_read(
    filename,
    *,
    target_crs: str | None = None,
) -> object | None:
    """Try to read a Shapefile using direct GPU SHP binary decode.

    When both .shp and .shx files exist, bypasses the pyogrio -> WKB ->
    GPU WKB decode roundtrip entirely.  Geometry is extracted directly
    from the SHP binary format on GPU.  Attributes come from the GPU
    DBF parser.

    Falls back to None on failure (missing .shx, unsupported shape type,
    etc.), allowing the caller to try the pyogrio + GPU WKB path.
    """
    try:
        payload = _try_shapefile_shp_direct_gpu_read_native(
            filename,
            target_crs=target_crs,
        )
        if payload is None:
            return None
        gdf = _materialize_native_file_read_result(payload)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="shapefile_shp_direct_gpu_adapter",
            reason=(
                "GPU-native Shapefile read: direct SHP binary decode on GPU "
                "(no WKB intermediate), GPU DBF parser for attributes."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_shapefile_shp_direct_gpu_read_native(
    filename,
    *,
    target_crs: str | None = None,
):
    file_path = Path(filename)
    is_zip = (
        file_path.suffix.lower() == ".zip"
        or str(file_path).lower().endswith(".shp.zip")
    )

    crs = None
    attrs_df = None

    if is_zip:
        import zipfile

        from .shp_gpu import _assemble_from_shp_bytes

        with zipfile.ZipFile(file_path) as zf:
            names = zf.namelist()
            shp_name = next((n for n in names if n.lower().endswith(".shp")), None)
            shx_name = next((n for n in names if n.lower().endswith(".shx")), None)
            if shp_name is None or shx_name is None:
                return None

            shp_bytes = zf.read(shp_name)
            shx_bytes = zf.read(shx_name)

            prj_name = next((n for n in names if n.lower().endswith(".prj")), None)
            if prj_name:
                try:
                    crs = zf.read(prj_name).decode("utf-8", errors="replace").strip()
                except Exception:
                    pass

            dbf_name = next((n for n in names if n.lower().endswith(".dbf")), None)
            dbf_bytes = zf.read(dbf_name) if dbf_name else None

        owned = _assemble_from_shp_bytes(shp_bytes, shx_bytes)
        del shp_bytes, shx_bytes

        if dbf_bytes is not None:
            from .dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu_from_bytes

            dbf_result = read_dbf_gpu_from_bytes(dbf_bytes)
            del dbf_bytes
            attrs_df = dbf_result_to_dataframe(dbf_result)
    else:
        from .shp_gpu import read_shp_gpu

        shx_path = file_path.with_suffix(".shx")
        if not shx_path.exists():
            return None

        owned = read_shp_gpu(file_path)

        prj_path = file_path.with_suffix(".prj")
        if prj_path.exists():
            try:
                crs = prj_path.read_text().strip()
            except Exception:
                pass

        dbf_path = file_path.with_suffix(".dbf")
        if dbf_path.exists():
            from .dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu

            dbf_result = read_dbf_gpu(dbf_path)
            attrs_df = dbf_result_to_dataframe(dbf_result)

    effective_crs = _resolve_target_crs_for_owned(
        owned,
        source_crs=crs,
        target_crs=target_crs,
    )
    return _native_file_result_from_owned(
        owned,
        crs=effective_crs,
        attributes=attrs_df,
    )


def _try_shapefile_dbf_gpu_read(
    filename,
    *,
    target_crs: str | None = None,
) -> object | None:
    """Try to read a Shapefile with GPU DBF attribute parsing.

    Uses pyogrio.read_arrow() for geometry (via GPU WKB decode) and
    the GPU DBF reader for attributes, bypassing the Arrow->pandas
    attribute conversion for numeric columns. Falls back to None on
    failure.
    """
    try:
        payload = _try_shapefile_dbf_gpu_read_native(
            filename,
            target_crs=target_crs,
        )
        if payload is None:
            return None
        gdf = _materialize_native_file_read_result(payload)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="shapefile_gpu_dbf_adapter",
            reason=(
                "GPU-dominant Shapefile read: pyogrio Arrow for geometry with "
                "GPU WKB decode, GPU DBF parser for numeric attributes."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_shapefile_dbf_gpu_read_native(
    filename,
    *,
    target_crs: str | None = None,
):
    import pyogrio

    from .arrow import decode_wkb_arrow_array_owned
    from .dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu

    file_path = Path(filename)
    dbf_path = file_path.with_suffix(".dbf")
    if not dbf_path.exists():
        return None

    metadata, table = pyogrio.read_arrow(filename)
    geom_idx, _ = _select_arrow_geometry_column(table, metadata)
    geom_column = table.column(geom_idx).combine_chunks()
    owned = decode_wkb_arrow_array_owned(geom_column)
    crs = _resolve_target_crs_for_owned(
        owned,
        source_crs=metadata.get("crs"),
        target_crs=target_crs,
    )

    dbf_result = read_dbf_gpu(dbf_path)
    attrs_df = dbf_result_to_dataframe(dbf_result)

    geometry_name = metadata.get("geometry_name") or "geometry"
    return _native_file_result_from_owned(
        owned,
        geometry_name=geometry_name,
        crs=crs,
        attributes=attrs_df,
    )


def _try_kml_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a KML file using the GPU byte-classification pipeline.

    KML files are XML-based with Placemark elements containing geometry.
    Returns a GeoDataFrame with a geometry column only, or None on failure.
    """
    try:
        payload = _try_kml_gpu_read_native(filename, target_crs=target_crs)
        gdf = _materialize_native_file_read_result(payload)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="kml_gpu_byte_classify_adapter",
            reason=(
                "GPU byte-classification: direct GPU parsing of KML with "
                "NVRTC kernels, XML structural analysis and coordinate extraction."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_kml_gpu_read_native(filename, *, target_crs: str | None = None):
    from .kml_gpu import read_kml_gpu
    from .kvikio_reader import read_file_to_device

    file_path = Path(filename)
    file_size = file_path.stat().st_size
    result = read_file_to_device(file_path, file_size)
    d_bytes = result.device_bytes

    kml_result = read_kml_gpu(d_bytes)
    crs = target_crs if target_crs is not None else None
    return _native_file_result_from_owned(
        kml_result.geometry,
        crs=crs,
        attributes=kml_result.attributes,
        row_count=kml_result.n_placemarks,
    )


def _try_osm_pbf_gpu_read_native(
    filename,
    *,
    target_crs: str | None = None,
    tags: bool | str = "ways",
    geometry_only: bool = False,
    layer=None,
    compatibility: bool = False,
):
    from .osm_bundle import build_osm_native_bundle
    from .osm_gpu import read_osm_pbf

    normalized_layer = _normalize_osm_layer(layer)
    osm_result = read_osm_pbf(
        filename,
        tags=tags,
        geometry_only=geometry_only,
        layer=layer,
    )

    crs = target_crs if target_crs is not None else "EPSG:4326"
    bundle = build_osm_native_bundle(
        osm_result,
        crs=crs,
        source=str(filename),
    )
    payload = bundle.to_native_tabular_result(
        layer=normalized_layer,
        compatibility=compatibility,
    )
    if payload is None or payload.geometry.row_count == 0:
        return None
    return payload


def _try_fgb_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a FlatGeobuf file using the GPU direct binary decoder.

    Parses the FGB binary format directly on GPU, bypassing the
    pyogrio -> WKB -> GPU WKB decode roundtrip.  FGB stores coordinates
    as flat arrays, making the GPU decode path very efficient.

    Returns a GeoDataFrame with geometry + attributes, or None on failure.
    """
    try:
        payload = _try_fgb_gpu_read_native(filename, target_crs=target_crs)
        gdf = _materialize_native_file_read_result(payload)

        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="flatgeobuf_gpu_direct_decode_adapter",
            reason=(
                "GPU direct FGB decode: binary FlatBuffer navigation on GPU via "
                "NVRTC kernels, bypassing pyogrio/WKB roundtrip."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_fgb_gpu_read_native(filename, *, target_crs: str | None = None):
    from .fgb_gpu import read_fgb_gpu

    fgb_result = read_fgb_gpu(filename)
    crs = _resolve_target_crs_for_owned(
        fgb_result.geometry,
        source_crs=fgb_result.crs,
        target_crs=target_crs,
    )
    return _native_file_result_from_owned(
        fgb_result.geometry,
        crs=crs,
        attributes=fgb_result.attributes,
        row_count=fgb_result.n_features,
    )


# Minimum file size (in bytes) for direct GPU byte-classify routing on formats
# that still need a coarse size gate. GeoJSON intentionally bypasses this:
# unfiltered public reads prefer the repo-owned GPU ingest path so the first
# downstream GPU consumer does not pay an immediate host-to-device promotion.
_GPU_MIN_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _prefer_pipeline_native_read(plan: VectorFilePlan) -> bool:
    return plan.format in {
        IOFormat.GEOJSON,
        IOFormat.SHAPEFILE,
    }


def _geojson_pipeline_prefers_gpu_for_source(filename) -> bool:
    if not isinstance(filename, str | Path):
        return False
    path = Path(filename)
    return path.exists() and path.is_file()


def _supports_explicit_pyogrio_native_read(plan: VectorFilePlan, engine, filename) -> bool:
    if not isinstance(filename, str | Path):
        return False
    if not Path(filename).exists():
        return False
    return engine == "pyogrio" and plan.format in {
        IOFormat.GEOPACKAGE,
        IOFormat.FILE_GEODATABASE,
        IOFormat.FLATGEOBUF,
        IOFormat.GML,
        IOFormat.GPX,
        IOFormat.TOPOJSON,
        IOFormat.GEOJSONSEQ,
    }


def _supports_direct_flatgeobuf_native_read(
    filename,
    *,
    engine,
    bbox,
    columns,
    rows,
    kwargs,
) -> bool:
    if engine is not None or bbox is not None or columns is not None or rows is not None:
        return False
    if kwargs:
        return False
    if not isinstance(filename, str | Path):
        return False
    return "://" not in str(filename)


def _try_gpu_read_file_native(
    filename,
    *,
    plan,
    engine=None,
    bbox,
    columns,
    rows,
    target_crs: str | None = None,
    compatibility: bool = False,
    **kwargs,
):
    """Try to read a vector file into the shared native tabular boundary."""
    import pyogrio

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.runtime import get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        return None

    runtime = get_cuda_runtime()
    if not runtime.available():
        return None

    osm_layer = kwargs.get("layer")
    osm_tags = kwargs.get("tags", "ways")
    osm_geometry_only = bool(kwargs.get("geometry_only", False))
    osm_pyogrio_default = (
        plan.format is IOFormat.OSM_PBF
        and bbox is None
        and columns is None
        and rows is None
        and _osm_uses_pyogrio_compat_default(
            layer=osm_layer,
            tags=osm_tags,
            geometry_only=osm_geometry_only,
        )
    )
    use_pyogrio_osm_layer = (
        plan.format is IOFormat.OSM_PBF
        and _osm_uses_pyogrio_compat_layer(layer=osm_layer, tags=osm_tags)
    )

    if osm_pyogrio_default:
        payload = _read_osm_pbf_supported_layers_native(
            filename,
            target_crs=target_crs,
            geometry_only=osm_geometry_only,
            tags=osm_tags,
        )
        if payload.geometry.row_count > 0:
            record_dispatch_event(
                surface="geopandas.read_file",
                operation="read_file",
                implementation="osm_pbf_pyogrio_supported_layers_gpu_wkb",
                reason=(
                    "Public OSM PBF default read combines the supported pyogrio layers "
                    "(points, lines, multilinestrings, multipolygons, other_relations) "
                    "through Arrow container parsing, GPU WKB decode, native concat, "
                    "and explicit public export."
                ),
                selected=ExecutionMode.GPU,
            )
            return payload

        payload = _try_osm_pbf_gpu_read_native(
            filename,
            target_crs=target_crs,
            tags=osm_tags,
            geometry_only=osm_geometry_only,
            layer=osm_layer,
            compatibility=compatibility,
        )
        if payload is not None:
            record_dispatch_event(
                surface="geopandas.read_file",
                operation="read_file",
                implementation="osm_pbf_gpu_hybrid_adapter",
                reason=(
                    "GPU hybrid OSM PBF: CPU protobuf parsing with GPU varint "
                    "decoding, coordinate assembly, Way/Relation coordinate "
                    "resolution via binary-search kernel, MultiPolygon "
                    "assembly from Relation Way members, and bounded public "
                    "tag export through promoted columns plus lossless "
                    "other_tags payloads."
                ),
                selected=ExecutionMode.GPU,
            )
        return payload

    if (
        use_pyogrio_osm_layer
        and bbox is None
        and columns is None
        and rows is None
    ):
        payload = _read_osm_pbf_pyogrio_layer_native(
            filename,
            layer=_normalize_osm_layer(osm_layer),
            target_crs=target_crs,
            geometry_only=osm_geometry_only,
            tags=osm_tags,
        )
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation=f"{plan.format.value}_pyogrio_arrow_gpu_wkb",
            reason=(
                "GPU-dominant read for a standard OSM public layer through pyogrio "
                "Arrow container parsing, GPU WKB decode when supported, and "
                "explicit compatibility bridging for unsupported layer geometry "
                "families."
            ),
            selected=ExecutionMode.GPU,
        )
        return payload

    if bbox is None and columns is None and rows is None and not use_pyogrio_osm_layer:
        file_path = Path(filename)
        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        if plan.format is IOFormat.WKT:
            payload = _try_wkt_gpu_read_native(filename, target_crs=target_crs)
            record_dispatch_event(
                surface="geopandas.read_file",
                operation="read_file",
                implementation="wkt_gpu_byte_classify_adapter",
                reason=(
                    "GPU byte-classification: direct GPU parsing of WKT with "
                    "NVRTC kernels, one geometry per line."
                ),
                selected=ExecutionMode.GPU,
            )
            return payload

        if plan.format is IOFormat.OSM_PBF:
            payload = _try_osm_pbf_gpu_read_native(
                filename,
                target_crs=target_crs,
                tags=osm_tags,
                geometry_only=bool(kwargs.get("geometry_only", False)),
                layer=osm_layer,
                compatibility=compatibility,
            )
            if payload is not None:
                record_dispatch_event(
                    surface="geopandas.read_file",
                    operation="read_file",
                    implementation="osm_pbf_gpu_hybrid_adapter",
                    reason=(
                        "GPU hybrid OSM PBF: CPU protobuf parsing with GPU varint "
                        "decoding, coordinate assembly, Way/Relation coordinate "
                        "resolution via binary-search kernel, MultiPolygon "
                        "assembly from Relation Way members, and bounded public "
                        "tag export through promoted columns plus lossless "
                        "other_tags payloads."
                    ),
                    selected=ExecutionMode.GPU,
                )
            return payload

        if plan.format is IOFormat.CSV and file_size > _GPU_MIN_FILE_SIZE:
            selection = _select_csv_gpu_read_native(filename, target_crs=target_crs)
            if selection.payload is not None:
                record_dispatch_event(
                    surface="geopandas.read_file",
                    operation="read_file",
                    implementation=selection.implementation,
                    reason=selection.reason,
                    selected=ExecutionMode.GPU,
                )
                return selection.payload

        if plan.format is IOFormat.KML and file_size > _GPU_MIN_FILE_SIZE:
            payload = _try_kml_gpu_read_native(filename, target_crs=target_crs)
            if payload is not None:
                record_dispatch_event(
                    surface="geopandas.read_file",
                    operation="read_file",
                    implementation="kml_gpu_byte_classify_adapter",
                    reason=(
                        "GPU byte-classification: direct GPU parsing of KML with "
                        "NVRTC kernels, XML structural analysis and coordinate extraction."
                    ),
                    selected=ExecutionMode.GPU,
                )
                return payload

        if plan.format is IOFormat.SHAPEFILE and _prefer_pipeline_native_read(plan):
            payload = read_shapefile_native(
                file_path,
                target_crs=target_crs,
            )
            record_dispatch_event(
                surface="geopandas.read_file",
                operation="read_file",
                implementation="shapefile_native_pipeline_gpu_adapter",
                reason=(
                    "Pipeline-oriented Shapefile ingest prefers the repo-owned native "
                    "read plan so public read_file can use direct SHP GPU decode when "
                    "available and fall back to Arrow/WKB only when required."
                ),
                selected=ExecutionMode.GPU,
            )
            return payload

        if plan.format is IOFormat.GEOJSON and _prefer_pipeline_native_read(plan):
            geojson_plan = plan_geojson_ingest(
                prefer="auto" if _geojson_pipeline_prefers_gpu_for_source(file_path) else "fast-json",
                objective="pipeline",
            )
            try:
                payload = read_geojson_native(
                    file_path,
                    prefer=geojson_plan.selected_strategy,
                    objective=geojson_plan.objective,
                    target_crs=target_crs,
                )
                selected = (
                    ExecutionMode.GPU
                    if geojson_plan.selected_strategy == "gpu-byte-classify"
                    else ExecutionMode.CPU
                )
            except Exception as exc:
                record_fallback_event(
                    surface="geopandas.read_file",
                    reason=(
                        "GPU byte-classify GeoJSON ingest failed; falling back to the "
                        "repo-owned fast-json path"
                    ),
                    detail=str(exc),
                    selected=ExecutionMode.CPU,
                    pipeline="io/read_file",
                    d2h_transfer=False,
                )
                geojson_plan = plan_geojson_ingest(prefer="fast-json", objective="pipeline")
                payload = read_geojson_native(
                    file_path,
                    prefer=geojson_plan.selected_strategy,
                    objective=geojson_plan.objective,
                    target_crs=target_crs,
                )
                selected = ExecutionMode.CPU
            record_dispatch_event(
                surface="geopandas.read_file",
                operation="read_file",
                implementation=f"{geojson_plan.implementation}_adapter",
                reason=geojson_plan.reason,
                selected=selected,
            )
            return payload

        if plan.format is IOFormat.FLATGEOBUF and _supports_direct_flatgeobuf_native_read(
            filename,
            engine=engine,
            bbox=bbox,
            columns=columns,
            rows=rows,
            kwargs=kwargs,
        ):
            try:
                payload = _try_fgb_gpu_read_native(filename, target_crs=target_crs)
            except Exception as exc:
                record_fallback_event(
                    surface="geopandas.read_file",
                    reason=(
                        "GPU direct FlatGeobuf decode failed; falling back to the "
                        "pyogrio Arrow/WKB native bridge"
                    ),
                    detail=str(exc),
                    selected=ExecutionMode.GPU,
                    pipeline="io/read_file",
                    d2h_transfer=False,
                )
            else:
                record_dispatch_event(
                    surface="geopandas.read_file",
                    operation="read_file",
                    implementation="flatgeobuf_gpu_direct_decode_adapter",
                    reason=(
                        "Pipeline-oriented FlatGeobuf ingest prefers the repo-owned "
                        "direct FlatBuffer GPU decoder for eligible local unfiltered "
                        "reads; explicit engine='pyogrio' stays on the shared Arrow/WKB "
                        "native boundary."
                    ),
                    selected=ExecutionMode.GPU,
                )
                return payload

    skip_features, max_features = _normalize_feature_window(rows)
    arrow_columns = columns
    if use_pyogrio_osm_layer:
        arrow_columns, arrow_kwargs = _prepare_osm_pyogrio_kwargs(
            layer=osm_layer,
            tags=osm_tags,
            columns=columns,
            kwargs=kwargs,
        )
    else:
        arrow_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
    if bbox is not None:
        if hasattr(bbox, "total_bounds"):
            meta = pyogrio.read_info(filename, layer=arrow_kwargs.get("layer"))
            crs = meta.get("crs")
            if crs is not None and bbox.crs is not None:
                bbox = tuple(bbox.to_crs(crs).total_bounds)
            else:
                bbox = tuple(bbox.total_bounds)
        elif hasattr(bbox, "bounds"):
            bbox = bbox.bounds
        arrow_kwargs["bbox"] = bbox

    metadata, table = pyogrio.read_arrow(
        filename,
        columns=arrow_columns,
        skip_features=skip_features,
        max_features=max_features,
        **arrow_kwargs,
    )
    payload = _pyogrio_arrow_wkb_to_native_tabular_result(
        table,
        metadata,
        target_crs=target_crs,
    )
    record_dispatch_event(
        surface="geopandas.read_file",
        operation="read_file",
        implementation=f"{plan.format.value}_pyogrio_arrow_gpu_wkb",
        reason=(
            "GPU-dominant read: pyogrio Arrow container parse, device-side WKB geometry "
            "decode via pylibcudf, native tabular read boundary, explicit public export."
        ),
        selected=ExecutionMode.GPU,
    )
    return payload


def _try_gpu_read_file(
    filename,
    *,
    plan,
    engine=None,
    bbox,
    columns,
    rows,
    target_crs: str | None = None,
    build_index: bool = False,
    **kwargs,
):
    """Try to read a vector file using the GPU-dominant owned path.

    Uses pyogrio.read_arrow() for container parsing, then GPU WKB decode
    for geometry and lowers the result through the shared native tabular
    boundary before the explicit public GeoDataFrame materialization point.
    Falls back to None if anything fails, triggering the vendored CPU path.
    """
    try:
        payload = _try_gpu_read_file_native(
            filename,
            plan=plan,
            engine=engine,
            bbox=bbox,
            columns=columns,
            rows=rows,
            target_crs=target_crs,
            compatibility=True,
            **kwargs,
        )
        if payload is None:
            return None
        gdf = _materialize_native_file_read_result(payload)

        if build_index:
            _attach_gpu_spatial_index(gdf)
        return gdf
    except Exception as exc:
        record_fallback_event(
            surface="geopandas.read_file",
            reason=f"GPU-dominant file read failed: {exc}",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/read_file",
            d2h_transfer=False,
        )
        return None


def _latest_read_file_gpu_failure(start_index: int):
    events = get_fallback_events()
    if start_index < 0:
        start_index = 0
    for event in reversed(events[start_index:]):
        if event.surface == "geopandas.read_file" and event.pipeline == "io/read_file":
            return event
    return None


def read_vector_file_native(
    filename,
    bbox=None,
    mask=None,
    columns=None,
    rows=None,
    engine=None,
    *,
    target_crs: str | None = None,
    **kwargs,
):
    """Read a spatial file into the shared native tabular boundary."""
    from vibespatial.api._native_results import _spatial_to_native_tabular_result

    plan = plan_vector_file_io(filename, operation=IOOperation.READ)
    normalized = _normalize_driver(filename)

    if normalized == "GeoParquet":
        from .arrow import read_geoparquet_native

        parquet_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
        if columns is not None:
            parquet_kwargs["columns"] = columns
        if bbox is not None:
            parquet_kwargs["bbox"] = bbox
        return read_geoparquet_native(filename, **parquet_kwargs)

    if normalized in ("Feather", "Arrow"):
        from vibespatial.api.io.arrow import _read_feather

        feather_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
        if columns is not None:
            feather_kwargs["columns"] = columns
        return _spatial_to_native_tabular_result(_read_feather(filename, **feather_kwargs))

    _GPU_DISPATCH_FORMATS = {
        IOFormat.GEOJSON,
        IOFormat.SHAPEFILE,
        IOFormat.WKT,
        IOFormat.CSV,
        IOFormat.KML,
        IOFormat.OSM_PBF,
        IOFormat.GEOPACKAGE,
        IOFormat.FILE_GEODATABASE,
        IOFormat.FLATGEOBUF,
        IOFormat.GML,
        IOFormat.GPX,
        IOFormat.TOPOJSON,
        IOFormat.GEOJSONSEQ,
    }
    osm_layer = kwargs.get("layer")
    osm_tags = kwargs.get("tags", "ways")
    osm_geometry_only = bool(kwargs.get("geometry_only", False))
    osm_pyogrio_default = (
        plan.format is IOFormat.OSM_PBF
        and bbox is None
        and columns is None
        and rows is None
        and mask is None
        and _osm_uses_pyogrio_compat_default(
            layer=osm_layer,
            tags=osm_tags,
            geometry_only=osm_geometry_only,
        )
    )
    osm_pyogrio_compat = (
        plan.format is IOFormat.OSM_PBF
        and mask is None
        and _osm_uses_pyogrio_compat_layer(layer=osm_layer, tags=osm_tags)
    )
    allow_native_with_explicit_pyogrio = _supports_explicit_pyogrio_native_read(plan, engine, filename)
    if (
        plan.format in _GPU_DISPATCH_FORMATS
        and mask is None
        and (engine is None or allow_native_with_explicit_pyogrio)
    ):
        payload = _try_gpu_read_file_native(
            filename,
            plan=plan,
            engine=engine,
            bbox=bbox,
            columns=columns,
            rows=rows,
            target_crs=target_crs,
            **kwargs,
        )
        if payload is not None:
            return payload

    if plan.format is IOFormat.WKT:
        raise RuntimeError(
            f"Cannot read WKT file '{filename}': GPU runtime is required for raw "
            "WKT files (pyogrio/GDAL does not support this format). Ensure a CUDA "
            "GPU is available and CuPy is installed."
        )

    if plan.format is IOFormat.OSM_PBF:
        if not osm_pyogrio_compat and not osm_pyogrio_default:
            raise RuntimeError(
                f"Cannot read OSM PBF file '{filename}': GPU runtime is required for "
                "full-data OSM PBF reads. Standard layered reads "
                "(`points`, `lines`, `multipolygons`) can use the pyogrio "
                "compatibility path; `layer=\"all\"` and native-only tag modes "
                "require the GPU OSM reader."
            )

    record_dispatch_event(
        surface="geopandas.read_file",
        operation="read_file",
        implementation=plan.implementation,
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    chosen_engine = engine
    if plan.format in {IOFormat.GEOJSON, IOFormat.SHAPEFILE} and engine is None:
        chosen_engine = "pyogrio"
    if chosen_engine is not None:
        from vibespatial.api.io import file as api_file

        chosen_engine = api_file._check_engine(chosen_engine, "'read_file' function")
    from vibespatial.api.io.file import _read_file

    if osm_pyogrio_default:
        return _spatial_to_native_tabular_result(
            _read_osm_pbf_supported_layers_public(
                filename,
                target_crs=target_crs,
                geometry_only=osm_geometry_only,
                tags=osm_tags,
            )
        )

    read_kwargs = dict(kwargs)
    read_columns = columns
    if osm_pyogrio_compat:
        read_columns, read_kwargs = _prepare_osm_pyogrio_kwargs(
            layer=osm_layer,
            tags=osm_tags,
            columns=columns,
            kwargs=kwargs,
        )

    gdf = _read_file(
        filename,
        bbox=bbox,
        mask=mask,
        columns=read_columns,
        rows=rows,
        engine=chosen_engine,
        **read_kwargs,
    )
    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)
    elif target_crs is not None and gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    return _spatial_to_native_tabular_result(gdf)


def read_vector_file(
    filename,
    bbox=None,
    mask=None,
    columns=None,
    rows=None,
    engine=None,
    *,
    target_crs: str | None = None,
    build_index: bool = False,
    **kwargs,
):
    """Read a spatial file into a GeoDataFrame.

    Supports GeoParquet, Feather/Arrow, Shapefile, GeoPackage, File
    Geodatabase, FlatGeobuf, GeoJSON, GeoJSON-Seq, GML, GPX, TopoJSON,
    WKT, CSV, KML, OSM PBF, and any format readable by pyogrio/fiona.

    GPU acceleration is automatic for GeoJSON, Shapefile, FlatGeobuf, WKT,
    CSV, KML, and OSM PBF formats. GeoJSON and Shapefile auto-routing now
    optimize for pipeline shape rather than isolated read latency: eligible
    unfiltered reads prefer the repo-owned native ingest path so downstream
    GPU work does not immediately pay a host-to-device promotion. FlatGeobuf
    now follows the same policy for eligible local unfiltered reads, using
    the repo-owned direct FlatBuffer decoder by default. CSV and KML still
    use a coarse 10 MB direct-GPU threshold where launch overhead dominates
    on very small files. WKT and full-data OSM PBF reads use the native GPU
    path. Standard OSM layers (``points``, ``lines``, ``multipolygons``)
    may use the pyogrio compatibility path when the native all-data parser
    is not required.

    ``mask`` still disables the GPU-native read path. ``bbox``, ``columns``,
    and ``rows`` continue to work on the promoted pyogrio-backed vector
    containers through the shared native Arrow/WKB boundary. Explicit
    ``engine="pyogrio"`` stays on that native pyogrio-backed boundary for
    GeoPackage, FileGDB, FlatGeobuf, GML, GPX, TopoJSON, and GeoJSON-Seq;
    other explicit engine selections fall through to the compatibility path.

    Aliased as ``vibespatial.read_file()``.

    Parameters
    ----------
    filename : str or Path
        Path to the vector file.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial filter bounding box.  Disables the GPU fast path.
    mask : Geometry or GeoDataFrame, optional
        Spatial filter mask geometry.  Disables the GPU fast path.
    columns : list of str, optional
        Subset of columns to read.  Disables the GPU fast path.
    rows : int or slice, optional
        Subset of rows to read.  Disables the GPU fast path.
    engine : str, optional
        Force a specific I/O engine (``"pyogrio"`` or ``"fiona"``).
        Disables GPU auto-routing.
    target_crs : str, optional
        Target CRS to reproject coordinates into (e.g. ``"EPSG:3857"``).
        When the GPU path is used, the reprojection is fused with ingest
        via vibeProj GPU transform (no separate pass required).  When the
        CPU path is used, the result is reprojected via ``gdf.to_crs()``
        as a post-read step.  For formats without an embedded CRS (WKT,
        CSV, KML, OSM PBF), the target CRS is set as a label without
        reprojection.
    build_index : bool, default False
        When True and the GPU path is used, build a GPU-resident packed
        Hilbert R-tree spatial index fused with ingest.  The index is
        accessible via the ``GeoDataFrame.gpu_spatial_index`` property.
    **kwargs
        Passed through to the underlying engine. For OSM PBF GPU reads, the
        repo-owned path also accepts:

        - ``tags``: ``True``, ``False``, or ``"ways"`` to control tag decode
        - ``geometry_only``: skip tag and ID export for geometry-only reads
        - ``layer``: ``"points"``, ``"lines"``, ``"multipolygons"``,
          ``"ways"``, ``"relations"``, ``"multilinestrings"``,
          ``"other_relations"``, or ``"all"``

    Returns
    -------
    GeoDataFrame
    """
    plan = plan_vector_file_io(filename, operation=IOOperation.READ)

    # Delegate GeoParquet/Feather/Arrow to their dedicated readers.
    # These formats have their own optimized pipelines (GPU-native for
    # Parquet via pylibcudf, Arrow-native for Feather/IPC) and are not
    # handled by the pyogrio vector file path.
    normalized = _normalize_driver(filename)
    if normalized == "GeoParquet":
        from .arrow import read_geoparquet

        parquet_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
        if columns is not None:
            parquet_kwargs["columns"] = columns
        if bbox is not None:
            parquet_kwargs["bbox"] = bbox
        return read_geoparquet(filename, **parquet_kwargs)
    if normalized in ("Feather", "Arrow"):
        from vibespatial.api.io.arrow import _read_feather

        feather_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
        if columns is not None:
            feather_kwargs["columns"] = columns
        return _read_feather(filename, **feather_kwargs)

    # Try GPU-dominant owned path for supported formats.
    _GPU_DISPATCH_FORMATS = {
        IOFormat.GEOJSON,
        IOFormat.SHAPEFILE,
        IOFormat.WKT,
        IOFormat.CSV,
        IOFormat.KML,
        IOFormat.OSM_PBF,
        IOFormat.GEOPACKAGE,
        IOFormat.FILE_GEODATABASE,
        IOFormat.FLATGEOBUF,
        IOFormat.GML,
        IOFormat.GPX,
        IOFormat.TOPOJSON,
        IOFormat.GEOJSONSEQ,
    }
    fallback_start = len(get_fallback_events())
    osm_layer = kwargs.get("layer")
    osm_tags = kwargs.get("tags", "ways")
    osm_geometry_only = bool(kwargs.get("geometry_only", False))
    osm_pyogrio_default = (
        plan.format is IOFormat.OSM_PBF
        and bbox is None
        and columns is None
        and rows is None
        and mask is None
        and _osm_uses_pyogrio_compat_default(
            layer=osm_layer,
            tags=osm_tags,
            geometry_only=osm_geometry_only,
        )
    )
    osm_pyogrio_compat = (
        plan.format is IOFormat.OSM_PBF
        and mask is None
        and _osm_uses_pyogrio_compat_layer(layer=osm_layer, tags=osm_tags)
    )
    allow_native_with_explicit_pyogrio = _supports_explicit_pyogrio_native_read(plan, engine, filename)
    if (
        plan.format in _GPU_DISPATCH_FORMATS
        and mask is None
        and (engine is None or allow_native_with_explicit_pyogrio)
    ):
        gpu_result = _try_gpu_read_file(
            filename,
            plan=plan,
            engine=engine,
            bbox=bbox,
            columns=columns,
            rows=rows,
            target_crs=target_crs,
            build_index=build_index,
            **kwargs,
        )
        if gpu_result is not None:
            return gpu_result

    # WKT files have no CPU fallback (pyogrio/GDAL does not support raw WKT).
    # If the GPU path returned None, raise an informative error.
    if plan.format is IOFormat.WKT:
        gpu_failure = _latest_read_file_gpu_failure(fallback_start)
        if gpu_failure is not None:
            detail = gpu_failure.detail or gpu_failure.reason
            raise RuntimeError(
                f"Cannot read WKT file '{filename}': GPU WKT read failed: {detail}"
            )
        raise RuntimeError(
            f"Cannot read WKT file '{filename}': GPU runtime is required for raw "
            "WKT files (pyogrio/GDAL does not support this format). Ensure a CUDA "
            "GPU is available and CuPy is installed."
        )

    # OSM PBF files have no CPU fallback (pyogrio/GDAL does not support PBF).
    if plan.format is IOFormat.OSM_PBF:
        gpu_failure = _latest_read_file_gpu_failure(fallback_start)
        if osm_pyogrio_default and gpu_failure is not None:
            detail = gpu_failure.detail or gpu_failure.reason
            raise RuntimeError(
                f"Cannot read OSM PBF file '{filename}': GPU OSM PBF read failed: {detail}"
            )
        if not osm_pyogrio_compat and not osm_pyogrio_default:
            if gpu_failure is not None:
                detail = gpu_failure.detail or gpu_failure.reason
                raise RuntimeError(
                    f"Cannot read OSM PBF file '{filename}': GPU OSM PBF read failed: {detail}"
                )
            raise RuntimeError(
                f"Cannot read OSM PBF file '{filename}': GPU runtime is required for "
                "full-data OSM PBF reads. Standard layered reads "
                "(`points`, `lines`, `multipolygons`) can use the pyogrio "
                "compatibility path; `layer=\"all\"` and native-only tag modes "
                "require the GPU OSM reader."
            )

    record_dispatch_event(
        surface="geopandas.read_file",
        operation="read_file",
        implementation=plan.implementation,
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    chosen_engine = engine
    if plan.format in {IOFormat.GEOJSON, IOFormat.SHAPEFILE} and engine is None:
        chosen_engine = "pyogrio"
    if chosen_engine is not None:
        from vibespatial.api.io import file as api_file

        chosen_engine = api_file._check_engine(chosen_engine, "'read_file' function")
    from vibespatial.api.io.file import _read_file

    if osm_pyogrio_default:
        return _read_osm_pbf_supported_layers_public(
            filename,
            target_crs=target_crs,
            geometry_only=osm_geometry_only,
            tags=osm_tags,
        )

    read_kwargs = dict(kwargs)
    read_columns = columns
    if osm_pyogrio_compat:
        read_columns, read_kwargs = _prepare_osm_pyogrio_kwargs(
            layer=osm_layer,
            tags=osm_tags,
            columns=columns,
            kwargs=kwargs,
        )

    gdf = _read_file(
        filename,
        bbox=bbox,
        mask=mask,
        columns=read_columns,
        rows=rows,
        engine=chosen_engine,
        **read_kwargs,
    )

    # CPU post-read: reproject if target_crs was requested.
    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)
    elif target_crs is not None and gdf.crs is None:
        # No source CRS known; set the target CRS directly (user asserts
        # the data is already in target_crs).
        gdf = gdf.set_crs(target_crs)

    return gdf


def write_vector_file(
    df,
    filename,
    driver=None,
    schema=None,
    index=None,
    **kwargs,
):
    from vibespatial.api import GeoDataFrame, GeoSeries
    from vibespatial.api._native_results import (
        _spatial_to_native_tabular_result,
        to_native_tabular_result,
    )
    from vibespatial.api.geo_base import _is_geometry_like_dtype
    from vibespatial.api.io import file as api_file

    plan = plan_vector_file_io(filename, operation=IOOperation.WRITE, driver=driver)
    chosen_engine = kwargs.pop("engine", None)
    if plan.format in {IOFormat.GEOJSON, IOFormat.SHAPEFILE} and chosen_engine is None:
        chosen_engine = "pyogrio"
    chosen_engine = api_file._check_engine(chosen_engine, "'to_file' method")
    record_dispatch_event(
        surface="geopandas.geodataframe.to_file",
        operation="to_file",
        implementation=plan.implementation,
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )

    payload = None
    is_public_spatial = isinstance(df, GeoDataFrame | GeoSeries)
    if hasattr(df, "dtypes") and hasattr(df, "columns"):
        geometry_column_count = int(df.dtypes.map(_is_geometry_like_dtype).sum())
        if geometry_column_count <= 1:
            payload = to_native_tabular_result(df)
            if payload is None and hasattr(df, "_geometry_column_name"):
                payload = _spatial_to_native_tabular_result(df)
    else:
        payload = to_native_tabular_result(df)
    if payload is not None and chosen_engine == "pyogrio" and not is_public_spatial:
        column_names = [*payload.attributes.columns, payload.geometry_name]
        driver_name = driver or plan.driver
        api_file._check_metadata_supported(kwargs.get("metadata"), chosen_engine, driver_name)
        if driver_name == "ESRI Shapefile" and any(len(str(column)) > 10 for column in column_names):
            import warnings

            warnings.warn(
                "Column names longer than 10 characters will be truncated when saved to "
                "ESRI Shapefile.",
                stacklevel=3,
            )
        return _write_vector_file_native_pyogrio(
            payload,
            filename,
            driver=driver,
            schema=schema,
            index=index,
            **kwargs,
        )

    if payload is not None and not is_public_spatial:
        df = payload.to_geodataframe()

    from vibespatial.api.io.file import _to_file

    return _to_file(
        df,
        filename,
        driver=driver,
        schema=schema,
        index=index,
        engine=chosen_engine,
        **kwargs,
    )


def benchmark_shapefile_ingest(
    *,
    geometry_type: str = "point",
    rows: int = 100_000,
    repeat: int = 5,
    seed: int = 0,
) -> list[ShapefileIngestBenchmark]:
    import pyogrio

    from .arrow import decode_wkb_owned

    if geometry_type == "point":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_points

        dataset = generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed))
    elif geometry_type == "line":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

        dataset = generate_lines(SyntheticSpec("line", "grid", count=rows, seed=seed, vertices=8))
    elif geometry_type == "polygon":
        from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

        dataset = generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=rows, seed=seed, vertices=6)
        )
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}")

    gdf = dataset.to_geodataframe()
    import tempfile

    results: list[ShapefileIngestBenchmark] = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.shp"
        gdf.to_file(path, driver="ESRI Shapefile")

        start = perf_counter()
        for _ in range(repeat):
            pyogrio.read_dataframe(path)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            ShapefileIngestBenchmark(
                implementation="pyogrio_host",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        start = perf_counter()
        last_table = None
        last_meta = None
        for _ in range(repeat):
            last_meta, last_table = pyogrio.read_arrow(path)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            ShapefileIngestBenchmark(
                implementation="pyogrio_arrow_container",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        assert last_meta is not None and last_table is not None
        geometry_index, _ = _select_arrow_geometry_column(last_table, last_meta)
        geometry_column = last_table.column(geometry_index).combine_chunks()
        wkb_values = list(geometry_column.to_pylist())

        # Benchmark the steady-state owned path, not one-time WKB kernel setup.
        read_shapefile_owned(path)

        start = perf_counter()
        for _ in range(repeat):
            read_shapefile_owned(path)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            ShapefileIngestBenchmark(
                implementation="shapefile_owned_native",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        # Native WKB decode shares the same kernel stack; exclude cold-start cost.
        decode_wkb_owned(wkb_values)
        start = perf_counter()
        for _ in range(repeat):
            decode_wkb_owned(wkb_values)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            ShapefileIngestBenchmark(
                implementation="native_wkb_decode",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )
    return results
