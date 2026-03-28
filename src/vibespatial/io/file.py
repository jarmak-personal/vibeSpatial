from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from vibespatial.geometry.owned import (
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event

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


def plan_shapefile_ingest(*, prefer: str = "arrow-wkb") -> ShapefileIngestPlan:
    selected_strategy = "arrow-wkb" if prefer == "auto" else prefer
    implementation = {
        "arrow-wkb": "shapefile_arrow_wkb_owned",
    }.get(selected_strategy, "shapefile_arrow_wkb_owned")
    return ShapefileIngestPlan(
        implementation=implementation,
        selected_strategy=selected_strategy,
        uses_pyogrio_container=True,
        uses_arrow_batch=True,
        uses_native_wkb_decode=True,
        reason=(
            "Use host-side pyogrio container parsing for the Shapefile sidecar ecosystem, "
            "then land geometry through Arrow WKB batches into owned buffers without per-row "
            "Shapely construction. Keep attributes in a columnar table."
        ),
    )


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


def read_shapefile_owned(
    source: str | Path,
    *,
    bbox=None,
    columns=None,
    rows=None,
    **kwargs,
) -> ShapefileOwnedBatch:
    import pyogrio

    from .arrow import decode_wkb_arrow_array_owned

    plan = plan_shapefile_ingest()
    skip_features, max_features = _normalize_feature_window(rows)
    record_dispatch_event(
        surface="vibespatial.io.shapefile",
        operation="read_owned",
        implementation=plan.implementation,
        reason=plan.reason,
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


def _normalize_driver(filename, driver: str | None = None) -> str:
    if driver is not None:
        return driver
    if not isinstance(filename, str | Path):
        return "GDAL-legacy"
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
        reason = "GeoJSON stays a staged hybrid path with pyogrio-first routing."
    elif normalized_driver == "ESRI Shapefile":
        io_format = IOFormat.SHAPEFILE
        implementation = "shapefile_hybrid_adapter"
        reason = "Shapefile stays an explicit hybrid path because the container remains host-oriented."
    elif normalized_driver == "WKT":
        io_format = IOFormat.WKT
        implementation = "wkt_gpu_hybrid_adapter"
        reason = "WKT uses GPU byte-classification for geometry parsing with host fallback."
    elif normalized_driver == "CSV":
        io_format = IOFormat.CSV
        implementation = "csv_gpu_hybrid_adapter"
        reason = "CSV uses GPU byte-classification for structural analysis and coordinate extraction with host fallback."
    elif normalized_driver == "KML":
        io_format = IOFormat.KML
        implementation = "kml_gpu_hybrid_adapter"
        reason = "KML uses GPU byte-classification for structural analysis and coordinate extraction with host fallback."
    elif normalized_driver == "OSM-PBF":
        io_format = IOFormat.OSM_PBF
        implementation = "osm_pbf_gpu_hybrid_adapter"
        reason = "OSM PBF uses CPU protobuf parsing with GPU varint decoding and coordinate assembly."
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
        from .arrow import geoseries_from_owned
        from .kvikio_reader import read_file_to_device

        file_path = Path(filename)
        file_size = file_path.stat().st_size
        result = read_file_to_device(file_path, file_size)
        d_bytes = result.device_bytes

        from .wkt_gpu import read_wkt_gpu

        owned = read_wkt_gpu(d_bytes)

        # WKT has no embedded CRS.  When target_crs is requested, set it
        # as the CRS label on the output (no reprojection possible without
        # a known source CRS).
        crs = target_crs if target_crs is not None else None
        geom_series = geoseries_from_owned(owned, name="geometry", crs=crs)

        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(geometry=geom_series)
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


def _try_csv_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a CSV file using the GPU byte-classification pipeline.

    Performs GPU structural analysis to find lat/lon or WKT geometry
    columns, then extracts spatial data and assembles a GeoDataFrame.
    Returns None on failure to fall through to CPU.
    """
    try:
        from .arrow import geoseries_from_owned
        from .kvikio_reader import read_file_to_device

        file_path = Path(filename)
        file_size = file_path.stat().st_size
        result = read_file_to_device(file_path, file_size)
        d_bytes = result.device_bytes

        from .csv_gpu import read_csv_gpu

        csv_result = read_csv_gpu(d_bytes)

        # CSV has no embedded CRS.  When target_crs is requested, set it
        # as the CRS label on the output.
        crs = target_crs if target_crs is not None else None
        geom_series = geoseries_from_owned(csv_result.geometry, name="geometry", crs=crs)

        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(geometry=geom_series)
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="csv_gpu_byte_classify_adapter",
            reason=(
                "GPU byte-classification: GPU structural analysis and "
                "coordinate extraction for CSV spatial data."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


def _try_shapefile_dbf_gpu_read(filename) -> object | None:
    """Try to read a Shapefile with GPU DBF attribute parsing.

    Uses pyogrio.read_arrow() for geometry (via GPU WKB decode) and
    the GPU DBF reader for attributes, bypassing the Arrow->pandas
    attribute conversion for numeric columns. Falls back to None on
    failure.
    """
    try:
        import pyogrio

        from .arrow import decode_wkb_arrow_array_owned, geoseries_from_owned
        from .dbf_gpu import dbf_result_to_dataframe, read_dbf_gpu

        file_path = Path(filename)
        dbf_path = file_path.with_suffix(".dbf")
        if not dbf_path.exists():
            return None

        # Read geometry via pyogrio Arrow + GPU WKB decode
        metadata, table = pyogrio.read_arrow(filename)
        geom_idx, _ = _select_arrow_geometry_column(table, metadata)
        geom_column = table.column(geom_idx).combine_chunks()
        owned = decode_wkb_arrow_array_owned(geom_column)
        crs = metadata.get("crs")
        geom_series = geoseries_from_owned(
            owned,
            name=table.schema.field(geom_idx).name,
            crs=crs,
        )

        # Read attributes via GPU DBF parser
        dbf_result = read_dbf_gpu(dbf_path)
        attrs_df = dbf_result_to_dataframe(dbf_result)

        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(attrs_df, geometry=geom_series)
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


def _try_kml_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read a KML file using the GPU byte-classification pipeline.

    KML files are XML-based with Placemark elements containing geometry.
    Returns a GeoDataFrame with a geometry column only, or None on failure.
    """
    try:
        from .arrow import geoseries_from_owned
        from .kvikio_reader import read_file_to_device

        file_path = Path(filename)
        file_size = file_path.stat().st_size
        result = read_file_to_device(file_path, file_size)
        d_bytes = result.device_bytes

        from .kml_gpu import read_kml_gpu

        owned = read_kml_gpu(d_bytes)

        # KML has no embedded CRS.  When target_crs is requested, set it
        # as the CRS label on the output.
        crs = target_crs if target_crs is not None else None
        geom_series = geoseries_from_owned(owned, name="geometry", crs=crs)

        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(geometry=geom_series)
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


def _try_osm_pbf_gpu_read(filename, *, target_crs: str | None = None) -> object | None:
    """Try to read an OSM PBF file using the GPU hybrid pipeline.

    OSM PBF files contain DenseNodes (Points), Ways (LineStrings, Polygons),
    and Relations (MultiPolygons) extracted via CPU protobuf parsing with
    GPU varint decoding, coordinate assembly, and binary-search-based
    coordinate resolution.  Returns a GeoDataFrame with mixed geometry
    types and OSM ID columns, or None on failure.
    """
    try:
        from .arrow import geoseries_from_owned
        from .osm_gpu import read_osm_pbf

        osm_result = read_osm_pbf(filename)
        has_nodes = osm_result.nodes is not None and osm_result.n_nodes > 0
        has_ways = osm_result.ways is not None and osm_result.n_ways > 0
        has_relations = osm_result.relations is not None and osm_result.n_relations > 0

        if not has_nodes and not has_ways and not has_relations:
            return None

        # OSM PBF has no embedded CRS; coordinates are WGS84 by definition.
        crs = target_crs if target_crs is not None else "EPSG:4326"

        import cupy as _cp

        import vibespatial.api as geopandas

        # Count how many element types we have -- if only one, return a
        # simple single-type GeoDataFrame; if multiple, concatenate with
        # osm_element / osm_id columns.
        n_types = int(has_nodes) + int(has_ways) + int(has_relations)

        if n_types == 1:
            if has_nodes:
                geom_series = geoseries_from_owned(osm_result.nodes, name="geometry", crs=crs)
                data: dict[str, object] = {}
                if osm_result.node_ids is not None:
                    data["osm_node_id"] = _cp.asnumpy(osm_result.node_ids)
                gdf = geopandas.GeoDataFrame(data, geometry=geom_series)
            elif has_ways:
                geom_series = geoseries_from_owned(osm_result.ways, name="geometry", crs=crs)
                data = {}
                if osm_result.way_ids is not None:
                    data["osm_way_id"] = _cp.asnumpy(osm_result.way_ids)
                gdf = geopandas.GeoDataFrame(data, geometry=geom_series)
            else:
                geom_series = geoseries_from_owned(osm_result.relations, name="geometry", crs=crs)
                data = {}
                if osm_result.relation_ids is not None:
                    data["osm_relation_id"] = _cp.asnumpy(osm_result.relation_ids)
                gdf = geopandas.GeoDataFrame(data, geometry=geom_series)
        else:
            # Multiple element types -- build per-type GeoDataFrames and concatenate.
            import pandas as pd

            frames = []

            if has_nodes:
                node_series = geoseries_from_owned(osm_result.nodes, name="geometry", crs=crs)
                node_data: dict[str, object] = {"osm_element": "node"}
                if osm_result.node_ids is not None:
                    node_data["osm_id"] = _cp.asnumpy(osm_result.node_ids)
                frames.append(geopandas.GeoDataFrame(node_data, geometry=node_series))

            if has_ways:
                way_series = geoseries_from_owned(osm_result.ways, name="geometry", crs=crs)
                way_data: dict[str, object] = {"osm_element": "way"}
                if osm_result.way_ids is not None:
                    way_data["osm_id"] = _cp.asnumpy(osm_result.way_ids)
                frames.append(geopandas.GeoDataFrame(way_data, geometry=way_series))

            if has_relations:
                rel_series = geoseries_from_owned(osm_result.relations, name="geometry", crs=crs)
                rel_data: dict[str, object] = {"osm_element": "relation"}
                if osm_result.relation_ids is not None:
                    rel_data["osm_id"] = _cp.asnumpy(osm_result.relation_ids)
                frames.append(geopandas.GeoDataFrame(rel_data, geometry=rel_series))

            gdf = geopandas.GeoDataFrame(
                pd.concat(frames, ignore_index=True),
            )

        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation="osm_pbf_gpu_hybrid_adapter",
            reason=(
                "GPU hybrid OSM PBF: CPU protobuf parsing with GPU varint "
                "decoding, coordinate assembly, Way/Relation coordinate "
                "resolution via binary-search kernel, and MultiPolygon "
                "assembly from Relation Way members."
            ),
            selected=ExecutionMode.GPU,
        )
        return gdf
    except Exception:
        return None


# Minimum file size (in bytes) for GPU fast-path routing.
# Below this threshold, let the CPU handle it -- kernel launch overhead
# dominates for small files.
_GPU_MIN_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _try_gpu_read_file(
    filename,
    *,
    plan,
    bbox,
    columns,
    rows,
    target_crs: str | None = None,
    build_index: bool = False,
    **kwargs,
):
    """Try to read a vector file using the GPU-dominant owned path.

    Uses pyogrio.read_arrow() for container parsing, then GPU WKB decode
    for geometry, and assembles a GeoDataFrame from the owned geometry and
    Arrow attribute columns. Falls back to None if anything fails,
    triggering the vendored CPU path.
    """
    import pyogrio

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.runtime import get_requested_mode

    from .arrow import decode_wkb_arrow_array_owned, geoseries_from_owned

    if get_requested_mode() is ExecutionMode.CPU:
        return None

    runtime = get_cuda_runtime()
    if not runtime.available():
        return None

    # Format-specific GPU fast paths: direct GPU parsing of the entire file,
    # bypassing pyogrio entirely.  Only for unfiltered reads.
    if bbox is None and columns is None and rows is None:
        file_path = Path(filename)
        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        # WKT: always route to GPU (pyogrio/GDAL does not support raw WKT)
        if plan.format is IOFormat.WKT:
            gdf = _try_wkt_gpu_read(filename, target_crs=target_crs)
            if gdf is not None and build_index:
                _attach_gpu_spatial_index(gdf)
            return gdf

        # OSM PBF: always route to GPU (pyogrio/GDAL does not support PBF)
        if plan.format is IOFormat.OSM_PBF:
            gdf = _try_osm_pbf_gpu_read(filename, target_crs=target_crs)
            if gdf is not None and build_index:
                _attach_gpu_spatial_index(gdf)
            return gdf

        # CSV: route to GPU for files above size threshold
        if plan.format is IOFormat.CSV and file_size > _GPU_MIN_FILE_SIZE:
            gpu_result = _try_csv_gpu_read(filename, target_crs=target_crs)
            if gpu_result is not None:
                if build_index:
                    _attach_gpu_spatial_index(gpu_result)
                return gpu_result
            # fall through to pyogrio CPU path

        # KML: route to GPU for files above size threshold
        if plan.format is IOFormat.KML and file_size > _GPU_MIN_FILE_SIZE:
            gpu_result = _try_kml_gpu_read(filename, target_crs=target_crs)
            if gpu_result is not None:
                if build_index:
                    _attach_gpu_spatial_index(gpu_result)
                return gpu_result
            # fall through to pyogrio CPU path

        # Shapefile: route to GPU DBF parser for attributes (large files)
        if plan.format is IOFormat.SHAPEFILE and file_size > _GPU_MIN_FILE_SIZE:
            gpu_result = _try_shapefile_dbf_gpu_read(filename)
            if gpu_result is not None:
                if target_crs is not None:
                    _reproject_gdf_gpu(gpu_result, target_crs)
                if build_index:
                    _attach_gpu_spatial_index(gpu_result)
                return gpu_result
            # fall through to pyogrio CPU path

        # GeoJSON GPU byte-classify fast path: direct GPU parsing of the
        # entire file for files > 10 MB.
        if plan.format is IOFormat.GEOJSON and file_size > _GPU_MIN_FILE_SIZE:
            try:
                from .geojson_gpu import read_geojson_gpu

                gpu_result = read_geojson_gpu(file_path, target_crs=target_crs)
                # GeoJSON is EPSG:4326 by spec (RFC 7946).  If target_crs
                # was set, coordinates are already reprojected.
                effective_crs = target_crs if target_crs is not None else "EPSG:4326"
                geom_series = geoseries_from_owned(
                    gpu_result.owned, name="geometry", crs=effective_crs,
                )
                props_df = gpu_result.extract_properties_dataframe()
                import vibespatial.api as geopandas

                gdf = geopandas.GeoDataFrame(props_df, geometry=geom_series)
                record_dispatch_event(
                    surface="geopandas.read_file",
                    operation="read_file",
                    implementation="geojson_gpu_byte_classify_adapter",
                    reason=(
                        "GPU byte-classification: direct GPU parsing of GeoJSON with "
                        "NVRTC kernels, bypassing pyogrio for geometry."
                    ),
                    selected=ExecutionMode.GPU,
                )
                if build_index:
                    _attach_gpu_spatial_index(gdf)
                return gdf
            except Exception:
                pass  # fall through to pyogrio GPU WKB path

    try:
        skip_features, max_features = _normalize_feature_window(rows)
        arrow_kwargs = {k: v for k, v in kwargs.items() if k not in ("engine",)}
        if bbox is not None:
            if hasattr(bbox, "total_bounds"):
                # GeoDataFrame/GeoSeries bbox
                meta = pyogrio.read_info(filename, layer=arrow_kwargs.get("layer"))
                crs = meta.get("crs")
                if crs is not None and bbox.crs is not None:
                    bbox = tuple(bbox.to_crs(crs).total_bounds)
                else:
                    bbox = tuple(bbox.total_bounds)
            elif hasattr(bbox, "bounds"):
                # Shapely geometry
                bbox = bbox.bounds
            arrow_kwargs["bbox"] = bbox
        metadata, table = pyogrio.read_arrow(
            filename,
            columns=columns,
            skip_features=skip_features,
            max_features=max_features,
            **arrow_kwargs,
        )
        # Find the geometry column.
        geom_idx, _ = _select_arrow_geometry_column(table, metadata)
        geom_column = table.column(geom_idx).combine_chunks()

        # Try GPU WKB decode.
        owned = decode_wkb_arrow_array_owned(geom_column)

        # Fused reproject: transform on device before GeoSeries assembly.
        source_crs = metadata.get("crs")
        if target_crs is not None and source_crs is not None:
            _reproject_owned_inplace(owned, src_crs=source_crs, dst_crs=target_crs)
            effective_crs = target_crs
        else:
            effective_crs = source_crs

        # Build GeoSeries from owned geometry.
        geom_series = geoseries_from_owned(
            owned,
            name=table.schema.field(geom_idx).name,
            crs=effective_crs,
        )

        # ADR-0036 boundary: keep Arrow table through intermediate processing;
        # defer .to_pandas() to the GeoDataFrame construction point.
        import datetime

        attrs_table = table.remove_column(geom_idx)
        attrs_df = attrs_table.to_pandas()

        # Normalise timezone representation for columns that PyArrow annotated
        # with zoneinfo.ZoneInfo('UTC'): upstream GeoPandas and pyogrio expose
        # datetime.timezone.utc which is what the contract tests expect.
        for col in attrs_df.columns:
            s = attrs_df[col]
            if hasattr(s, "dt") and s.dt.tz is not None:
                try:
                    attrs_df[col] = s.dt.tz_convert(datetime.UTC)
                except Exception:
                    pass

        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(attrs_df, geometry=geom_series)

        # Record GPU dispatch event.
        gpu_impl = (
            "geojson_gpu_adapter" if plan.format is IOFormat.GEOJSON else "shapefile_gpu_adapter"
        )
        record_dispatch_event(
            surface="geopandas.read_file",
            operation="read_file",
            implementation=gpu_impl,
            reason=(
                "GPU-dominant read: pyogrio Arrow container parse, device-side WKB geometry "
                "decode via pylibcudf, owned-buffer GeoDataFrame assembly."
            ),
            selected=ExecutionMode.GPU,
        )
        if build_index:
            _attach_gpu_spatial_index(gdf)
        return gdf
    except Exception:
        return None


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
    """Read a vector file into a GeoDataFrame.

    Supports Shapefile, GeoPackage, GeoJSON, WKT, CSV, KML, OSM PBF,
    and any format readable by pyogrio/fiona.

    GPU acceleration is automatic for GeoJSON, Shapefile, WKT, CSV, KML,
    and OSM PBF formats.  For CSV, KML, GeoJSON, and Shapefile, GPU
    acceleration activates only for files larger than 10 MB (below this
    threshold, CPU is faster due to kernel launch overhead).  WKT and
    OSM PBF always use the GPU path -- there is no CPU fallback for
    these formats (pyogrio/GDAL does not support them).

    The GPU fast path is disabled when any of ``bbox``, ``columns``,
    ``rows``, or ``mask`` parameters are specified, or when ``engine``
    is explicitly set.  In those cases the reader falls through to the
    CPU pyogrio/fiona path.

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
        Passed through to the underlying engine.

    Returns
    -------
    GeoDataFrame
    """
    plan = plan_vector_file_io(filename, operation=IOOperation.READ)

    # Try GPU-dominant owned path for supported formats.
    _GPU_DISPATCH_FORMATS = {
        IOFormat.GEOJSON,
        IOFormat.SHAPEFILE,
        IOFormat.WKT,
        IOFormat.CSV,
        IOFormat.KML,
        IOFormat.OSM_PBF,
    }
    if plan.format in _GPU_DISPATCH_FORMATS and mask is None and engine is None:
        gpu_result = _try_gpu_read_file(
            filename,
            plan=plan,
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
        raise RuntimeError(
            f"Cannot read WKT file '{filename}': GPU runtime is required for raw "
            "WKT files (pyogrio/GDAL does not support this format). Ensure a CUDA "
            "GPU is available and CuPy is installed."
        )

    # OSM PBF files have no CPU fallback (pyogrio/GDAL does not support PBF).
    if plan.format is IOFormat.OSM_PBF:
        raise RuntimeError(
            f"Cannot read OSM PBF file '{filename}': GPU runtime is required for "
            "OSM PBF files (pyogrio/GDAL does not support this format). Ensure a "
            "CUDA GPU is available and CuPy is installed."
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
    from vibespatial.api.io.file import _read_file

    gdf = _read_file(
        filename,
        bbox=bbox,
        mask=mask,
        columns=columns,
        rows=rows,
        engine=chosen_engine,
        **kwargs,
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
    plan = plan_vector_file_io(filename, operation=IOOperation.WRITE, driver=driver)
    chosen_engine = kwargs.pop("engine", None)
    if plan.format in {IOFormat.GEOJSON, IOFormat.SHAPEFILE} and chosen_engine is None:
        chosen_engine = "pyogrio"
    record_dispatch_event(
        surface="geopandas.geodataframe.to_file",
        operation="to_file",
        implementation=plan.implementation,
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
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

        start = perf_counter()
        for _ in range(repeat):
            read_shapefile_owned(path)
        elapsed = (perf_counter() - start) / repeat
        results.append(
            ShapefileIngestBenchmark(
                implementation="shapefile_owned_batch",
                geometry_type=geometry_type,
                rows=rows,
                elapsed_seconds=elapsed,
                rows_per_second=rows / elapsed if elapsed else float("inf"),
            )
        )

        from .arrow import decode_wkb_owned

        wkb_values = list(geometry_column.to_pylist())
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
