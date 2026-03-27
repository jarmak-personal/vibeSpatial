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


def _try_gpu_read_file(filename, *, plan, bbox, columns, rows, **kwargs):
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

    # GeoJSON GPU byte-classify fast path: direct GPU parsing of the entire file,
    # bypassing pyogrio entirely. Only for unfiltered reads of files > 10 MB.
    if (
        plan.format is IOFormat.GEOJSON
        and bbox is None
        and columns is None
        and rows is None
    ):
        try:
            file_path = Path(filename)
            if file_path.stat().st_size > 10 * 1024 * 1024:
                from .geojson_gpu import read_geojson_gpu
                gpu_result = read_geojson_gpu(file_path)
                # GeoJSON is EPSG:4326 by spec (RFC 7946)
                geom_series = geoseries_from_owned(
                    gpu_result.owned, name="geometry", crs="EPSG:4326",
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

        # Extract CRS from metadata.
        crs = metadata.get("crs")

        # Build GeoSeries from owned geometry.
        geom_series = geoseries_from_owned(
            owned,
            name=table.schema.field(geom_idx).name,
            crs=crs,
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
    **kwargs,
):
    """Read a vector file into a GeoDataFrame.

    Supports Shapefile, GeoPackage, GeoJSON, and any format readable by
    pyogrio/fiona.  For GeoJSON and Shapefile inputs the reader attempts a
    GPU-accelerated owned path first; other formats fall back to pyogrio.

    Aliased as ``vibespatial.read_file()``.

    Parameters
    ----------
    filename : str or Path
        Path to the vector file.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial filter bounding box.
    mask : Geometry or GeoDataFrame, optional
        Spatial filter mask geometry.
    columns : list of str, optional
        Subset of columns to read.
    rows : int or slice, optional
        Subset of rows to read.
    engine : str, optional
        Force a specific I/O engine (``"pyogrio"`` or ``"fiona"``).
    **kwargs
        Passed through to the underlying engine.

    Returns
    -------
    GeoDataFrame
    """
    plan = plan_vector_file_io(filename, operation=IOOperation.READ)

    # Try GPU-dominant owned path for GeoJSON and Shapefile.
    if plan.format in {IOFormat.GEOJSON, IOFormat.SHAPEFILE} and mask is None and engine is None:
        gpu_result = _try_gpu_read_file(
            filename, plan=plan, bbox=bbox, columns=columns, rows=rows, **kwargs
        )
        if gpu_result is not None:
            return gpu_result

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

    return _read_file(
        filename,
        bbox=bbox,
        mask=mask,
        columns=columns,
        rows=rows,
        engine=chosen_engine,
        **kwargs,
    )


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
