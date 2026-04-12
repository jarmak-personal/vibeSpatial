from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IOFormat(StrEnum):
    GEOARROW = "geoarrow"
    GEOPARQUET = "geoparquet"
    WKB = "wkb"
    WKT = "wkt"
    CSV = "csv"
    GEOJSON = "geojson"
    KML = "kml"
    SHAPEFILE = "shapefile"
    OSM_PBF = "osm-pbf"
    GEOPACKAGE = "geopackage"
    FILE_GEODATABASE = "file-geodatabase"
    FLATGEOBUF = "flatgeobuf"
    GML = "gml"
    GPX = "gpx"
    TOPOJSON = "topojson"
    GEOJSONSEQ = "geojsonseq"
    GDAL_LEGACY = "gdal-legacy"


class IOPathKind(StrEnum):
    GPU_NATIVE = "gpu_native"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class IOOperation(StrEnum):
    READ = "read"
    WRITE = "write"
    SCAN = "scan"
    DECODE = "decode"
    ENCODE = "encode"


@dataclass(frozen=True)
class IOSupportEntry:
    format: IOFormat
    default_path: IOPathKind
    read_path: IOPathKind
    write_path: IOPathKind
    canonical_gpu: bool
    reason: str


@dataclass(frozen=True)
class IOPlan:
    format: IOFormat
    operation: IOOperation
    selected_path: IOPathKind
    canonical_gpu: bool
    reason: str


IO_SUPPORT_MATRIX: dict[IOFormat, IOSupportEntry] = {
    IOFormat.GEOARROW: IOSupportEntry(
        format=IOFormat.GEOARROW,
        default_path=IOPathKind.GPU_NATIVE,
        read_path=IOPathKind.GPU_NATIVE,
        write_path=IOPathKind.GPU_NATIVE,
        canonical_gpu=True,
        reason="GeoArrow is the canonical columnar geometry interchange and should align directly with owned GPU-friendly buffers.",
    ),
    IOFormat.GEOPARQUET: IOSupportEntry(
        format=IOFormat.GEOPARQUET,
        default_path=IOPathKind.GPU_NATIVE,
        read_path=IOPathKind.GPU_NATIVE,
        write_path=IOPathKind.HYBRID,
        canonical_gpu=True,
        reason="GeoParquet should read through GPU-native scanning and pushdown; writes may stay hybrid while metadata assembly remains host-heavy.",
    ),
    IOFormat.WKB: IOSupportEntry(
        format=IOFormat.WKB,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.HYBRID,
        canonical_gpu=False,
        reason="WKB is a compatibility bridge format; decode and encode should be GPU-accelerated but are not the preferred storage layout.",
    ),
    IOFormat.WKT: IOSupportEntry(
        format=IOFormat.WKT,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="WKT read uses GPU byte-classification and coordinate extraction; write stays fallback (no GPU WKT serializer yet).",
    ),
    IOFormat.CSV: IOSupportEntry(
        format=IOFormat.CSV,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="CSV read uses libcudf table parse for large geometry-column files and GPU byte-classification for the remaining spatial layouts; write stays fallback (no GPU CSV serializer).",
    ),
    IOFormat.GEOJSON: IOSupportEntry(
        format=IOFormat.GEOJSON,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.HYBRID,
        canonical_gpu=False,
        reason="GeoJSON parsing and serialization can stage GPU geometry work, but text tokenization still makes this a hybrid path.",
    ),
    IOFormat.KML: IOSupportEntry(
        format=IOFormat.KML,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="KML read uses GPU byte-classification for structural analysis and coordinate extraction; write stays fallback (no GPU KML serializer).",
    ),
    IOFormat.SHAPEFILE: IOSupportEntry(
        format=IOFormat.SHAPEFILE,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.HYBRID,
        canonical_gpu=False,
        reason="Shapefile should use an explicit hybrid pipeline because the container and sidecar files are legacy host-oriented structures.",
    ),
    IOFormat.OSM_PBF: IOSupportEntry(
        format=IOFormat.OSM_PBF,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="OSM PBF read uses CPU protobuf parsing with GPU varint decoding and coordinate assembly; write is not supported.",
    ),
    IOFormat.GEOPACKAGE: IOSupportEntry(
        format=IOFormat.GEOPACKAGE,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="GeoPackage read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.FILE_GEODATABASE: IOSupportEntry(
        format=IOFormat.FILE_GEODATABASE,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="File Geodatabase read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.FLATGEOBUF: IOSupportEntry(
        format=IOFormat.FLATGEOBUF,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="FlatGeobuf read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.GML: IOSupportEntry(
        format=IOFormat.GML,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="GML read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.GPX: IOSupportEntry(
        format=IOFormat.GPX,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="GPX read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.TOPOJSON: IOSupportEntry(
        format=IOFormat.TOPOJSON,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="TopoJSON read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.GEOJSONSEQ: IOSupportEntry(
        format=IOFormat.GEOJSONSEQ,
        default_path=IOPathKind.HYBRID,
        read_path=IOPathKind.HYBRID,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="GeoJSON-Seq read uses pyogrio Arrow container parse with GPU WKB geometry decode; write via pyogrio.",
    ),
    IOFormat.GDAL_LEGACY: IOSupportEntry(
        format=IOFormat.GDAL_LEGACY,
        default_path=IOPathKind.FALLBACK,
        read_path=IOPathKind.FALLBACK,
        write_path=IOPathKind.FALLBACK,
        canonical_gpu=False,
        reason="Non-targeted GDAL formats stay behind explicit fallback adapters until a GPU-native or justified hybrid path exists.",
    ),
}


def plan_io_support(format: IOFormat | str, operation: IOOperation | str) -> IOPlan:
    normalized_format = format if isinstance(format, IOFormat) else IOFormat(format)
    normalized_operation = operation if isinstance(operation, IOOperation) else IOOperation(operation)
    entry = IO_SUPPORT_MATRIX[normalized_format]
    if normalized_operation in {IOOperation.READ, IOOperation.SCAN, IOOperation.DECODE}:
        selected = entry.read_path
    elif normalized_operation in {IOOperation.WRITE, IOOperation.ENCODE}:
        selected = entry.write_path
    else:
        selected = entry.default_path
    return IOPlan(
        format=normalized_format,
        operation=normalized_operation,
        selected_path=selected,
        canonical_gpu=entry.canonical_gpu,
        reason=entry.reason,
    )
