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
        reason="CSV read uses GPU byte-classification for structural analysis and coordinate extraction; write stays fallback (no GPU CSV serializer).",
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
