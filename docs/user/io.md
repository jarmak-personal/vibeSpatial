# I/O

## Supported formats

| Format | Read | Write | GPU-accelerated |
|--------|------|-------|-----------------|
| GeoParquet | `read_parquet()` | `to_parquet()` | WKB decode/encode, GeoArrow codec, GPU scan where available |
| GeoArrow / Feather | `read_feather()` | `to_feather()` | Native GeoArrow codec |
| GeoJSON | `read_file()` | `to_file()` | GPU byte-classify geometry parser plus narrowed host property decode |
| Shapefile | `read_file()` | `to_file()` | Direct SHP/DBF native path or Arrow + GPU WKB bridge |
| FlatGeobuf | `read_file()` | `to_file()` | Direct FlatBuffer GPU decode for eligible reads, native Arrow/WKB write boundary |
| GeoPackage | `read_file()` | `to_file()` | Shared Arrow + GPU WKB native boundary for supported requests |
| GeoJSONSeq | `read_file()` | `to_file()` | Rewritten to GPU GeoJSON parser for eligible reads |
| OSM PBF | `read_file()` | driver-dependent fallback | Native full-data path and parallel supported-layer public reads |
| PostGIS | `read_postgis()` | `to_postgis()` | ADBC/WKB bridge when available |

## Reading files

```python
import vibespatial

# Auto-detect format
gdf = vibespatial.read_file("data.gpkg")

# GeoParquet (recommended for large datasets)
gdf = vibespatial.read_parquet("data.parquet")

# PostGIS
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@host/db")
gdf = vibespatial.read_postgis("SELECT * FROM my_table", engine)
```

## Writing files

```python
gdf.to_file("output.gpkg", driver="GPKG")
gdf.to_parquet("output.parquet")
gdf.to_postgis("my_table", engine)
```

## GeoParquet performance

GeoParquet is the recommended format for large datasets. vibeSpatial's
GeoParquet reader:

1. Plans row group selection based on spatial metadata
2. Decodes WKB geometry on GPU when available
3. Produces device-resident `OwnedGeometryArray` without host round-trips

## GeoJSON GPU acceleration

For eligible unfiltered GeoJSON reads, vibeSpatial uses a GPU byte-classify
parser (ADR-0038) whenever CUDA is available. The parser reads file bytes to
device, classifies JSON structure, detects geometry families, extracts
coordinates, and assembles Point, LineString, and Polygon geometries directly
into owned device buffers. Property extraction is narrowed to property-object
payloads and remains host-side today.

When [kvikio](https://github.com/rapidsai/kvikio) is installed, file-to-device
transfer uses parallel POSIX I/O with pinned bounce buffers for additional
throughput.
