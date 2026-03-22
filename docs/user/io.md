# I/O

## Supported formats

| Format | Read | Write | GPU-accelerated |
|--------|------|-------|-----------------|
| GeoParquet | `read_parquet()` | `to_parquet()` | WKB decode/encode, Arrow codec |
| GeoPackage | `read_file()` | `to_file()` | -- |
| Shapefile | `read_file()` | `to_file()` | Arrow fast path |
| GeoJSON | `read_file()` | `to_file()` | GPU byte-classify parser (Point, LineString, Polygon) |
| Feather | `read_feather()` | `to_feather()` | Arrow codec |
| PostGIS | `read_postgis()` | `to_postgis()` | -- |

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

For GeoJSON files larger than ~10 MB, vibeSpatial uses a GPU byte-classify
parser (ADR-0038) that parses JSON structure, extracts coordinates, and
assembles Point, LineString, and Polygon geometries directly on-device.
Property extraction stays on CPU via orjson.

When [kvikio](https://github.com/rapidsai/kvikio) is installed, file-to-device
transfer uses parallel POSIX I/O with pinned bounce buffers for additional
throughput.
