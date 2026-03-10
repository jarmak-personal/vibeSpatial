# I/O

## Supported formats

| Format | Read | Write | GPU-accelerated |
|--------|------|-------|-----------------|
| GeoParquet | `read_parquet()` | `to_parquet()` | WKB decode, Arrow codec |
| GeoPackage | `read_file()` | `to_file()` | -- |
| Shapefile | `read_file()` | `to_file()` | Arrow fast path |
| GeoJSON | `read_file()` | `to_file()` | GPU byte-classify (>10 MB) |
| Feather | `read_feather()` | `to_feather()` | Arrow codec |

## Reading files

```python
import vibespatial

# Auto-detect format
gdf = vibespatial.read_file("data.gpkg")

# GeoParquet (recommended for large datasets)
gdf = vibespatial.read_parquet("data.parquet")
```

## Writing files

```python
gdf.to_file("output.gpkg", driver="GPKG")
gdf.to_parquet("output.parquet")
```

## GeoParquet performance

GeoParquet is the recommended format for large datasets. vibeSpatial's
GeoParquet reader:

1. Plans row group selection based on spatial metadata
2. Decodes WKB geometry on GPU when available
3. Produces device-resident `OwnedGeometryArray` without host round-trips
