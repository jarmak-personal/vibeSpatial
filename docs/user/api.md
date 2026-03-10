# API Reference

## Core classes

### `vibespatial.GeoDataFrame`

A pandas DataFrame with a geometry column backed by GPU-resident storage.
Supports all standard GeoPandas methods with automatic GPU dispatch.

### `vibespatial.GeoSeries`

A pandas Series of geometries. Provides the full set of geometry operations
(area, buffer, centroid, predicates, etc.).

## Top-level functions

| Function | Description |
|----------|-------------|
| `vibespatial.read_file(path)` | Read vector file (Shapefile, GeoPackage, etc.) |
| `vibespatial.read_parquet(path)` | Read GeoParquet file |
| `vibespatial.sjoin(left, right)` | Spatial join |
| `vibespatial.sjoin_nearest(left, right)` | Nearest spatial join |
| `vibespatial.overlay(df1, df2, how)` | Spatial overlay |
| `vibespatial.clip(gdf, mask)` | Clip geometries to mask |
| `vibespatial.points_from_xy(x, y)` | Create point geometries from coordinates |

## Projection

`GeoDataFrame.to_crs()` reprojects geometries using
{external+vibeproj:doc}`vibeProj <index>` for GPU-accelerated coordinate
transforms. See the
{external+vibeproj:doc}`vibeSpatial integration guide <user/vibespatial>`
for the zero-copy `transform_buffers()` API.

## Raster

Install {external+vibespatial-raster:doc}`vibespatial-raster <index>` for
GPU raster operations under the `vibespatial.raster` namespace. See the
{external+vibespatial-raster:doc}`integration guide <user/vibespatial>`
for shared module details.

## Runtime inspection

| Function | Description |
|----------|-------------|
| `vibespatial.select_runtime()` | Query current GPU/CPU selection |
| `vibespatial.has_gpu_runtime()` | Check if GPU is available |
| `vibespatial.get_dispatch_events()` | Retrieve dispatch event log |
| `vibespatial.clear_dispatch_events()` | Clear the dispatch event log |
| `vibespatial.get_fallback_events()` | Retrieve fallback event log |
| `vibespatial.clear_fallback_events()` | Clear the fallback event log |

## Execution modes

```python
from vibespatial import ExecutionMode

ExecutionMode.AUTO  # Default: GPU when available, CPU otherwise
ExecutionMode.GPU   # Force GPU (raises if unavailable)
ExecutionMode.CPU   # Force CPU
```
