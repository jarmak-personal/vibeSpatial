# Quickstart

## Create a GeoDataFrame

```python
import vibespatial
from shapely.geometry import Point, Polygon

gdf = vibespatial.GeoDataFrame(
    {"city": ["A", "B", "C"], "pop": [100, 200, 300]},
    geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
)
```

## Geometry operations

All standard GeoPandas geometry operations are available. When a GPU is
present, operations dispatch to CUDA kernels automatically.

```python
# Buffer
buffered = gdf.geometry.buffer(0.5)

# Binary predicates
contains = gdf.geometry.contains(Point(0.5, 0.5))

# Area, length, centroid
areas = gdf.geometry.area
centroids = gdf.geometry.centroid
```

## Spatial joins

```python
result = vibespatial.sjoin(gdf, other_gdf, predicate="intersects")
```

## Overlay operations

```python
intersection = vibespatial.overlay(gdf1, gdf2, how="intersection")
union = vibespatial.overlay(gdf1, gdf2, how="union")
```

## Dissolve

```python
dissolved = gdf.dissolve(by="city", aggfunc="sum")
```

## I/O

```python
# Read
gdf = vibespatial.read_file("data.gpkg")
gdf = vibespatial.read_parquet("data.parquet")

# Write
gdf.to_file("output.gpkg")
gdf.to_parquet("output.parquet")
```

## Observing dispatch

vibeSpatial records which operations went to GPU vs CPU:

```python
vibespatial.clear_dispatch_events()

gdf.geometry.buffer(1.0)

for event in vibespatial.get_dispatch_events():
    print(f"{event.surface}: {event.implementation} ({event.selected})")
```
