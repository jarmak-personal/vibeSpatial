# API Reference

This page provides a categorized overview of the vibeSpatial public API.
Every entry links to the full auto-generated documentation with parameters,
return types, and docstrings.

For the complete module-by-module reference, see the
{doc}`full API docs </apidocs/index>`.

## Core Classes

| Class | Description |
|-------|-------------|
| {py:class}`~vibespatial.api.geodataframe.GeoDataFrame` | pandas DataFrame with a geometry column backed by GPU-resident storage |
| {py:class}`~vibespatial.api.geoseries.GeoSeries` | pandas Series of geometries with the full set of spatial operations |
| {py:class}`~vibespatial.api.geometry_array.GeometryArray` | ExtensionArray wrapping Shapely geometry objects |
| {py:class}`~vibespatial.geometry.owned.OwnedGeometryArray` | GPU-resident geometry array with zero-copy device access |

## Top-Level Functions

### I/O

| Function | Description |
|----------|-------------|
| {py:func}`~vibespatial.io.file.read_vector_file` | Read vector file (Shapefile, GeoPackage, GeoJSON, etc.) -- aliased as `vibespatial.read_file()` |
| {py:func}`~vibespatial.io.arrow.read_geoparquet` | Read GeoParquet file -- aliased as `vibespatial.read_parquet()` |
| {py:func}`~vibespatial.api.io.arrow._read_feather` | Read Feather file -- aliased as `vibespatial.read_feather()` |
| {py:func}`~vibespatial.api.io.file._list_layers` | List layers in a vector file -- aliased as `vibespatial.list_layers()` |
| {py:func}`~vibespatial.api.io.sql._read_postgis` | Read from PostGIS database -- aliased as `vibespatial.read_postgis()` |
| {py:func}`~vibespatial.io.arrow.write_geoparquet` | Write GeoParquet file |
| {py:func}`~vibespatial.io.file.write_vector_file` | Write vector file |

### Spatial Operations

| Function | Description |
|----------|-------------|
| {py:func}`~vibespatial.api.tools.sjoin.sjoin` | Spatial join of two GeoDataFrames |
| {py:func}`~vibespatial.api.tools.sjoin.sjoin_nearest` | Nearest-neighbor spatial join |
| {py:func}`~vibespatial.api.tools.overlay.overlay` | Spatial overlay (intersection, union, difference, etc.) |
| {py:func}`~vibespatial.api.tools.clip.clip` | Clip geometries to a mask extent |
| {py:func}`~vibespatial.api.geometry_array.points_from_xy` | Create point geometries from x, y (and optional z) coordinates |

### Runtime

| Function | Description |
|----------|-------------|
| {py:func}`~vibespatial.runtime.select_runtime` | Query current GPU/CPU runtime selection |
| {py:func}`~vibespatial.runtime.has_gpu_runtime` | Check if a CUDA GPU is available |
| {py:func}`~vibespatial.runtime.set_execution_mode` | Set the execution mode (AUTO, GPU, CPU) |
| {py:func}`~vibespatial.runtime.get_requested_mode` | Get the currently requested execution mode |
| {py:func}`~vibespatial.runtime.dispatch.get_dispatch_events` | Retrieve the dispatch event log |
| {py:func}`~vibespatial.runtime.dispatch.clear_dispatch_events` | Clear the dispatch event log |
| {py:func}`~vibespatial.runtime.fallbacks.get_fallback_events` | Retrieve the fallback event log |
| {py:func}`~vibespatial.runtime.fallbacks.clear_fallback_events` | Clear the fallback event log |
| {py:func}`~vibespatial.api.tools._show_versions.show_versions` | Print package version info |

## GeoDataFrame Methods

Full class documentation: {py:class}`~vibespatial.api.geodataframe.GeoDataFrame`

### Constructors

| Method | Description |
|--------|-------------|
| `GeoDataFrame(data, geometry=, crs=)` | Standard constructor |
| `GeoDataFrame.from_dict()` | Construct from dictionary |
| `GeoDataFrame.from_file()` | Construct from vector file |
| `GeoDataFrame.from_features()` | Construct from GeoJSON features |
| `GeoDataFrame.from_postgis()` | Construct from PostGIS query |
| `GeoDataFrame.from_arrow()` | Construct from Arrow table |

### Geometry Management

| Method | Description |
|--------|-------------|
| `.set_geometry(col)` | Set the active geometry column |
| `.rename_geometry(col)` | Rename a geometry column |
| `.active_geometry_name` | Name of the active geometry column |
| `.geometry` | Access the active geometry as a GeoSeries |
| `.crs` | Coordinate reference system |

### Spatial Operations

| Method | Description |
|--------|-------------|
| `.dissolve(by=)` | Dissolve geometries by group with GPU-accelerated union |
| `.explode()` | Explode multi-part geometries to single parts |
| `.sjoin()` | Spatial join with another GeoDataFrame |
| `.sjoin_nearest()` | Nearest spatial join |
| `.clip(mask)` | Clip geometries by a mask |
| `.overlay(right, how=)` | Spatial overlay operation |

### Export

| Method | Description |
|--------|-------------|
| `.to_parquet(path)` | Write to GeoParquet |
| `.to_feather(path)` | Write to Feather |
| `.to_file(path)` | Write to vector file (Shapefile, GPKG, GeoJSON, etc.) |
| `.to_postgis(name, con)` | Write to PostGIS table |
| `.to_json()` | Export to GeoJSON string |
| `.to_wkb()` | Export geometries as WKB |
| `.to_wkt()` | Export geometries as WKT |
| `.to_arrow()` | Export to Arrow table |
| `.to_crs(crs)` | Reproject geometries to a different CRS |
| `.set_crs(crs)` | Assign or override CRS |
| `.to_geo_dict()` | Export as GeoJSON-compatible dict |

## GeoSeries Methods

Full class documentation: {py:class}`~vibespatial.api.geoseries.GeoSeries`

### Constructors

| Method | Description |
|--------|-------------|
| `GeoSeries(data, crs=)` | Standard constructor |
| `GeoSeries.from_wkb(data)` | Construct from WKB binary |
| `GeoSeries.from_wkt(data)` | Construct from WKT strings |
| `GeoSeries.from_xy(x, y)` | Construct points from coordinate arrays |
| `GeoSeries.from_file(path)` | Construct from vector file |
| `GeoSeries.from_arrow(arr)` | Construct from Arrow array |

### Coordinate Access

| Property | Description |
|----------|-------------|
| `.x` | X coordinate (points only) |
| `.y` | Y coordinate (points only) |
| `.z` | Z coordinate (if present) |
| `.m` | M coordinate (if present) |

## Geometry Operations

These methods are available on both {py:class}`~vibespatial.api.geoseries.GeoSeries`
and {py:class}`~vibespatial.api.geodataframe.GeoDataFrame` via the
{py:class}`~vibespatial.api.geo_base.GeoPandasBase` mixin. See the full class
docs for parameter details.

### Properties & Measurements

| Member | Description |
|--------|-------------|
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.area` | Area of each geometry |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.length` | Length of each geometry |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.bounds` | Bounding box (minx, miny, maxx, maxy) per geometry |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.total_bounds` | Total bounding box of all geometries |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.geom_type` | Geometry type string per geometry |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_valid` | Validity check |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.is_valid_reason` | Reason for invalidity |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_empty` | Empty check |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_simple` | Simplicity check |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_ring` | Ring check |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_closed` | Closed check |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.is_ccw` | Counter-clockwise winding check |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.has_z` | Has Z coordinates |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.has_m` | Has M coordinates |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.count_coordinates` | Total coordinate count |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.count_geometries` | Sub-geometry count |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.count_interior_rings` | Interior ring count |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.get_precision` | Grid precision |

### Binary Predicates

| Method | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.contains` | Contains test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.contains_properly` | Contains properly (excludes boundary) |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.covers` | Covers test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.covered_by` | Covered by test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.crosses` | Crosses test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.disjoint` | Disjoint test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.intersects` | Intersects test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.overlaps` | Overlaps test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.touches` | Touches test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.within` | Within test |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.dwithin` | Within distance threshold |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.geom_equals` | Geometry equality |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.geom_equals_exact` | Exact equality with tolerance |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.geom_equals_identical` | Identical equality |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.relate` | DE-9IM relationship string |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.relate_pattern` | Match a DE-9IM pattern |

### Constructive Operations

| Method | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.buffer` | Buffer geometries by distance |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.centroid` | Centroid point |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.convex_hull` | Convex hull |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.concave_hull` | Concave hull |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.envelope` | Minimum bounding rectangle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.minimum_rotated_rectangle` | Minimum rotated bounding rectangle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.minimum_bounding_circle` | Minimum bounding circle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.maximum_inscribed_circle` | Maximum inscribed circle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.minimum_bounding_radius` | Minimum bounding radius |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.boundary` | Geometry boundary |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.exterior` | Exterior ring (polygons) |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.interiors` | Interior rings (polygons) |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.representative_point` | Point guaranteed inside geometry |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.extract_unique_points` | All unique vertices |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.simplify` | Douglas-Peucker simplification |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.simplify_coverage` | Simplify preserving coverage topology |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.make_valid` | Repair invalid geometries |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.normalize` | Normalize ring direction |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.orient_polygons` | Orient polygon rings |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.reverse` | Reverse coordinate order |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.segmentize` | Densify by max segment length |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.offset_curve` | Offset a line by distance |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.clip_by_rect` | Clip to a rectangle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.line_merge` | Merge connected line segments |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.remove_repeated_points` | Remove duplicate vertices |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.set_precision` | Set coordinate grid precision |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.delaunay_triangles` | Delaunay triangulation |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.constrained_delaunay_triangles` | Constrained Delaunay triangulation |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.voronoi_polygons` | Voronoi diagram |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.build_area` | Build polygonal area from linework |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.polygonize` | Polygonize line segments |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.sample_points` | Random point sampling |

### Set Operations

| Method | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.difference` | Set difference |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.symmetric_difference` | Symmetric difference |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.union` | Pairwise union |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.intersection` | Pairwise intersection |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.union_all` | Union all geometries in the series |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.intersection_all` | Intersect all geometries in the series |

### Distance & Relationships

| Method | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.distance` | Distance to another geometry |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.hausdorff_distance` | Hausdorff distance |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.frechet_distance` | Frechet distance |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.shortest_line` | Shortest line between geometries |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.project` | Project point onto line |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.interpolate` | Interpolate point along line |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.snap` | Snap to another geometry |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.shared_paths` | Shared path segments |

### Transformations

| Method | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.affine_transform` | Apply affine transformation matrix |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.translate` | Translate by x/y/z offsets |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.rotate` | Rotate by angle |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.scale` | Scale by x/y/z factors |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.skew` | Skew by x/y angles |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.transform` | Apply coordinate transformation function |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.force_2d` | Drop Z/M, force 2D |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.force_3d` | Add Z coordinate |

### Coordinate Access

| Member | Description |
|--------|-------------|
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.get_coordinates` | Get coordinate arrays |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.cx` | Coordinate-based indexer |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.hilbert_distance` | Hilbert curve distance for spatial ordering |
| {py:meth}`~vibespatial.api.geo_base.GeoPandasBase.get_geometry` | Get sub-geometry by index |

### Spatial Index

| Member | Description |
|--------|-------------|
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.sindex` | Spatial index (STRtree) |
| {py:attr}`~vibespatial.api.geo_base.GeoPandasBase.has_sindex` | Whether spatial index is built |

## Execution Modes

```python
from vibespatial import ExecutionMode, set_execution_mode

set_execution_mode(ExecutionMode.AUTO)  # Default: GPU when available
set_execution_mode(ExecutionMode.GPU)   # Force GPU (raises if unavailable)
set_execution_mode(ExecutionMode.CPU)   # Force CPU
```

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `VIBESPATIAL_EXECUTION_MODE` | `auto` | Force execution mode (`auto`, `gpu`, `cpu`) |
| `VIBESPATIAL_STRICT_NATIVE` | disabled | Set `1` to error on any CPU fallback |
| `VIBESPATIAL_DETERMINISM` | disabled | Set `1` for deterministic (reproducible) results |
| `VIBESPATIAL_TRACE_WARNINGS` | disabled | Set `1` to emit warnings from execution traces |
| `VIBESPATIAL_EVENT_LOG` | disabled | Path to write structured dispatch event log |
| `VIBESPATIAL_GPU_POOL_LIMIT` | unset | Limit GPU memory pool size (bytes) |
| `VIBESPATIAL_PROVENANCE_REWRITES` | enabled | Set `0` to disable automatic query rewrites |
| `VIBESPATIAL_CCCL_CACHE` | enabled | Set `0` to disable CCCL CUBIN disk cache |
| `VIBESPATIAL_NVRTC_CACHE` | enabled | Set `0` to disable NVRTC CUBIN disk cache |
| `VIBESPATIAL_CCCL_CACHE_DIR` | `~/.cache/vibespatial/cccl` | Override CCCL cache directory |
| `VIBESPATIAL_NVRTC_CACHE_DIR` | `~/.cache/vibespatial/nvrtc` | Override NVRTC cache directory |
| `VIBESPATIAL_PRECOMPILE` | enabled | Set `0` to disable background kernel pre-compilation |

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
