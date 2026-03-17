# vibeSpatial

vibeSpatial is a **GPU-first spatial analytics library** for Python.  Change
one import line and your existing GeoPandas code runs on CUDA — binary
predicates, buffer, overlay, dissolve, make-valid, spatial joins, and I/O all
dispatch to GPU kernels automatically, with transparent CPU fallback when no
GPU is available.

> [!WARNING]
> vibeSpatial is very early in development. Operations may be unoptimized or have multiple Host/Device transfers causing reduced performance. [File an issue](https://github.com/jarmak-personal/vibeSpatial/issues) if you hit a problem!

### Install

```bash
pip install vibespatial              # CPU-only (GeoPandas drop-in)
pip install vibespatial[cu12]        # CUDA 12 GPU acceleration
pip install vibespatial[cu13]        # CUDA 13 GPU acceleration
```

### Quick start

```python
import vibespatial as gpd

gdf = gpd.read_file("my_data.gpkg")
buffered = gdf.buffer(100)
joined = gpd.sjoin(gdf, buffered)
gdf.to_parquet("out.parquet")
```

### Real-world example: 7.2 million buildings

Load every building footprint in Florida, reproject to UTM, find all buildings
within 1 km of a random pick, and export to GeoParquet.  The full script is
at [`examples/nearby_buildings.py`](examples/nearby_buildings.py).

```python
import vibespatial as gpd

# Read 7.2M buildings from Microsoft US Building Footprints
gdf = gpd.read_file("Florida.geojson")

# Reproject to UTM for metric distances
gdf_utm = gdf.to_crs(gdf.geometry.estimate_utm_crs())

# Pick a random building and find everything within 1 km
seed = gdf_utm.geometry.iloc[random.randrange(len(gdf_utm))]
nearby = gdf_utm[gdf_utm.geometry.dwithin(seed.centroid, 1_000)]

# Export to GeoParquet
nearby.to_crs(epsg=4326).to_parquet("nearby_buildings.parquet")
```

**vibeSpatial is a drop-in replacement for GeoPandas.** Here is the only diff:

```diff
-import geopandas as gpd
+import vibespatial as gpd

 gdf = gpd.read_file("Florida.geojson")
 gdf_utm = gdf.to_crs(gdf.geometry.estimate_utm_crs())
 seed = gdf_utm.geometry.iloc[random.randrange(len(gdf_utm))]
 nearby = gdf_utm[gdf_utm.geometry.dwithin(seed.centroid, 1_000)]
 nearby.to_crs(epsg=4326).to_parquet("nearby_buildings.parquet")
```

**Performance on 7.2M polygons (RTX 4090 vs GeoPandas on i9-13900k):**

| Step | GeoPandas | vibeSpatial | Speedup |
|---|---|---|---|
| Read GeoJSON | 57.7 s | 11.7 s | **4.9x** |
| Reproject to UTM | 8.2 s | 0.4 s | **21x** |
| Select within 1 km | 0.2 s | 0.5 s | -- |
| Write GeoParquet | 0.2 s | 0.1 s | 2x |
| **End-to-end** | **66.3 s** | **12.7 s** | **5.2x** |

GeoJSON reading uses GPU byte-classification: 10 NVRTC kernels parse JSON
structure, extract coordinates, and assemble geometry directly on-device in
**1.8 s** (32x vs pyogrio); property extraction stays on CPU via orjson.
Reprojection uses [vibeProj](https://github.com/jarmak-personal/vibeProj)
fused GPU kernels via `transform_buffers()` -- no host round-trip.
Spatial queries use device-resident bounding-box prefilter + GPU distance
kernels.

### Tech stack

| Layer | Technology |
|---|---|
| GPU kernels | NVRTC (runtime-compiled CUDA C via `cuda-python`) |
| GPU primitives | CCCL (`cccl` — scan, sort, reduce, select) |
| GPU arrays | CuPy (device memory, element-wise ops, prefix sums) |
| GPU JSON parse | Custom byte-classification kernels (ADR-0038) |
| GPU projection | [vibeProj](https://github.com/jarmak-personal/vibeProj) |
| GPU Parquet/Arrow | pylibcudf (WKB decode, GeoArrow codec) |
| CPU compatibility | GeoPandas API (vendored upstream test suite) |
| JSON parsing | orjson (property extraction) |
| File I/O | pyogrio (Shapefile, GPKG, small GeoJSON) |
| Packaging | uv, hatchling |

All GPU kernels are **pure Python** — CUDA C source strings compiled at
runtime via NVRTC with background warmup (ADR-0034).  No compiled extensions,
no `nvcc` build step.  The entire suite ships as pure-Python wheels:

| Package | Wheel size |
|---|---|
| vibespatial | 612 KB |
| vibeproj | 57 KB |
| vibespatial-raster | 51 KB |
| **Total** | **720 KB** |

### Documentation

See the [documentation](https://vibespatial.github.io/vibeSpatial/) for the
full API reference, GPU acceleration guide, and I/O format support matrix.

---

## Contributing

```bash
uv sync --group dev
uv run python scripts/check_docs.py --refresh
uv run python scripts/vendor_geopandas_tests.py
uv run pytest tests/upstream/geopandas/tests/test_config.py
```

### Dependency groups

- `dev`: local development and pytest tooling
- `upstream-optional`: heavier I/O and visualization extras for broader coverage
- `gpu-optional`: CUDA runtime, CuPy, pylibcudf

### Layout

- `src/vibespatial/`: package code
- `src/geopandas/`: GeoPandas compatibility shim
- `tests/`: repo-owned tests
- `tests/upstream/geopandas/`: vendored upstream GeoPandas test suite
- `docs/`: architecture docs and ADRs
- `examples/`: benchmarks and usage examples
