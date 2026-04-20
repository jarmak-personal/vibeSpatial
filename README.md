# vibeSpatial

vibeSpatial is a **GPU-first spatial analytics library** for Python.  Change
one import line and your existing GeoPandas code runs on CUDA — binary
predicates, buffer, overlay, dissolve, make-valid, spatial joins, and I/O all
dispatch to GPU kernels automatically, with **explicit, observable** CPU
compatibility fallback only when the native GPU path is unavailable or
unsupported.

> [!WARNING]
> vibeSpatial is still early, but the public GPU path is now the design center:
> the April 20, 2026 local GPU health gate reports **95.09% value-weighted
> GPU acceleration** across tracked public dispatches. [File an issue](https://github.com/jarmak-personal/vibeSpatial/issues)
> if you hit a fallback, correctness mismatch, or unexpected host transfer.
>
> The repository enforces fallback observability: once a workflow is on device,
> hidden host exits are treated as bugs, and strict-native tests fail if a path
> materializes to host without first recording an explicit fallback or
> compatibility boundary. The maintained warmed `10k` public shootout suite
> under [`benchmarks/shootout/`](benchmarks/shootout/) currently passes with
> matching fingerprints on local RTX 4090 runs; heavier workflows show clear
> wins while tiny CPU-shaped workflows are treated as crossover signals rather
> than benchmark theater.

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

**Performance on 7.2M polygons (GeoPandas CPU baseline vs current public
vibeSpatial run on local RTX 4090 / i9-13900K):**

| Step | GeoPandas | vibeSpatial | Speedup |
|---|---|---|---|
| Read GeoJSON | 57.7 s | 6.7 s | **8.6x** |
| Reproject to UTM | 8.2 s | 0.1 s | **82x** |
| Select within 1 km | 0.2 s | 0.2 s | 1.0x |
| **End-to-end including GeoParquet export** | **66.3 s** | **8.0 s** | **8.3x** |

The vibeSpatial column is the public
[`examples/nearby_buildings.py`](examples/nearby_buildings.py) path, not a
private benchmark hook. GeoJSON reading uses GPU byte-classification: NVRTC
kernels parse JSON structure, detect geometry families, extract coordinates,
and assemble geometry into owned device buffers. Property payloads are decoded
through a narrowed host seam. Reprojection uses
[vibeProj](https://github.com/jarmak-personal/vibeProj) fused GPU kernels via
`transform_buffers()` -- no host round-trip. Spatial queries use
device-resident bounding-box prefilter + GPU distance kernels.

### Current GPU coverage

The April 20, 2026 local GPU health gate reports **95.09% value-weighted
GPU acceleration** across tracked public dispatches:

| Surface | GPU work coverage |
|---|---:|
| I/O write | 99.88% |
| Query | 95.51% |
| I/O read | 94.71% |
| Other public APIs | 94.48% |
| Constructive | 85.92% |
| Overlay | 76.14% |
| Dissolve | 54.43% |

The remaining work is concentrated in exact constructive/overlay/dissolve
paths and uncommon compatibility boundaries. Silent CPU fallback is not an
accepted success mode.

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
| File I/O | native GPU/hybrid routes for GeoJSON, Shapefile, FlatGeobuf, GeoJSONSeq, OSM PBF; pyogrio for GDAL compatibility |
| Packaging | uv, hatchling |

All GPU kernels are **pure Python** — CUDA C source strings compiled at
runtime via NVRTC with background warmup (ADR-0034).  Compiled CUBINs are
cached on disk so the JIT cost is paid only once per install.  No compiled
extensions, no `nvcc` build step.  The entire suite ships as pure-Python
wheels:

| Package | Wheel size |
|---|---|
| vibespatial | 612 KB |
| vibeproj | 57 KB |
| vibespatial-raster | 51 KB |
| **Total** | **720 KB** |

### Pre-compilation

The first time a GPU operation runs, CUDA kernels are JIT-compiled in the
background (~2-3 s wall time on 8 threads).  Compiled CUBINs are cached on
disk so subsequent process starts are near-instant.  To pre-populate the
caches (e.g. in CI or after install):

```python
from vibespatial.cccl_precompile import precompile_all
precompile_all()  # compiles all 21 CCCL specs + 61 NVRTC kernels, blocks until done
```

Or from the command line:

```bash
uv run python -c "from vibespatial.cccl_precompile import precompile_all; precompile_all()"
```

See [GPU Kernel Caching](docs/architecture/gpu-kernel-caching.md) for the
full design and environment variables.

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
