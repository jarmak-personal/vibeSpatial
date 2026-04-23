# vibeSpatial

vibeSpatial is an early **GPU-first spatial analytics library** for Python. It
keeps GeoPandas-style workflows on CUDA where native paths exist, and makes CPU
compatibility fallback explicit when they do not.

The strongest paths today are bulk I/O, CRS transforms, device-backed geometry
buffers, predicates, selected constructive/overlay/dissolve workloads, and
Arrow/Parquet export. Performance is workload-dependent: some public workflows
are already faster than GeoPandas, while others are still limited by
compatibility boundaries, composition overhead, or missing physical operators.

> [!WARNING]
> vibeSpatial is still under active development. Public API compatibility and
> GPU residency are improving quickly, but GPU coverage is not the same thing
> as end-to-end speed on every GeoPandas workload.
>
> Fallbacks should be observable. If a workflow silently leaves the GPU path,
> produces a correctness mismatch, or loses badly where the shape should be
> accelerated, please [file an issue](https://github.com/jarmak-personal/vibeSpatial/issues).

### Install

```bash
pip install vibespatial              # CPU-only GeoPandas-compatible API
pip install vibespatial[cu12]        # CUDA 12 GPU acceleration
pip install vibespatial[cu13]        # CUDA 13 GPU acceleration
```

### Quick Start

```python
import vibespatial as gpd

gdf = gpd.read_file("my_data.gpkg")
buffered = gdf.buffer(100)
joined = gpd.sjoin(gdf, buffered)
gdf.to_parquet("out.parquet")
```

### Example: 7.2 Million Buildings

Load every building footprint in Florida, reproject to UTM, find all buildings
within 1 km of a random pick, and export to GeoParquet.  The full script is
at [`examples/nearby_buildings.py`](examples/nearby_buildings.py).

```python
import vibespatial as gpd
import random

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

For compatible workflows, code can often stay close to GeoPandas:

```diff
-import geopandas as gpd
+import vibespatial as gpd

 gdf = gpd.read_file("Florida.geojson")
 gdf_utm = gdf.to_crs(gdf.geometry.estimate_utm_crs())
 seed = gdf_utm.geometry.iloc[random.randrange(len(gdf_utm))]
 nearby = gdf_utm[gdf_utm.geometry.dwithin(seed.centroid, 1_000)]
 nearby.to_crs(epsg=4326).to_parquet("nearby_buildings.parquet")
```

This public example is currently I/O and reprojection dominated, which is where
vibeSpatial is strongest. On a local RTX 4090 / i9-13900K run:

| Step | GeoPandas | vibeSpatial | Speedup |
|---|---|---|---|
| Read GeoJSON | 57.7 s | 6.7 s | **8.6x** |
| Reproject to UTM | 8.2 s | 0.1 s | **82x** |
| Select within 1 km | 0.2 s | 0.2 s | 1.0x |
| End-to-end including GeoParquet export | 66.3 s | 8.0 s | 8.3x |

This is one representative public path, not a blanket performance claim. The
maintained shootout workflows in [`benchmarks/shootout/`](benchmarks/shootout/)
are used to track where performance generalizes and where more physical-plan
work is still needed.

### Current Focus

- Keep geometry device-resident across public workflows instead of repeatedly
  materializing pandas/Shapely intermediates.
- Expand reusable physical shapes such as semijoins, anti-semijoins,
  many-few overlay, mask clip, and grouped geometry reduction.
- Preserve GeoPandas compatibility while making CPU fallback and host/device
  transfers visible.
- Use vendored GeoPandas tests and public workflow shootouts as the correctness
  and performance contract.

### Tech Stack

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
| File I/O | Native GPU/hybrid routes for GeoJSON, Shapefile, FlatGeobuf, GeoJSONSeq, and OSM PBF; pyogrio for GDAL compatibility |
| Packaging | uv, hatchling |

GPU kernels are shipped as Python source strings and compiled at runtime with
NVRTC. Compiled CUBINs are cached on disk, so the JIT cost is paid once per
install. No compiled extensions or `nvcc` build step are required.

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
