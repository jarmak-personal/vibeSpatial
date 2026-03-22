# File Format IO

<!-- DOC_HEADER:START
Scope: File-based vector format routing for GeoJSON, Shapefile, and legacy GDAL adapters.
Read If: You are changing read_file, to_file, GeoJSON ingest, Shapefile ingest, or file-format routing.
STOP IF: Your task already has the specific format adapter open and only needs local implementation detail.
Source Of Truth: File-format IO architecture for GeoJSON, Shapefile, and GDAL legacy adapters.
Body Budget: 210/280 lines
Document: docs/architecture/io-files.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-17 | Request Signals |
| 18-23 | Open First |
| 24-29 | Verify |
| 30-35 | Risks |
| 36-44 | Decision |
| 45-54 | Performance Notes |
| 55-120 | Current Behavior |
| 121-210 | Measured Local Baseline |
DOC_HEADER:END -->

## Intent

Define how file-based vector formats should route through the repo while
keeping GPU-native formats primary and legacy formats explicit.

## Request Signals

- io file
- geojson
- shapefile
- read_file
- to_file
- file format
- gdal

## Open First

- docs/architecture/io-files.md
- src/vibespatial/io/file.py
- tests/test_io_file.py

## Verify

- `uv run pytest tests/test_io_file.py`
- `uv run python scripts/benchmark_io_file.py --suite smoke`
- `uv run python scripts/check_docs.py --check`

## Risks

- Legacy GDAL formats masquerading as native hides work that bypasses the GPU stack.
- GeoJSON geometry ingest is now GPU-accelerated (ADR-0038); remaining bottleneck is CPU property extraction.
- Shapefile adapter losing speed leadership if the raw Arrow-binary fast path regresses.

## Decision

- GeoJSON is a first-class hybrid path. Files >10 MB auto-route to GPU
  byte-classification (ADR-0038); smaller files and filtered reads use pyogrio.
- Shapefile is a first-class hybrid path with pyogrio-first routing.
- Other GDAL vector formats stay behind an explicit legacy fallback adapter.
- Public `geopandas.read_file` and `GeoDataFrame.to_file` should dispatch
  through repo-owned wrappers so the chosen path is observable.

## Performance Notes

- GeoJSON files >10 MB now auto-route to GPU byte-classification (ADR-0038)
  for geometry, with pyogrio as fallback. Shapefile stays pyogrio-first.
- The pyogrio bias is retained for Shapefile and small GeoJSON because that
  path keeps us closer to Arrow- and columnar-oriented follow-on work.
- Legacy GDAL formats should not masquerade as native; the extra explicit
  fallback event is part of the performance contract because it exposes work
  that still bypasses the GPU-oriented stack.

## Current Behavior

- `geopandas.read_file` now classifies GeoJSON, Shapefile, and legacy GDAL
  paths through one repo-owned router.
- `GeoDataFrame.to_file` uses the same routing policy.
- GeoJSON and Shapefile record dispatch events without fallback events.
- Legacy formats such as GPKG emit explicit fallback events.
- Repo-owned GeoJSON ingest now also has an internal staged owned path:
  - `auto` now selects the `fast-json` strategy: `orjson` for parsing (when
    available, otherwise CPython `json`) plus vectorized per-family coordinate
    extraction directly into numpy owned buffers. This eliminates the old
    per-feature `_append_geojson_geometry` loop and is 2.4-2.6x faster than
    the previous `full-json` default, and 3.5-3.9x faster than `pyogrio`.
  - `prefer="chunked"` splits the features array into byte-range chunks,
    parses each chunk with orjson, and extracts coordinates with vectorized
    numpy. Slightly slower than single-pass fast-json but reduces peak memory.
  - `prefer="full-json"` remains available as the legacy host path using
    `json.loads` plus per-element native geometry assembly
  - `prefer="pylibcudf"` uses host feature-span discovery plus `pylibcudf`
    bulk JSON-path extraction and family-local GPU parsing; now slower than
    `fast-json` because host-side span discovery dominated GPU savings
  - `prefer="pylibcudf-arrays"` exposes a cleaner splitter-free GPU prototype
    that extracts `$.features[*].geometry.type` and
    `$.features[*].geometry.coordinates` directly from the full
    `FeatureCollection` and assembles owned buffers from concatenated typed
    columns
  - `prefer="pylibcudf-rowized"` exposes an experimental device-rowization
    prototype for homogeneous feature arrays, but it is intentionally not the
    default GPU route
  - the GPU path uses coordinates-only parsing for point/line families and
    full-geometry parsing for polygon families, because coordinates-only parsing
    loses ring structure for polygons
  - property dictionaries are materialized lazily on the owned batch, so
    geometry-only callers do not pay host-side property decode by default
  - `prefer="gpu-byte-classify"` uses 12 NVRTC kernels for GPU byte
    classification, structural scanning, geometry type detection, coordinate
    extraction, and ASCII-to-fp64 parsing directly on device-resident file
    bytes. Supports homogeneous and mixed Point, LineString, and Polygon
    files. Type detection scans for `"type":` keys at geometry depth,
    classifies per-feature, then partitions into family-local decode batches
    (per io-acceleration.md policy). Property extraction stays on CPU via
    orjson (hybrid design per ADR-0038).
    Geometry parse: **1.8s** for 2.16 GB / 7.2M polygons (32x vs pyogrio).
    Total read including properties: **11.7s** (4.9x vs pyogrio).
    File-to-device transfer uses kvikio when installed (parallel POSIX reads
    with pinned bounce buffers, no GDS required), falling back to
    `cp.asarray` otherwise. Thread count is tunable via `KVIKIO_NTHREADS`.
  - the `read_file` GPU path auto-selects `gpu-byte-classify` for GeoJSON
    files >10 MB when a CUDA device is available, before falling back to
    the pyogrio GPU WKB path
  - the stream tokenizer and structural feature-span tokenizer remain available
    as explicit strategies
  - assemble geometry directly into owned buffers without Shapely objects
  - keep property rows separate from geometry assembly
- Repo-owned Shapefile ingest now also has an internal batch-first owned path:
  - `read_shapefile_owned(...)` uses `pyogrio.read_arrow(...)` for host container
    parsing of `.shp/.shx/.dbf`
  - geometry lands through Arrow `geoarrow.wkb` batches into owned buffers via
    the repo-owned native WKB decoder
  - homogeneous point Shapefiles now use a raw Arrow-binary point fast path
    before any `to_pylist()` materialization
  - attributes stay in a columnar Arrow table instead of materializing a
    GeoDataFrame during ingest
  - the public `read_file(..., driver=\"ESRI Shapefile\")` route stays on
    `pyogrio` until the owned batch path is measurably faster end-to-end

## Measured Local Baseline

On this machine the `fast-json` strategy is the clear GeoJSON ingest winner.
It uses `orjson` for parsing and vectorized per-family coordinate extraction
directly into numpy arrays, eliminating the old per-feature assembly loop.

- point-heavy GeoJSON at `100K` rows:
  - `pyogrio`: about `300K` rows/s
  - **`fast-json` (orjson + vectorized)**: about `1,141K` rows/s
  - full `json.loads` plus native assembly (old default): about `451K` rows/s
  - `pylibcudf` GPU tokenizer-native: about `542K` rows/s
  - staged stream-native: about `331K` rows/s
  - structural tokenizer-native: about `126K` rows/s
- point-heavy GeoJSON at `1M` rows:
  - `pyogrio`: about `290K` rows/s
  - **`fast-json` (orjson + vectorized)**: about `1,041K` rows/s
  - full `json.loads` plus native assembly: about `439K` rows/s
- polygon-heavy GeoJSON at `20K` rows:
  - `pyogrio`: about `161K` rows/s
  - **`fast-json` (orjson + vectorized)**: about `745K` rows/s
  - full `json.loads` plus native assembly: about `282K` rows/s
  - `pylibcudf` GPU tokenizer-native: about `307K` rows/s

The `fast-json` path achieves `3.5-3.9x` speedup over `pyogrio` and
`2.4-2.6x` over the old `full-json` default. The `pylibcudf` GPU path is now
slower than `fast-json` because host-side span discovery and `PyArrow` string
column construction dominated GPU compute savings. The remaining bottleneck is
`orjson.loads()` itself, which for future work could be addressed by:
- CCCL-backed byte classification and span planning on GPU
- `simdjson` integration for ~2-4x faster host parsing
- direct-to-device coordinate decode bypassing Python objects entirely

Current Shapefile numbers on this machine now clear the published ingest floors:

- point-heavy Shapefile at `10K` rows:
  - `pyogrio.read_dataframe`: about `1.18M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.97M` rows/s
  - repo-owned native WKB decode only: about `1.65M` rows/s
  - full owned batch ingest: about `925K` rows/s
- point-heavy Shapefile at `100K` rows after the raw Arrow-binary point fast path:
  - `pyogrio.read_dataframe`: about `1.25M` rows/s
  - `pyogrio.read_arrow` container parse: about `4.41M` rows/s
  - repo-owned native WKB decode only: about `1.56M` rows/s
  - full owned batch ingest: about `4.10M` rows/s
- point-heavy Shapefile at `1M` rows:
  - `pyogrio.read_dataframe`: about `1.12M` rows/s
  - `pyogrio.read_arrow` container parse: about `4.48M` rows/s
  - repo-owned native WKB decode only: about `1.43M` rows/s
  - full owned batch ingest: about `4.08M` rows/s
- line-heavy Shapefile at `10K` rows:
  - `pyogrio.read_dataframe`: about `1.10M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.32M` rows/s
  - repo-owned native WKB decode only: about `611K` rows/s
  - full owned batch ingest: about `3.20M` rows/s
- line-heavy Shapefile at `1M` rows:
  - `pyogrio.read_dataframe`: about `1.02M` rows/s
  - `pyogrio.read_arrow` container parse: about `3.73M` rows/s
  - repo-owned native WKB decode only: about `617K` rows/s
  - full owned batch ingest: about `3.40M` rows/s
- polygon-heavy Shapefile at `5K` rows:
  - `pyogrio.read_dataframe`: about `858K` rows/s
  - `pyogrio.read_arrow` container parse: about `2.37M` rows/s
  - repo-owned native WKB decode only: about `502K` rows/s
  - full owned batch ingest: about `2.29M` rows/s
- polygon-heavy Shapefile at `250K` rows:
  - `pyogrio.read_dataframe`: about `893K` rows/s
  - `pyogrio.read_arrow` container parse: about `2.51M` rows/s
  - repo-owned native WKB decode only: about `452K` rows/s
  - full owned batch ingest: about `2.24M` rows/s

The main change was shifting non-point families off the generic per-row Arrow
WKB bridge and onto uniform raw-buffer fast paths. That turned the owned path
from "points only" into a broad Shapefile ingest win:

- point-heavy ingest now runs about `3.63x` faster than the current host
  baseline at `1M` rows
- line-heavy ingest now runs about `3.32x` faster than the current host
  baseline at `1M` rows
- polygon-heavy ingest now runs about `2.51x` faster than the current host
  baseline at `250K` rows

The GPU byte-classification path (ADR-0038) now handles geometry extraction in
**1.8s** for the 2.16 GB Florida.geojson benchmark (32x vs pyogrio). The
remaining bottleneck is CPU property extraction at **9.2s** — 7.2M
`orjson.loads()` calls dominated by Python interpreter overhead (function call
dispatch, dict construction), not JSON parsing throughput. POC evaluation of
parallel orjson, multiprocessing, pylibcudf, and coordinate stripping showed
no meaningful improvement (see ADR-0038 Consequences). The next acceleration
step would be a native (Rust/C) columnar property extractor, which is deferred
because it changes the pure-Python build story and the cost is already lazy.
