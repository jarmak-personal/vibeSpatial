# File Format IO

<!-- DOC_HEADER:START
Scope: File-based vector format routing for GeoJSON, Shapefile, and legacy GDAL adapters.
Read If: You are changing read_file, to_file, GeoJSON ingest, Shapefile ingest, or file-format routing.
STOP IF: Your task already has the specific format adapter open and only needs local implementation detail.
Source Of Truth: File-format IO architecture for GeoJSON, Shapefile, and GDAL legacy adapters.
Body Budget: 275/280 lines
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
| 36-54 | Decision |
| 55-64 | Performance Notes |
| 65-185 | Current Behavior |
| 186-275 | Measured Local Baseline |
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
- Promoted pyogrio-backed vector containers such as GeoPackage, FileGDB, GML,
  GPX, TopoJSON, GeoJSON-Seq, and FlatGeobuf stay hybrid rather than being
  treated as canonical GPU-native formats.
- Untargeted legacy GDAL vector formats stay behind an explicit fallback
  adapter.
- Public `geopandas.read_file` and `GeoDataFrame.to_file` should dispatch
  through repo-owned wrappers so the chosen path is observable.
- On the `pyogrio` write path, terminal export should prefer the shared
  native tabular Arrow boundary over rebuilding a GeoDataFrame-shaped host
  export when the input is already a native result. Public
  `GeoDataFrame.to_file(...)` remains on the compatibility writer unless the
  repo-owned native sink can match GeoPandas file semantics exactly. Fiona
  remains an explicit host compatibility boundary.

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
- `read_vector_file_native(...)` is now the shared native file-read surface for
  promoted vector formats. It returns a `NativeTabularResult` at the read
  boundary and lets public `read_file(...)` materialize a `GeoDataFrame` only
  at the explicit compatibility/export step.
- The repo-owned file router already makes the promoted read-boundary
  classification explicit through `plan_vector_file_io(...).selected_path`:
  - `hybrid`: GeoJSON, Shapefile, WKT, CSV, KML, OSM PBF, GeoPackage,
    FileGDB, FlatGeobuf, GML, GPX, TopoJSON, GeoJSON-Seq
  - `fallback`: untargeted legacy GDAL formats
- `GeoDataFrame.to_file` uses the same routing policy.
- Repo-owned `to_file(..., engine="pyogrio")` now writes through
  `pyogrio.write_arrow(...)` from the shared native tabular boundary whenever
  a native result is available, instead of rebuilding a host GeoDataFrame as
  the execution model for terminal export.
- Public `GeoDataFrame.to_file(..., engine="pyogrio")` stays on
  `pyogrio.write_dataframe(...)` for compatibility-sensitive cases such as
  append mode, timezone-preserving datetime fields, unsupported metadata
  combinations, and other legacy driver semantics that the native Arrow sink
  does not yet match exactly.
- Repo-owned `read_file` GPU branches that can already produce owned geometry
  plus columnar attributes now lower through the shared `NativeTabularResult`
  boundary before terminal `GeoDataFrame` materialization. That now includes
  the pyogrio Arrow + GPU WKB compatibility path, direct WKT/CSV/KML readers,
  both Shapefile GPU paths, the GeoJSON byte-classify path after property
  extraction, and the OSM PBF hybrid path after protobuf/tag extraction.
- The OSM PBF public boundary now uses a bounded, lossless tag projection instead of widening every observed tag key into its own eager object column. Common OSM keys stay first-class and the remainder stay in `other_tags`, avoiding the previous Florida-scale `2843`-column host explosion.
- Public OSM standard layers (`points`, `lines`, `multilinestrings`, `multipolygons`, `other_relations`) now prefer `pyogrio` container reads through the shared native boundary. Those supported-layer scans run in parallel for the default public `read_file("*.osm.pbf")` path, so the user-facing wall time is no longer dominated by five serial OSM driver passes. Layers with native-supported geometry stay on Arrow + GPU WKB; `other_relations` uses an explicit compatibility bridge because real PBF data still carries `GeometryCollection`. The repo-owned hybrid OSM parser remains the path for `layer="all"` and the full-data native contract.
- Default public `read_file("*.osm.pbf")` now combines those supported public layers by default instead of forcing the full mixed all-data parser into one eager frame. Small node-only fixtures and other empty supported-layer cases explicitly fall back to the full native parser so data is not lost.
- OSM PBF native reads now keep tag projection lazy until explicit export. The low-level reader still returns host-resident tag dicts today, but the shared native file boundary no longer eagerly rebuilds a giant pandas object table.
- Full-data OSM native reads now normalize through an internal partitioned bundle before any public layer projection or `GeoDataFrame` assembly. That keeps parser-shaped node/way/relation output out of the public boundary while preserving a reusable full-data seam for future views.
- Large geometry-column CSV now prefers a `pylibcudf` / `libcudf` table parse
  instead of the older whole-file byte-classify path. That keeps the CSV
  container parse on device, avoids the Florida-scale WKT concatenation memory
  blowup, and then routes the geometry column into the native GPU WKT/WKB
  decode path. The byte-classify reader still owns small files and lat/lon
  layouts.
- FlatGeobuf now defaults to the pyogrio Arrow + GPU WKB path on public
  `read_file(...)` / `read_vector_file_native(...)`. The repo still has a
  direct GPU FlatBuffer decoder in `fgb_gpu.py`, but the Florida shootout
  shows the Arrow path is materially faster today, so the direct route is not
  the default execution shape.
- Direct format-native helpers now exist for the promoted compatibility readers that already have a credible shared boundary: `read_geojson_native(...)` and `read_shapefile_native(...)`.
- GeoJSON remains an explicit hybrid compatibility boundary because property
  extraction is still CPU-side even though geometry decode is GPU-native.
  The shared native read boundary now preserves GeoJSON properties lazily
  until explicit attribute access or terminal public export, so geometry-only
  native consumers do not pay the property parse cost up front.
  The GPU byte-classify path now accepts both filesystem paths and in-memory
  RFC 7946 text/bytes sources, avoiding synthetic write-then-read loops when
  the payload is already resident on host.
  Explicit `track_properties=False` reads now drop property retention
  completely; accessing properties after a geometry-only read is an explicit
  contract error and requires a re-read with `track_properties=True`.
- Shapefile remains an explicit hybrid compatibility boundary because the
  container and DBF attribute story are still legacy-oriented even when
  geometry decode is native; untargeted legacy GDAL formats stay outside the
  GPU-native promise and route through explicit fallback compatibility adapters.
- Repo-owned GeoJSON ingest now also has an internal staged owned path:
  - `auto` prefers `gpu-byte-classify` when a GPU runtime is available,
    producing device-resident geometry via NVRTC kernels. On CPU-only hosts,
    `auto` falls back to `fast-json`: `orjson` for parsing (when available,
    otherwise CPython `json`) plus vectorized per-family coordinate extraction
    directly into numpy owned buffers. The `fast-json` path is 2.4-2.6x
    faster than the previous `full-json` default, and 3.5-3.9x faster than
    `pyogrio`.
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
- Repo-owned Shapefile ingest now also has an internal native-first owned path:
  - `read_shapefile_owned(...)` now prefers direct SHP binary decode on GPU plus
    the GPU DBF parser for plain local-file reads with no bbox, column
    projection, row window, or pyogrio-specific kwargs
  - when the request needs those pyogrio container features, it falls back to
    `pyogrio.read_arrow(...)` plus the repo-owned native WKB decoder
  - attributes stay columnar through the read boundary instead of materializing
    a GeoDataFrame during ingest
  - the direct SHP path and the Arrow-WKB fallback now share the same public
    owned/native read boundary instead of reader-local frame assembly

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
