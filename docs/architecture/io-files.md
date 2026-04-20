# File Format IO

<!-- DOC_HEADER:START
Scope: File-based vector format routing for GeoJSON, Shapefile, and legacy GDAL adapters.
Read If: You are changing read_file, to_file, GeoJSON ingest, Shapefile ingest, or file-format routing.
STOP IF: Your task already has the specific format adapter open and only needs local implementation detail.
Source Of Truth: File-format IO architecture for GeoJSON, Shapefile, and GDAL legacy adapters.
Body Budget: 280/280 lines
Document: docs/architecture/io-files.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-16 | Request Signals |
| 17-22 | Open First |
| 23-28 | Verify |
| 29-34 | Risks |
| 35-55 | Decision |
| 56-64 | Performance Notes |
| 65-195 | Current Behavior |
| 196-280 | Measured Local Baseline |
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
- GeoJSON geometry ingest is now GPU-accelerated (ADR-0038); remaining hybrid seam is staged host property decode.
- Shapefile adapter losing speed leadership if the raw Arrow-binary fast path regresses.

## Decision
- GeoJSON is a first-class hybrid path. Unfiltered public reads auto-route to
  the repo-owned GPU byte-classify path whenever a GPU runtime is available
  because `read + downstream GPU consumer` is the planning objective for
  `read_file`; filtered reads still use pyogrio.
- Shapefile is a first-class hybrid path. Eligible public reads prefer the
  repo-owned native plan: direct SHP GPU decode first, Arrow/WKB fallback
  second.
- Promoted pyogrio-backed vector containers such as GeoPackage, FileGDB, GML,
  GPX, TopoJSON, GeoJSON-Seq, and FlatGeobuf stay hybrid rather than being
  treated as canonical GPU-native formats.
- Untargeted legacy GDAL vector formats stay behind an explicit fallback
  adapter.
- Public `geopandas.read_file` and `GeoDataFrame.to_file` should dispatch
  through repo-owned wrappers so the chosen path is observable.
- On the `pyogrio` write path, terminal export should prefer the shared
  native tabular Arrow boundary over rebuilding a GeoDataFrame-shaped host
  export. Public device-backed GeoJSON, Shapefile, GeoPackage, and FlatGeobuf
  writes may use that sink when request semantics match pyogrio exactly; CPU,
  append, and legacy metadata cases remain explicit compatibility. Fiona
  remains a host boundary.
## Performance Notes
- GeoJSON public `read_file(...)` now prefers pipeline-optimal routing over a
  coarse file-size heuristic. The 10k bar is parity-or-better on the public
  `read_file + first GPU stage` path, not isolated parser throughput.
- `fast-json` remains the measured standalone GeoJSON parser winner, so its
  benchmark rails still act as the host baseline.
- Legacy GDAL formats should not masquerade as native; explicit fallback
  events expose work that still bypasses the GPU-oriented stack.

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
- Repo-owned `to_file(..., engine="pyogrio")` writes through
  `pyogrio.write_arrow(...)` from the shared native tabular boundary whenever
  a native result is available. Public device-backed GeoJSON, Shapefile,
  GeoPackage, and FlatGeobuf writes now use the same native Arrow/WKB sink and
  force device WKB encode so small exports do not fall through the generic host
  encoder threshold.
- Public CPU-backed `GeoDataFrame.to_file(..., engine="pyogrio")` stays on
  `pyogrio.write_dataframe(...)` for compatibility-sensitive cases such as
  append mode, timezone-preserving datetime fields, unsupported metadata
  combinations, and other legacy driver semantics.
- Repo-owned `read_file` GPU branches that can already produce owned geometry
  plus columnar attributes now lower through the shared `NativeTabularResult`
  boundary before terminal `GeoDataFrame` materialization. That now includes
  the pyogrio Arrow + native WKB bridge, direct WKT/CSV/KML readers, both
  Shapefile GPU paths, the GeoJSON byte-classify path after property
  extraction, and the OSM PBF hybrid path after protobuf/tag extraction.
- The OSM PBF public boundary now uses a bounded, lossless tag projection instead of widening every observed tag key into its own eager object column. Common OSM keys stay first-class and the remainder stay in `other_tags`, avoiding the previous Florida-scale `2843`-column host explosion.
- Public OSM standard layers (`points`, `lines`, `multilinestrings`, `multipolygons`, `other_relations`) now prefer `pyogrio` container reads through the shared native boundary. Those supported-layer scans run in parallel for the default public `read_file("*.osm.pbf")` path, so the user-facing wall time is no longer dominated by five serial OSM driver passes. Layers with native-supported geometry stay on Arrow + GPU WKB; `other_relations` skips the unsupported owned WKB decode and uses an explicit compatibility bridge because real PBF data still carries `GeometryCollection`. The repo-owned hybrid OSM parser remains the path for `layer="all"` and the full-data native contract.
- Default public `read_file("*.osm.pbf")` now combines those supported public layers by default instead of forcing the full mixed all-data parser into one eager frame. Small node-only fixtures and other empty supported-layer cases explicitly fall back to the full native parser so data is not lost.
- OSM PBF native reads now keep tag projection lazy until explicit export. The low-level reader still returns host-resident tag dicts today, but the shared native file boundary no longer eagerly rebuilds a giant pandas object table.
- Full-data OSM native reads now normalize through an internal partitioned bundle before any public layer projection or `GeoDataFrame` assembly. That keeps parser-shaped node/way/relation output out of the public boundary while preserving a reusable full-data seam for future views.
- Large geometry-column CSV now prefers a `pylibcudf` / `libcudf` table parse
  before native GPU WKT/WKB decode. Public `GeoSeries.from_wkt(...)` also uses
  the GPU WKT parser for large clean arrays, so the common `pd.read_csv` plus
  WKT constructor idiom no longer stays pure Shapely at Florida scale.
- FlatGeobuf now defaults to the repo-owned direct FlatBuffer GPU decoder for
  eligible local unfiltered public `read_file(...)` / `read_vector_file_native(...)`
  calls and uses a typed dense-property extractor for common numeric plus
  repeated-string schemas. Explicit `engine="pyogrio"` and container-shaped
  requests stay on the shared Arrow + GPU WKB native boundary.
- GeoJSONSeq now routes eligible local unfiltered public reads through the GPU
  GeoJSON parser by rewriting newline-delimited feature records into a
  FeatureCollection byte payload. Filtered or explicit pyogrio-shaped requests
  stay on the shared Arrow + GPU WKB native boundary.
- GeoJSON remains an explicit hybrid compatibility boundary because property
  extraction is still host-side even though geometry decode is GPU-native.
  Public `read_file(...)` now tries that repo-owned GPU path for all eligible
  unfiltered GeoJSON reads, including explicit `engine="pyogrio"` when the
  request shape stays native-compatible. If the GPU parser fails, the public
  boundary falls back to repo-owned `fast-json` before reaching vendored
  pyogrio. The shared native read boundary preserves properties lazily, accepts
  both filesystem and in-memory RFC 7946 sources, and treats
  `track_properties=False` as an explicit geometry-only contract.
- Shapefile remains an explicit hybrid compatibility boundary because the container and DBF
  attribute story are still legacy-oriented even when geometry decode is native. Public automatic
  reads prefer direct SHP, while explicit `engine="pyogrio"` reads stay on the Arrow/WKB bridge
  for pyogrio-shaped requests. Untargeted legacy GDAL formats stay explicit compatibility.
- Public GeoPackage reads and the promoted pyogrio-backed vector-container
  family now keep `mask` and safe `layer` filters on the shared native Arrow/WKB boundary whenever
  the request stays native-compatible; invalid `bbox` plus `mask` fails before dispatch accounting.
  The public boundary asks pyogrio for datetime strings so naive datetime fields and timezone-aware
  roundtrips survive without forced UTC Arrow timestamps. Unsupported public geometry families such
  as `Point Z` and `Unknown` still route through explicit compatibility.
- Repo-owned GeoJSON ingest now also has an internal staged owned path:
  - `auto` now has two explicit objectives:
    - `pipeline` prefers `gpu-byte-classify` when a GPU runtime is available so
      the first downstream GPU consumer does not pay an immediate promotion
    - `standalone` prefers `fast-json`, which is still the measured isolated
      ingest winner on this machine
    On CPU-only hosts both objectives fall back to `fast-json`: `orjson` (when
    available, otherwise CPython `json`) plus vectorized per-family coordinate
    extraction into numpy owned buffers. That path is 2.4-2.6x faster than the
    previous `full-json` default and 3.5-3.9x faster than `pyogrio`.
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
    (per io-acceleration.md policy). The GPU path now also captures
    `$.properties` object spans while structural state is still on device, so
    the host only decodes the small property-object payloads instead of
    reparsing full feature JSON.
    Geometry parse: **1.8s** for 2.16 GB / 7.2M polygons (32x vs pyogrio).
    Total read including properties: **7.1s** (55.2s GeoPandas -> 7.1s
    vibeSpatial GPU on the latest local Florida run).
    File-to-device transfer uses kvikio when installed (parallel POSIX reads
    with pinned bounce buffers, no GDS required), falling back to
    `cp.asarray` otherwise. Thread count is tunable via `KVIKIO_NTHREADS`.
  - the public `read_file` GeoJSON path now auto-selects `gpu-byte-classify`
    for eligible unfiltered reads whenever a CUDA device is available because
    the planning objective is end-to-end pipeline shape rather than naked file
    parse speed
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
On this machine `fast-json` is still the clear standalone GeoJSON ingest
winner. The public-path KPI is different: GeoJSON is measured on
`read_file(...) + first downstream GPU stage`, with `10k` as the minimum
acceptance scale, because that determines whether we read on CPU and
immediately pay a promotion tax.

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
construction dominated GPU compute savings. The remaining bottleneck is
`orjson.loads()` itself, which future work could address by:
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
**1.8s** for the 2.16 GB Florida.geojson benchmark, and the staged
property-object decode keeps the full public Florida read near **6.5s**. The
next acceleration step would be a native columnar property extractor, deferred
because it changes the pure-Python build story.
