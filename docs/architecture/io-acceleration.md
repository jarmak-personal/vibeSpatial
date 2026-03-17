<!-- DOC_HEADER:START
Scope: Post-Phase-6b GPU-native IO execution model, staged decode policy, and format-level speed targets.
Read If: You are changing GeoArrow, GeoParquet, WKB, GeoJSON, or Shapefile performance strategy or decode architecture.
STOP IF: Your task already has the routed IO implementation files open and only needs local adapter detail.
Source Of Truth: IO acceleration policy for turning repo-owned adapters into GPU-dominant ingest and emission paths.
Body Budget: 144/260 lines
Document: docs/architecture/io-acceleration.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Purpose |
| 7-11 | Intent |
| 12-20 | Request Signals |
| 21-28 | Open First |
| 29-34 | Verify |
| 35-41 | Risks |
| 42-60 | Decision |
| 61-74 | Execution Model |
| 75-102 | Format Strategy |
| 103-115 | CCCL Preference Order |
| 116-134 | Performance Targets |
| 135-144 | Non-Negotiable Constraints |
DOC_HEADER:END -->

## Purpose

Define the post-Phase-6b IO acceleration program so GeoArrow, GeoParquet, WKB,
GeoJSON, and Shapefile all converge on one GPU-first execution model instead of
growing as unrelated adapters.

## Intent

Turn repo-owned IO support into GPU-dominant ingest and emission paths with one
shared decode architecture and explicit format-level floor targets.

## Request Signals

- io acceleration
- geoparquet performance
- geoarrow decode
- wkb decode
- geojson ingest
- shapefile ingest

## Open First

- docs/architecture/io-acceleration.md
- docs/architecture/io-arrow.md
- docs/architecture/io-files.md
- src/vibespatial/io_arrow.py
- src/vibespatial/io_file.py

## Verify

- `uv run pytest tests/test_decision_log.py`
- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "gpu native io acceleration roadmap"`

## Risks

- Treating every format as bespoke work will fragment the fast path and dilute GPU effort.
- Decoding before pruning will erase most of the potential GeoParquet win.
- A generic mixed-family decoder will drag homogeneous fast paths down to the slow case.
- Text and legacy container support can quietly reintroduce per-row Python work if not measured.

## Decision

- Owned geometry buffers remain the only canonical in-memory destination.
- IO planning is metadata-first: prune row groups, pages, or feature batches
  before full geometry decode whenever the source format allows it.
- Geometry decode is family-specialized:
  - point and multipoint
  - linestring and multilinestring
  - polygon and multipolygon
- Truly mixed inputs should scan tags first, then partition into family-local
  decode batches instead of using one generic mixed decoder.
- GeoArrow and GeoParquet are the primary GPU-native paths.
- WKB is the primary compatibility bridge and should still be GPU-native on the
  decode and encode steps.
- GeoJSON and Shapefile remain hybrid, but must be batch-oriented and must not
  materialize Shapely objects during normal ingest or emission.
- CCCL primitives are the default building blocks for scans, compaction,
  partitioning, prefix sums, scatters, run-length encoding, and reductions.

## Execution Model

Every format should map onto the same staged pipeline:

1. source read
2. structural scan or metadata planning
3. row-group, page, or feature-batch pruning
4. family tagging and optional partition
5. output-size scan
6. family-specialized decode into owned buffers
7. optional lazy materialization of properties or host objects

The critical rule is that decode happens after pruning, not before it.

## Format Strategy

### GeoArrow and GeoParquet

- Prefer zero-copy or single-copy buffer adoption when offsets, validity, and
  coordinate buffers already match the owned schema.
- Push bbox and covering filters into row-group or page planning before decode.
- Decode only surviving rows into owned buffers.

### WKB

- Treat WKB as a byte-stream compatibility bridge.
- Use GPU header scans, size scans, and family partitions before decode.
- Compact unsupported or ambiguous rows into an explicit fallback pool.

### GeoJSON

- Separate text tokenization from geometry assembly.
- Keep property columns and geometry assembly on independent tracks so geometry
  can become GPU-native even while some attribute handling remains hybrid.

### Shapefile

- Keep container parsing explicit on host.
- Batch geometry record decode and attribute assembly.
- Land decoded geometry directly in owned buffers without per-feature Python
  object construction.

## CCCL Preference Order

Reach for these before custom raw kernels:

- `cub::DeviceScan` for offsets and output sizing
- `cub::DeviceSelect` and `cub::DevicePartition` for survivor and family pools
- `cub::DeviceRadixSort` for key-grouped ordering
- `cub::DeviceRunLengthEncode` for tag ranges
- `cub::DeviceReduce` and segmented reductions for planning summaries

Custom kernels should be reserved for the actual geometry decode, encode, and
format-specific math after the data has already been laid out by CCCL passes.

## Performance Targets

These are the floor targets for supported NVIDIA GPU environments. All targets
are end-to-end relative to the current repo-owned host path or the dominant
host baseline for the same format, whichever is faster.

| Format / Path | Floor Target | Aspirational Target | Reference Scale |
|---|---:|---:|---|
| GeoArrow aligned import or export | `5x` faster | `10x` faster | `10M` points / `1M` polygons |
| GeoParquet unfiltered native scan | `3x` faster | `5x` faster | `10M` points / `1M` polygons |
| GeoParquet selective scan with bbox pushdown | decode `<= 15%` of rows at `< 10%` selectivity | decode `<= 5%` | row-group dataset with covering metadata |
| GeoArrow native decode or encode | `4x` faster | `8x` faster | `10M` points / `1M` polygons |
| WKB decode | `4x` faster | `8x` faster | `10M` points / `1M` polygons |
| WKB encode | `3x` faster | `5x` faster | `10M` points / `1M` polygons |
| GeoJSON point or line ingest | `2x` faster | `4x` faster | `1M` features |
| GeoJSON polygon ingest | `1.25x` faster | `2x` faster | `250K` polygons |
| Shapefile point or line ingest | `1.5x` faster | `3x` faster | `1M` records |
| Shapefile polygon ingest | `1.1x` faster | `2x` faster | `250K` polygons |

## Non-Negotiable Constraints

- No silent Shapely materialization in fast paths.
- No per-row Python decode loops in supported formats.
- No host-side full decode before a metadata or bbox prune step when the source
  format exposes enough planning information to avoid it.
- Mixed-family support must not force the homogeneous fast paths onto a generic
  decoder.
- Out-of-core and chunked execution must compose with `o17.2.9` and
  `o17.6.10`, not bypass them.
