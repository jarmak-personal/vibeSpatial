# Spatial Query And Join Assembly

<!-- DOC_HEADER:START
Scope: Spatial-index query assembly, sjoin result semantics, bounded nearest strategy, and output-format policy.
Read If: You are changing sindex query behavior, spatial join materialization, dwithin, or nearest result assembly.
STOP IF: You already have the spatial-query engine and vendored join helpers open and only need local implementation detail.
Source Of Truth: Phase-4 spatial query and join assembly policy before broader API dispatch work.
Body Budget: 129/220 lines
Document: docs/architecture/spatial-joins.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Request Signals |
| 14-21 | Open First |
| 22-30 | Verify |
| 31-36 | Risks |
| 37-41 | Intent |
| 42-62 | Decision |
| 63-102 | Query Strategy |
| 103-113 | Nearest Strategy |
| 114-123 | Pandas Semantics |
| 124-129 | Consequences |
DOC_HEADER:END -->

## Request Signals

- sindex query
- sjoin
- dwithin
- nearest
- pandas semantics
- join assembly
- spatial index
- cccl

## Open First

- docs/architecture/spatial-joins.md
- src/vibespatial/spatial_query.py
- src/vibespatial/_vendor/geopandas/sindex.py
- src/vibespatial/_vendor/geopandas/tools/sjoin.py
- docs/architecture/binary-predicates.md

## Verify

- `uv run pytest tests/test_spatial_query.py`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "predicate or nearest"`
- `uv run python scripts/benchmark_spatial_query.py --rows 20000 --overlap-ratio 0.2`
- `uv run python scripts/profile_fixture_spatial_query.py --fixture polygons-regular-grid-rows100000 --operation query --ensure`
- `uv run python scripts/check_docs.py --check`

## Risks

- Query/index speedups are meaningless if join assembly breaks index names, suffixes, or geometry-column retention.
- A nearest implementation without a bounded-search fast path will not scale.
- Falling back to Shapely/STRtree for unsupported inputs must stay explicit enough that future GPU work is not hidden.

## Intent

Define how repo-owned spatial query primitives plug into vendored GeoPandas
`sindex` and `sjoin` surfaces while preserving pandas-compatible output rules.

## Decision

Land a repo-owned spatial-query engine and keep the vendored DataFrame-level
join assembly.

This means:

- the repo-owned spatial-query engine defines how future GPU-friendly query,
  `dwithin`, and bounded-nearest work should be staged
- regular-grid, axis-aligned rectangle polygon indexes may register an
  explicit GPU point-query specialization instead of paying the generic
  bbox-tiling cost
- vendored `SpatialIndex` stays on STRtree by default for current host
  execution because the owned host path is not yet performance-competitive
- `sjoin` continues to rely on vendored `_frame_join(...)` so suffixes, index
  restoration, and geometry-column rules stay GeoPandas-compatible
- `dwithin` uses distance-expanded bounds as the coarse pass, then exact
  `shapely.dwithin` on the compacted candidate set
- bounded nearest uses exact-distance reduction over compacted candidates
- unsupported geometry families fall back to the original Shapely/STRtree path

## Query Strategy

The owned engine provides:

- build and cache a flat spatial index over owned geometry buffers
- allow the generic flat-index order build to use GPU morton-key sort when the
  runtime selection explicitly requests GPU execution
- use bbox candidate generation for scalar and bulk query
- refine candidate rows with the exact binary-predicate engine
- support `indices`, `dense`, and `sparse` output formats

For benchmark-style joins where the tree is a regular grid of non-overlapping
rectangles and the query side is points, the engine may skip generic candidate
generation entirely:

- detect the regular-grid rectangle layout while building the flat index
- launch a point-to-cell GPU kernel that emits up to four polygon hits per
  point for edge and corner cases
- return polygon row ids directly because point-vs-rectangle `intersects`
  semantics match the cell lookup exactly
- compact duplicate polygon ids before any downstream dissolve/materialization
  work so repeated point hits do not dominate CPU join assembly

The planned integration seam makes this query
stack prefer index-driven coarse filtering where possible and routes
point-vs-polygon refinement through the dedicated point predicate pipeline
instead of always using the generic binary-predicate path.

This maps cleanly onto a future GPU/CCCL path:

- bounds compare: transform-style primitive
- candidate compaction: `DeviceSelect`
- exact predicate refine on compacted rows
- scatter back into requested output representation

Profile public-path query changes against cached GeoParquet fixtures in
`.benchmark_fixtures/` before trusting synthetic-only kernel rails. The fixture
workflow is the source of truth for host/device traversal because it exercises
`read_parquet -> sindex -> query` through the public GeoPandas surface.

## Nearest Strategy

Nearest in the owned engine has two modes:

- unbounded nearest: keep the existing STRtree nearest path for now
- bounded nearest (`max_distance` set): use distance-expanded bounds to prune,
  then exact distance reduction per input geometry

The bounded path is the GPU-friendly design center because it can be expressed
as candidate generation plus a reduce-by-key minimum over candidate distances.

## Pandas Semantics

GeoPandas-compatible result assembly remains in the vendored layer:

- `how` semantics for `left`, `right`, and `inner`
- index restoration and unnamed-index handling
- overlapping-column suffix policy
- geometry-column retention rules
- optional distance-column insertion for nearest joins

## Consequences

- `o17.4.3` now has one owned spatial-query engine that future GPU work can target
- targeted upstream `sindex` and `sjoin` families validate the result semantics
- the default host adapter avoids a measured regression by staying on STRtree for now
- future GPU work can replace query/nearest internals without re-implementing DataFrame join behavior
