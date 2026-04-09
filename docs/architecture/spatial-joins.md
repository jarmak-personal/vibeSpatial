# Spatial Query And Join Assembly

<!-- DOC_HEADER:START
Scope: Spatial-index query assembly, sjoin result semantics, bounded nearest strategy, and output-format policy.
Read If: You are changing sindex query behavior, spatial join materialization, dwithin, or nearest result assembly.
STOP IF: You already have the spatial-query engine and vendored join helpers open and only need local implementation detail.
Source Of Truth: Phase-4 spatial query and join assembly policy before broader API dispatch work.
Body Budget: 149/220 lines
Document: docs/architecture/spatial-joins.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Request Signals |
| 14-21 | Open First |
| 22-30 | Verify |
| 31-36 | Risks |
| 37-42 | Intent |
| 43-71 | Decision |
| 72-111 | Query Strategy |
| 112-129 | Nearest Strategy |
| 130-141 | Pandas Semantics |
| 142-149 | Consequences |
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
- src/vibespatial/spatial/query.py
- src/vibespatial/api/sindex.py
- src/vibespatial/api/tools/sjoin.py
- docs/architecture/binary-predicates.md

## Verify

- `uv run pytest tests/test_spatial_query.py`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "predicate or nearest"`
- `uv run vsbench run spatial-query --rows 20000 --arg overlap_ratio=0.2`
- `uv run python scripts/profile_fixture_spatial_query.py --fixture polygons-regular-grid-rows100000 --operation query --ensure`
- `uv run python scripts/check_docs.py --check`

## Risks

- Query/index speedups are meaningless if join assembly breaks index names, suffixes, or geometry-column retention.
- A nearest implementation without a bounded-search fast path will not scale.
- Falling back to Shapely/STRtree for unsupported inputs must stay explicit enough that future GPU work is not hidden.

## Intent

Define how repo-owned spatial query primitives plug into GeoPandas-compatible
`sindex` and `sjoin` surfaces while preserving pandas-compatible output rules
without making pandas assembly the internal execution model.

## Decision

Land a repo-owned spatial-query engine and make relation results the internal
join boundary, with explicit GeoPandas export only at the public surface.

This means:

- the repo-owned spatial-query engine defines how future GPU-friendly query,
  `dwithin`, and bounded-nearest work should be staged
- regular-grid, axis-aligned rectangle polygon indexes may register an
  explicit GPU point-query specialization instead of paying the generic
  bbox-tiling cost
- vendored `SpatialIndex` stays on STRtree by default for current host
  execution because the owned host path is not yet performance-competitive
- `sjoin` and `sjoin_nearest` build low-level `RelationJoinResult` objects
  first, then wrap them in a deferred export result that owns join context
  until the explicit `GeoDataFrame` boundary so suffixes, index restoration,
  and geometry-column rules stay GeoPandas-compatible
- `RelationJoinExportResult -> NativeTabularResult` now builds a native
  attribute payload directly, so terminal Arrow-family and file sinks do not
  need to go back through the joined-frame materializer just to emit join rows
- those deferred join exports can also lower into the shared
  `NativeTabularResult` boundary so terminal writers do not need to rebuild a
  public frame just to emit join results
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

Nearest uses a three-tier dispatch for point-point data:

1. **Zero-copy GPU grid nearest** — extracts coords directly from Shapely
   arrays (bypassing `_to_owned`), builds a uniform-grid spatial hash on
   device, and runs ring-expansion search entirely on the GPU.  Handles both
   bounded and unbounded nearest.  Falls through for non-point inputs or
   degenerate grids.
2. **Indexed GPU nearest** — uses `_to_owned` arrays with a sorted-X sweep
   and CCCL lower/upper bound for tie counting.  Fallback when the grid
   path declines.
3. **STRtree host path** — used for non-point or mixed-geometry inputs, or
   when the GPU is unavailable.

The bounded path (`max_distance` set) can also use distance-expanded bounds to
prune candidates, then exact distance reduction per input geometry.

## Pandas Semantics

GeoPandas-compatible result assembly is an explicit export step from native
relation results:

- relation pairs stay separate from join/export context until the boundary
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
- downstream native workflows can consume join relation results without paying
  immediate pandas materialization
