# Native Physical Shape Ledger

<!-- DOC_HEADER:START
Scope: Physical workload shape ledger for native 100ms canaries and hot stage classification.
Read If: You are selecting a native 100ms canary, mapping hot shootout stages to Native* carriers, or checking whether a performance change improves physical shape.
STOP IF: You only need the high-level 100ms plan or one operation-local kernel detail.
Source Of Truth: Working ledger for native physical workload shapes, canaries, and export boundaries.
Body Budget: 180/180 lines
Document: docs/dev/native-physical-shape-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-22 | Request Signals |
| 23-36 | Open First |
| 37-45 | Verify |
| 46-56 | Risks |
| 57-69 | Ledger |
| 70-74 | Canary Maintenance |
| 75-180 | PRD Execution Notes |
DOC_HEADER:END -->

## Intent

Map the hot native performance stages to reusable physical workload shapes.
This ledger is the working table for the 100ms push: it names the current
shape, required shape, carriers, export boundary, and canary before any local
optimization is counted as progress.

The ledger deliberately tracks stage families, not benchmark-script branches.
A row is useful only when it explains how the shape helps unknown downstream
work remain native.

## Request Signals

- physical shape ledger
- native 100ms canary
- relation consumer
- overlay cached pairs
- grouped reduce
- native composition

## Open First

- docs/dev/native-physical-shape-ledger.md
- docs/dev/native-100ms-physical-shape-plan.md
- docs/dev/private-native-execution-substrate-plan.md
- docs/dev/native-format-library-plan.md
- docs/decisions/0044-private-native-execution-substrate.md
- docs/decisions/0046-gpu-physical-workload-shape-contracts.md
- src/vibespatial/api/_native_results.py
- src/vibespatial/api/tools/sjoin.py
- src/vibespatial/api/tools/overlay.py
- src/vibespatial/api/tools/_pair_cache.py
- tests/test_overlay_api.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "native physical shape ledger relation overlay cached pairs"`
- `uv run pytest tests/test_pipeline_benchmarks.py -q -k overlay_relation_constructive`
- `uv run pytest tests/test_overlay_api.py -q -k "reuses_cached_sjoin_pairs or reuses_cached_pairs_when_only_nonparticipating"`
- `uv run vsbench shootout benchmarks/shootout --repeat 3 --scale 10k`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Counting cached host arrays as a native win can hide that the real target is
  relation consumption before public row assembly.
- A canary can become workflow-specific if it names a benchmark step but not
  the native carrier and sanctioned downstream consumer.
- Removing one materialization event without reducing wall time is a diagnostic
  improvement, not a completed physical-shape change.
- Public export boundaries are acceptable only when they are terminal for the
  user-visible operation.

## Ledger

| Stage family | Current physical shape | Required physical shape | Native input carriers | Native output carrier | Public export boundary | Shape canary | Profile signal |
|---|---|---|---|---|---|---|---|
| Spatial join relation consumers | `NativeRelation` feeds semijoin, anti-join, counts, and reductions. Dynamic distance predicates now produce `NativeDeviceSelection`; `NativeRelationSelection` gathers pair columns at source capacity and reduces match counts plus distance count/min with a device-resident logical cardinality. No compact relation allocation or host count fence occurs before the terminal oracle/export. Public joined rows remain common only for terminal `sjoin`. | Relation-consume over pair capacity with device logical length; use rowset or tabular projection only at sanctioned consumers. | `NativeFrameState`, `NativeSpatialIndex`, `NativeRelation`, `NativeRelationSelection`, `NativeDeviceSelection`, `NativeIndexPlan` | `NativeDeviceSelection`, `NativeExpression`, `NativeFrameState`, or `NativeTabularResult` | `RelationJoinExportResult.to_geodataframe()` and explicit IO export | `relation-bridge-consumer`, `relation-semijoin`, `relation-distance-expression` | Static and CPU-safe tests enforce that the selected relation consumer does not compact or export its logical count. Accelerator timing passes in the full profile. The latest 1M relation-semijoin reports zero materialization and zero D2H in consumer stages; public retail joined-row/index exports remain classified as terminal. |
| Overlay cached-pair reuse | Public `sjoin` may export relation pairs, while overlay can still rebuild candidate pairs if the cache is missed. | Reuse the already-produced relation pairs as the overlay candidate relation; avoid a second spatial query when lineage and validity checks admit it. | `RelationIndexResult` or `NativeRelation`, left/right `NativeFrameState`, cached geometry validity metadata | `NativeTabularResult` with pairwise constructive provenance | Overlay `NativeTabularResult.to_geodataframe()` | `test_overlay_intersection_reuses_cached_sjoin_pairs*` | Canary guards device sjoin pairs that skip the pre-export host cache; public export now seeds the overlay cache once relation pairs are already host-visible. |
| Many/few overlay constructive | `overlay-relation-constructive` keeps candidate generation, refinement, constructive geometry, and tabular projection in native carriers before the native shape reference check. Public overlay intersection consumes `NativeSpatialIndex.query_relation`; device pair rows feed owned gathers and tabular projection directly; grouped difference radix-orders one full-source `NativeGrouped`; collective topology owns containment, coverage, holes, multipart complements, exact crossing rows, and paging; pairwise line/polygon work shares one split-event carrier; and lower-dimensional remnants stay in `NativeGeometryComposition` until public export. Indexed topology remains row-indirected, with named device physicalization only for contiguous-family consumers. Exact topological equality reuses bidirectional native difference for structurally unresolved lineal and polygonal rows. Host pair/group exports, Python group loops, compact retry partitions, exception-driven switches, Shapely repair, and duplicate topology engines are deleted. | `NativeSpatialIndex`/metadata to candidate `NativeRelation`, refine relation, constructive provenance, native row or attribute projection. | `NativeSpatialIndex`, `NativeGeometryMetadata`, `NativeRelation`, left/right `NativeFrameState` | `NativeTabularResult` with `NativeGeometryProvenance` | Explicit overlay export or IO write | Relation constructive, grouped spatial overlay/complement, pair-cache, boundary-composition, indexed-view, DE-9IM, equality, and exception-atomicity canaries | The full accelerator profile passes with zero compute materialization or D2H; upstream overlay passes 128 tests. No active physical-shape debt remains in the admitted overlay contract. |
| Grouped geometry reduce | `NativeGrouped` feeds constructive work through sorted offsets. Polygonal complements, mixed exact partitions, rectangle strips, bounded fragments, residual repair, validity repair, paging, component plans, and microcell topology retain grouped, rowset, part, event, or output-byte capacity. Pair/carry/round counts are algebraic; invalid outputs repair atomically; global union lowers once; buffered-line dissolve buffers once; make-valid reuses grouped topology; and exact SoA radix passes replace mixed host keys. Host CSR regroup, bbox coloring, Python row loops, Shapely retry trees, and post-repair recomputation are deleted. | Segmented reduction plus bounded fragments, sparse rowsets, capacity-backed ring/segment/concat buffers, streamed candidate classification, externally merged split-event runs, complete-row, interval-component, connected-page, or segmented-multirow microcell topology, and exact boundary atoms. | `NativeGrouped`, `NativeDeviceSelection`, `NativeGroupedSelection`, `NativeFrameState`, `NativeGeometryMetadata`, `NativeGeometryComposition` | `NativeGrouped` result, geometry composition, or native tabular output | Dissolve/grouped export and terminal GeometryCollection assembly | Grouped topology/reducer/global-union/residual/coordinate-capacity and point-pair capacity guards, buffered-line single-carrier/no-switch guards, page/component/microcell tests, and the `grouped-capacity-partitions` profile canary | Accelerator execution passes. Latest 1M grouped disjoint setup, mixed union, partition setup, positive/degenerate union, grouped union, and difference are 65.88, 50.64, 47.36, 46.40, 33.91, and 10.86ms with zero compute D2H/materialization. |
| Native composition copy and filter | Exact rowset transitions preserve `NativeFrameState`; `NativeDeviceSelection` carries source-capacity positions plus a device logical count and rebases gathered results to an active compact prefix without reading that count. `NativeRelationSelection` and `NativeGroupedSelection` consume it directly. Relation selection construction null-pads pair geometry, gathers attributes/provenance at capacity, and returns `NativeTabularSelection`. `NativeTabularSelection` holds an exact capacity-sized `NativeTabularResult`, concatenates partition prefixes and source-orders active rows without a count read, and compacts only for an explicit consumer/export. Generic rename, concat, and symmetric-difference adapters preserve the capacity result plus selection, so dynamic producers survive ordinary constructive composition without a producer-specific escape hatch. Polygon exact continuation, point-mask filtering, mask-cover valid/nonempty passthrough, and rectangle output are dynamic tabular producers. Rectangle line output fuses segment traversal with source-row run/coordinate scans, preserves source part breaks, and retains bounded coordinate capacity without a segment table or allocation fence. Aligned line-line intersection consumes shared same-row segment-classifier pages, nodes overlap spans through grouped atomic-line capacity, deduplicates and line-suppresses points, and emits empty, point, line, or mixed rows through `NativeGeometryComposition`; GeometryCollection creation and typed-empty selection remain terminal. Aligned line/polygon intersection and difference share one all-ring split-event carrier; midpoint and event predicates feed capacity line assembly, isolated point deduplication, line suppression, and native mixed composition. Shared paths uses the same classifier and separate forward/backward atomic-line capacities; ordered collection positions preserve valid-empty MultiLineString slots until terminal GeometryCollection export. The old row-shaped line-line kernel, first-ring line/polygon kernel, bespoke shared-path kernels, and their allocation fences are deleted. Point, polygon, and line capacities compose through one device row-indirection map; the mixed public rectangle adapter consumes the combined carrier without family compaction. Device geometry takes preserve structural metadata; indexed views propagate validity and cached bounds by family-row indirection. | Use capacity carriers for dynamic filters without weakening exact frame/tabular invariants. | `NativeFrameState`, `NativeRowSet`, `NativeDeviceSelection`, `NativeRelationSelection`, `NativeTabularSelection`, `NativeIndexPlan`, `NativeAttributeTable`, `NativeExpression` | Exact frame/result, capacity selection, or owned row-indirection | Unknown pandas operations, explicit compact-rowset conversion, and public export | relation-distance-expression, relation-selection constructive, line-line, line/polygon, and shared-path topology, generic dynamic-tabular rename/concat/order, polygon/point/line/mixed clip, mask-cover passthrough, fixed nested take, capacity partition selection, and fused multi-scatter canaries | CPU/static guards pass; constructive accelerator execution passes. |
| Mask clip and area filtering | Native mask and cleanup filters preserve candidate, exact, positive-area, boundary, source-lineage, and relation-coverage capacity selections. `NativeDeviceSelection` now stores a stable selected prefix and rejected tail, so inactive lanes retain unique source provenance. Polygon device candidates keep inside, exact, positive-area, and boundary masks as capacity selections; fused predicate declines continue through aligned exact device intersection and native tabular assembly instead of host partition reconstruction; spatial ordering uses exact SoA radix columns without stacked float row ids. Native-to-spatial cleanup combines valid/nonempty, positive-area, and keep-geometry-type masks over source capacity and returns `NativeTabularSelection`; missing device rowsets are synthesized from already-known source positions instead of declining to host cleanup. Degenerate line repair emits one point per collapsed LineString or MultiLineString part at line-part capacity, performs segmented exact deduplication plus compensated fp64 grouped means, and selects the repaired Point partition through row indirection. Polygon area plus lower-dimensional boundary remnants map concrete owned parts through `NativeGeometryComposition`; overlay and clip use the same fixed-capacity composition factory, with GeometryCollection construction terminal. Rectangle-cell boundary splitting retains source-row and component capacity, derives nested logical prefixes on device, and selects MultiPolygon replacements through one row-indirection map; dense polygon-rectangle output vertices size from fixed-width metadata. Polygon/MultiPolygon keep-type compatibility selects row-aligned family capacities through one device map. Non-keep rectangle Polygon/MultiPolygon partitions reuse the shared `clip_by_rect_owned` candidate-capacity executor; the duplicate trusted-bounds boundary algorithm is deleted. Lazy grouped masks keep covered and unresolved rows at source capacity. One two-int aggregate admission packet skips empty unresolved work and compares relation-first pair capacity with union-first source capacity using native coordinate, segment, pair, group, output-byte, and temporary-byte estimates; no logical count sizes geometry output. Compatibility tails export sparse nonmissing positions from the same rowset and synthesize identity rows without transfer. Rectangle boundary point contacts retain Point/MultiPoint row capacity and select null-padded exact line-overlap rows without a cardinality read. General boundary intersections explode line/point parts at physical capacity, convert zero-length lines to point capacity, and combine point sources before Point/MultiPoint packing. Device polygon-mask compatibility reuses the canonical candidate-capacity executor; its host correction probes, sparse exports, reconstruction, reupload, and compact pack/combine/regroup family are deleted. Point-contact slivers use one polygon slot per boundary row, and final geometry/tabular output stays at candidate capacity under `NativeTabularSelection`. The terminal exact-rebuild hook is deleted. Accelerator verification is complete. | Metadata/expression/relation to rowset, then native owned/tabular/composition assembly. | `NativeGeometryMetadata`, `NativeExpression`, `NativeRelation`, `NativeFrameState`, `NativeRowSet`, `NativeDeviceSelection`, `NativeGeometryComposition` | `NativeRowSet`, capacity-selected `NativeTabularResult`, filtered `NativeFrameState`, or `GeometryNativeResult` | Terminal clip/overlay export and explicit invalid-input GEOS compatibility | Polygon device-candidate exact-continuation, semantic-cleanup capacity selection, degenerate-line part-capacity repair, rectangle boundary-split capacity, area-plus-boundary composition, overlay remnant, sparse/dense collective shape selection, lazy grouped-mask partition, and no-scalar-admission and general boundary-capacity canaries | CPU-safe composition and dispatch contracts pass. Accelerator measurements pass in the full profile. |
| Dispatch shape estimates | Every adaptive planner call supplies `PhysicalWorkEstimate`; shared metadata-only estimators cover coordinates, coordinate pairs, segments, segment pairs, parts, part pairs, rings, candidate/relation pairs, groups, output rows/bytes, and scratch bytes. Unary metrics/constructives, validity/repair, buffers, linear reference, pairwise predicates, spatial indexes, overlay relation expansion, and bounded polygon kernels expose their dominant physical shape. Authoritative device families override host stubs and indexed views scale to logical rows. Host compatibility and static defaults use explicit bootstrap estimates. | Keep dynamic output sizing capacity-backed and enforce row-indirected execution or named non-mutating device physicalization. | `OwnedGeometryArray`, `NativeRelation`, `NativeGrouped`, `NativeGeometryMetadata` | `AdaptivePlan`/`RuntimeSelection` | None for compute planning; explicit terminal public export only | runtime policy, authoritative-device, pair-product, part-graph, buffer-output, grouped-union, overlay-relation, and AST coverage tests | Zero planner calls omit `work_estimate`; `ARCH009` rejects direct compute-path `_device_resolve`; polygon buffer returns stream-ordered state without a forced sync. |
| Terminal native export | Native Arrow, Parquet, Feather, and GeoDataFrame exports exist as explicit public boundaries. Device-resident GeoParquet writes avoid the small Arrow shortcut, do not force device `total_bounds` for optional bbox metadata, and assemble row-indirected GeoArrow offsets from device family rowsets. Dynamic public pandas results read one exact device count because pandas requires a Python length; possible GeometryCollection outputs perform terminal multiplicity certification before owned-device physicalization. | Keep export terminal and measured separately from compute-stage targets. | `NativeFrameState`, `NativeTabularResult`, `NativeIndexPlan` | Public GeoDataFrame, Arrow, Parquet, Feather | The export operation itself | native IO/export rails | Latest full profile reports zero compute materialization and zero compute D2H. Across both scales, terminal public writes account for the only totals: 2 materializations and zero runtime D2H. Terminal wall time is tracked separately from compute-shape decisions. |

## Canary Maintenance
1. Keep relation coverage and grouped union/disjoint/difference as 100ms canaries.
2. Keep physicalization guards green and terminal export measured separately.
3. Reopen only for host work, a stale carrier, or a native stage above 100ms.

## PRD Execution Notes
- Row-isolated face output-hole, ring, polygon, and public-row assembly stays at structural capacity; dynamic non-row-isolated `OwnedGeometryArray` length is terminal physicalization because row count is a Python integer.
- Indexed Polygon/MultiPolygon/Point/Line explode uses fixed widths, per-row maxima, or conservative structural bounds and never physicalizes then retries.
- Rectangle-strip/exact and positive-area/degenerate grouped union are complementary device partitions; both two-flag packets and global Python switches are deleted.
- Compact `NativeRowSet` conversion is an explicit `NativeTabularSelection` consumer/export boundary; grouped-union scalar correctness admissions remain separately classified.
- Grouped difference carries proved indexed part and complement capacities;
  unproved rows continue through row-indirected collective topology.
- Clip and aligned overlay use device rowsets and native boundary composition; clip
  predicates preserve indexed source and candidate-local rows through device-counted
  capacity selections. Rectangle specialization does not compact or scalar-probe tags; GEOS is terminal.
- Overlay strict polygon difference sends the complete candidate relation,
  including boundary-only contacts, into grouped topology without a duplicate
  intersection/area prefilter. Pairwise lineal/polygonal intersection and
  difference atomize every aligned ring/part in one split-event plan, assemble
  row-ordered fragments at capacity, and carry isolated point contacts through
  native composition; grouped pair ids remain resident. Grouped
  topology keeps indexed polygon/multipolygon rows as row-indirected logical
  carriers; exact left-covered groups prune to valid empty polygons; direct
  single-ring holes, one-hole right donuts, and grouped-union single-ring holes
  emit from grouped offsets and structural metadata.
  Multi-hole/multipart rights now use row-indirected `NativeDeviceSelection`
  part capacity plus `NativeGroupedSelection`; inactive part/ring/interior/
  coordinate lanes are masked and output families retain public-row capacity.
  Component, make-valid, clip, grouped-disjoint, known-coverage, and residual guards
  pass. Overlay consumes atomic repair directly; host postchecks/patches are deleted.
- Lazy grouped-union clip keeps covered/unresolved rows as source capacity and
  exports one aggregate planner packet. Relation pair output reduces by source:
  valid-empty grouped polygon area, area-subtracted/noded atomic lines, deduplicated
  points with area/line suppression, and terminal `NativeGeometryComposition`.
- Cached overlay subset pairs use `NativeRelationSelection`; grouped counts
  prove participating-row validity, and geometry, attributes, area/type
  filters, and boundary composition stay at pair capacity. Static
  no-compaction guards and accelerator execution pass.
- Clip degenerate-line repair is now one line-part-capacity carrier. It emits
  one candidate for a collapsed LineString and one for each nonempty
  MultiLineString part, performs segmented exact coordinate deduplication,
  computes compensated fp64 unique-point means, and selects the row-aligned
  Point partition through one device index map. `extract_unique_points` also
  preserves indexed inputs and retains input-coordinate capacity; neither path
  exports valid rows, active totals, or compact coordinate indices. The
  distinct collapsed-part canary prevents regression to first-coordinate
  repair. Accelerator execution passes in the broad clip gate.
- Native grouped dissolve union feeds one group-row carrier to coverage and
  split matching. Buffered-line provenance uses one buffer/grouped-union
  execution; invalid outputs use atomic sparse repair. Touching groups, lines,
  partial rows, strict-disjoint assembly, bounded spans, and rectangle strips
  or holes consume `NativeGrouped` offsets plus device metadata. Rejected
  shortcuts include weak coverage-union admission, predicate/area empty skips,
  exact-containment-only plans, and bbox-risk partitioning because they changed
  GeoPandas-equivalent counts.
- Overlay difference radix-sorts relation pairs once into dense full-source-row
  `NativeGrouped` offsets; zero spans preserve no-neighbor rows and topology owns
  paging. Direct hole/donut masks fuse with one complementary exact carrier;
  external batching/scatter, scalar admission, and group-size export are deleted.
- Overlay keep-geometry-type tolerance compares overlap/source areas on device
  and exports only the final public keep mask/rows. Cover-probe candidates use
  fp64 device area masks and fused broadcast-capable `NativeExpression`
  predicate vectors without the old area-candidate mask export. Device-backed
  sparse exact-cache positions stay sparse for semantic inputs. Failed device
  predicates decline to the conservative native warning path instead of CPU.
  Device-backed unresolved semantic probes now decline before host geometry
  materialization; intersection attribute pair exports remain terminal public
  pandas assembly. GeometryCollection-preserving invalid rectangle repair
  remains an explicit strict-native-blocked GEOS fallback.
- Overlay many/few candidate-pair assembly now keeps few-right device pairs in
  the native carrier through owned gathers and native tabular construction.
  Exact few-right topology consumes indexed right rows through the canonical
  bounded device physicalizer; the unused duplicate segment cache, coordinate
  expansion, crossover, allocation fence, and rejected-row vectors are deleted.
  Numeric dataframe-backed projected attributes promote to device tables for
  downstream native expressions; mixed/string attributes use a terminal loader
  so candidate rows are not exported before the public GeoDataFrame boundary.
  Constructed overlay outputs keep aligned pair-source provenance so warning
  filters do not export `DeviceSpatialJoinResult` rows to rebuild pair context.
- Owned device state carries trusted family-domain and unique family-row proofs.
  Mixed-family constructive physicalizes family pairs at public-row capacity and
  fuses them through one row-indirected selection; ordinary same-category type
  probes no longer cross devices. Aligned polygon operations retain null-padded
  valid empties; union uses one same-side-split exact topology plan. The five-way
  rectangle/SH/swapped-SH/exact router is capacity-backed; host classifiers,
  compact rows, count reads, rescue switches, and preference flags are deleted.
- Multipart polygon regroup has one `NativeGrouped` topology executor. The
  circular `segmented_union_all` retry, its opt-in flag, and both device
  group-offset exports are deleted; topology failures and exceptions are atomic.
- Point/MultiPoint-Polygon intersection and difference now flatten point
  members into a device source-row carrier, consume exact native `intersects`,
  deduplicate `(source, x, y)` tuples, and pack Point/MultiPoint buffers from
  device counts. Offset, validity, coordinate, predicate-mask, and Shapely
  export/reupload paths are deleted; null and valid-empty rows are distinct.
- `PhysicalWorkEstimate` now covers rows, coordinates, coordinate pairs,
  segments, segment pairs, parts, part pairs, rings, relation density, groups,
  logical indexed expansion, output, and scratch pressure. Every adaptive
  planner call supplies one; only host compatibility and static defaults use
  explicit row bootstrap. Same-row warp admission uses structural span proofs.
  Capacity selections feed relation/grouped reduction and exact continuation
  without host counts. Part graphs, bounded SH/rectangle clip, contained-hole,
  stroke-buffer, and segmentize outputs retain physical capacity. Generic
  offset-slice gathers use guarded row-range kernels rather than inactive lanes.
  Unknown-size parser gathers carry explicit allocation reasons. Line/polygon
  construction consumes split-event capacity without an output-total read.
  Grouped-difference part explosion now uses named, non-mutating device-row
  physicalization; `ARCH009` confines direct generic resolution to owned-carrier internals.
  Clip candidate masks retain stable capacity; grouped-mask planning exports one two-int packet.
  Repeated-right overlay keeps canonical indexed physicalization. Segmentize
  exports one exact-allocation packet and scatters coordinate lanes directly.
  Host Arrow/WKB is native CPU; GPU IO requires an executable runtime.
