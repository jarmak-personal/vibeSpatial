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
| Spatial join relation consumers | Pair-preserving consumers use `NativeRelation`/`NativeRelationSelection`. Existential/count consumers use `NativeSpatialIndex` range-sliced Morton traversal: span buckets order query ids, each launch scans only that bucket's interval slice, and count/scan/scatter writes bbox hits to a fixed-capacity prefix. Mixed candidates are radix-partitioned once by packed family-pair code; specialized point and polygon kernels consume shared sorted arrays through device source offsets/counts, and aggregate launch capacity is bounded by one structural tile. Atomics reduce live candidates directly to row outputs. Existential output remains a capacity-backed `NativeDeviceSelection`; antijoin complements its source mask on device, and clip preserves its selected prefix/rejected tail through spatial ordering and point/line/polygon/mixed construction. No full pair carrier, dense query-by-tree tile, repeated family-pair capacity scan, dynamic output compaction, or device-controlled host loop remains. | Choose relation or reduction before candidate allocation; partition mixed family pairs once and consume grouped spans directly. | `NativeFrameState`, `NativeSpatialIndex`, `NativeRelation`, `NativeRelationSelection`, `NativeRelationFamilyPartition`, `NativeDeviceSelection`, `NativeIndexPlan` | `NativeDeviceSelection`, `NativeExpression`, `NativeFrameState`, or `NativeTabularResult` | `RelationJoinExportResult.to_geodataframe()` and explicit IO export | range-sliced/span-bucket/candidate-prefix AST guard, grouped-family one-pass/aggregate-capacity canary, precision-plan propagation, dense semijoin/count accelerator canary, clip capacity-candidate canaries | At 10K, direct site semijoin exactly matches 3,302 unique rows from 778,271 pairs. At 1M, site, redevelopment, retail, and the mixed-family canary complete without full relations; the mixed canary records one radix partition and zero D2H. One count probe processes 32M exact predicates with one 24-byte planning packet. The current mandatory 100K/1M suite records zero materializations or fallbacks and 40 bounded topology-planning D2H packets totaling 41,056 bytes. Eager public relations require 46.51-70.84GiB. |
| Overlay cached-pair reuse | Public `sjoin` may export relation pairs, while overlay can still rebuild candidate pairs if the cache is missed. Cache retention is a byte-bounded LRU and counts reverse-key aliases once, so a multi-GB one-use public relation cannot pin frames and device columns indefinitely. | Reuse already-produced relation pairs when lineage, validity, and cache capacity admit them; avoid a second query without making cache residency unbounded. | `RelationIndexResult` or `NativeRelation`, left/right `NativeFrameState`, cached geometry validity metadata | `NativeTabularResult` with pairwise constructive provenance | Overlay `NativeTabularResult.to_geodataframe()` | `test_overlay_intersection_reuses_cached_sjoin_pairs*`, `test_intersection_pair_cache_evicts_relations_over_byte_budget` | Small reusable relations remain device-cached; relations above the global physical byte budget are evicted atomically with their reverse aliases. |
| Many/few overlay constructive | Candidate generation, refinement, constructive geometry, and projection remain native. Grouped difference uses one full-source `NativeGrouped`; signed winding deltas survive partial-overlap renoding; exact cycle orientation, fixed-capacity indexed containment, and O(E) boundary peeling replace host probes and quadratic face relations. Prepared one-mask exact topology lowers unresolved row/segment evidence to a CUDA-resident `NativeRelation` of complete candidate rings. Each hole candidate adds its ancestor shell as the canonical face-walk winding baseline, and only those ring segments enter row-isolated fp64 topology. Indexed grouped-topology batches cross one exact nested-offset allocation packet before mutable topology, so retained constructive coordinate capacity cannot size the gather. Grouped row-isolated split planning derives exact per-row event weights and segment maxima from device ownership even when optional host span metadata is absent, then retires complete-row topology pages inside the active query-memory envelope. Coverage-only containment adapts on device between one warp per shallow root and one 256-lane traversal for a lone large root. Lower-dimensional remnants remain in `NativeGeometryComposition` until export. | Retain `NativeSpatialIndex`/metadata to candidate `NativeRelation`, refine relation, constructive provenance, and native row or attribute projection. Exact prepared-mask work must stay bounded by complete candidate-ring segments, and grouped exact topology must be bounded by authoritative nested-offset spans and complete-row split-event pages rather than backing coordinate capacity or one global event merge. | `NativeSpatialIndex`, `NativeGeometryMetadata`, `NativeRelation`, left/right `NativeFrameState` | `NativeTabularResult` with `NativeGeometryProvenance` | Explicit overlay export or IO write | Relation constructive, complete-ring winding-baseline, padded-capacity grouped-topology physicalization, missing-span-summary page planning, default-pool bounded grouped difference, grouped complement, adaptive face containment, boundary composition, indexed view, DE-9IM, equality, and exception-atomicity canaries | The complete-ring canary lowers 18 logical mask segments to 12; the adversarial multi-component/hole case lowers 72 to 20 while matching the exact oracle. The grouped paging canary derives 4 left and 8 right segments per row on device, lowers three logical rows to three independently retired exact pages, and matches fp64 difference topology using only compact page-weight/work-summary planning packets. A padded-capacity canary retains 1,000,000 coordinate lanes around 10 authoritative polygon coordinates and proves exact grouped-topology physicalization allocates only the live nested-offset spans. The current same-digest 14-workflow 100K artifact `/tmp/native-consolidation-current/core-100k-r3.json` is 14/14 exact: 167.2154s GeoPandas versus 13.4651s vibeSpatial, 12.418x aggregate and 4.989x geomean. Redevelopment is 29.83x and site suitability 115.05x; habitat (0.969x) and insurance (0.630x) are the only losses. The current bounded 100K/1M profile has zero compute materializations and fallbacks; its only compute D2H is 40 compact page-weight/work-summary packets totaling 41,056 bytes, and no stage exceeds 77.15ms. |
| Grouped geometry reduce | `NativeGrouped` feeds constructive work through sorted offsets. Scalar and grouped coverage reductions share one noded grouped-boundary assembler, preserving topology and structural output ordering across public entry points. Buffered multivertex line dissolve executes one collective topology row below measured segment-peer pressure. Above it, a two-dimensional regular grid builds an exact segment-to-tile `NativeRelation`; scanline coverage proves full tiles, and sparse boundary tiles alone enter rectangle clipping and local exact topology. Adaptive tile dimensions derive from measured local segment-peer pressure. Active neighboring tiles share one relation-grouped clip/topology plan while summed segment-peer pressure stays under the topology budget; offsets and exact tile pressures cross in one compact packet, and dense tiles remain isolated. Local and online seam results are device-physicalized to exact concrete prefixes before retention. Singular union retains only online seam levels; reusable downstream coverage retains only tile rows. Grouped-mask clip admission uses a bounded disjoint Morton-prefix cover before exact work. An admitted exact relation is retained, refined once, sorted once, and consumed in complete-source-row pages; each page refreshes admission against current free device memory so earlier retained outputs reduce the next page budget. Fixed-point atomic renoding runs only for same-row paired events with fp64 coordinate keys inside the ULP risk window. | Keep singular and reusable-coverage ownership exclusive. A grouped-mask clip plans from original mask members first and chooses retained exact relation/group reduction or one-time union before topology reduction; it must not eagerly replace local members with complex tiled coverage or repeat exact candidate work. | `NativeGrouped`, `NativeRelation`, `NativeDeviceSelection`, `NativeGroupedSelection`, `NativeFrameState`, `NativeGeometryMetadata`, `NativeGeometryComposition` | `NativeGrouped` result, geometry composition, or native tabular output | Dissolve/grouped export and terminal GeometryCollection assembly | Candidate-tile relation/pressure/full-tile guards, scalar/grouped coverage equivalence, aggregate-pressure batch-span canary, direct/tiled collective guards, exclusive coverage ownership guard, Morton-prefix interval disjointness, retained-relation single-query/single-sort guards, dynamic complete-row page admission, exact physicalization retention guard, planarity-risk/renoding guards, face-queue guards, memory-derived page planner, and grouped topology/reducer/global-union/residual canaries | Vegetation 1M is exact at 52.73s in the comparable timed-section harness versus the historical 359.83s GeoPandas reference. Transit is exact at 191.44s versus the historical 411s gate. Habitat 100K repeat-three is exact at 2.5081s versus 2.4383s GeoPandas (0.972x) in `/tmp/native-habitat-final/habitat-100k-page-formal-r3.json`, with 332.8MB peak device memory and no fallback. Its synchronized replay is 2.5702s with 43/43 GPU steps, zero CPU steps, and no hotpath above 34.16ms. |
| Native composition copy and filter | Exact rowset transitions preserve `NativeFrameState`; `NativeDeviceSelection` carries source-capacity positions plus a device logical count and rebases gathered results to an active compact prefix without reading that count. `NativeRelationSelection` and `NativeGroupedSelection` consume it directly. Relation selection construction null-pads pair geometry, gathers attributes/provenance at capacity, and returns `NativeTabularSelection`. `NativeTabularSelection` holds an exact capacity-sized `NativeTabularResult`, concatenates partition prefixes and source-orders active rows without a count read, and compacts only for an explicit consumer/export. Generic rename, concat, and symmetric-difference adapters preserve the capacity result plus selection, so dynamic producers survive ordinary constructive composition without a producer-specific escape hatch. Polygon exact continuation, point-mask filtering, mask-cover valid/nonempty passthrough, and rectangle output are dynamic tabular producers. Rectangle line output fuses segment traversal with source-row run/coordinate scans, preserves source part breaks, and retains bounded coordinate capacity without a segment table or allocation fence. Aligned line-line intersection consumes shared same-row segment-classifier pages, nodes overlap spans through grouped atomic-line capacity, deduplicates and line-suppresses points, and emits empty, point, line, or mixed rows through `NativeGeometryComposition`; GeometryCollection creation and typed-empty selection remain terminal. Aligned line/polygon intersection and difference share one all-ring split-event carrier; midpoint and event predicates feed capacity line assembly, isolated point deduplication, line suppression, and native mixed composition. Shared paths uses the same classifier and separate forward/backward atomic-line capacities; ordered collection positions preserve valid-empty MultiLineString slots until terminal GeometryCollection export. The old row-shaped line-line kernel, first-ring line/polygon kernel, bespoke shared-path kernels, and their allocation fences are deleted. Point, polygon, and line capacities compose through one device row-indirection map; the mixed public rectangle adapter consumes the combined carrier without family compaction. Device geometry takes preserve structural metadata; indexed views propagate validity and cached bounds by family-row indirection. | Use capacity carriers for dynamic filters without weakening exact frame/tabular invariants. | `NativeFrameState`, `NativeRowSet`, `NativeDeviceSelection`, `NativeRelationSelection`, `NativeTabularSelection`, `NativeIndexPlan`, `NativeAttributeTable`, `NativeExpression` | Exact frame/result, capacity selection, or owned row-indirection | Unknown pandas operations, explicit compact-rowset conversion, and public export | relation-distance-expression, relation-selection constructive, line-line, line/polygon, and shared-path topology, generic dynamic-tabular rename/concat/order, polygon/point/line/mixed clip, mask-cover passthrough, fixed nested take, capacity partition selection, and fused multi-scatter canaries | CPU/static guards pass; constructive accelerator execution passes. |
| Mask clip and area filtering | Native mask and cleanup filters preserve candidate, exact, positive-area, boundary, source-lineage, and relation-coverage selections. `NativeDeviceSelection` stores a stable selected prefix and rejected tail. One-mask polygon classification builds one device segment index, uses range-sliced Morton span buckets for boundary and exact-ray candidates, refines only count/scan/scattered candidate prefixes under an explicit authoritative-fp64 `PrecisionPlan`, and retains row flags on device. Boundary-unresolved rows pack into a selection prefix; one aggregate exact-allocation packet sizes concrete coordinate buffers, then a device `NativeRelation` selects complete local rings and ancestor-shell winding baselines for exact row-isolated topology. Device positions scatter results and metadata back. Lazy grouped masks query original member bounds before reduction. A fixed 16-interval Morton-prefix packet compares structural relation and union work before exact allocation. If relation work wins, the single retained relation is refined and executed by a dynamic complete-source-row page plan; only capacity rejection or a later page-admission failure discards it and materializes the union path. Polygon candidates otherwise keep inside, exact, positive-area, and boundary masks at capacity; native cleanup returns `NativeTabularSelection`; degenerate line repair is line-part shaped; area plus boundary remnants remain `NativeGeometryComposition`; GeometryCollection construction is terminal. Host correction probes, logical-count admission, eager tiled-coverage admission, duplicate exact prepasses, sparse exports, reconstruction, reupload, compact regroup, and terminal exact rebuild are deleted. | Indexed mask metadata to capacity candidate relation and row flags, then `NativeDeviceSelection` exact-prefix physicalization, complete-ring relation topology, or original-member grouped reduction selected before any mask union/coverage topology. | `NativeSpatialIndex`, `NativeGeometryMetadata`, `NativeExpression`, `NativeRelation`, `NativeFrameState`, `NativeRowSet`, `NativeDeviceSelection`, `NativeGeometryComposition` | Compact exact owned prefix, `NativeRowSet`, capacity-selected `NativeTabularResult`, filtered `NativeFrameState`, or `GeometryNativeResult` | Terminal clip/overlay export and explicit invalid-input GEOS compatibility | Indexed-mask lane-bound/exact-ray, complete-ring winding-baseline, Morton-prefix cover, retained-relation single-query/single-sort, dynamic complete-row page admission, original-member grouped-mask admission, precision-plan, polygon exact continuation, cleanup capacity, degenerate repair, boundary composition, GeometryCollection grouped union, rectangle split, and no-scalar-admission canaries | Habitat 100K repeat-three is exact at 2.5081s versus 2.4383s GeoPandas (0.972x), down from the 14.1656s amplified tiled-coverage path, with no fallback and 332.8MB peak device memory. The full replay records 43 GPU steps, zero CPU steps, zero H2D, and 1.43MB of bounded D2H metadata in 1.87ms. Broad focused GPU gates pass. Vegetation 10K is exact at 187.4ms. The current mandatory 100K/1M profile has zero compute materializations or fallbacks; six grouped-topology stages emit only bounded planning packets, and no active stage exceeds 77.15ms. |
| Dispatch shape estimates | Every adaptive planner call supplies `PhysicalWorkEstimate`; shared metadata-only estimators cover coordinates, coordinate pairs, segments, segment pairs, parts, part pairs, rings, candidate/relation pairs, groups, output rows/bytes, and scratch bytes. Unary metrics/constructives, validity/repair, buffers, linear reference, pairwise predicates, spatial indexes, overlay relation expansion, and bounded polygon kernels expose their dominant physical shape. Authoritative device families override host stubs and indexed views scale to logical rows. Host compatibility and static defaults use explicit bootstrap estimates. Raw driver, pylibcudf, and CCCL work shares the active CuPy stream. One completion service coalesces explicit-stream windows, records PTDS events in the submitting thread under a retained lifetime-unique thread token, and retires every cached/one-shot CCCL operand plus scratch token independently of later API calls; cross-stream pylibcudf consumers wait on producer-recorded readiness events. Metric precision summaries reduce logical row bounds; indexed bounds and metrics compact unique family rows, bin their selected coordinate spans on device, and scatter through an inverse without scanning unselected ancestral coordinates. Completed grouped outputs publish valid-family-row injectivity, so indexed polygon-part assembly consumes exact physical root capacity without a row-width export or dense `rows * maximum` allocation. | Keep dynamic output sizing capacity-backed and enforce row-indirected execution or named non-mutating device physicalization. | `OwnedGeometryArray`, `NativeRelation`, `NativeGrouped`, `NativeGeometryMetadata` | `AdaptivePlan`/`RuntimeSelection` | None for compute planning; explicit terminal public export only | runtime policy, authoritative-device, pair-product, part-graph, buffer-output, grouped-union, overlay-relation, stream-ordering, completion-retirement, indexed-metric precision, and AST coverage tests | Zero planner calls omit `work_estimate`; `ARCH009` rejects direct compute-path `_device_resolve`. The current mandatory 100K/1M profile has zero compute materializations or fallbacks, 41,056 bytes of classified planning D2H, and no stage above 77.15ms. |
| Terminal native export | Native Arrow, Parquet, Feather, and GeoDataFrame exports exist as explicit public boundaries. Native WKB export certifies valid composition-page spans in one compact packet, ignores all-invalid capacity carriers, emits exact physical pages whole from exact structural byte bounds, and bounds partial indexed pages. Certified contiguous GeoParquet read compositions reuse their row-group partition carrier without a row-count-sized multiplicity selector. | Keep export terminal, page-shaped, memory-bounded, and measured separately from compute-stage targets. | `NativeFrameState`, `NativeTabularResult`, `NativeGeometryComposition`, `NativeIndexPlan` | Public GeoDataFrame, Arrow, Parquet, Feather | The export operation itself | native IO/export rails, null-capacity page certification, contiguous read-carrier reuse | The 30.55M-row transit output writes and reads exactly on 24GB in 191.44s versus the historical 411s GeoPandas gate. The latest full profile reports zero compute materializations and only 41,056 bytes of classified bounded planning D2H; terminal wall time remains separate. |

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
- Lazy grouped-union clip plans original mask members before union or tiled
  coverage, keeps source capacity, and exports one aggregate planner packet. Relation pair output reduces by source:
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
- Overlay keep-geometry-type preserves every finite positive fp64 polygon area
  and exports only final public rows; no relative threshold or source-interior
  probe discards exact slivers. Cover probes use fp64 device area masks and
  fused broadcast-capable `NativeExpression` vectors without mask export.
  Sparse exact-cache positions stay sparse, and failed device predicates decline
  to the conservative native warning path instead of CPU.
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
