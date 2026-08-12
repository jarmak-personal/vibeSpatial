# Native 100ms Physical Shape Plan

<!-- DOC_HEADER:START
Scope: Tracking plan for generalized native performance work after the ADR0044 rich baseline.
Read If: You are planning native substrate performance work, interpreting 10k shootouts, or deciding whether a change improves generalized execution.
STOP IF: You only need a local kernel implementation detail already routed by intake.
Source Of Truth: Reach-goal tracking plan for native physical workload shapes and 100ms-stage performance targets.
Body Budget: 320/320 lines
Document: docs/dev/native-100ms-physical-shape-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-14 | Intent |
| 15-25 | Request Signals |
| 26-35 | Open First |
| 36-41 | Verify |
| 42-52 | Risks |
| 53-71 | Principles |
| 72-97 | Baseline Reading |
| 98-116 | Reach Goals |
| 117-174 | Workstreams |
| 175-241 | Next Autonomous Push Queue |
| 242-259 | Acceptance |
| 260-271 | Tracking |
| 272-310 | Fresh Session Handoff |
| ... | (1 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Track the next generalized performance push around reusable physical workload
shapes, not benchmark-specific workflow tuning. The reach goal is that major
native compute stages in public GeoPandas-compatible workflows can run at
100ms or less at the relevant 10k shootout scale, and remain structurally able
to scale to new unknown workflows.

This plan exists because native carriers alone are not the goal. A change is
valuable when it makes downstream unknown work more likely to stay in a device
physical shape with explicit export boundaries.

## Request Signals

- native performance
- 100ms target
- physical workload shape
- shootout regression
- ADR0044 baseline
- materialization increase
- D2H increase
- generalized perf

## Open First

- docs/dev/native-100ms-physical-shape-plan.md
- docs/dev/native-physical-shape-ledger.md
- docs/dev/private-native-execution-substrate-plan.md
- docs/dev/native-format-library-plan.md
- docs/decisions/0044-private-native-execution-substrate.md
- docs/decisions/0046-gpu-physical-workload-shape-contracts.md
- docs/ops/intake-index.json

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run vsbench shootout benchmarks/shootout --repeat 3 --scale 10k`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Treating shootout counters as the target can produce local wins that do not
  improve unknown-work performance.
- Optimizing public object assembly can hide the need for relation, rowset,
  grouped, segment, ring, candidate-pair, or byte-shaped native execution.
- Fixed row-count thresholds can regress once geometry complexity or output
  cardinality changes.
- Native carrier preservation without stale-state tests can silently produce
  incorrect downstream composition.

## Principles

- Shootouts are guardrails, not the design target. They catch regressions and
  expose weak generalization, but they do not define the implementation shape.
- A change may be kept without a large shootout win if it improves a reusable
  physical shape, preserves a native carrier, or removes an asymptotic blocker
  for downstream work.
- Counter improvements are insufficient. Reducing D2H or materialization counts
  is useful only when it also improves wall time or preserves a better execution
  shape.
- Public GeoDataFrame, GeoSeries, pandas, Shapely, Arrow, and GeoParquet are
  ingress, fallback, debug, or terminal export surfaces. They are not hot
  internal execution currency for GPU-selected native paths.
- Native work should be shaped as relation, rowset, grouped, segment, ring,
  candidate-pair, or output-byte work where that is the real physical cost.
- Fixed row-count thresholds are bootstrap policy only. Dispatch decisions
  should move toward shape-level estimates: coordinates, segments, relation
  pairs, groups, output rows, output bytes, and temporary bytes.

## Baseline Reading

The ADR0044 rich baseline remains the floor. The August 11 branch is exact
across all 14 10k shootouts but runs at 0.826x parity versus the August 7 rich
checkpoint's 1.201x, with more explicit materialization surfaces.

The interpretation is:

- aggregate wall time is the final regression guard
- physical-shape health is the implementation target
- materialization and transfer counters are diagnostic signals
- changes that improve counters but lose wall time are rejected unless they
  remove a proven structural blocker

The high-value signal from current 10k and 1M profiles is that relation,
many/few overlay, grouped geometry, composition, and clip carriers are active,
but 1M capacity is not complete. Three shootouts request eager public joins of
6.24B, 8.06B, and 9.51B rows; their two int32 pair columns alone require
46.51GiB, 60.08GiB, and 70.84GiB. Reducible consumers must bypass those
relations. `NativeSpatialIndex.query_left_semijoin()` launches one thread per
query over its Morton-range slice, then count/scan/scatters bbox hits into a
capacity-backed prefix. Exact kernels consume its device logical count and
return `NativeDeviceSelection` without dynamic compaction. Antijoin complements
the selection on device, and clip consumes its capacity partition plus active
mask. Public joined-row export remains an explicit terminal cardinality limit.

## Reach Goals

These are intentionally aggressive. They are meant to force approach changes,
not polish existing wrappers.

| Stage family | Current issue | Reach goal |
|---|---|---:|
| Many/few overlay | Core relation-to-constructive canary is green; broader predicate-refine families still need the same shape. | <=100ms |
| Spatial join | Relation consumers are green; public joined rows remain terminal/export behavior. | <=100ms |
| Grouped geometry reduce | NativeGrouped union, disjoint-assembly, and grouped-difference canaries are green; broader dissolve cases need segmented carriers. | <=100ms |
| Copy and tabular filter | Zero-transfer canary is green; admitted pandas composition needs continued stale-state guards. | <=100ms combined |
| Mask clip and area filtering | Device rowset paths are green; terminal GEOS typing and unsupported shapes must stay explicit boundaries. | <=100ms combined |
| Dispatch shape estimates | Runtime planner accepts physical work estimates; more callers need to pass dominant work units. | Expand |
| Terminal native export | IO is a separate terminal boundary, not a compute-stage target. | Track separately |

The 100ms target applies to reusable stage families, not to every individual
line in a workflow profile. IO-heavy stages and explicit user exports should be
reported separately so they do not distort compute-shape decisions.

## Workstreams

### 1. Physical Shape Ledger

Create and maintain a table that maps each hot shootout stage to:

- current physical shape
- required physical shape
- native input carriers
- native output carrier
- public export boundary
- shape canary
- 10k and 1M profile signal

This ledger should explain why a change helps future unknown work. It should
not be a list of workflow-specific special cases.
The working ledger lives in `docs/dev/native-physical-shape-ledger.md`.

### 2. Relation Consumers

Use `NativeRelation` when downstream semantics require pair flow. Use direct
`NativeRowSet`/`NativeExpression` reduction when the consumer needs only
existence, anti-existence, counts, or another reduction. Building a relation
that the next operation immediately deduplicates is wrong physical shape.
Public joined rows remain terminal/export behavior.

Do not force small public sjoins through a slower device export path just to
improve counters. The native win is downstream relation consumption, not public
row assembly for its own sake.

### 3. Many/Few Overlay Pipeline

Reframe overlay as:

```text
NativeSpatialIndex / NativeGeometryMetadata
-> candidate NativeRelation
-> predicate/refine relation
-> constructive provenance output
-> native row/attribute projection
-> explicit terminal export
```

Early host export of candidate pairs or public index arrays is a shape break
unless the next consumer is a public export.

### 4. Grouped Geometry Reduce

Move grouped geometry work toward `NativeGrouped` as the execution state:
sorted rows, group offsets, family partitions, and segmented geometry assembly.
Avoid optimizing Shapely-shaped tree reduction as the long-term path.

### 5. Native Composition

Treat copy, projection, boolean filtering, `.iloc`/`take`, and admitted label
selection as rowset/view/projection transitions over `NativeFrameState`. Unknown
pandas operations should continue to drop native state conservatively.

## Next Autonomous Push Queue

Use this queue as the `$autonomous-execution` mandate. Work top-down unless profiling
proves a lower item is the blocker, and finish each changed carrier family.

| Priority | Remaining work | Correct shape | First acceptance gate |
|---|---|---|---|
| P0 | Active. Homogeneous direct left/right existential, anti-existence, and count consumers select range-sliced Morton reductions before relation construction. Query threads scan only their own interval slice; count/scan/scatter emits a fixed-capacity prefix, and exact kernels guard geometry work by its device logical count. Existential outputs remain `NativeDeviceSelection` through antijoin and clip consumers. Pair-preserving consumers retain relations. Mixed-family direct reduction still needs one capacity-backed family partition carrier. Eager public pair flow remains only when the API requests joined rows, with capacity failure explicit. | `NativeSpatialIndex` Morton ranges -> range-sliced candidate count/scan/scatter -> logical-count exact refine -> `NativeDeviceSelection`/`NativeExpression`; capacity-backed family partitions for mixed inputs. | Site, redevelopment, and retail homogeneous reductions complete on 24GB without a full relation and match 10K relation semantics; AST guards reject dense query-by-tree tiles and dynamic output compaction; mixed-family reduction must avoid full relation allocation. |
| P1 | Completed: fixed nested takes size from structural metadata; boundary line/point families now pack directly from part capacity, and mixed rows remain `NativeGeometryComposition` until terminal export. The old dynamic compact/regroup helpers are deleted. Continue only if a new variable nested rowset path exposes a non-terminal sizing fence. | `OwnedGeometryArray` rowset view or gathered-buffer carrier with public row order, family-local row indirection, logical coordinate sizes, and explicit terminal materialization. | Clip boundary line/point/mixed canaries assert no non-terminal boundary allocation or offset-slice fences; mixed polygon/multipolygon row-indirected GeoDataFrame/Parquet canary remains green. |
| P2 | Completed. Cover/exact-cache probes, many/few candidate relations, grouped polygon difference, collective line/polygon constructive, polygon-part explosion, boundary remnants, keep-type refinement, and public-row assembly all retain relation, grouped, rowset, part, or composition capacity. Indexed exact topology is row-indirected; named physicalization is used only where a contiguous family buffer is physically required. Host group offsets, per-row constructive loops, compact retry paths, exception-driven algorithm switches, post-hoc Shapely repair, and sparse metadata reconstruction are deleted. Exact topological equality now resolves structurally unresolved lineal and polygonal rows through bidirectional native constructive difference after bounded device physicalization, so redundant vertices and reordered multipart components remain GEOS-compatible without host topology. | `NativeSpatialIndex`/metadata -> `NativeRelation` -> predicate/refine relation -> constructive provenance -> native geometry composition/projection. | Grouped complement, collective line/polygon, pair-cache, boundary-composition, indexed-view, exception-atomicity, and full upstream overlay gates pass on the accelerator. |
| P3 | The grouped polygonal-complement output-byte carrier is implemented: exact union rows explode to polygon parts, group-local ring parents preserve nested islands, and the output builder restores Polygon/MultiPolygon rows. Mixed complement/exact, rectangle-strip/exact, bounded strip-difference fragments, and sparse touching failures assemble natively. Known-coverage union and segment extraction size from structure/capacity; the grouped reducer derives pair/carry/next-round counts algebraically and uses sparse validity/degenerate rowsets. Grouped-union validity repair passes sparse invalid-row positions directly to atomic `GPURepairResult`; the original carrier is scattered once, and incomplete repair fails the admitted native plan instead of invoking host recomputation. Grouped-union coverage failure metadata is now input-row and output-group `NativeDeviceSelection` capacity, while residual geometry remains source-row aligned, reduces through one `NativeGroupedSelection` carrier, and merges through one valid-empty coverage union. Global polygon and coverage union now lower directly to one all-observed `NativeGrouped` sorted-offset carrier; fixed observed offsets bypass dynamic group compaction, public reduction wrappers do not retry Shapely after native admission, and the duplicate dissolve tree reducer, host bbox/color decomposition, spatial sorting, pairwise retry, and empty substitution paths are deleted. Make-valid validity-expression positions, invalid family/global mappings, valid-repair filtering, and duplicate indexed-row repair remain aligned device rowsets; invalid MultiPolygon parts remain at physical capacity through repair, real groups reduce through grouped fp64 topology, and inactive lanes occupy one sentinel group whose result is discarded by a static device take. `GPURepairResult` is complete-result-only: any unrepaired requested row causes an atomic native decline, recorded before host materialization. Residual-row export, Shapely patching, reupload, and mixed scatter assembly are deleted. GPU repair establishes device state once; the dead host coordinate/offset builder and Python ring reconstruction branch are deleted. Ring closure allocates one extra coordinate per ring, duplicate removal scan/scatters within retained capacity, and ring offsets carry the logical active prefix. Invalid normalized rows polygonize through shared overlay topology rather than the old quadratic make-valid split/rebuild engine. Linework mode composes repaired polygonal area with collapsed/internal source-boundary remnants through `NativeGeometryComposition`, so GeometryCollection construction is terminal. General, outlier, bounded same-row, grouped right-right, and same-side segment sweeps classify candidate pages, emit compact event runs, release each classified page immediately, and externally merge sorted unique runs with device lower/upper bounds instead of rebuilding one classified relation or globally sorting one concatenated event array. Nonpaged classification uses the same emitter and now has explicit device-state ownership. Row-isolated topology exceeding the memory-derived live-event target returns `PagedOverlayExecutionPlan`, derives page boundaries algebraically, and builds/releases one complete-row graph at a time. Half-edge nodes use the source/twin invariant once, and exact stable radix passes replace stacked fp64 lexsort keys throughout split, graph, face assembly, candidate-pair ordering, and grouped dissolve ordering. Oversized aligned and grouped single rows with strictly separated combined polygon-part x intervals return `ComponentOverlayExecutionPlan`; part grouping and disjoint result packing retain physical capacity, while one explicit component-count admission scalar selects the Python plan variant. Each interval becomes an independent synthetic row, grouped right parts retain same-side topology, and disjoint results pack back without another union. A connected oversized aligned row now returns `MicrocellOverlayExecutionPlan`: complete x intervals page at a fixed segment-membership budget, selected trapezoids emit exact slanted atoms, vertical interfaces atomize by signed `(row, x, y)` endpoint scans, duplicate atoms cancel by streamed radix keys, and disconnected contours classify nesting before canonical half-edge polygon assembly. Aligned multirow contraction now lowers segment endpoints and exact intersection events into one device-indirected `(row, interval, segment)` relation, compacts active memberships, and computes segmented left/right parity without row-span exports or a Python row loop. Exact positive-area semantics preserve nonzero slivers. The obsolete host union-find and grouped cell-union reconstruction were deleted. Buffered two-point line dissolve deduplicates source endpoints as a device rowset, buffers once, and executes one grouped topology reduction; host bounds coloring, partial unions, tree retry, and exception-driven execution switches are deleted. Device owned concat compacts active coordinate and nested-offset prefixes into retained capacity without terminal-offset scalar exports. `NativeDeviceSelection` represents dynamic ordered positions at source capacity plus a device logical count and can rebase gathered results onto their compact active prefix without reading that count. `NativeRelationSelection` and `NativeGroupedSelection` consume that capacity directly; relation selections now physicalize pair geometry, construct, gather attributes/provenance, and return `NativeTabularSelection` at capacity. `NativeTabularSelection` preserves the exact `NativeTabularResult` invariant while carrying dynamic logical rows over a capacity result; partition concat, rename, symmetric-difference assembly, and selected source ordering remain device-only, and compact `NativeRowSet` conversion is an explicit consumer/export boundary. Generic constructive adapters preserve the capacity result instead of forcing producer-specific compaction. Shared paths now reuse same-row fp64 segment classification, orient overlap capacity by the left source, reduce forward/backward atomic lines separately, and retain two ordered MultiLineString slots in native composition until terminal GeometryCollection export; the bespoke kernels, count fence, seven intermediate exports, and Python segment loops are deleted. Segmentize now counts one lane per physical input coordinate, scans int64 contribution capacity, gathers output span offsets directly, and scatters one lane per output coordinate. Mixed-family totals cross once in a compact exact-allocation packet because contiguous owned coordinate buffers require host-sized allocation; no geometry or row metadata crosses that boundary. Legacy multi-group public reduction lowers host CSR metadata once to the same native executor. | `NativeGrouped` offsets/codes, `NativeDeviceSelection`, `NativeRelationSelection`, `NativeTabularSelection`, row-indirected polygon parts, sparse rowsets, capacity-backed ring/segment/concat buffers, streamed candidate classification, externally merged split-event runs, complete-row, interval-component, and connected or segmented-multirow microcell topology, exact boundary atoms, segmented output-byte assembly, and logical-row or ordered-collection geometry composition. | Grouped topology/reducer/global-union guards, capacity-selection CPU/static guards, relation-selection constructive and dynamic-tabular no-compaction guards, CPU-safe page/component/microcell/concat/source-contract tests, duplicate-indexed repair/static rowset guards, buffered-line single-carrier/no-switch guards, shared-path capacity and ordered-composition guards, segmentize coordinate-capacity guards, multipart linework composition canaries, residual-capacity guards, and forced-budget, hole, nested-island, and sliver GPU canaries cover the shape; accelerator execution passes across broad grouped topology and full-profile gates. |
| P4 | Completed. Polygon candidates retain inside, exact-area, boundary, lineage, and relation-coverage selections at device capacity. One-mask classification now queries a reusable segment index for boundary MBR and exact ray candidates; Morton span buckets bound scheduled lanes, count/scan/scatter keeps candidate counts on device, an explicit fp64 predicate `PrecisionPlan` governs exact orientation, and exact topology crosses one aggregate allocation packet into a compact concrete prefix before device scatter-back. Rectangle/general masks share native point/line/polygon assembly; degenerate repair is line-part shaped; lower-dimensional remnants stay in `NativeGeometryComposition`; cleanup returns `NativeTabularSelection`. Host correction probes, boundary export/reupload, logical-count admission, compact regroup, repeated semantic takes, and terminal exact rebuild are deleted. | `NativeGeometryMetadata`/`NativeExpression` -> indexed candidate relation/`NativeDeviceSelection` -> exact physicalized prefix -> native tabular, owned, or composition assembly, with GEOS typing only at public export. | Indexed-mask lane-bound/exact-ray/precision-plan, polygon/point/line/mixed, cleanup, degenerate repair, area-plus-boundary, rectangle split, grouped-mask, and no-scalar-admission canaries pass; broad clip and upstream gates pass on the accelerator. |
| P5 | Completed for the admitted composition contract. Device indexed views propagate all-valid caches without host row reads; multi-partition owned scatter fuses replacements into one row-indirected carrier; public `assign`, `__setitem__`, `insert`, concat, exact/duplicate label selection, object-backed loader deferral, numeric rowset takes, arithmetic/filter, geom-type, area, scalar-dwithin, and public Series-mask sidecars preserve exact native state. Broad `query`, `eval`, `merge`, `join`, and other unknown pandas operations intentionally drop state. Reopen only for a stale-state failure or a newly admitted exact operation. | `NativeFrameState` + `NativeRowSet`/projection transitions with exact invalidation. | Zero-transfer assignment/concat, loader deferral, duplicate-label selection, device-take scatter, and fused multi-scatter canaries stay green; stale or unknown pandas operations conservatively drop native state. |
| P6 | Every `plan_dispatch_selection` caller now supplies a physical estimate or an explicit host/bootstrap estimate. Shared carriers cover coordinates, coordinate pairs, segments, segment pairs, parts, part pairs, rings, candidate/relation pairs, groups, output rows, output bytes, and temporary bytes; authoritative device families and logical indexed expansion are used without metadata export. Buffer, validity, repair, metric, linear-reference, spatial-index, predicate, overlay, and polygon constructive wrappers report their actual scan, quadratic, relation, or bounded-output shape. Polygon buffer remains stream ordered. Grouped-difference polygon explosion and every production compute caller now use row indirection or the named non-mutating device-row physicalization boundary. Direct mutating `_device_resolve` calls are restricted to owned-carrier internals by `ARCH009`. | Shared estimates plus named native-carrier physicalization. | Runtime policy tests prove scan, quadratic pair, grouped, indexed, output, and scratch pressure can dominate without host scans or local row gates; AST audit reports zero planner calls without `work_estimate`. |
| P7 | Current checkpoint split compute, terminal, and reference counters and moved default-profile terminal geometry writes onto native device export rails. Continue here only for user-visible export breadth, not compute-path accounting. | Explicit terminal export from native carriers, measured separately from compute. | Export benchmark/canary reports wall time separately and does not hide compute-stage host work. |

### Reconciled Gate Status

The queue is active. The August 11 1M shootout has 11 successful exact
workflows and three eager joined-row capacity failures. Direct native semijoin
reductions now complete those logical selections on 24GB: site suitability is
6.64s for 350,223 rows, redevelopment is 9.18s for 414,447 rows, and retail is
11.58s for transit plus 12.23s for competitor exclusion. At 10K, the site
rowset exactly matches the 3,302 unique left rows from the 778,271-pair public
relation. The anti-join API complements the exact matched rowset on device and
the count API scans every bounded tile into one int64 value per left row. A
1M-row/32-tree-row probe evaluates 32M exact predicates and 16M matches in
59.85ms with one 24-byte planning packet and no pair carrier. Public scripts that explicitly
request billions of joined rows remain expected terminal failures until their
API contract requests a reduction.

The nullable homogeneous device-placeholder take bug is fixed; the 1M site
difference overlay now completes in 5.91s for 352,648 rows. Buffered-line
dissolve now uses bounded binary aggregate levels and improves the 1M exact
corridor materialization from 65.40s to 34.32s with the same valid native shape.
Fresh public vegetation is 34.64s and habitat is 55.95s. Across 11 exact public 1M workflows, vibeSpatial is 133.57s
versus 963.60s for the unchanged GeoPandas baseline, a 7.21x aggregate speedup.
The fresh mandatory full profile passes 11 active 1M pipelines in 0.623s
combined. No stage exceeds 73.3ms; compute has zero materializations and zero
fallbacks. Grouped topology emits 11 bounded planning packets totaling 8,376
bytes, and peak tracked device allocation is 1.38GB.

The fresh post-review 10K gate is 14/14 exact: GeoPandas is 3.553s and vibeSpatial
is 4.301s. Site is 2.47x, retail 2.18x, nearby buildings 1.74x, flood exposure
1.50x, and transit 1.20x faster. Vegetation is 860.2ms versus 294.9ms;
redevelopment is 739.0ms versus 731.0ms. The profile records 253 materializations,
705 runtime D2H events, and 143 materialization D2H events. Both totals moved
upward during storage activity; unchanged physical counters are the reliable comparison.

Regression recovery remains active: corridor/network grouped construction,
vegetation exact overlay, and repeated public composition are the largest
losses. The 0.826x result fails the rich floor; fix carrier roots without
workflow special cases or resident-data GEOS redirects.

### Queue Rules

- A native decline must precede host assembly, remain observable, and never hide
  Shapely-shaped work.
- Scalar fences are acceptable only for cheap proof/admission decisions. If a
  fence appears in profiles or sizes work from host rows, replace it with
  device metadata, sparse rowsets, or a physical work estimate.
- Public-object materialization is terminal only for user-visible result assembly;
  mid-pipeline materialization is a shape regression.
- Each completed queue item must update the ledger row, the Fresh Session
  Handoff, and at least one canary or targeted test that would fail if the path
  regressed to host-shaped compute.

## Acceptance

For a generalized performance change to count, it needs at least one of:

- a new or improved native physical-shape canary
- reduced asymptotic work for a reusable shape
- preserved native carrier through a sanctioned downstream consumer
- eliminated mid-pipeline public-object assembly
- improved dispatch decision using shape-level estimates

And it must satisfy all of:

- no silent CPU fallback
- no stale native state risk
- no benchmark-specific branches
- no loss against the ADR0044 rich baseline outside measurement noise
- no counter-only win that loses wall time without a structural reason

## Tracking

| Workstream | Shape canary | Primary guard | Status |
|---|---|---|---|
| Physical shape ledger | Ledger table | Intake routes hot stages to shapes | Complete; maintain with profiles |
| Relation consumers | Direct range-sliced Morton left/right existential and anti selections plus count expressions | 10K <=100ms; 1M bounded by tile memory and Morton intervals, not relation cardinality | Active; homogeneous range-sliced reductions and device-count outputs are complete, mixed-family capacity partitioning remains |
| Many/few overlay | Overlay relation-to-constructive profile | Many/few overlay <=100ms | Complete; canary green |
| Grouped geometry reduce | NativeGrouped union/disjoint/difference and buffered-line binary reduction profiles | 10K <=100ms; bounded fan-in at 1M | Active; general canaries green, 1M buffered corridor 34.32s |
| Native composition | Zero-transfer rowset/profile | Copy + filter <=100ms | Complete; canary green |
| Mask clip and area filtering | Predicate-heavy and clip rowset canaries | Mask/area cleanup <=100ms | Complete; canaries green |
| Terminal export | Native Arrow/Parquet profile | Report separately | Tracked separately |

## Fresh Session Handoff

- Core shape: overlay consumes relations, clip consumes rowsets, grouped reduce
  consumes `NativeGrouped`, and GeometryCollection/GEOS typing is terminal.
- Completed topology uses relation/group/segment/ring capacity, row indirection,
  paged exact events, segmented radial merges, and atomic decline after admission.
  Signed source winding deltas survive partial-overlap renoding; exact cycle
  orientation labels bounded faces; fixed-capacity indexed containment and an
  O(E) boundary peel replace face-pair probes and host convergence. Exact
  construction preserves every positive fp64 sliver. Do not restore tolerances,
  host regroup, retry, semantic repair, or Shapely repair paths.
- Runtime ordering is explicit across driver, pylibcudf, and CCCL on the active
  CuPy stream. Planner calls carry authoritative physical estimates, and
  `ARCH009` confines mutating device resolution to owned internals.
- Rich 10K repeat-3 (August 7, 2026): 14/14 exact, GeoPandas 3328.6ms versus
  vibeSpatial 2771.8ms (1.201x), zero failures. Site suitability is 3.53x,
  retail 2.19x, redevelopment 2.00x, and vegetation 1.16x faster than
  GeoPandas.
- Current 10K repeat-3 (August 11, 2026): 14/14 exact, GeoPandas 3552.8ms versus
  vibeSpatial 4301.4ms (0.826x). Site is 2.47x, retail 2.18x, nearby 1.74x, flood
  1.50x, and transit 1.20x faster. Exact-prefix physicalization lowers vegetation
  from 1169.0ms to 860.2ms. The profile records 253 materializations, 705 runtime
  D2H events, and 143 materialization D2H events.
- August 11 1M capacity checkpoint: 11 public shootouts complete exactly in
  133.57s versus 963.60s for GeoPandas, a 7.21x aggregate speedup. Site, redevelopment, and retail
  stop before public export because their eager joins contain 9.51B, 8.06B,
  and 6.24B pairs. The direct native semijoin carrier completes their intended
  unique-left reductions in 6.64s, 9.18s, and 11.58s/12.23s respectively.
  Vegetation corridor is no longer a `make_valid` issue: deferred exact union
  dominates, and bounded binary aggregate reduction lowers it from 65.40s to
  34.32s; the fresh public workflow is 34.64s. The full mandatory profile
  passes all 11 active 1M pipelines with no stage above 73.3ms, zero compute
  materializations/fallbacks, and 8,376 bytes of bounded grouped planning
  packets.
- Correctness gates pass: strict-native upstream is 1,971 passed / 423 skipped /
  5 xfailed; contract health passes every surface; the focused carrier suite is
  612 passed; and the uninterrupted local plus vendored-upstream suite is 7,028
  passed / 434 skipped / 7 xfailed with zero failures.

## Completion State

The PRD is active. Immediate remaining work is root-cause recovery to at least
the 1.201x rich 10K floor, beginning with corridor/network grouped constructive
reduction, vegetation exact overlay, and repeated public
composition/materialization. Broad compatibility verification remains part of
each landed recovery. Eager public
joins with multi-billion output rows are an explicit terminal cardinality
limit on a 24GB device; this does not block reduced native consumers, but it
cannot be described as a successful public 1M export.
