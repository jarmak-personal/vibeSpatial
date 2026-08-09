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
| 72-94 | Baseline Reading |
| 95-113 | Reach Goals |
| 114-169 | Workstreams |
| 170-203 | Next Autonomous Push Queue |
| 204-221 | Acceptance |
| 222-233 | Tracking |
| 234-313 | Fresh Session Handoff |
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

The ADR0044 rich baseline remains the floor for public workflow performance.
The current native branch is already faster in aggregate on the 10k shootouts,
but it achieves that while exposing more explicit materialization surfaces.

The interpretation is:

- aggregate wall time is the final regression guard
- physical-shape health is the implementation target
- materialization and transfer counters are diagnostic signals
- changes that improve counters but lose wall time are rejected unless they
  remove a proven structural blocker

The high-value signal from current 10k and 1M profiles is that the first
reusable shapes are now active: relation consumers, many/few overlay,
grouped geometry reduce, native composition, and mask clip all have green
canaries. The remaining frontier is expanding those shapes beyond the current
admitted paths while keeping public-object assembly terminal. Runtime dispatch
now has a reusable `PhysicalWorkEstimate` carrier, so new native paths should
feed coordinates, segments, pairs, groups, rows, or byte estimates into the
shared planner instead of adding operation-local row thresholds.

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

Make `NativeRelation` the default internal currency for spatial join consumers:
semijoin, anti-join, grouped counts, relation projection, and relation-backed
attribute reduction. Public joined rows should be terminal/export behavior.

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
| P0 | Keep the ledger and handoff rebased after each autonomous push. Classify every >100ms total as native compute, setup, reference/oracle, or terminal export. | Profile rows mapped to ledger stage families, not benchmark names. | Updated ledger/handoff with stage names, times, fallback counts, and terminal-export classification. |
| P1 | Completed: fixed nested takes size from structural metadata; boundary line/point families now pack directly from part capacity, and mixed rows remain `NativeGeometryComposition` until terminal export. The old dynamic compact/regroup helpers are deleted. Continue only if a new variable nested rowset path exposes a non-terminal sizing fence. | `OwnedGeometryArray` rowset view or gathered-buffer carrier with public row order, family-local row indirection, logical coordinate sizes, and explicit terminal materialization. | Clip boundary line/point/mixed canaries assert no non-terminal boundary allocation or offset-slice fences; mixed polygon/multipolygon row-indirected GeoDataFrame/Parquet canary remains green. |
| P2 | Completed. Cover/exact-cache probes, many/few candidate relations, grouped polygon difference, collective line/polygon constructive, polygon-part explosion, boundary remnants, keep-type refinement, and public-row assembly all retain relation, grouped, rowset, part, or composition capacity. Indexed exact topology is row-indirected; named physicalization is used only where a contiguous family buffer is physically required. Host group offsets, per-row constructive loops, compact retry paths, exception-driven algorithm switches, post-hoc Shapely repair, and sparse metadata reconstruction are deleted. Exact topological equality now resolves structurally unresolved lineal and polygonal rows through bidirectional native constructive difference after bounded device physicalization, so redundant vertices and reordered multipart components remain GEOS-compatible without host topology. | `NativeSpatialIndex`/metadata -> `NativeRelation` -> predicate/refine relation -> constructive provenance -> native geometry composition/projection. | Grouped complement, collective line/polygon, pair-cache, boundary-composition, indexed-view, exception-atomicity, and full upstream overlay gates pass on the accelerator. |
| P3 | The grouped polygonal-complement output-byte carrier is implemented: exact union rows explode to polygon parts, group-local ring parents preserve nested islands, and the output builder restores Polygon/MultiPolygon rows. Mixed complement/exact, rectangle-strip/exact, bounded strip-difference fragments, and sparse touching failures assemble natively. Known-coverage union and segment extraction size from structure/capacity; the grouped reducer derives pair/carry/next-round counts algebraically and uses sparse validity/degenerate rowsets. Grouped-union validity repair passes sparse invalid-row positions directly to atomic `GPURepairResult`; the original carrier is scattered once, and incomplete repair fails the admitted native plan instead of invoking host recomputation. Grouped-union coverage failure metadata is now input-row and output-group `NativeDeviceSelection` capacity, while residual geometry remains source-row aligned, reduces through one `NativeGroupedSelection` carrier, and merges through one valid-empty coverage union. Global polygon and coverage union now lower directly to one all-observed `NativeGrouped` sorted-offset carrier; fixed observed offsets bypass dynamic group compaction, public reduction wrappers do not retry Shapely after native admission, and the duplicate dissolve tree reducer, host bbox/color decomposition, spatial sorting, pairwise retry, and empty substitution paths are deleted. Make-valid validity-expression positions, invalid family/global mappings, valid-repair filtering, and duplicate indexed-row repair remain aligned device rowsets; invalid MultiPolygon parts remain at physical capacity through repair, real groups reduce through grouped fp64 topology, and inactive lanes occupy one sentinel group whose result is discarded by a static device take. `GPURepairResult` is complete-result-only: any unrepaired requested row causes an atomic native decline, recorded before host materialization. Residual-row export, Shapely patching, reupload, and mixed scatter assembly are deleted. GPU repair establishes device state once; the dead host coordinate/offset builder and Python ring reconstruction branch are deleted. Ring closure allocates one extra coordinate per ring, duplicate removal scan/scatters within retained capacity, and ring offsets carry the logical active prefix. Invalid normalized rows polygonize through shared overlay topology rather than the old quadratic make-valid split/rebuild engine. Linework mode composes repaired polygonal area with collapsed/internal source-boundary remnants through `NativeGeometryComposition`, so GeometryCollection construction is terminal. General, outlier, bounded same-row, grouped right-right, and same-side segment sweeps classify candidate pages, emit compact event runs, release each classified page immediately, and externally merge sorted unique runs with device lower/upper bounds instead of rebuilding one classified relation or globally sorting one concatenated event array. Nonpaged classification uses the same emitter and now has explicit device-state ownership. Row-isolated topology exceeding the memory-derived live-event target returns `PagedOverlayExecutionPlan`, derives page boundaries algebraically, and builds/releases one complete-row graph at a time. Half-edge nodes use the source/twin invariant once, and exact stable radix passes replace stacked fp64 lexsort keys throughout split, graph, face assembly, candidate-pair ordering, and grouped dissolve ordering. Oversized aligned and grouped single rows with strictly separated combined polygon-part x intervals return `ComponentOverlayExecutionPlan`; part grouping and disjoint result packing retain physical capacity, while one explicit component-count admission scalar selects the Python plan variant. Each interval becomes an independent synthetic row, grouped right parts retain same-side topology, and disjoint results pack back without another union. A connected oversized aligned row now returns `MicrocellOverlayExecutionPlan`: complete x intervals page at a fixed segment-membership budget, selected trapezoids emit exact slanted atoms, vertical interfaces atomize by signed `(row, x, y)` endpoint scans, duplicate atoms cancel by streamed radix keys, and disconnected contours classify nesting before canonical half-edge polygon assembly. Aligned multirow contraction now lowers segment endpoints and exact intersection events into one device-indirected `(row, interval, segment)` relation, compacts active memberships, and computes segmented left/right parity without row-span exports or a Python row loop. Exact positive-area semantics preserve nonzero slivers. The obsolete host union-find and grouped cell-union reconstruction were deleted. Buffered two-point line dissolve deduplicates source endpoints as a device rowset, buffers once, and executes one grouped topology reduction; host bounds coloring, partial unions, tree retry, and exception-driven execution switches are deleted. Device owned concat compacts active coordinate and nested-offset prefixes into retained capacity without terminal-offset scalar exports. `NativeDeviceSelection` represents dynamic ordered positions at source capacity plus a device logical count and can rebase gathered results onto their compact active prefix without reading that count. `NativeRelationSelection` and `NativeGroupedSelection` consume that capacity directly; relation selections now physicalize pair geometry, construct, gather attributes/provenance, and return `NativeTabularSelection` at capacity. `NativeTabularSelection` preserves the exact `NativeTabularResult` invariant while carrying dynamic logical rows over a capacity result; partition concat, rename, symmetric-difference assembly, and selected source ordering remain device-only, and compact `NativeRowSet` conversion is an explicit consumer/export boundary. Generic constructive adapters preserve the capacity result instead of forcing producer-specific compaction. Shared paths now reuse same-row fp64 segment classification, orient overlap capacity by the left source, reduce forward/backward atomic lines separately, and retain two ordered MultiLineString slots in native composition until terminal GeometryCollection export; the bespoke kernels, count fence, seven intermediate exports, and Python segment loops are deleted. Segmentize now counts one lane per physical input coordinate, scans int64 contribution capacity, gathers output span offsets directly, and scatters one lane per output coordinate. Mixed-family totals cross once in a compact exact-allocation packet because contiguous owned coordinate buffers require host-sized allocation; no geometry or row metadata crosses that boundary. Legacy multi-group public reduction lowers host CSR metadata once to the same native executor. | `NativeGrouped` offsets/codes, `NativeDeviceSelection`, `NativeRelationSelection`, `NativeTabularSelection`, row-indirected polygon parts, sparse rowsets, capacity-backed ring/segment/concat buffers, streamed candidate classification, externally merged split-event runs, complete-row, interval-component, and connected or segmented-multirow microcell topology, exact boundary atoms, segmented output-byte assembly, and logical-row or ordered-collection geometry composition. | Grouped topology/reducer/global-union guards, capacity-selection CPU/static guards, relation-selection constructive and dynamic-tabular no-compaction guards, CPU-safe page/component/microcell/concat/source-contract tests, duplicate-indexed repair/static rowset guards, buffered-line single-carrier/no-switch guards, shared-path capacity and ordered-composition guards, segmentize coordinate-capacity guards, multipart linework composition canaries, residual-capacity guards, and forced-budget, hole, nested-island, and sliver GPU canaries cover the shape; accelerator execution passes across broad grouped topology and full-profile gates. |
| P4 | Completed. Polygon device candidates retain inside, exact-area, positive-area, boundary, source-lineage, and relation-coverage partitions as `NativeDeviceSelection` capacity. Rectangle and general polygon masks share native point/line/polygon assembly; degenerate line repair is line-part shaped; area plus lower-dimensional remnants stay in `NativeGeometryComposition`; semantic cleanup returns `NativeTabularSelection`; and grouped masks use one aggregate admission packet that never sizes geometry. Host correction probes, boundary-row export/reupload, compact regroup, repeated semantic takes, and terminal exact rebuild are deleted. Dynamic public results perform only the exact count read required by pandas, while possible GeometryCollection rows perform a terminal multiplicity certification before owned-device physicalization; neither is compute-path assembly. | `NativeGeometryMetadata`/`NativeExpression` -> `NativeRelation`/`NativeDeviceSelection` -> exact or dynamic native tabular, owned, or composition assembly, with GEOS typing only at public export. | Polygon/point/line/mixed, semantic cleanup, degenerate repair, area-plus-boundary, rectangle split, grouped-mask, and no-scalar-admission canaries pass; broad clip and upstream gates pass on the accelerator. |
| P5 | Completed for the admitted composition contract. Device indexed views propagate all-valid caches without host row reads; multi-partition owned scatter fuses replacements into one row-indirected carrier; public `assign`, `__setitem__`, `insert`, concat, exact/duplicate label selection, object-backed loader deferral, numeric rowset takes, arithmetic/filter, geom-type, area, scalar-dwithin, and public Series-mask sidecars preserve exact native state. Broad `query`, `eval`, `merge`, `join`, and other unknown pandas operations intentionally drop state. Reopen only for a stale-state failure or a newly admitted exact operation. | `NativeFrameState` + `NativeRowSet`/projection transitions with exact invalidation. | Zero-transfer assignment/concat, loader deferral, duplicate-label selection, device-take scatter, and fused multi-scatter canaries stay green; stale or unknown pandas operations conservatively drop native state. |
| P6 | Every `plan_dispatch_selection` caller now supplies a physical estimate or an explicit host/bootstrap estimate. Shared carriers cover coordinates, coordinate pairs, segments, segment pairs, parts, part pairs, rings, candidate/relation pairs, groups, output rows, output bytes, and temporary bytes; authoritative device families and logical indexed expansion are used without metadata export. Buffer, validity, repair, metric, linear-reference, spatial-index, predicate, overlay, and polygon constructive wrappers report their actual scan, quadratic, relation, or bounded-output shape. Polygon buffer remains stream ordered. Grouped-difference polygon explosion and every production compute caller now use row indirection or the named non-mutating device-row physicalization boundary. Direct mutating `_device_resolve` calls are restricted to owned-carrier internals by `ARCH009`. | Shared estimates plus named native-carrier physicalization. | Runtime policy tests prove scan, quadratic pair, grouped, indexed, output, and scratch pressure can dominate without host scans or local row gates; AST audit reports zero planner calls without `work_estimate`. |
| P7 | Current checkpoint split compute, terminal, and reference counters and moved default-profile terminal geometry writes onto native device export rails. Continue here only for user-visible export breadth, not compute-path accounting. | Explicit terminal export from native carriers, measured separately from compute. | Export benchmark/canary reports wall time separately and does not hide compute-stage host work. |

### Reconciled Gate Status

The physical-shape queue is complete. Broad compatibility, 10K repeat-3, and full
1M gates are green. Reopen only for measured host-shaped compute or a >100ms stage.

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
| Relation consumers | Native relation semijoin/reduce profiles | Spatial join stage <=100ms | Complete; canary green |
| Many/few overlay | Overlay relation-to-constructive profile | Many/few overlay <=100ms | Complete; canary green |
| Grouped geometry reduce | NativeGrouped union/disjoint/difference profiles | Grouped reduce <=100ms | Complete; canaries green |
| Native composition | Zero-transfer rowset/profile | Copy + filter <=100ms | Complete; canary green |
| Mask clip and area filtering | Predicate-heavy and clip rowset canaries | Mask/area cleanup <=100ms | Complete; canaries green |
| Terminal export | Native Arrow/Parquet profile | Report separately | Tracked separately |

## Fresh Session Handoff

- Core shape: overlay consumes relations, clip consumes rowsets, grouped reduce
  consumes `NativeGrouped`, and GeometryCollection/GEOS typing is terminal.
- P2: grouped polygon difference treats containment and collective coverage as
  topology. Rectangle-hole, polygon-hole, and polygon-donut builders now return
  public-row-capacity results plus device support masks. Exact topology consumes
  the complementary metadata-masked groups once, and one device index map fuses
  all direct/exact partitions without scalar admission, group compaction, or the
  former two-int group-size export. Device pairs radix-sort once into dense
  source-row `NativeGrouped` offsets; zero spans preserve no-neighbor rows, and
  external unique-left batching, pool trims, concatenation, and scatter are
  deleted because grouped topology owns paging. The aligned polygon router uses
  five device-counted capacity partitions, including non-rectangle and
  `keep_geom_type=False` batches, before native boundary composition. Pairwise
  line/polygon work shares split-event capacity with row-indirected null lanes.
- P3: paged split events, exact radix half-edge assembly, component plans, and
  connected microcell boundary reconstruction are implemented. Multirow
  contraction uses a segmented device relation instead of exporting row spans
  and iterating rows on the host. Buffered-line dissolve and invalid grouped
  output repair are one-shot grouped/atomic carriers with no execution retry.
  Device concat/gather and physical take, polygon boundary/interiors, bounded
  SH/rectangle/contained-hole and buffer assembly, line merge, point-pair and
  point/line part expansion, and make-valid retain capacity/logical lengths
  without scalar-sized allocation. Generic capacity offset-slice gathering now
  uses guarded row-range kernels, so inactive allocation lanes never feed
  search/gather indices. Its lower-level slice planner requires structure or
  capacity; unknown-size generic value gathers require a caller-owned reason,
  and only mixed WKT/KML/GeoJSON input-family assembly uses that boundary.
  Line/polygon construction now uses shared split-event capacity across all
  rings and parts; its first-ring count/scatter kernel and allocation fences are
  deleted. Segmentize now counts and scans physical coordinate lanes, batches
  all family totals into one exact-allocation packet, and scatters output lanes
  directly; the packet controls contiguous device allocation and carries no
  geometry or row metadata.
  Aligned line-line intersection consumes the shared same-row segment classifier page-by-page and carries mixed results through ordered native composition. Repeated-right variable-width polygon, point, and line consumers use fixed/max structural metadata or conservative row-indirected bounds; explode no longer physicalizes and retries.
  Segmentize uses constructive precision policy; make-valid output stays native. Row-isolated face output extracts one selected-side boundary-cycle carrier: positive cycles are shells and negative cycles are holes. The inverse excluded-face carrier, full-capacity ring merge, collapsed-excluded repair, and duplicate-hole signature kernels are deleted.
  Face assembly is device-only after native admission; the selected-face host bridge is explicit debug/export. Geometry-only spatial overlay lowers device candidate columns atomically into canonical `NativeRelation`, rejects host-backed pair carriers, radix-orders that relation, executes pair intersections collectively, and runs each difference side through one full-source grouped topology plan.
  Identity, planar union, and symmetric difference compose those native components. The old pairwise-union semantics, no-candidate early return, per-group stream executor, boundary exports, centroid rescue, and Shapely fallback are deleted; the explicit CPU oracle is isolated in `overlay.host_fallback`.
  Grouped rectangle-strip/exact and positive-area/degenerate routing are complementary capacity partitions with no two-flag packet. Grouped-union residual closure retains one initial logical-count admission to avoid an empty regroup/merge; the duplicate post-repair difference/area pass and scalar are deleted because exact closure is `C union (I difference C)`.
  Compact `NativeRowSet` conversion occurs only at the explicit `NativeTabularSelection` consumer/export boundary; structural constructive failures remain atomic.
- P6: every planner call carries physical shape, prefers authoritative device
  families, and scales indexed logical rows. Compute uses row indirection or named
  physicalization; `ARCH009` confines `_device_resolve` to owned internals.
- Unary stroke admission physicalizes resident indexed logical rows before family dispatch; unreferenced ancestral buffers cannot trigger mixed fallback, and admitted kernel failures are atomic.
- P4: polygon clip predicates carry indexed source rows and candidate-local outputs
  through device-counted capacity selections; rectangle specialization does not compact or scalar-probe tags.
  Grouped-mask clip keeps covered/unresolved source and predicate-pair capacities; one
  two-int plan packet never sizes geometry. Pair intersections reduce by source:
  valid-empty grouped polygon identities, area-subtracted/noded atomic line edges,
  deduplicated points with area/line suppression, and terminal native composition.
  Mixed-family constructive physicalizes family-pair rows at public capacity
  and fuses one row-indirected result; trusted family-domain and unique-row
  proofs avoid ordinary semantic probes. Clip cleanup combines nonempty, area,
  and keep-type filters in one capacity selection. Degenerate line parts use
  segmented exact deduplication and compensated fp64 Point-capacity reduction.
- Current instrumented 10K repeat-3 checkpoint (August 7, 2026): 14/14 exact,
  GeoPandas 3527.5ms versus vibeSpatial 2826.2ms (1.248x), zero fallbacks,
  219 total materializations, and 467 runtime D2H transfers. The 211
  stage-attributed materializations are 154 public exports and 57 explicit
  public-operation compatibility conversions; 24 of the latter are eight-byte
  dynamic row-count reads required to construct exact-length pandas results.
  These totals are boundary diagnostics, not native-compute work. Site
  suitability is 3.78x, retail screening 2.29x, redevelopment 2.04x, and
  vegetation 1.19x faster than GeoPandas.
- Full 1M sparkline (August 7, 2026): maximum active stage 70.50ms; grouped
  disjoint setup 65.88ms, mixed union 50.64ms, grouped union 33.91ms, grouped
  difference 10.86ms, and relation intersection 1.84ms. Native compute reports
  zero materializations and zero D2H; terminal output reports one
  materialization and zero runtime D2H at 1M. Two raster cases remain expected
  feature deferrals.
- Correctness hardening now covers equal-cardinality reordered indexed metrics,
  copy-on-write native attribute mutation, detached native-expression assignment,
  canonical MultiLineString/MultiPoint normalization, strict-interior polygon
  representative points, and GEOS-compatible lineal/polygonal topological
  equality. The complete upstream overlay gate passes 128 tests and the
  remaining upstream tail passes 419 tests. The final uninterrupted local and
  vendored-upstream suite passes 6,979 tests with 434 optional-dependency skips,
  7 expected xfails, and zero failures.

## Completion State

The PRD is complete when deterministic gates and the uninterrupted full suite
remain green at landing. The measured acceptance evidence above already proves
the physical-shape conditions: no native compute materialization or D2H, no
fallbacks in the 10K or full profile, no active 1M stage above 100ms, exact 10K
fingerprints, and only explicit public compatibility/export boundaries.
