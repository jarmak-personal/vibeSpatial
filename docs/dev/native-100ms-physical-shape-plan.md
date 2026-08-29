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
| 72-100 | Baseline Reading |
| 101-119 | Reach Goals |
| 120-132 | Workstreams |
| 133-230 | Next Autonomous Push Queue |
| 231-248 | Acceptance |
| 249-260 | Tracking |
| 261-308 | Fresh Session Handoff |
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

The ADR0044 rich baseline remains the floor. The August 12 branch is exact
across all 14 10k shootouts and runs at 1.271x aggregate parity. The latest
repeat-3 gate is 2.800s for vibeSpatial versus 3.559s for GeoPandas. Both
runtimes were 4-5% slower immediately after the full suite, so the aggregate
ratio is stable while the machine-sensitive absolute result is 1.0% above the
2.772s rich checkpoint.

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
| Mask clip and area filtering | Device rowset paths are green. Lazy grouped masks now plan from original members, selecting exact relation/group reduction or one-time union before topology work; habitat 100K is exact at 2.5196 s versus 2.4663 s GeoPandas, down from 14.1656 s. Terminal GEOS typing and unsupported shapes remain explicit boundaries. | <=100ms combined |
| Dispatch shape estimates | Runtime planner accepts physical work estimates; more callers need to pass dominant work units. | Expand |
| Terminal native export | IO is a separate terminal boundary, not a compute-stage target. | Track separately |

The 100ms target applies to reusable stage families, not to every individual
line in a workflow profile. IO-heavy stages and explicit user exports should be
reported separately so they do not distort compute-shape decisions.

## Workstreams

The detailed map is `docs/dev/native-physical-shape-ledger.md`. Its contracts are:

- Pair-preserving consumers use `NativeRelation`; reductions use
  `NativeDeviceSelection`, `NativeRowSet`, or `NativeExpression` before pair allocation.
- Overlay remains index/metadata -> candidate relation -> refine -> constructive
  provenance -> native projection -> explicit export.
- Grouped constructive work uses `NativeGrouped` offsets and family partitions;
  grouped-mask clip admits original-member relations before pages or tiles.
- Copy, filtering, and admitted selection remain `NativeFrameState` transitions;
  unknown pandas operations conservatively drop native state.

## Next Autonomous Push Queue

Use this queue as the `$autonomous-execution` mandate. Work top-down unless profiling
proves a lower item is the blocker, and finish each changed carrier family.

| Priority | Remaining work | Correct shape | First acceptance gate |
|---|---|---|---|
| P0 | Completed. Direct left/right existential, anti-existence, and count consumers select range-sliced Morton reductions before relation construction. Query threads scan only their own interval slice; count/scan/scatter emits a fixed-capacity prefix, and exact kernels guard geometry work by its device logical count. Mixed inputs classify active candidates once into an all-family grouped relation partition; family classifiers consume shared sorted pair arrays through device offsets/counts, with aggregate launch capacity bounded by one tile. Existential outputs remain `NativeDeviceSelection` through antijoin and clip consumers. Pair-preserving consumers retain relations, and eager public pair flow occurs only when the API requests joined rows. | `NativeSpatialIndex` Morton ranges -> range-sliced candidate count/scan/scatter -> one grouped family partition -> logical-count exact refine -> `NativeDeviceSelection`/`NativeExpression`. | Site, redevelopment, and retail reductions complete on 24GB without a full relation and match 10K relation semantics; AST guards reject dense query-by-tree tiles and dynamic output compaction; the 1M mixed-family canary proves one partition pass, tile-bounded aggregate classifier capacity, and zero D2H. |
| P1 | Completed: fixed nested takes size from structural metadata; boundary line/point families now pack directly from part capacity, and mixed rows remain `NativeGeometryComposition` until terminal export. The old dynamic compact/regroup helpers are deleted. Continue only if a new variable nested rowset path exposes a non-terminal sizing fence. | `OwnedGeometryArray` rowset view or gathered-buffer carrier with public row order, family-local row indirection, logical coordinate sizes, and explicit terminal materialization. | Clip boundary line/point/mixed canaries assert no non-terminal boundary allocation or offset-slice fences; mixed polygon/multipolygon row-indirected GeoDataFrame/Parquet canary remains green. |
| P2 | Completed. Cover/exact-cache probes, many/few candidate relations, grouped polygon difference, collective line/polygon constructive, polygon-part explosion, boundary remnants, keep-type refinement, and public-row assembly all retain relation, grouped, rowset, part, or composition capacity. Indexed exact topology is row-indirected; named physicalization is used only where a contiguous family buffer is physically required. Host group offsets, per-row constructive loops, compact retry paths, exception-driven algorithm switches, post-hoc Shapely repair, and sparse metadata reconstruction are deleted. Exact topological equality now resolves structurally unresolved lineal and polygonal rows through bidirectional native constructive difference after bounded device physicalization, so redundant vertices and reordered multipart components remain GEOS-compatible without host topology. | `NativeSpatialIndex`/metadata -> `NativeRelation` -> predicate/refine relation -> constructive provenance -> native geometry composition/projection. | Grouped complement, collective line/polygon, pair-cache, boundary-composition, indexed-view, exception-atomicity, and full upstream overlay gates pass on the accelerator. |
| P3 | Grouped polygonal complement, exact union, rectangle-strip, degenerate, make-valid, paged topology, tiled collective union, and dynamic tabular assembly are native. Scalar and grouped coverage reduction share the canonical grouped noded-boundary assembler. Completed grouped outputs now publish a valid-family-row injectivity proof; disjoint Polygon/MultiPolygon merges consume physical root part/ring/coordinate capacity directly instead of exporting exact row-width packets or allocating `rows * conservative_width`. The 1M grouped-capacity canary has zero compute D2H/materialization and a 32MiB peak. Continue only when profiles expose a new grouped topology shape rather than extending legacy retries. | `NativeGrouped` offsets/codes, injective row-indirected owned carriers, `NativeDeviceSelection`, `NativeGroupedSelection`, streamed candidate pages, and exact output-byte assembly. | Grouped topology/reducer/global-union, mixed strip/exact, positive/degenerate, residual-capacity, make-valid, page/component/microcell, hole/island/sliver, scalar/grouped coverage equivalence, and full-profile gates pass; the grouped-capacity stages publish `trusted_unique_family_rows`. |
| P4 | Completed. Polygon candidates retain inside, exact-area, boundary, lineage, and relation-coverage selections at device capacity. One-mask classification now queries a reusable segment index for boundary MBR and exact ray candidates; Morton span buckets bound scheduled lanes, count/scan/scatter keeps candidate counts on device, an explicit fp64 predicate `PrecisionPlan` governs exact orientation, and exact topology crosses one aggregate allocation packet into a compact concrete prefix before device scatter-back. Rectangle/general masks share native point/line/polygon assembly; degenerate repair is line-part shaped; lower-dimensional remnants stay in `NativeGeometryComposition`; cleanup returns `NativeTabularSelection`. Host correction probes, boundary export/reupload, logical-count admission, compact regroup, repeated semantic takes, and terminal exact rebuild are deleted. | `NativeGeometryMetadata`/`NativeExpression` -> indexed candidate relation/`NativeDeviceSelection` -> exact physicalized prefix -> native tabular, owned, or composition assembly, with GEOS typing only at public export. | Indexed-mask lane-bound/exact-ray/precision-plan, polygon/point/line/mixed, cleanup, degenerate repair, area-plus-boundary, rectangle split, grouped-mask, and no-scalar-admission canaries pass; broad clip and upstream gates pass on the accelerator. |
| P5 | Completed for the admitted composition contract. Device indexed views propagate all-valid caches without host row reads; multi-partition owned scatter fuses replacements into one row-indirected carrier; public `assign`, `__setitem__`, `insert`, concat, exact/duplicate label selection, object-backed loader deferral, numeric rowset takes, arithmetic/filter, geom-type, area, scalar-dwithin, and public Series-mask sidecars preserve exact native state. Broad `query`, `eval`, `merge`, `join`, and other unknown pandas operations intentionally drop state. Reopen only for a stale-state failure or a newly admitted exact operation. | `NativeFrameState` + `NativeRowSet`/projection transitions with exact invalidation. | Zero-transfer assignment/concat, loader deferral, duplicate-label selection, device-take scatter, and fused multi-scatter canaries stay green; stale or unknown pandas operations conservatively drop native state. |
| P6 | Every `plan_dispatch_selection` caller now supplies a physical estimate or an explicit host/bootstrap estimate. Shared carriers cover coordinates, coordinate pairs, segments, segment pairs, parts, part pairs, rings, candidate/relation pairs, groups, output rows, output bytes, and temporary bytes; authoritative device families and logical indexed expansion are used without metadata export. Buffer, validity, repair, metric, linear-reference, spatial-index, predicate, overlay, and polygon constructive wrappers report their actual scan, quadratic, relation, or bounded-output shape. Polygon buffer remains stream ordered. Grouped-difference polygon explosion and every production compute caller now use row indirection or the named non-mutating device-row physicalization boundary. Direct mutating `_device_resolve` calls are restricted to owned-carrier internals by `ARCH009`. | Shared estimates plus named native-carrier physicalization. | Runtime policy tests prove scan, quadratic pair, grouped, indexed, output, and scratch pressure can dominate without host scans or local row gates; AST audit reports zero planner calls without `work_estimate`. |
| P7 | Active only for remaining export breadth. Terminal WKB export now certifies valid composition-page spans with one compact device packet, ignores all-invalid capacity carriers, emits exact physical pages whole, and bounds only partial indexed pages. Physical pages size WKB from exact coordinate/ring totals rather than `largest row x page rows`. Certified contiguous GeoParquet read compositions bypass row-count-sized multiplicity selectors. Intersection-pair cache retention is byte-bounded LRU state. | Explicit page-shaped terminal export from native composition and row-group carriers, measured separately from compute. | The 30.55M-row transit output writes and reads exactly on 24GB without whole-result physicalization; export benchmark/canary reports wall time separately and does not hide compute-stage host work. |

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

The nullable homogeneous device-placeholder take bug is fixed; the historical
1M site difference overlay completes in 5.91s for 352,648 rows. The former
buffered-line binary tree and one-dimensional slab planner are deleted.
Single-group collective union now builds a two-dimensional segment-to-tile
`NativeRelation`, proves full tiles by scanline coverage, and sends only sparse
boundary tiles through exact rectangle clipping and local topology. Tile size
is derived from measured local segment-peer pressure rather than source-row
count. Active neighboring tiles now enter one relation-grouped clip/topology
plan until their summed segment-peer pressure reaches the existing topology
budget; the compact offset and pressure vectors cross in one planning packet.
Dense tiles remain isolated. Every local and seam result is device-physicalized to its exact concrete
prefix before retention, so bounded live seam objects no longer retain nested
candidate-capacity coordinate buffers. Split-event paging retains its original
throughput budget while bounding the upstream candidate page by the live event
capacity. Same-row paired events request fixed-point renoding only when their
fp64 coordinate keys expose an actual ULP-scale planarity risk. Coverage-only
face containment adapts on device between eight shallow roots per block and one
256-lane traversal for a lone large root. Singular union and reusable coverage
now have exclusive ownership: the singular path retains only its online seam
levels, while the coverage path retains only its downstream tile carrier.

The former 1M vegetation OOM completes exactly in 52.73s in the comparable
timed-section harness, with the expected 100,000 rows and fingerprint. Against
the historical 359.83s GeoPandas reference this is 6.82x. A
20K isolated collective probe completes in 58.44s with 2.78GB peak tracked
allocation; the earlier carrier retained more than 24GB and failed at 1M.
Habitat now consumes reusable tiled mask coverage and completes exactly in
38.18s with the seven-row reference fingerprint, versus the recent 39.17s
checkpoint and historical 157.49s GeoPandas reference. Transit
now completes its exact 30,553,577-row public write/read in 191.44s versus the
historical 411s GeoPandas gate. Its former terminal OOM was a wrong-shaped WKB
page admission and read-composition selector, not topology capacity.

The August 27 mandatory full profile passes every active 1M pipeline with zero
compute materializations, D2H packets, or fallbacks. The 72.33ms maximum is
mixed-strip exact union; other exact stages are 67.07ms positive-degenerate
union, 51.70ms small-group union, 10.69ms grouped difference, 2.86ms
relation-overlay constructive, and 0.58ms point-in-polygon.

The current 10K repeat-3 gate is 14/14 exact: GeoPandas is 3.559s and
vibeSpatial is 2.800s, or 1.271x aggregate. Transit is 191.0ms versus 239.9ms,
vegetation is 187.4ms versus 304.0ms, and site suitability is 187.8ms versus
657.6ms. The statement-level profile reports 239 materialization events: 175
explicit user exports and 64 internal host conversions. It also reports 614
D2H stage events across fixture IO, planning packets, public assembly, terminal IO,
and verification. Those whole-profile counters are not comparable to the older
compute-only 24/66 checkpoint; the 64 internal conversions are the actionable
debt. Its absolute wall time is 1.0% above the 2.772s rich floor.

The PRD remains active. Relation-grouped multi-tile construction, reusable tile
coverage, mixed-family reductions, zero-transfer compute, and ring-local
winding are complete. The 64 internal host conversions follow. Do
not add resident-data GEOS redirects or workflow branches.

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
| Relation consumers | Direct range-sliced Morton left/right existential and anti selections plus count expressions | 10K <=100ms; 1M bounded by tile memory and Morton intervals, not relation cardinality | Complete; homogeneous and one-pass grouped mixed-family reductions are green |
| Many/few overlay | Overlay relation-to-constructive profile plus complete-ring winding canary | Many/few overlay <=100ms; exact topology bounded by candidate-ring segments, not unresolved rows times all mask segments | Complete; relation and ring-local canaries green |
| Grouped geometry reduce | NativeGrouped union/disjoint/difference and direct/tiled collective profiles | 10K <=100ms; memory-bounded exact topology at 1M | Complete; grouped multi-tile execution and habitat tiled-mask reuse are green |
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
- Prepared one-mask exact topology lowers unresolved row/segment evidence to a CUDA-resident `NativeRelation` of complete candidate rings. Candidate holes
  include their ancestor shell for the canonical fp64 face-walk baseline. The
  canaries record 12 versus 18 segments and an adversarial 20 versus 72.
- Runtime ordering is explicit across driver, pylibcudf, and CCCL on the active
  CuPy stream. Planner calls carry authoritative physical estimates, and
  `ARCH009` confines mutating device resolution to owned internals.
- Rich 10K repeat-3 (August 7, 2026): 14/14 exact, GeoPandas 3328.6ms versus
  vibeSpatial 2771.8ms (1.201x), zero failures. Site suitability is 3.53x,
  retail 2.19x, redevelopment 2.00x, and vegetation 1.16x faster than
  GeoPandas.
- Current 10K repeat-3 (August 12, 2026): 14/14 exact, GeoPandas 3558.6ms versus
  vibeSpatial 2799.9ms (1.271x). Transit is 191.0ms, vegetation is 187.4ms,
  and site suitability is 187.8ms. The post-suite run remains aggregate-positive;
  both libraries were 4-5% slower and vibeSpatial is 1.0% above the rich 2.772s
  floor on this machine-sensitive absolute measurement.
- August 11 1M capacity checkpoint: 11 public shootouts complete exactly in
  133.57s versus 963.60s for GeoPandas, a 7.21x aggregate speedup. Site, redevelopment, and retail
  stop before public export because their eager joins contain 9.51B, 8.06B,
  and 6.24B pairs. The direct native semijoin carrier completes their intended
  unique-left reductions in 6.64s, 9.18s, and 11.58s/12.23s respectively.
  The old public collective rerun had 8 exact passes, three known eager-relation
  limits, and habitat/transit/vegetation OOMs. The replacement two-dimensional
  collective carrier closes vegetation at 52.73s in the comparable timed-section
  harness versus the historical 359.83s GeoPandas reference; reusable tiled
  coverage closes habitat at 38.18s versus the historical 157.49s reference; transit's
  30.55M-row public result completes exactly in 191.44s after page-shaped WKB
  export and row-group read composition.
- August 27 full-profile refresh: every active 1M pipeline passes with zero
  compute materializations, D2H packets, or fallbacks. Exact stage times are
  72.33ms mixed-strip union, 67.07ms positive-degenerate union, 51.70ms
  small-group union, 10.69ms grouped difference, 2.86ms relation-overlay
  constructive, and 0.58ms point-in-polygon.
- Correctness gates pass: strict-native upstream is 1,971 passed / 423 skipped /
  5 xfailed; contract health passes every surface; the focused carrier suite is
  612 passed; and the uninterrupted local plus vendored-upstream suite is 7,158
  passed / 434 skipped / 7 xfailed with zero failures.

## Completion State

The PRD is active; the previously failing 1M vegetation, habitat, and transit
workflows are recovered, and active boundary tiles execute through pressure-bounded
relation-grouped clip/topology batches with reusable downstream coverage.
Ring-local winding baselines are complete. Immediate remaining work is the 64
statement-profile internal host conversions. Mixed-family direct reductions
are complete through one grouped relation partition with device span metadata.
Broad compatibility verification remains part of each landed recovery. Eager public
joins with multi-billion output rows are an explicit terminal cardinality
limit on a 24GB device; this does not block reduced native consumers, but it
cannot be described as a successful public 1M export.
