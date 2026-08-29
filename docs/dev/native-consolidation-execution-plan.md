# Native Consolidation Execution Plan

<!-- DOC_HEADER:START
Scope: Scale-aware execution plan for closing the Native feature hold, generalizing pylibcudf tabular execution, and restoring trustworthy delivery gates.
Read If: You are choosing the next Native completion task, removing public-workflow host composition, setting 10K/100K/1M/SF100 gates, or deciding whether the feature hold can lift.
STOP IF: You only need an operation-local kernel detail or the completed historical SF100 implementation record.
Source Of Truth: Ordered execution plan for the final Native consolidation push beneath the Native full-coverage PRD.
Body Budget: 300/300 lines
Document: docs/dev/native-consolidation-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-24 | Request Signals |
| 25-36 | Open First |
| 37-46 | Verify |
| 47-60 | Risks |
| 61-91 | Current Evidence Snapshot |
| 92-118 | Scale-Aware Success Contract |
| 119-142 | Target Execution Architecture |
| 143-282 | Ordered Work Plan |
| 283-293 | Explicit Deferrals |
| 294-300 | Handoff Record |
DOC_HEADER:END -->

## Intent

Finish and reconcile near-complete work before adding feature breadth. This plan
is subordinate to `docs/dev/native-full-coverage-prd.md`: that PRD owns the
feature hold and acceptance gates; this document owns the remaining work order.

Generalize the architecture already proven by SpatialBench. Native* carriers
remain the geometry, lineage, index, and physical-shape substrate; pylibcudf is
the default private tabular plane beneath them. pandas and Arrow host objects
remain public compatibility and terminal export boundaries, not compute formats.

## Request Signals

- Native consolidation
- what is next
- finish the feature hold
- pylibcudf across public workflows
- tabular host composition
- 10K, 100K, 1M, or SF100 performance gates
- strict-native truth
- release readiness

## Open First

- `docs/dev/native-full-coverage-prd.md`
- `docs/dev/native-100ms-physical-shape-plan.md`
- `docs/dev/pylibcudf-sf100-execution-plan.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/testing/performance-tiers.md`
- `docs/testing/native-coverage.md`
- `docs/testing/pipeline-benchmarks.md`
- `docs/dev/native-format-inventory.md`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/health.py --tier contract --check`
- `uv run pytest tests/test_strict_native_mode.py tests/test_private_native_substrate.py -q`
- `VIBESPATIAL_STRICT_NATIVE=1 uv run python scripts/upstream_native_coverage.py --grouped --group-by file --json`
- `uv run vsbench shootout benchmarks/shootout --scale 10k --repeat 3 --reuse-geopandas <validated-baseline> --json --output <current-artifact>`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- use the matching 100K, 1M, external-corpus, and SF100 commands before claiming those scales

## Risks

- Treating a 10K loss as a scaling failure can trade large-scale architecture for launch and compatibility overhead.
- A large-scale win cannot excuse `auto` dispatch that is slower than an available CPU path at 10K.
- pylibcudf beside rather than beneath Native* would create a second execution model and repeat the rejected public planner.
- A device primitive is not a win if public pair export, pandas ordering, duplicate reduction, or index assembly dominates.
- Aggregate totals can hide a wrong answer, failed query, capacity limit, or severely regressed workflow.
- A clean narrow canary can hide strict upstream failures or a broken CPU-only
  CI collector.
- Unbounded repeat-three million-row comparators can exhaust host memory and
  trigger `systemd-oomd`; run one heavy lane at a time in a memory-capped
  transient service, persist each sample immediately, and record capacity
  limits as evidence rather than retrying inside the editor/agent scope.

## Current Evidence Snapshot

This rolling August 28, 2026 candidate snapshot is planning evidence, not a
reusable baseline. Bounded shootouts share source digest
`ebb9ac6c6d3bc66a188f59eaec409c7d051985af21aaf48e076effd07a4986c7`;
later correctness, relation-provenance, and pickle-state fixes changed it to
`c8a58295928e5bc8bcf0623b45f86824ca0a8e4e35c64ab2631b37be26b210fd`.
See `benchmark_results/native-consolidation/2026-08-28/candidate-evidence-manifest.md`.
Neither digest substitutes for a clean accepted revision; reuse comparators only
when workload, data, environment, host, measurement, and fingerprint identity match.

| Surface | Current evidence | Interpretation |
|---|---|---|
| Core public shootout, **10K**, repeat 3, current bounded candidate | 14/14 exact; vibeSpatial 2.7161 s versus GeoPandas 3.4681 s; 1.277x aggregate and 1.004x geomean; 6/14 workflows are individually slower. Session artifact: `/tmp/native-consolidation-current/core-10k-r3.json`. | The suite is aggregate-positive at 10K. Accessibility, corridor, emergency, habitat, insurance, and parcel remain CPU-faster at this scale; all except habitat and insurance are below 357 ms on vibeSpatial, and the two material large-shape lanes already win. This is crossover evidence, not a claim about 1M scaling. |
| External corpus, **10K**, repeat 3, current bounded candidate | 6/6 exact from the current worktree source digest; GeoPandas 1.2625 s versus vibeSpatial 1.1089 s; 1.139x aggregate and 1.029x geomean. Session artifact: `/tmp/native-consolidation-current/external-10k-r3.json`. | CMAB is 9.17x. GeoLife, nested IO, WKT summary, and Power are CPU-faster but complete below 259 ms on vibeSpatial; OSM enrichment is the material exception at 0.292x. Generalization is aggregate-positive while fixed IO/runtime costs remain visible. |
| CMAB external lane, **10K**, repeat 3, indexed-slice candidate | Ordered-v3 exact; vibeSpatial 0.0780 s versus GeoPandas 0.6963 s, or 8.93x. Partial device slices and flat spatial-index construction have zero geometry D2H/H2D/materialization; only bounded scalar fences precede the terminal public index. Artifact: `/tmp/cmab-flat-sindex-auto-10k-repeat3.json`. | Transient source lowering, multipart pylibcudf consumption, device writer row indirection, row-indirected slicing, and residency-aware flat indexing close the relation-shaping path. |
| Power external lane, **10K**, repeat 3, indexed-slice candidate | `power-nearest-v1` exact; vibeSpatial 0.0585 s versus GeoPandas 0.0430 s, or 0.736x. Both `iloc` stages have zero geometry D2H/H2D/materialization. Artifact: `/tmp/power-slice-indirection-10k-repeat3.json`. | CPU remains faster for this small workload. Keep fixed/runtime and nearest dispatch costs visible; do not contort the GPU path for a 10K headline win. |
| Power scale probe, **20K/100K** | Exact selected IDs, index, attributes, order, schema, and CRS now match. The Power-specific verifier admits only bounded projection numerics: two fp64 ULPs at the EPSG:3857 world bound for coordinates and the propagated Euclidean bound for distance. | The global ordered-v3 fingerprint remains visibly different because it intentionally has a broader numeric policy. Power timings are comparable only when the fail-closed `power-nearest-v1` contract matches. |
| Habitat core lane, **100K**, repeat 3, page-aware exact-relation candidate | Exact; vibeSpatial 2.5081 s versus GeoPandas 2.4383 s, or 0.972x; no fallback; 332,788,564-byte peak device memory. Session artifact: `/tmp/native-habitat-final/habitat-100k-page-formal-r3.json`. | A bounded 16-interval Morton-prefix packet chooses the physical shape before exact work. When relation work wins, one retained exact relation is refined once, sorted once, and executed in complete-source-row pages whose admission is refreshed against current free device memory. This reaches practical parity without a workflow-specific row threshold or a duplicate exact prepass. |
| Habitat structural-planner probes, **100K**, diagnostic | One safe Morton interval is too loose at 15.8049 s; bbox-count and exact-count prepasses are worse at 15.8642 s and 23.5199 s. A 16-prefix cover still chose amplified tiled coverage and took 15.5741 s. The page-aware retained-relation candidate instead takes 2.5081 s repeat-three. | Conservative structure alone is necessary but insufficient. Compare relation and union work before exact allocation, retain any admitted exact relation, and page complete source rows. Never compute exact counts and then repeat candidate/refinement work. |
| Habitat page-aware profile, **100K**, full synchronized replay | Exact 2.5702 s replay; 43/43 GPU steps, zero CPU steps, zero fallback/offramp, zero H2D, and 79 bounded D2H transfers totaling 1,429,914 bytes and 1.87 ms. The largest instrumented hotpath is `overlay.plan.faces` at 34.16 ms. Session artifact: `/tmp/native-habitat-final/habitat-100k-page-profile.json`. | The parity result is not hiding host composition or a disproportionate CPU stage. The D2H traffic is planning/allocation metadata and remains visible for later consolidation. |
| Redevelopment comparator, **1M**, isolated capacity probe | One GeoPandas sample ran for 27 minutes, reached 30.06 GiB cgroup memory plus 558 MiB swap, and accumulated 252 `memory.high` events under a 28 GiB-high/36 GiB-max service before controlled termination. Durable packet: `benchmark_results/native-consolidation/2026-08-28/redevelopment-1m-geopandas-capacity.json`. | The comparator is capacity-limited under the safe host-memory contract; it is neither a vibeSpatial failure nor a completed timing. Do not rerun repeat three without a separately approved larger envelope. |
| Core public shootout, **100K**, repeat 3, bounded benchmark digest | 14/14 exact; GeoPandas 167.2154 s versus vibeSpatial 13.4651 s; 12.418x aggregate and 4.989x geomean. A current-source insurance replay is also exact at 0.2353 s versus 0.1448 s. Session artifacts: `/tmp/native-consolidation-current/core-100k-r3.json` and `insurance-100k-current-profile.json`. | Habitat and insurance are the only losses. Insurance's nominal 100K input becomes 5,601 post-clip rows and 1,131 overlay rows; all 37 operations are GPU-selected, no stage exceeds 98 ms, and there is no fallback. This is a 235-ms crossover, not hidden host composition. Every other workflow wins. |
| External corpus, **100K**, repeat 3, bounded benchmark digest | 6/6 correctness contracts pass; GeoPandas 10.4341 s versus vibeSpatial 1.5185 s; 6.871x aggregate and 1.576x geomean. Current-source OSM enrichment is exact at 0.5786 s versus 0.1644 s. Session artifacts: `/tmp/native-consolidation-current/external-100k-r3.json` and `osm-enrichment-100k-current-profile.json`. | CMAB is 26.47x and Power is 2.88x. OSM is capped at 1,934 rows and forms two complex groups from 86,426 source segments. All 56 operations are GPU-selected; the synchronized >1 s `to_parquet` stage is deferred multipart-union topology realized at export, not Parquet IO or CPU work. Other sub-260-ms losses remain ordinary crossover cases. |
| External corpus, requested **1M** / asset-capped effective rows, repeat 3, current bounded candidate | 6/6 correctness contracts pass from the current worktree source digest; GeoPandas 11.8627 s versus vibeSpatial 1.9332 s; 6.136x aggregate and 1.735x geomean. Session artifact: `/tmp/native-consolidation-current/external-1m-r3.json`. | Effective input rows are CMAB 104,640; GeoLife 1,000,000 of 2,507,357; OSM enrichment 1,934; nested IO 1,934; WKT summary 627; and Power 164,374. CMAB is 26.72x, GeoLife crosses to 1.18x, and Power is 3.74x with 81,803 verified output rows. This is not six synthetic million-row evidence and does not satisfy the core 1M gate. |
| Full pipeline, **100K and 1M**, landing tree | 22 active results pass and two Phase 8 raster lanes are deferred. There are zero compute materializations and zero fallbacks. Forty classified D2H planning packets total 41,056 bytes; the slowest stage is 76.02 ms and the slowest 1M stage is 73.70 ms. Artifact: `benchmark_results/native-consolidation/2026-08-28/full-pipeline-landing-tree.json`. | The carrier and physical-shape substrate is healthy, and the remaining transfers are bounded page-weight/work-summary packets rather than row-shaped composition. This rail is not a GeoPandas speedup comparison. |
| SpatialBench, **SF100**, landing-tree candidate | Fail-closed verification passes all 12 queries: vibeSpatial 426.58 s versus immutable GeoPandas 8,086.00 s, or 18.955x. Q6 is 13.13 s after selective zone streaming plus exact, two-phase-admitted variable-width row-view compaction; Q9 remains the intentional 0.14 s small-work exception. Candidate/acceptance SHA256: `4cbc9c48...` / `7a071e27...`. | Large-scale acceptance passes on worktree source digest `2107dee8...`. Final hold closure still requires the accepted clean revision; do not infer that GPU is always fastest for Q9-like work. |
| Strict upstream coverage, landing-tree grouped file sweep | 2,239 passed, 54 failed, 410 skipped, 6 xfailed; 97.39% native pass rate. The sweep confirms the pickle-state and NYBB fixes on the landing tree. | Fifty declines are intentional 2D/transform-contract limits, three are harness defects, and the separate `build_area` workstream is the only open admitted capability. Nothing is unclassified. Complete JSON publication remains tied to the clean candidate revision. |
| Core statement profile, **10K** | 69 internal host conversions, 179 user exports, 631 D2H events, zero fallback | User exports are expected. Internal conversion count is a ledger input, not a standalone optimization target; wall time and physical shape decide priority. |
| Delivery | CPU collection is repaired: 5,699 tests collect and the CPU lane selects 4,961 without importing CuPy from a GPU-only module. Current local contract health passes all eight required surfaces plus the 46-test optional performance rail; full overlay is 398 passed/2 skipped and the adjacent Native/shape suite is 740 passed. Pipeline workflows run CPU base/current comparisons on pull requests and `main`; GPU comparison is explicit when the labeled self-hosted runner is declared available. | Local delivery gates are green. Final closure still needs a successful CI run reference tied to the accepted revision. |

## Scale-Aware Success Contract

Every result must state scale, data family, requested mode, actual backend,
correctness, capacity status, wall time, peak device memory, D2H,
materialization, and fallback/offramp events. Cached GeoPandas timings are
immutable comparator evidence; the current vibeSpatial revision is always
remeasured.

| Scale | Purpose | Success interpretation |
|---|---|---|
| 10K | Crossover, launch, import, public assembly, and dispatch gate | Aim for per-workflow parity and require aggregate parity. A measured GPU loss is acceptable only when `auto` selects a faster exact CPU implementation observably, or the record names the fixed cost and shows the crossover curve. Do not demand large GPU speedups from inherently sub-second work. |
| 100K | Reusable-shape crossover gate | The reusable GPU path should normally have crossed over. Require exact results, aggregate and geomean wins, no unexplained individual regression, and no host-shaped intermediate dominating an admitted native path. |
| 1M | Throughput, scaling, and capacity gate | Require a clear aggregate win, bounded memory, and successful exact completion whenever the public output itself fits the device contract. Prefer at least 5x aggregate for the core suite; classify output-cardinality limits separately from compute failures. |
| SF100 | Large irregular workload proof | Preserve the existing 10x suite-total contract and per-query correctness. Q9-like sub-second work stays in correctness, suite-total, dispatch, and 5% no-regression gates without a forced GPU-only or per-query 10x requirement. |

The scale gates complement operation tiers. They do not erase the Tier 1-5
expectations for reusable kernels, nor permit a large-scale aggregate to hide a
wrong or severely regressed workload.

`auto` and strict-native answer different questions:

- `auto` chooses the fastest admitted exact backend for the measured physical
  shape. A CPU choice is allowed, observable, and tested; silent fallback is not.
- strict-native measures repo-owned native compatibility and fails explicit
  host fallback or hidden materialization. It is a coverage/debug contract, not
  a requirement that every small operation execute on GPU in `auto`.

## Target Execution Architecture

```text
public GeoPandas-compatible API
  -> NativeFrameState and NativeIndexPlan
  -> NativeRowSet / NativeRelation / NativeGrouped / NativeExpression
  -> pylibcudf tabular and relational primitives
  -> CCCL or custom CUDA spatial/irregular primitives
  -> native carrier result
  -> explicit terminal pandas/GeoPandas/Arrow export
```

pylibcudf should own stable ordering, sort, first/last-per-key, duplicate
elimination, gather/scatter, projection, filter, join, top-k, groupby,
reduction, and supported nullable/string/categorical/datetime operations.
CCCL remains appropriate for compact/scan/sort/reduce shapes that pylibcudf
cannot express efficiently. Custom CUDA remains appropriate for spatial
candidates, exact refinement, topology, dynamic geometry assembly, and fused
shapes whose generic intermediate cardinality is structurally wrong.

This is not a public cuDF API or a general lazy planner. Native* owns lineage,
physical shape, geometry, index semantics, readiness, and export boundaries;
pylibcudf is an implementation plane consumed through those contracts.

## Ordered Work Plan

### M0: Restore Repository Truth — local gates complete; CI/revision acceptance open

- Make CPU-only test collection independent of optional GPU imports and add a
  collect-only guard that would catch another module-level CuPy dependency.
- Reconcile current CI behavior, benchmark automation docs, Native PRD status,
  and generated health surfaces so each describes what actually runs.
- Check in or explicitly reference identity-complete comparator artifacts for
  each accepted scale. Record current vibeSpatial measurements separately.
- Make all performance summaries name the scale; never publish “six workflows
  are slower” without “at 10K.”

Exit: required CPU CI is green on supported Python versions, docs checks pass,
and no automated benchmark or coverage claim exceeds the active workflow.

### M1: Reconcile Strict-Native Completion — one admitted lane open

- Turn the 78 strict-upstream failures into an owned ledger: missing native
  capability, correctness defect, metadata/schema defect, intentional
  unsupported contract, optional dependency, or harness defect.
- Fix correctness and metadata defects before optimizing their paths. Expand
  the inventory beyond a surface-level covered label when dimensionality,
  null, geometry-family, dtype, or index contracts differ.
- Preserve the completed ring-local winding-baseline path and finish any
  remaining P7 terminal export breadth from the active Native plans. The
  prepared-mask path now lowers unresolved row/segment evidence to a
  device-resident `NativeRelation` of complete rings, adds each candidate
  hole's ancestor shell as its face-walk winding baseline, and materializes
  only those ring segments for exact fp64 topology.
- Re-run the broad strict sweep and update the PRD completion record from fresh
  evidence. Do not infer completion from focused strict canaries alone.
- Finish mixed line/polygon `build_area`; zero-copy lint and upstream
  `test_build_area` must pass before this milestone closes.

Exit: every failure has an explicit disposition, admitted coverage passes, the
Native inventory and upstream sweep agree, and all PRD acceptance evidence is
present for maintainer review.

Ring-local checkpoint (August 27, 2026): the existing physical-shape canary
shrinks three rows against a six-segment mask from the former 18 logical
segments to 12 segments from two complete candidate rings. The adversarial
MultiPolygon/hole/component canary shrinks the old 72-segment logical bound to
20 segments from five complete rings while matching the exact Shapely oracle.
Both relation columns remain CUDA-resident. Candidate-relation and exact
topology sizing use two named scalar allocation packets; no coordinate,
predicate, winding, or topology payload crosses to the host.

### M2: Inventory The Tabular Execution Plane — complete for admitted workflows

- Trace every Native producer-to-consumer path that exports pairs, row
  positions, group labels, keys, attributes, or index labels before terminal
  public assembly.
- Record physical shape, cardinality, bytes, dtype/null/index/order/duplicate
  semantics, current backend, D2H, materialization, and wall-time share.
- Rank reusable shapes by end-to-end wall time across core, external, 1M, and
  SF100 evidence. Do not prioritize by raw conversion count or one benchmark.
- For each path, choose pylibcudf, CCCL, custom CUDA, admitted scalar fence, or
  terminal export and record why.

First target family:

```text
NativeRelation
  -> stable device order
  -> first/last/unique-per-key or grouped reduction
  -> device gather of attributes and index labels
  -> NativeFrameState / NativeRowSet / NativeExpression
  -> terminal public assembly
```

Exit: there is one checked-in ledger with no unclassified internal host
conversion and an ordered set of cross-workflow primitive gaps.

### M3: Generalize pylibcudf Consumers — complete for the bounded plan

- Implement the highest-value stable sort/dedup/first-per-key relation consumer
  first, preserving exact duplicate, null, ordering, and index semantics.
- Extend the same private adapters to grouped reduction, bounded top-k,
  gather/scatter, joins, projection/filter, and required dtype families.
- Return Native carriers so consecutive public operations compose without
  pandas/Arrow reconstruction; invalidate state conservatively for unknown
  pandas operations.
- Share the existing RMM resource, stream-readiness, admission, and memory
  budget contracts. Do not introduce a second allocator or eager gather.
- Add physical-shape canaries and public compatibility tests for every admitted
  consumer, including empty, nullable, duplicate-label, and stable-order cases.

Exit: admitted public pipelines have no pandas sort/groupby/dedup/index
assembly before terminal export, and their end-to-end wall time improves at
the scale that motivated the work without cross-scale regression.

M3 checkpoint (August 28, 2026): owned joins lower transient attributes into
pylibcudf stable sort/distinct, grouped first/last, bounded top-k, and writers.
Explicit nullable/string/categorical/decimal/temporal policies preserve stable
order, null/NaN equality, duplicate policy, and source index. Restricted
query/eval lowers exact integer/bool expressions to Native carriers; bounded
stable many-to-one merge and aligned left index join use pylibcudf and device
gathers. Invalid public inputs preserve pandas errors; unsupported valid shapes
decline observably and strict mode rejects them.

Evidence is scale-specific. CMAB is 8.93x at 10K; Power is 0.736x at 10K,
2.22x at 100K, and 2.87x at requested 1M. Top-k is 0.399x/0.840x/4.988x
pandas at 10K/100K/1M. Query is 0.386x/0.710x/4.230x and merge is
0.521x/1.109x/6.903x. CPU wins the small shapes while the admitted GPU work
dominates at 1M. The same-source full audit has 22/22 passes, no
fallback/compute materialization, 40 packets/41,056 bytes, and a 74.54 ms max
1M stage. This completes bounded M3, not unrestricted pandas expression or join breadth.

### M4: Remove Or Classify Remaining Host Composition — complete

- Re-profile the core and external suites after M3. Each internal conversion
  must be eliminated, batched into an inherent named fence, or proven terminal.
- Keep user-requested exports separate from compute debt. A lower counter is
  not success if wall time or memory regresses.
- Finish near-complete composition/export work before opening new public API
  breadth. Delete transitional host helpers once their last consumer is native.

Exit: zero unexplained internal host composition in admitted workflows, with
bounded exceptions named by operation, bytes, purpose, and canary.
Current-source exit: core 10K preserves the classified 66-event ledger. All six
external workflows pass at 10K/100K/requested-1M with zero fallback and only two
terminal join-index exports plus one 8-byte fence per scale. The 627-row WKT
dissolve explicitly selects CPU: observable crossover, not fallback.

### M5: Install Scale-Aware Ratchets And Close The Hold — local evidence complete

- Current-source core/external scale records, per-workload correctness, bounded
  full-pipeline evidence, and frozen SF100 acceptance are durable and explicit.
- SF100 passes 12/12 at 426.58 s versus 8,086.00 s (18.955x); Q6's
  warmup-plus-three protocol completes without ancestral-capacity allocation,
  and Q9 remains exempt.
- The core 1M GeoPandas leg is a recorded capacity limit and must not be retried
  inside an editor/agent scope without an approved larger memory envelope.
- Remaining delivery gates are one clean accepted revision, complete same-source
  artifacts, CI run references, and explicit maintainer approval.

Exit: truth, compatibility, physical-shape, composition, scale, and delivery
gates all pass from one accepted revision. Only then resume feature breadth.

## Explicit Deferrals

- New public API breadth while the Native PRD is active.
- A public cuDF object model or broad lazy planner.
- Benchmark/query-specific production branches.
- Raster-to-vector merely to make a deferred pipeline count look complete.
- Generic caching before the derived-carrier reuse audit proves ownership,
  invalidation, memory, and cross-workflow value.
- A release that packages known-red required CI or an unfilled PRD completion
  record.

## Handoff Record

Update this plan after each milestone with the accepted revision, artifacts,
changed ledger rows, scale-specific results, strict-native counts, CI status,
and next highest reusable shape. Historical measurements remain labeled by
date and scale; never overwrite them with a candidate from a different
measurement contract.
