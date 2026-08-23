# Archived Adaptive Point-Quadtree Production Implementation Plan

<!-- DOC_HEADER:START
Scope: Archived implementation plan and rejection record for the benchmark-gated adaptive point-quadtree provider.
Read If: You are reviewing why the experimental quadtree provider was removed from production or reusing its safety findings.
STOP IF: You need current production point-region behavior or settled public spatial predicate semantics.
Source Of Truth: Historical productionization contract; the active runtime supports grid-to-Morton selection instead.
Body Budget: 344/350 lines
Document: docs/archive/2026-08-21-quadtree-experiments/adaptive-point-quadtree-production-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-18 | Intent |
| 19-28 | Request Signals |
| 29-41 | Open First |
| 42-50 | Verify |
| 51-63 | Evidence Boundary |
| 64-76 | Risks |
| 77-97 | Decision |
| 98-129 | Scope And Non-Goals |
| 130-165 | Physical Workload Contract |
| 166-197 | Precision And Superset Contract |
| 198-228 | Capacity, Memory, And Fault Contract |
| 229-292 | Implementation Milestones |
| 293-330 | Implementation Outcome (2026-08-20) |
| ... | (1 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

Status: archived after end-to-end evidence found no winning production
selection region. The shared grid/runtime safety work remains production code;
the quadtree provider is preserved only in the ignored experiment capsule.

## Intent

Turn the benchmark-gated adaptive point quadtree into one safe production
candidate provider beneath existing public spatial-index APIs. The first
landing is deliberately narrow: it hardens the fixed quadtree implementation,
repairs dense-grid admission, and uses quadtree only when an otherwise eligible
dense grid is declined before submission.

This is NativeSpatialIndex completion work under the Native* feature hold. It
adds no public API, device-product lookup, general planner, or Q11-specific
branch.

## Request Signals

- production adaptive point quadtree
- quadtree versus Morton
- point-region scaling
- point-grid admission or OOM
- bounded point-region reduction
- public query_pair_aggregate acceleration
- Q10 or Q11 point-region execution

## Open First

- `docs/archive/2026-08-21-quadtree-experiments/quadtree-scaling-evidence.md`
- `docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md`
- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0002-dual-precision-dispatch.md`
- `src/vibespatial/spatial/point_quadtree_index.py`
- `src/vibespatial/spatial/point_grid_index.py`
- `src/vibespatial/spatial/spatial_index_device.py`
- `src/vibespatial/api/_native_metadata.py`

## Verify

- `uv run ruff check`
- `uv run python scripts/check_docs.py --check`
- targeted point-region, spatial-query, device-index, predicate, and upstream
  spatial-index tests listed by intake at implementation time
- the public 10K, 1M, constrained 10M, and SF100 no-regression gates
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Evidence Boundary

Current RTX 4090 evidence proves that fixed-parameter quadtree can beat direct
Morton by 7.1x at 1M and 18x at 10M on the clustered-extent canary, while it is
slower than the current automatic path on the protected uniform 1M shape. It
also exposes a production dense-grid admission defect: the grid is admitted
from an incomplete estimate and later asks CUB for about 12 GiB at 10M.

That evidence is enough to add a safe escape path for a grid that cannot be
admitted. It is not enough to learn a general grid/quadtree/Morton winner model,
hardware policy, or arbitrary tuning parameters. H200, 100M, 1B, Nsight gain,
and Q10/Q11 speedups remain follow-up evidence, not landing prerequisites.

## Risks

- An incomplete peak-memory model can admit a provider that fails only after
  GPU submission, where safe retry is impossible.
- Stale prepared state or missing stream dependencies can turn cache reuse into
  wrong results, use-after-free, or asynchronous device faults.
- A non-conservative bbox transform can silently remove exact-true pairs.
- Raw counts reused with the wrong index or query slice can overrun scatter
  capacity before host code observes the mismatch.
- A selector generalized from one skewed 4090 fixture can regress ordinary or
  differently provisioned hardware, so the first rule intentionally does not
  predict a winner when grid is admissible.

## Decision

Retain all three physical implementations with one local production rule:

1. If the point-region shape is eligible and the repaired dense grid is fully
   pre-admitted, execute the dense grid.
2. If that same grid shape is declined before any grid submission, attempt the
   fixed quadtree after its own complete pre-admission.
3. Otherwise execute the existing GPU Morton baseline.

Selection happens once, before provider submission. No implementation may
fail over after it launches. A pre-submission decline is normal GPU routing; a
post-launch allocation failure, device error, or asynchronous fault propagates.
CPU fallback retains its existing observable meaning and is never used merely
because one GPU partition variant declined.

The rule is operation-local and subordinate to AdaptivePlan: AdaptivePlan owns
execution family, residency, and precision; this rule chooses only the private
point-partition variant after native admission. The registry advertises kernel
capability and does not make another choice.

## Scope And Non-Goals

The first landing covers homogeneous Polygon/MultiPolygon query rows against
homogeneous indexed Point rows for existing `right_count` and aligned
`right_pair_count` consumers. The admitted predicates are exactly
`intersects`, `contains`, `covers`, `contains_properly`, and `touches`: each
true result requires unexpanded bbox overlap, and exact refinement remains
authoritative.

`predicate=None`, `disjoint`, `dwithin`, distance arrays, other predicates,
mixed families, and nonfinite aggregate bounds decline quadtree before
submission and continue through the current GPU baseline. A later `dwithin`
extension requires outward-rounded fp64 distance-expanded query bounds and its
own proof. Null, empty, or nonfinite rows are not silently filtered into a new
semantic path; mixed/nonfinite inputs use the baseline until finite-row
partitioning is separately designed and proven.

Not in scope:

- a public index type, tuning argument, or forced execution control
- a generic candidate-provider protocol or device-planning abstraction
- arbitrary quadtree parameters, online calibration, or product-name policy
- joins, nearest, bbox-only query, non-point tree indexes, or quadtree relation
  output
- deeper keys, streamed billion-row construction, or dense-grid removal
- renaming broad reducer families or widening `SpatialQueryExecution`

Forced grid, quadtree, and Morton controls exist only as private test injection.
Public dispatch records the selected implementation and named reason in the
existing `SpatialQueryExecution` fields. Detailed work and memory counters
belong in opt-in profiler artifacts, not the runtime metadata contract.

## Physical Workload Contract

Logical contract: preserve public result values, original row order, duplicate
multiplicity, index, CRS, null/empty behavior, strict-native behavior, and
predicate semantics.

Physical shape: a reusable NativeSpatialIndex point-partition derivative feeding
bounded candidate/refine reductions.

NativeSpatialIndex is the sole logical production owner of the private prepared
point-partition cache. `FlatSpatialIndex` may remain a transitional build/view
reference but may not own independently mutable production grid or quadtree
state. Both sides of paired reduction cross the reducer boundary as
NativeSpatialIndex. A cache key binds:

- geometry lineage/version and row count
- Morton key generation and sorted row order
- fp64 bounds generation and fixed quadtree parameters
- CUDA device/context and producer readiness dependency

The first implementation has one fixed parameter set: the measured 16-level
Morton key depth and leaf-size 256. A cache entry is published atomically only
after build checks pass and readiness is recorded. Invalidation drops the whole
entry; partial or failed builds are never visible.

Persistent state contains Morton-derived leaf prefixes/levels, sorted point-row
spans, per-leaf counts, fp64 leaf bounds, and row-to-leaf mapping required by
aligned exclusion. Morton keys and sorted order remain canonical substrate, not
a fallback to CPU.

Work estimates and profiler output use physical units: point rows, finite rows,
leaves, maximum leaf occupancy, query rows, `query_rows * leaves` bbox tests,
query-leaf descriptors, represented candidate lanes, exact lanes, persistent
bytes, transient bytes, and result rows. Candidate pairs exist only inside one
admitted tile and are reduced before the next tile.

## Precision And Superset Contract

The quadtree changes only candidate generation. Two distinct PrecisionPlans
apply:

- a COARSE plan registered as an fp64-only quadtree-bounds variant; AUTO may
  not silently resolve it to fp32
- the existing PREDICATE plan for exact point-region classification, whose
  current indexed-point result remains authoritative fp64

If policy cannot admit the fp64 COARSE variant, quadtree declines before
submission. No local precision boolean or cast may override either plan.

Every admitted point row belongs to exactly one leaf, row remapping is stable,
and every exact-true pair appears in the candidate superset exactly once.
Leaf/query min-max values and overlap tests remain fp64. Exhaustive mechanical
bbox-oracle tests cover nextafter boundaries, zero extents, translated large
coordinates, holes and vertices, duplicate coordinates, maximum-depth
collisions, asymmetric leaf assignment, and randomized fixtures. Final public
results use the Shapely/upstream oracle, not handwritten GIS truth tables.

For each aligned output row `r`, paired reduction must satisfy:

- `L[r] = sum_q P(query[q], left[r])`
- `R[r] = sum_q P(query[q], right[r])`
- `S[r] = sum_q (P(query[q], left[r]) AND P(query[q], right[r]))`

The second pass evaluates exactly `C_right \\ C_left`; if an oversized left
query row was evaluated against every point, `C_left` is the full point set.
Tests cover asymmetric partitions, identical and duplicate coordinates,
boundary hits, empty inputs, and all tile edges.

## Capacity, Memory, And Fault Contract

First repair dense-grid admission at its physical root. The grid count primitive
must either have owned bounded scratch or expose an accurate worst-case scratch
estimate. An underestimated CUB allocation is not valid evidence for choosing
another provider.

Replace raw `precomputed_query_counts` and `pair_capacity` arguments with a
narrow immutable private query-slice token. It binds prepared-index identity,
provider variant, fixed parameters, query-bounds identity and slice, device
counts, exact capacity, and producer dependency. Tokens cannot be reused across
indexes, providers, query slices, or cache generations.

All add, multiply, prefix-sum, and downcast operations are checked. Scatter
kernels receive explicit capacity and segment limits, guard every write, and
set a device error flag on overflow or provenance mismatch. Tests exercise
INT32/INT64 boundary arithmetic without allocating boundary-sized arrays, plus
mismatched tokens and deliberately undersized capacities.

The stage lifetime model accounts for every simultaneously live buffer:
Morton-order reuse, sorted gathers, prefix/leaf construction, row-leaf ids,
query counts, offsets, descriptors, cursors, candidate tiles, exclusion masks,
exact-refinement scratch, and output columns. Planning metadata has a fixed byte
cap; an unbounded query-row packet declines before submission.

Every producer records its stream/event. Same-stream consumers rely on stream
order; cross-stream consumers wait on that event. Completion retention owns the
source geometry, both NativeSpatialIndexes, query bounds, token counts/offsets,
cursors, masks, prepared arrays, outputs, and exact scratch through completion.
GC plus allocation-churn tests cover same- and cross-stream use.

## Implementation Milestones

### P0. Freeze evidence and repair grid admission

- preserve source/input/device/result fingerprints for current scaling evidence
- replace the incomplete grid estimate with checked per-stage peak admission
- prove grid preflight never launches a stage whose declared peak exceeds the
  active allocation envelope
- inject pre-launch OOM and post-launch asynchronous faults; only the former may
  cause a named provider decline

Exit: dense-grid eligibility is trustworthy and no post-launch failure retries.

### P1. Harden one fixed quadtree

- move sole prepared ownership and invalidation to NativeSpatialIndex
- add the cache key, atomic publication, readiness, and completion retention
- implement the fp64 COARSE PrecisionPlan variant and precompile/inventory entry
- implement checked memory admission and the bound query-slice token
- add capacity guards, device error reporting, and leaf/superset invariants

Exit: privately forced quadtree is exact, bounded, lifetime-safe, and never
materializes a full relation. Automatic selection remains disabled.

### P2. Wire the existing bounded consumers

- pass both paired sides as NativeSpatialIndex
- parameterize only right-count and aligned paired-count internals enough to use
  grid, fixed quadtree, or Morton without callbacks or a provider framework
- preserve bounded tile reduction and exact second-pass exclusion equations
- propagate actual implementation/reason through public dispatch instead of a
  hard-coded grid or Morton message

Exit: public consumers can exercise each private variant with identical outputs.

### P3. Enable the narrow local rule

- apply `admitted grid -> grid`, `grid preflight decline -> admitted quadtree`,
  `otherwise -> Morton`
- build at most one new derivative per invocation
- make all unsupported/unknown/pressure cases named pre-submission declines
- add no hardware heuristics, cache-reuse prediction, candidate-inflation model,
  or postexecution feedback to selection

Exit: the choice is static, explainable, and independent of GPU product name.

### P4. Validate and document

- forced-variant bbox-superset and public-result oracle tests
- strict-native, zero hidden compute D2H/materialization, cache invalidation,
  stream/lifetime, duplicate, boundary, oversized-row, and fault tests
- protected 10K/1M ordinary automatic paths remain unchanged within the 5% rail
- constrained 10M clustered-extent execution remains bounded; forced fixed
  quadtree beats direct Morton by at least 20% complete-stage time. Automatic
  selection continues to obey the local rule even when the repaired grid is
  now genuinely admitted, so this canary is not allowed to introduce a winner
  heuristic merely to force quadtree selection
- Q10, Q11, SF100, upstream spatial-index tests, and the mandatory full pipeline
  profile are correctness and no-regression gates, not required speedup claims
- update architecture notes, precision ledger, kernel inventory, and evidence

Exit: the public APIs gain a production quadtree escape path without regressing
ordinary workflows. Morton ordering/baseline and dense grid both remain.

## Implementation Outcome (2026-08-20)

P0-P3 are implemented. NativeSpatialIndex solely owns prepared derivatives.
FP64 COARSE plans, readiness/retention, checked memory math, guarded scatters,
both paired carriers, formal exclusion, provider telemetry, and the narrow
selector are live behind public aggregate APIs.

The pair-shaped dense-grid path for public relation consumers shares that
ownership, admission, provenance, scatter, readiness, and retention contract.
Tokens bind owner, provider, query shape, cache key, and a byte ceiling, so
cache invalidation cannot reuse cached-state admission for a rebuild. After the
count fence, relation execution admits its complete 73-byte-per-pair live
footprint; later declines or faults propagate instead of retrying Morton. This
is not quadtree relation support: exact refinement remains mandatory and Morton
is used only when the grid relation is ineligible before submission.

Fresh RTX 4090 evidence resolves one stale expectation. Without the unbounded
`bincount`, automatic 10M grid completes in 56.15 s cold and 262.11 ms warm, so
the locked rule does not select quadtree. Across five calls, forced quadtree is
135.06 ms cold and 10.76 ms steady-state versus Morton's 195.62 ms and
126.67 ms: 31.0% faster cold and 11.78x warm. Choosing it over an admitted grid
still requires the deferred winner model. Every result matched both oracles
with zero fallback.

The final protected corpus is stable. At 10K, all 14 fingerprints are byte-identical, aggregate vS time changed by +0.96%,
and every workflow is within 5% after a seven-repeat resolution of the noisy
transit sample. At 1M, all 14 fingerprints are byte-identical, aggregate time
changed by +0.12%, and the worst slowdown is +4.79%; redevelopment changed by
-0.04% and transit by +0.30%.

The final all-query SF100 run measured 484.71 s versus the prior 481.06 s
(+0.76%). Every query is within 1.85%: Q10 improved from 125.84 s to 124.60 s
and Q11 changed from 237.59 s to 241.98 s. Eleven normalized outputs are
byte-identical; Q6 differs only in printed floating-point digits, with maximum
absolute numeric delta `6.78e-21`. The mandatory 1M full-pipeline profile has
zero fallbacks, zero compute D2H transfers, zero compute materializations, and
a 71.22 ms maximum stage time across all implemented pipelines.

## Review Disposition

Three independent Sol reviews covered architecture fit, accuracy/safety, and
YAGNI. Their initial verdicts were REVISE. This version adopts their common
recommendation: single NativeSpatialIndex ownership, explicit stream lifetime,
fp64 coarse versus predicate plans, predicate admissibility, capacity-bound
query tokens and guarded scatters, no post-launch failover, repaired grid
preflight, formal paired-count semantics, no metadata widening, and the narrow
grid-decline-only selector. Final review additionally required complete
relation-stage admission, cache-invalidation byte-growth validation, and a
propagated post-count memory error that cannot fall through to Morton. The
general winner selector and cross-device tuning campaign are deferred until
production evidence demonstrates a need. Final review returned APPROVE from
all three perspectives with no remaining blocking findings.
