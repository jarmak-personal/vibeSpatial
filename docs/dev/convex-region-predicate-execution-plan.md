# Exact Convex-Region Predicate Execution Plan

<!-- DOC_HEADER:START
Scope: Evidence-first plan for certifying convex hole-free polygonal regions and lowering exact containment predicates to grouped point classification through existing public APIs.
Read If: You are changing polygon containment predicates, reusable geometry shape metadata, grouped point-in-region execution, or evaluating a quadtree/grid/Morton provider for polygon vertices.
STOP IF: You are designing approximate arithmetic or user-visible accuracy tolerances; use the bounded-accuracy plan instead.
Source Of Truth: Active implementation and evidence plan for the exact convex-region predicate fast path.
Body Budget: 360/360 lines
Document: docs/dev/convex-region-predicate-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-44 | Intent |
| 45-57 | Request Signals |
| 58-76 | Open First |
| 77-89 | Verify |
| 90-103 | Risks |
| 104-126 | Relationship To Other Work |
| 127-168 | Exact Semantic Contract |
| 169-213 | Certification Contract |
| 214-252 | Physical Execution Shape |
| 253-269 | Public API Boundary |
| 270-340 | Milestones And Exit Gates |
| 341-351 | Workspace And Evidence Isolation |
| 352-360 | Completion Condition |
DOC_HEADER:END -->

## Intent

Implementation status (2026-08-23): milestones C1-C4 are implemented for the
measured direct/broadcast provider. Exact selective orientation refinement,
bounded offset-native grouped `ALL`, conservative cached convex certification,
simple/nonzero source certification, complete-stage memory admission, and the
public predicate lowering are live. The grouped reducer assigns one warp per
segment, so a skewed group does not serialize on one lane. The selector declines
source rings above 65 coordinates to bound certification. Grid, quadtree, and
Morton remain outside this broadcast shape, which has no candidate search.

Current production evidence for a 16-vertex target is:

| Source rows | Archived general predicate | Production convex lowering | Speedup | Oracle |
|---:|---:|---:|---:|---|
| 10K | 2.850 ms | 0.598 ms | 4.77x | exact |
| 1M | 23.912 ms | 3.013 ms | 7.94x | exact |
| 10M | 247.988 ms | 28.017 ms | 8.85x | exact |

The 10M corpus has zero differences. Production selection is limited to at
least 10K rows and 64 target vertices by complete-stage evidence.

Final source SHA-256 is `7dd29f6f053d672054e1fda3b75345da7e8e308e35ce21c2a7b18b7e1196e30c`.
10K is 14/14 exact and 2.663 s versus GeoPandas at 3.542 s; 1M is 14/14 exact
and 494.36 s versus the prior accepted 576.50 s. SF100 is 12/12 exact with zero
fallback at 469.01 s; Q11 is 227.30 s versus the prior 237.65 s.
The full 1M profile has zero compute D2H/materialization/fallback; its maximum
stage is 70.73 ms, plus one intentional 128-row terminal GeoArrow export.

Accelerate exact polygonal containment when the containing region is certified
as a valid, convex, hole-free Polygon. The optimization must remain behind
existing public GeoPandas-compatible APIs and preserve exact predicate results.

The reusable idea is not detection of mathematically regular polygons. Equal
sides and angles are irrelevant. The useful property is convexity: a convex set
that contains every vertex of another polygon contains that polygon.

The first implementation lowers eligible polygon containment to batched
boundary-inclusive point classification followed by a device-resident grouped
ALL reduction. Dense grid, quadtree, and Morton providers are execution
alternatives, not public controls and not assumed winners.

## Request Signals

- convex polygon
- regular polygon fast path
- polygon within polygon
- polygon covered by polygon
- contains convex mask
- grouped point in polygon
- vertex containment
- reusable polygon mask
- quadtree polygon predicate
- shape certification

## Open First

- `docs/dev/convex-region-predicate-execution-plan.md`
- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md`
- `docs/architecture/point-predicates.md`
- `docs/decisions/0004-robustness-strategy.md`
- `docs/decisions/0011-binary-predicate-refine-pipeline.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `src/vibespatial/api/_native_metadata.py`
- `src/vibespatial/predicates/binary.py`
- `src/vibespatial/kernels/predicates/point_in_polygon.py`
- `src/vibespatial/spatial/point_grid_index.py`
- `src/vibespatial/spatial/spatial_index_device.py`

The archived production-quadtree experiment is evidence, not implementation
authority. Start from its true-hierarchy measurements and rejection record.

## Verify

- `uv run ruff check`
- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_binary_predicates.py tests/test_point_in_polygon.py -q`
- `uv run pytest tests/test_spatial_query.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "predicate"`
- `uv run vsbench run binary-predicates --scale 10k`
- `uv run vsbench run binary-predicates --scale 1m`
- run the public 10K, 1M, and SF100 regression gates
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- A false-positive convexity certificate makes an exact predicate wrong.
- Point `within` excludes a polygon boundary, while polygon containment may
  permit boundary contact; using the wrong point predicate changes semantics.
- Vertex containment is insufficient for concave or holed containing regions.
- Vertex PIP alone cannot decide polygon `intersects`, `touches`, or `overlaps`.
- Flattening coordinates without parent-row provenance breaks grouped results.
- Shape detection can cost more than it saves when the containing geometry is
  not reused.
- A synthetic quadtree win may remain an end-to-end public-workflow loss.
- Cached certification without source lineage or stream readiness can become
  stale or race a consumer.

## Relationship To Other Work

This plan is an exact optimization and remains independent of
`bounded-accuracy-execution-plan.md`.

- Exact convex certification never depends on a user tolerance.
- Exact public behavior remains the baseline for every accuracy experiment.
- A future bounded-accuracy path may consume certified shape metadata, but it
  must not change the certificate or the exact selector.
- Work should use an independent branch or worktree and an independent evidence
  directory. GPU benchmarks run sequentially with other experiment tracks so
  allocator state, compilation, and system contention do not contaminate data.

The existing point-region program remains authoritative for point-versus-region
classification. This plan introduces a new polygon-containment consumer of that
physical shape; it does not create a second point-region planner.

SF100 Q11 is not the motivating acceptance workload. Q11 already classifies
points against general regions, so polygon concavity does not prevent PIP. Its
archived quadtree experiment showed that a synthetic hierarchy win did not beat
the complete public grid path. This plan must prove value on polygonal
containment workloads and still protect Q11 from regression.

## Exact Semantic Contract

Let `A` be a valid, nonempty Polygon or MultiPolygon and let `B` be a valid,
nonempty, convex, hole-free Polygon. If every exterior vertex of every polygonal
part of `A` is covered by `B`, then every point of `A` lies in `B`.

Reason: `B` contains the vertices, convexity makes it contain their convex
hull, and `A` is a subset of that hull. Holes in `A` do not invalidate this
proof, although processing all ring vertices may be a simpler first physical
layout. Holes in `B` do invalidate it.

The first admissible predicate set is deliberately narrow:

| Public predicate | Exact lowering |
|---|---|
| `A.covered_by(B)` | boundary-inclusive PIP for all `A` exterior vertices, then grouped ALL |
| `A.within(B)` | same lowering for valid nonempty polygonal `A`; preserve existing interior rule |
| `B.covers(A)` | inverse of `A.covered_by(B)` |
| `B.contains(A)` | inverse of `A.within(B)` |

Do not initially lower:

- point or line sources, whose boundary/interior rules differ
- `contains_properly`, which needs strict boundary exclusion
- `intersects`, where crossing polygons may contain no opposing vertex
- `touches`, `overlaps`, `crosses`, or arbitrary DE-9IM masks
- a containing MultiPolygon with more than one nonempty component
- invalid, empty, nonfinite, or unclassified containing geometry

Boundary-inclusive point classification is required for the vertex proof.
Calling point `within` is wrong for vertices on `B`'s boundary. The exact PIP
primitive may expose boundary tags privately, but public polygon semantics are
resolved by the grouped consumer.

Axis-aligned rectangles remain a stronger shape class. Exact bounds tests
should win when existing metadata certifies that the rectangle equals its
bounds. Generic convex PIP must not replace a cheaper exact-bounds path.

If both polygons are convex, a later `intersects` proposal should evaluate a
separate Separating Axis Theorem physical shape. It is not bundled here merely
because the same certificate could admit it.

## Certification Contract

Certification is conservative and reusable. Each row has one of:

- `exact_bounds`: a valid hole-free polygon proven equal to its axis-aligned bounds
- `convex_simple`: a valid nonempty hole-free Polygon with one exterior ring
- `general`: proven ineligible for the vertex-containment theorem
- `unknown`: insufficient metadata or numerically ambiguous proof

Only positive certificates admit the optimization. `unknown` and `general`
select the existing exact path without fallback or materialization.

The device certification pass must establish:

1. Polygon family and one polygonal component.
2. Validity and nonemptiness from trusted native metadata.
3. Exactly one ring, so the containing region has no holes.
4. Closed-ring structural validity and at least three distinct vertices.
5. Consistent nonzero turn orientation after ignoring duplicate and collinear
   vertices.
6. No ambiguous orientation result under the exact robustness policy.

Validity is a prerequisite because consistent local turns alone are not a proof
that an arbitrary self-intersecting ring represents a convex set.

The certificate belongs in `NativeGeometryMetadata` or a typed derivative
owned by it. It must carry source token, row mapping, residency, stream
readiness, and invalidation lineage. A Python dictionary in `shape_summary` may
record aggregate profiling facts, but a production row-level certificate must
be a typed device-capable array rather than an unvalidated convention.

Production stores typed convex-mask and simple-source certificates on the
immutable `OwnedGeometryDeviceState`. Each carrier validates its device-state
token, owner, family row mapping, coordinate generation, residency, and
readiness event before reuse. A separate source certificate admits only one
simple, non-empty, positive-area ring (and one part for MultiPolygon); a failed
source summary declines the batch to the general exact predicate. Collapsed
invalid rings retain their GEOS-compatible boundary/interior semantics in the
general GPU predicate rather than entering the vertex theorem.

The certification scan is vertex-shaped and runs once per source generation.
It must not download row flags merely to choose the GPU path. A compact device
summary or existing trusted host structural metadata may guide admission at a
named boundary.

## Physical Execution Shape

The public result is row-aligned, but execution is vertex, candidate, and group
shaped:

1. Resolve predicate orientation and certified containing rows.
2. Flatten source exterior vertices while preserving source-row and part
   provenance.
3. Produce point-region candidates through an admitted provider.
4. Run exact boundary-inclusive point classification.
5. Reduce classifications with grouped logical ALL by source row.
6. Merge exceptional rows natively; decline indexed views and validate ordered/source bounds separately.
7. Export only at the public terminal boundary.

The native output is `NativeExpression` for aligned predicates and may lower to
`NativeRowSet` or `NativeRelation` for sanctioned consumers. Do not materialize
a public bool Series between PIP and grouped reduction.

Provider comparison is part of the complete stage:

- direct/broadcast exact classification for tiny or dense regular work
- dense grid when its complete admitted footprint and candidate shape win
- true hierarchy quadtree when reuse and clustered/skewed extent evidence win
- Morton as the existing general baseline

Forced variants exist only in tests and benchmark instrumentation. Production
selection uses existing runtime owners and measured shape facts such as:

- source vertex count and containing-region vertex count
- number of source rows and target reuse count
- cached certificate and index residency
- extent occupancy/skew and candidate inflation
- predicted exact classifications and grouped reduction work
- complete peak-live bytes, not only persistent index bytes

Preparation, flattening, index construction, PIP, grouped reduction, and export
must be timed separately and together. A provider is eligible only if the
complete public stage wins after build amortization.

## Public API Boundary

Acceleration is automatic behind existing APIs:

- `GeoSeries.within`
- `GeoSeries.covered_by`
- `GeoSeries.contains`
- `GeoSeries.covers`
- native expression and sanctioned spatial-query consumers reached from those APIs

There is no public `convex=True`, provider, leaf-capacity, depth, or forced-path
argument. Users request semantics; vS certifies the shape and chooses execution.

Dispatch observability records the certificate class, physical provider,
source vertices, target reuse, candidate count, complete-stage bytes, and
selection reason without naming a dataset or query.

## Milestones And Exit Gates

### C0. Freeze Exact Baselines

- record current public predicate results and timings at 10K and 1M
- preserve relevant SF100 and full-profile evidence as protected regressions
- add deterministic rectangle, convex, concave, holed, multipart, and uncertain
  shape generators

Exit: later results distinguish acceleration from changed inputs or baselines.

### C1. Prove Conservative Certification

- implement device-resident row classification with exact orientation handling
- bind certificate lifetime and invalidation to native metadata
- certify the source theorem prerequisites, including positive area and ring
  simplicity, independently from target convexity
- compare against an independent host convexity/validity oracle
- include near-collinear, duplicate, huge-offset, reversed-winding, invalid,
  empty, null, and nonfinite cases

Exit: the detector has zero false-positive `exact_bounds` or `convex_simple`
certificates. False negatives are permitted and measured.

### C2. Implement Forced Grouped Vertex Containment

- flatten source exterior vertices with row provenance
- reuse the exact point-region classifier
- reduce directly into a native row-aligned result
- use a warp-cooperative segmented reducer with explicit int32 capacity checks
  and admissions for every output, offset, and group-ID allocation
- retain every producer through stream completion
- preserve null, empty, index, and boundary semantics

Exit: forced baseline and forced convex lowering match the host oracle exactly.

### C3. Compare Complete Providers

- force direct, grid, true-quadtree, and Morton variants over identical batches
- sweep 10K, 1M, and 10M vertices where memory permits
- vary target reuse, vertex counts, skew, candidate density, and mask complexity
- run on RTX 4090 and H100/H200-class hardware when available
- include cold build, warm reuse, complete-stage, memory, launch, sync, and D2H
  evidence

Exit: identify at least one stable public shape where a non-baseline provider
wins complete-stage time on both device classes, or reject that provider.

### C4. Add Minimal Automatic Selection

- add only the measured discriminator to existing adaptive ownership
- select exact-bounds, convex lowering, or the general exact path
- select a point provider only from complete-stage evidence
- fail closed before submission when shape, memory, or readiness is uncertain
- propagate post-submission CUDA faults without retry

Exit: automatic selection contains no product, dataset, query, or GPU-name rule.

### C5. Validate Public Value

- require zero oracle differences across the correctness matrix
- require no unexpected D2H, materialization, or fallback
- protect simple/general predicate cases within a 5% bounded-noise rail
- require at least 20% complete-stage improvement in the admitted target region
- rerun public 10K, 1M, SF100, upstream predicate suites, and the mandatory full
  profile
- inspect every 1M sparkline stage for unexpected CPU-heavy work

If the complete public operation does not win, remove the production selector
and retain only independently useful metadata or safety improvements.

## Workspace And Evidence Isolation

Use a dedicated branch or worktree for this track. Keep raw artifacts under a
track-specific ignored directory such as
`benchmark_results/experiments/convex-region-predicates/` and commit only the
reviewed summary evidence required by the plan.

Do not run this track's GPU measurements concurrently with bounded-accuracy,
SF100, Nsight, or other allocator-heavy work. Record device, driver, clocks,
allocator configuration, warmup, and active processes for every comparison.

## Completion Condition

The plan is complete when conservative native certification automatically
selects an exact grouped vertex-containment path for a measured winning public
shape, all supported public results remain exact, cross-device evidence
justifies any provider choice, and protected public workflows do not regress.

If no complete-stage winning region exists, the correct completion is a
documented rejection with the production fast path removed.
