# Archived Adaptive Exact Point-Region Refinement Plan

<!-- DOC_HEADER:START
Scope: Archived design exploration for a multi-variant adaptive exact point-region refinement engine.
Read If: You are auditing the superseded ADR-0047 adopter proposal or researching alternatives rejected as premature.
STOP IF: You are implementing current point-region work; use the evidence-first plan instead.
Source Of Truth: Historical design record only; not implementation authority.
Body Budget: 638/710 lines
Document: docs/archive/2026-08-18-device-planning/adaptive-point-region-refinement-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-7 | Preamble |
| 8-22 | Intent |
| 23-37 | Request Signals |
| 38-54 | Open First |
| 55-67 | Verify |
| 68-85 | Risks |
| 86-106 | Mission |
| 107-127 | Public API Boundary |
| 128-148 | Logical Contract |
| 149-191 | Physical Contract |
| 192-221 | Current Algorithm And Evidence |
| 222-265 | Structurally Different Solutions |
| 266-301 | Prepared Region Metadata |
| 302-361 | Work Queue Design |
| ... | (9 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

> **Archived 2026-08-18.** This design exploration is not implementation
> authority. It proposed several unproven changes at once and depended on the
> superseded ADR-0047 planner. The active evidence-first direction is
> `docs/dev/evidence-first-point-region-execution-plan.md`.

## Intent

Build a reusable exact point-versus-Polygon/MultiPolygon refinement engine
behind existing public vibeSpatial and GeoPandas-compatible APIs.

SF100 Q11 exposed the dominant physical cost and motivates the investigation,
but it does not define the implementation. This plan targets the broader
problem family: candidate point-region classification with simple, dense,
sparse, multipart, skewed, pair-producing, and reduction-producing outputs.

Hardware-sensitive choices belong to the higher-level device execution planner
defined by ADR-0047 and its implementation plan. This document defines the
operation's semantics, physical variants, work statistics, native carriers,
and evidence. It must not contain product-specific thresholds.

## Request Signals

- point in polygon
- point in multipolygon
- point-region refinement
- prepared polygon index
- y-edge bins
- candidate-part work queue
- warp-per-polygon
- adaptive exact predicate
- spatial query refinement
- sjoin point polygon
- query_any / query_aggregate / query_pair_aggregate
- SF100 Q11

## Open First

- `docs/decisions/0047-device-execution-planning.md`
- `docs/archive/2026-08-18-device-planning/device-execution-planning-implementation-plan.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0010-point-predicate-pipeline.md`
- `docs/decisions/0011-binary-predicate-refine-pipeline.md`
- `docs/architecture/point-predicates.md`
- `docs/architecture/spatial-joins.md`
- `docs/dev/native-physical-shape-ledger.md`
- `src/vibespatial/predicates/point_location_index.py`
- `src/vibespatial/predicates/point_location_index_kernels.py`
- `src/vibespatial/predicates/point_relations.py`
- `src/vibespatial/predicates/point_relations_kernels.py`
- `src/vibespatial/spatial/query.py`
- `src/vibespatial/api/sindex.py`

## Verify

- `uv run ruff check`
- `uv run pytest tests/test_point_in_polygon.py tests/test_binary_predicates.py -q`
- `uv run pytest tests/test_spatial_query.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "predicate"`
- `uv run vsbench run gpu-pip --scale 1m`
- `uv run vsbench run point-predicates --scale 10k`
- `uv run vsbench run spatial-query --rows 20000 --arg overlap_ratio=0.2`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- run the public 10K, 1M, and SF100 shootout gates recorded in the SF100 plan

## Risks

- Improving the Q11 partition geometry directly would create a benchmark
  specialization rather than a reusable predicate engine.
- Always constructing candidate-part queues can regress simple polygons and
  small public calls.
- Flattening MultiPolygons globally can lose parent-row semantics or multiply
  relation size before selectivity is known.
- A fixed y-bin count can trade one skew problem for another and duplicate
  long vertical edges excessively.
- A fixed lane/warp/block cutoff can overfit the development GPU.
- fp32 shortcuts can change boundary, hole, or crossing parity semantics unless
  ambiguity is conservative and fp64 remains authoritative.
- Materializing tri-state results before existential or grouped reductions can
  preserve unnecessary memory traffic even after the kernel improves.
- A public benchmark helper that reaches private native methods would hide
  whether ordinary users receive the acceleration.

## Mission

Make exact point-region classification a reusable private physical shape that
automatically accelerates existing public APIs while preserving exact public
semantics and native result carriers.

The plan succeeds only if:

- no Q11, SpatialBench, dataset, or partition identifier appears in production
  dispatch
- no new optimization-only public API is introduced
- simple workloads retain the current low-overhead lane path
- skewed multipart and long-edge workloads can expose parallelism below the
  candidate level
- classification is performed once when several native consumers need the same
  membership state
- pair, selection, and aggregate consumers remain bounded and device-native
- public 10K, 1M, and SF100 suites remain correct and do not regress
- consumer and datacenter devices may select different exact variants through
  ADR-0047

## Public API Boundary

The engine is private and is reached only through existing public behavior:

- `GeoSeries.within`
- `GeoSeries.covered_by`
- region-on-left `contains`, `covers`, and `contains_properly`
- `intersects`, `touches`, and `disjoint` for admitted point-region pairs
- `GeoSeries.sindex.query`
- `geopandas.sjoin`
- `SpatialIndex.query_any`
- `SpatialIndex.query_aggregate`
- `SpatialIndex.query_pair_aggregate`

Public benchmarks and workflow code may import only public vibeSpatial or
GeoPandas-compatible surfaces. Private native methods may be used by internal
unit tests but are not an acceptable performance harness.

The implementation does not add `optimized_pip`, explicit preparation calls,
device names, bin counts, or cooperative-width knobs to the public API.

## Logical Contract

The canonical internal classification is one byte per admitted candidate:

- `0`: exterior
- `1`: boundary
- `2`: interior

Predicate mapping occurs after classification:

- point `within` region and region `contains` point require interior
- `covered_by`, `covers`, and `intersects` include boundary
- `touches` requires boundary
- `disjoint` requires exterior
- point-region `crosses` and `overlaps` are false

Null and empty handling remains at the established predicate boundary. Indexed
views must preserve original row mapping and lineage. MultiPolygon part
reduction must be deterministic and match the canonical current/public oracle,
including valid touching components, holes, and boundary cases.

## Physical Contract

Logical inputs:

- points or point-family owned geometry
- Polygon or MultiPolygon owned geometry
- candidate point/region row pairs, possibly capacity-backed and range-sliced
- exact predicate or native consumer shape

Native inputs:

- `NativeSpatialIndex` or aligned candidate source
- `NativeGeometryMetadata`
- owned point and region buffers
- prepared point-region metadata cached on region lineage
- `NativeRelation` or capacity-backed candidate arrays when already available

Primitive work units:

- candidate pairs
- candidate-part pairs after conservative part-bounds screening
- active y-bin edge memberships
- exact orientation evaluations
- ambiguous candidate-parts requiring authoritative refinement
- relation pairs or input-sized reduction rows

Native outputs:

- capacity-aligned tri-state bytes when a later consumer needs locations
- compact `NativeRelation` or `NativeRelationSelection`
- `NativeDeviceSelection` for existential consumers
- `NativeExpression` for count/numeric consumers
- fused aligned count expressions for two-index membership comparison

Memory must remain:

```text
O(prepared region metadata + one candidate tile + one work tile + native output)
```

The full query-wide candidate relation or candidate-by-part product is never a
required intermediate for reduction consumers.

## Current Algorithm And Evidence

The current prepared path builds eight uniform y bins per Polygon part when the
region family exceeds a fixed coordinate threshold. Every edge is stored in
each bin overlapped by its y interval. Query execution assigns one CUDA thread
to each candidate pair. A MultiPolygon candidate serially walks its parts; each
part selects one y bin and serially walks its edge memberships. Exact adaptive
fp64 orientation resolves crossings and boundary cases.

This provides abundant candidate-level parallelism but leaves variable
multipart and edge traversal inside one lane. Adjacent lanes often share a
query region because point-grid output is query-major, but their y bins, part
counts, edge counts, and early exits may differ.

The synchronized SF100 Q11 profile on the reference RTX 4090 recorded:

- 600 million trip rows in 154 shards
- 770 paired spatial reductions across five region partitions
- 293.63 seconds, or 94.1% of wall time, inside paired spatial reduction
- about 73.27 million exact candidate lanes for one representative shard
- a projected 11.28 billion exact candidate lanes for the full query
- a steady representative shard dominated 92.7% by
  `point_in_multipolygon_prepared_part_y_index`
- one region partition taking roughly twice the time of another with similar
  candidate count, demonstrating geometry-work skew rather than candidate-count
  scarcity

Preparation count/scatter is cached and small relative to repeated exact
execution. Optimizing preparation alone cannot address the dominant cost.

## Structurally Different Solutions

### A. Improve the prepared index and retain one lane per candidate

Use per-part bounds, adaptive y-bin counts, and better work estimates while
keeping direct candidate classification.

Advantages:

- small staging overhead
- natural result ordering
- strong fit for simple Polygon and uniform work
- minimal extra memory

Limitations:

- MultiPolygon parts remain serial
- long active bins remain serial
- warp duration remains controlled by the most expensive lane

### B. Expand bounded candidate-part work and cooperate over edges

Screen parts, emit candidate-part descriptors, bucket by work, and assign
lanes, warps, or blocks to descriptors. Reduce locations back to candidates.

Advantages:

- exposes part and edge parallelism
- normalizes warp work
- supports shared/coalesced reads for long edge lists
- makes skew visible to device planning

Limitations:

- count/scan/scatter and reduction overhead
- bounded candidate-part expansion requires careful admission
- can regress ordinary short polygons if used universally

### Decision: an adaptive hybrid

Retain A as the portable low-overhead baseline. Add B as an admitted variant
for skewed physical work. ADR-0047 chooses between them and selects their work
buckets, precision, tile, and launch policy for the active device.

## Prepared Region Metadata

Evolve `PreparedPolygonPartYIndex` into a richer private prepared point-region
structure while retaining lineage-bound caching.

Required metadata:

- full fp64 part bounds
- parent geometry row for each part
- ring and coordinate spans
- per-part bin count and bin-offset span
- bin edge counts and offsets
- encoded edge entries
- edge-membership total and device bytes
- part and bin edge-count summaries
- preparation plan/version and source lineage
- readiness event/stream ownership

Bin layout may be uniform initially, but bin count becomes per-part and
planner-selected from edge distribution, membership duplication, memory cost,
expected scans saved, and reuse. Long edges spanning many bins are explicitly
charged to preparation and query cost.

Preparation admission must replace the fixed total-coordinate threshold with
an amortization decision:

```text
predicted direct edge scans
    versus
preparation cost + predicted indexed scans + persistent bytes
```

A one-shot small call should retain direct classification even when the region
geometry is large. Repeated public queries may reuse prepared state when
lineage and layout version remain valid.

## Work Queue Design

### Stage 1: candidate-part counts

For each candidate pair:

- validate family-row mappings and empty state
- inspect the candidate region's part span
- conservatively reject parts whose full bounds exclude the point
- count surviving candidate-part descriptors

Polygon candidates naturally produce zero or one descriptor. MultiPolygon
candidates produce only bounds-admitted parts.

### Stage 2: bounded scan and scatter

Use count/scan/scatter within one candidate tile. Each descriptor stores:

- original candidate position
- part index
- selected bin key
- edge offset and count
- stable part order

The descriptor layout is structure-of-arrays. No Python loop, host count loop,
or per-candidate allocation is permitted.

### Stage 3: work classification

Produce compact work-size summaries and queues suitable for the device planner.
The operation declares lane-, warp-, and block-capable variants but does not
hard-code device thresholds.

The queue may remain candidate/query-major when that preserves sufficient edge
locality. Sorting by `(part, bin)` is a separate admissible variant whose sort
cost must be measured; it is not mandatory merely because it improves locality.

### Stage 4: exact classification

- lane variant: one lane serially evaluates one candidate-part
- warp variant: lanes cooperatively traverse edge memberships; crossing parity
  reduces by XOR and boundary state by OR
- block variant: reserved for exceptional long lists when resource and
  calibration evidence justify it

Each variant produces an exact part location or marks the descriptor for an
authoritative refinement pass.

### Stage 5: candidate reduction

Reduce part locations to the original candidate with stable semantics. A
candidate that has no admitted part is exterior. The reduction must preserve
the canonical behavior for boundary/interior observations and invalid inputs
covered by the public compatibility contract.

### Stage 6: native consumption

Map or fuse candidate locations into the requested native result shape without
unnecessary intermediate compaction.

## Precision And Robustness

The current indexed implementation remains authoritative fp64 until a safe
staged predicate exists.

Complete device-planner integration declares two exact alternatives:

- native fp64 classification
- conservative centered-fp32 classification plus selective authoritative fp64
  refinement

For staged precision:

- storage, part bounds, bin selection, and authoritative results remain fp64
- a candidate-part is definite only when every contributing crossing and
  boundary decision is certified
- any uncertain edge makes the candidate-part ambiguous
- ambiguous descriptors are compacted and rerun through the full fp64 exact
  path
- no tolerance may silently convert boundary into interior or exterior

On devices with favorable fp64 throughput, ADR-0047 may select native fp64 and
avoid staging entirely. Precision policy never uses a device name.

## Native Consumer Fusion

### Pair-preserving query and join

Refined true pairs remain capacity-backed or compact into `NativeRelation` as
required. Public `sindex.query` and `sjoin` export only at their established
public boundary.

### Existential selection

`query_any`, semijoin, and anti-join atomically or segment-reduce accepted
locations into input-sized native selection state. They do not construct a
full exact relation first.

### Counts and numeric aggregation

`query_aggregate` consumes accepted locations directly into device expressions.
The reducer retains public duplicate and multiplicity semantics.

### Aligned two-index membership

For `query_pair_aggregate`, classify membership for each indexed region side
once. Compute shared membership by intersecting exact device keys or consuming
co-grouped streams. Do not evaluate one side once for shared membership and a
second time for its own count.

This removes the redundant exact pass observed in Q11, but the reusable
contract is classification-once consumption for any aligned two-index public
aggregate.

## Device Planner Inputs

The operation supplies ADR-0047 with:

- candidate count and tile capacity
- part-count histogram and skew
- active-bin edge-count histogram and skew
- predicted candidate-part count
- prepared metadata bytes and reuse estimate
- output carrier and reducer shape
- fp64 coordinate statistics
- persistent and transient byte formulas per variant
- lane, warp, block, sorted-group, and precision alternatives
- portable exact baseline

The device planner returns:

- direct versus prepared execution
- lane/warp/block work buckets
- native fp64 versus staged exact precision
- candidate and work tile capacities
- block/grid/dynamic-shared-memory policy
- optional calibration record
- scratch admission and reason log

The operation must not override these fields after accepting the plan.

## Implementation Milestones

### R0. Freeze semantics, public surfaces, and evidence

- Add this plan and link ADR-0047.
- Capture the current public predicate/query/join correctness corpus.
- Save Q11 stage and representative-shard kernel evidence.
- Add workload-independent shape fixtures.
- Record current prepared metadata bytes, active edge scans, and exact kernel
  time where measurable.

Exit: the project can be evaluated without relying on a Q11 result alone.

### R1. Add exact device-side work instrumentation

- Count candidate parts considered and bounds-rejected.
- Histogram surviving part counts and active-bin edge counts.
- Count edges scanned, fast skips, exact orientation calls, boundary hits, and
  ambiguous refinements.
- Keep counters compact, optional, and device-side until the profiling boundary.
- Add per-region/complexity timing attribution without per-row export.

Exit: expensive partitions are explained in primitive work units.

### R2. Create one private refinement seam

- Centralize point-region candidate classification in one executor.
- Route indexed, non-indexed, relation, selection, and aggregate consumers
  through it.
- Preserve existing kernels as the portable baseline.
- Accept native capacity/logical-count/source-offset contracts directly.

Exit: public behavior and baseline performance are unchanged, and later
variants need one integration point.

### R3. Enrich prepared region metadata

- Add full part bounds and parent mappings.
- Add variable per-part bin layout support.
- Replace the fixed coordinate threshold with planner-ready cost and byte
  estimates while retaining the old layout as a bootstrap variant.
- Validate cache lineage, readiness, and memory admission.

Exit: preparation is reusable, observable, and shape-driven.

### R4. Add bounded candidate-part staging

- Implement count/scan/scatter descriptors per candidate tile.
- Preserve original candidate and stable part order.
- Add overflow, capacity, and integer-width guards.
- Produce device work histograms for ADR-0047.

Exit: MultiPolygon work can be scheduled independently without a full-query
candidate-part materialization.

### R5. Add lane, warp, and block exact variants

- Reuse the current lane algorithm for short work.
- Add warp parity/boundary reduction for long work.
- Add a block variant only after R1/R4 evidence proves a remaining shape.
- Register resource, scratch, and admissibility metadata with ADR-0047.
- Retain portable CUDA alternatives for devices lacking optional features.

Exit: skewed work no longer forces every lane in a warp through the longest
serial loop.

### R6. Complete ADR-0047 Level-3 adoption

- Delegate precision, preparation, bucket thresholds, tiles, and launches to
  the device planner.
- Add bounded calibration between materially distinct exact variants.
- Validate distinct consumer/datacenter plans on real devices.
- Remove operation-local hardware thresholds and capability probes.

Exit: point-region refinement is the first complete device-planner adopter.

### R7. Fuse native consumers and classification reuse

- Fuse existential and count reducers with exact refinement output.
- Preserve pair output only for pair-preserving consumers.
- Implement classify-once aligned membership reuse.
- Delete redundant exact evaluations and transitional full-relation paths.

Exit: native output shape is chosen before expensive candidate allocation.

### R8. Public rollout and cleanup

- Verify every listed public API through strict-native mode.
- Remove benchmark/private-helper dependencies.
- Update the native physical shape ledger and precision compliance status.
- Run public 10K, 1M, full-profile, and SF100 gates.
- Publish cross-device plan selections and performance evidence.

Exit: the acceleration is a vibeSpatial library feature, not a shootout path.

## Correctness Matrix

Use upstream GeoPandas tests and mechanical Shapely oracles for:

- Polygon and MultiPolygon
- shell, hole, boundary, vertex, and exterior points
- horizontal, vertical, zero-length, duplicate, and nearly collinear edges
- touching valid MultiPolygon components
- nulls, empties, and indexed views
- reversed public operand orientation
- sparse and dense candidate relations
- candidate tiles split at every structural boundary
- lane/warp/block equivalence
- native-fp64 and staged-exact equivalence
- relation, existential, count, and shared-count consumer equivalence
- repeated prepared-index reuse and invalidation

Invalid geometry behavior must match the existing public compatibility
contract. If a physical variant requires valid input, validity is an explicit
admissibility condition with observable fallback or an existing exact variant;
it is not assumed silently.

## Shape Canary Matrix

Canaries are named for physical work, not workflows:

- many points / few simple polygons
- many points / many simple polygons
- one region with many disjoint parts
- many regions with highly skewed part counts
- short uniform active bins
- long uniform active bins
- mixed short and pathological bins in the same relation
- high edge-duplication vertical geometry
- hole-heavy polygons
- sparse candidates with expensive regions
- dense candidates with cheap regions
- repeated index reuse versus one-shot query
- pair output versus any/count/shared reduction

Each canary runs only through a public API for performance acceptance. Private
kernel tests remain useful for correctness and resource diagnostics.

## Performance Gates

- Simple/uniform public workloads must retain the baseline path and stay within
  3% of their clean median unless the new path is faster.
- Skewed multipart and long-bin canaries must show a material exact-refinement
  improvement on both a consumer and a datacenter device, each against its own
  baseline.
- No device is required to select the same variant or achieve the same
  speedup.
- Preparation must amortize under the measured reuse count or be declined.
- Compute planning and refinement must not export candidate-, part-, or
  edge-sized data.
- Peak temporary memory remains within one admitted candidate/work tile.
- Public 10K and 1M shootouts must not materially regress individually or in
  aggregate.
- The mandatory full profile must remain free of hidden materialization,
  fallback, and unexplained CPU-heavy stages.
- Q11 must improve from its current exact-refinement profile, but its query
  name, zone partitioning, and SQL shape are forbidden from production policy.
- The full SF100 suite remains the regression gate so a Q11 win cannot hide
  damage elsewhere.

## Proposed Code Ownership

Likely implementation surfaces:

- `src/vibespatial/predicates/point_region_refinement.py`: private planning and
  execution seam
- `src/vibespatial/predicates/point_region_refinement_kernels.py`: staging and
  exact execution variants
- `src/vibespatial/predicates/point_location_index.py`: prepared metadata and
  cache migration
- `src/vibespatial/predicates/point_relations.py`: semantic mapping and legacy
  adapter removal
- `src/vibespatial/spatial/query.py`: candidate-tile and consumer integration
- `src/vibespatial/api/sindex.py`: existing public result boundaries only
- `src/vibespatial/runtime/device_planning.py`: generic ADR-0047 facility, with
  no point-region concepts

The final file layout may follow nearby implementation conventions, but the
runtime planner and point-region executor remain separate modules and separate
contracts.

## Handoff Evidence

Every completed milestone records:

- revision and environment
- public APIs exercised
- correctness corpus status
- physical work counters
- selected `DeviceExecutionPlan`
- preparation and refinement stage times
- kernel resource and occupancy data
- peak live/reserved memory and largest admission
- D2H, synchronization, materialization, and fallback events
- 10K, 1M, full-profile, and relevant SF100 comparisons
- explanation for every regression or variant-selection change
