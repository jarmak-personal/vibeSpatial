# Evidence-First Point-Region Execution Plan

<!-- DOC_HEADER:START
Scope: Evidence-first plan for proving one exact point-region execution alternative through public APIs before extracting generic device-planning infrastructure.
Read If: You are profiling or optimizing point-in-polygon refinement, prepared polygon traversal, Q11's dominant predicate stage, or cross-device point-region execution.
STOP IF: You only need settled public predicate semantics or unrelated generic runtime behavior.
Source Of Truth: Active execution plan for evidence-driven point-region optimization and the gate for any future reusable device planner.
Body Budget: 468/500 lines
Document: docs/dev/evidence-first-point-region-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-14 | Intent |
| 15-28 | Request Signals |
| 29-51 | Open First |
| 52-65 | Verify |
| 66-81 | Risks |
| 82-102 | Why The Earlier Design Was Archived |
| 103-124 | Starting Evidence |
| 125-148 | Scope |
| 149-167 | Ownership Boundaries |
| 168-189 | Safety Contract |
| 190-204 | Public API Boundary |
| 205-222 | Evidence Questions |
| 223-246 | Instrumentation Design |
| ... | (6 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Improve exact point-versus-Polygon/MultiPolygon execution behind existing
public APIs without first creating a second runtime planner.

SF100 Q11 is the motivating profile, not the production specialization. The
first objective is to identify which physical work actually dominates the
current prepared point-region path. The second is to implement exactly one
alternative execution shape justified by that evidence. Reusable planning
infrastructure is deferred until at least two kernel families prove that the
same decision contract is needed.

## Request Signals

- point in polygon
- point in multipolygon
- point-region refinement
- prepared polygon index
- Q11 profile
- candidate-part work
- warp-per-candidate
- edge traversal skew
- cross-device portability
- evidence-first GPU optimization
- public spatial query performance

## Open First

- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md`
- `docs/decisions/0032-point-in-polygon-gpu-utilization-diagnosis.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0007-probe-first-adaptive-runtime.md`
- `docs/decisions/0002-dual-precision-dispatch.md`
- `docs/architecture/adaptive-runtime.md`
- `docs/architecture/point-predicates.md`
- `docs/testing/profiling-rails.md`
- `src/vibespatial/runtime/adaptive.py`
- `src/vibespatial/runtime/precision.py`
- `src/vibespatial/runtime/kernel_registry.py`
- `src/vibespatial/predicates/point_location_index.py`
- `src/vibespatial/predicates/point_location_index_kernels.py`
- `src/vibespatial/predicates/point_relations.py`
- `src/vibespatial/predicates/point_relations_kernels.py`

Historical design exploration is preserved under
`docs/archive/2026-08-18-device-planning/`. ADR-0047 is superseded and is not
implementation authority.

## Verify

- `uv run ruff check`
- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_point_in_polygon.py tests/test_binary_predicates.py -q`
- `uv run pytest tests/test_spatial_query.py -q`
- `uv run pytest tests/test_adaptive_runtime.py tests/test_precision_policy.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q`
- `uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -k "predicate"`
- `uv run vsbench run binary-predicates --scale 1m`
- `uv run vsbench run binary-predicates --scale 10k`
- run the public 10K, 1M, and SF100 shootout gates recorded in the SF100 plan
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Optimizing Q11 directly would hide a benchmark specialization in production.
- A second generic planner would duplicate existing adaptive-runtime policy.
- Instrumentation can distort irregular kernels if it adds inner-loop atomics.
- Queue construction can cost more than it saves for simple polygons.
- One thread per candidate can strand parallelism inside multipart or long-bin
  traversal, but one block per descriptor can also leave large GPUs idle.
- A cutoff measured on one RTX 4090 can select poorly on H100, A100, RTX 3090,
  constrained allocators, or future devices.
- Staged fp32 predicates are correctness work, not a free performance variant.
- Observed free memory is not a reservation and cannot make a future allocation
  safe under concurrency.
- A post-launch CUDA fault cannot reliably recover by rerunning a baseline in
  the same context.

## Why The Earlier Design Was Archived

Two independent reviews reached the same redesign verdict from different
directions.

The YAGNI review found that ADR-0047 introduced a second plan object with
precision, variant, tiling, launch, calibration, caching, and rollout ownership
already held by `AdaptivePlan`, `PrecisionPlan`, `KernelVariantSpec`, and the
CUDA runtime. It built most of a generic framework before measuring a second
exact implementation.

The principal GPU-hardware review found that the proposed object was still too
small for safe execution. The first adopter is a graph of count, scan, scatter,
refine, and reduce stages, not one kernel launch. Correct planning would need
stage-specific resource profiles, peak-live memory, stream dependencies,
allocator-backed reservations, and joint precision/variant admissibility.

Both conclusions can be true: the proposed abstraction was premature and,
despite its size, not rich enough to be safe. This plan avoids that trap by
proving the operation before extracting infrastructure.

## Starting Evidence

The synchronized SF100 Q11 profile on the reference RTX 4090 recorded:

- 600 million trip rows in 154 shards
- 770 paired spatial reductions across five region partitions
- 293.63 seconds, or 94.1% of wall time, inside paired spatial reduction
- about 73.27 million exact candidate lanes for one representative shard
- a projected 11.28 billion exact candidate lanes for the full query
- one steady shard spending 92.7% of its time in
  `point_in_multipolygon_prepared_part_y_index`
- materially different partition times at similar candidate counts

The current prepared path assigns one CUDA lane to one candidate pair. A lane
serially traverses admitted MultiPolygon parts and the selected y-bin edges.
Preparation is cached and comparatively small.

This evidence proves candidate count alone is an inadequate work estimate. It
does not yet prove whether the dominant reusable limiter is multipart
traversal, long active bins, warp divergence, fp64 arithmetic, repeated
classification, or another stage.

## Scope

This plan includes:

- stage and physical-work instrumentation of current public paths
- a representative shape corpus and consumer/datacenter measurements
- exactly one new authoritative fp64 execution alternative selected from the
  evidence
- operation-private selection using existing runtime owners
- public-API correctness and performance validation
- an explicit decision gate for any later generic planner

This plan does not include:

- a new `DeviceExecutionPlan` or generic device-planning module
- session or persistent calibration
- confidence, expiry, hysteresis, or rollout state machines
- a generic analytic GPU cost model
- product-name or benchmark-name dispatch
- simultaneous lane, warp, block, sorted, adaptive-bin, staged-fp32, and fused
  consumer implementations
- a new optimization-only public API
- staged fp32 point predicates without a separate precision and exactness proof

## Ownership Boundaries

Existing components remain authoritative:

- `AdaptivePlan` owns execution selection, workload probing, chunking, and
  structured dispatch evidence.
- `PrecisionPlan` is the sole precision-policy owner. The first alternative
  remains authoritative fp64.
- `KernelVariantSpec` registers proven executable variants and their narrow
  admissibility metadata.
- the CUDA runtime owns device attributes, occupancy queries, stream order,
  allocation, and immediate memory admission.
- ADR-0046 `PhysicalWorkEstimate` carries compact operation work counts.
- point-region modules own prepared metadata, queue construction, exact
  classification, and native consumer semantics.

No point-region policy is added to the generic runtime. No generic runtime
contract is added merely to make the first experiment look reusable.

## Safety Contract

The existing prepared fp64 lane path is the permanent exact baseline during
this program.

The alternative must be independently exact. Selection may choose between
proven exact implementations; it cannot make an unsafe implementation
admissible. Unknown work shape, missing required launch facts, integer overflow,
memory pressure, or inconsistent metadata selects the existing GPU baseline.
Strict-native execution never converts uncertainty into a silent CPU fallback.

All resource checks occur immediately before allocation and launch. A sampled
allocator state is advisory unless bytes have actually been reserved or
preallocated. The first implementation should use bounded tiles and allocate
through the active pool rather than invent a reservation token.

Specialized kernels write private outputs. Results become visible only after
the required stream dependency or completion boundary. Fail-closed recovery is
limited to pre-submission planning, validation, and allocation failures.
Post-submission CUDA faults propagate; they do not retry the baseline in the
same call.

## Public API Boundary

All profiling and acceleration must be reached through existing public
behavior, including applicable forms of:

- `GeoSeries.within`, `covered_by`, `contains`, `covers`, and `intersects`
- `GeoSeries.sindex.query`
- `geopandas.sjoin`
- `SpatialIndex.query_any`
- `SpatialIndex.query_aggregate`
- `SpatialIndex.query_pair_aggregate`

Private instrumentation may observe these paths. Benchmarks may not call a
private optimized executor directly and claim public acceleration.

## Evidence Questions

The first milestone must answer:

1. Is runtime proportional to candidate pairs, surviving candidate-parts,
   active-bin edge visits, or the maximum work assigned to one lane?
2. How much time is count/index preparation versus exact traversal versus
   result reduction?
3. Does multipart serialization or long-bin traversal dominate the slowest
   partitions?
4. How much redundant classification occurs across pair, existential, count,
   and paired-aggregate consumers?
5. Does the same physical metric predict performance on RTX 4090 and H100?
6. Is one common static crossover adequate, or does a currently available
   capability such as fp64 throughput ratio materially change it?
7. Does the proposed alternative improve the complete public stage after all
   staging, scan, scratch, and reduction costs?

## Instrumentation Design

Profiling mode records bounded stage times and aggregate work counters:

- input rows and candidate pairs
- Polygon/MultiPolygon candidate mix
- considered and bounds-surviving candidate-parts
- selected y-bin edge memberships
- exact edge/orientation evaluations
- maximum, p50, p95, and p99 edge work per candidate
- boundary and ambiguity events
- preparation, candidate generation, exact refinement, reduction, and export
  durations
- transient and persistent bytes
- launches, synchronization, D2H, materialization, and fallback events

Device instrumentation uses register-, warp-, or block-local accumulation and
one reduced update per block, or an existing CCCL reduction. It must not add a
global atomic inside the edge loop. Disabled instrumentation compiles out.

Only a fixed-size control packet may cross to the host at a declared profiling
boundary. Production selection must not introduce a candidate-, part-, or
edge-sized D2H transfer.

## Shape Corpus

Profile through public APIs with deterministic cases that separate likely
causes:

- many points against one simple short Polygon
- many uniform short Polygon candidates
- one very long active bin
- one large Polygon whose edges distribute evenly across bins
- sparse points against a many-part MultiPolygon
- dense points against a many-part MultiPolygon
- similar candidate counts with different surviving-part counts
- similar surviving-part counts with different active-edge counts
- pair-producing, existential, count, and paired-count consumers
- null, empty, boundary, hole, nested-ring, and degenerate compatibility cases

Run small and large scales. Tiny/simple shapes protect launch and staging
overhead; skewed shapes expose latent parallelism.

## Milestones

### E0. Freeze The Baseline

- capture the current public 10K, 1M, and SF100 results
- preserve the current Q11 stage profile and selected implementation
- record device, driver, allocator mode, warmup state, and transfer/fallback
  evidence
- add no new execution variant

Exit: later measurements can distinguish improvement from changed inputs,
warmup, allocator policy, or public workflow shape.

### E1. Attribute Physical Cost

- implement bounded profiling counters and stage timing
- run the shape corpus on RTX 4090 and H100-class hardware
- use simulated safety-contract profiles for A100, RTX 3090, constrained
  memory, and unknown optional attributes
- identify the smallest physical metric that predicts the slow cases
- publish the raw evidence and a short conclusion

Exit: one dominant hypothesis is supported strongly enough to choose one
alternative. If no hypothesis is supported, stop and improve measurement.

### E2. Choose Exactly One Alternative

Choose from evidence, not preference:

| Observed limiter | First alternative |
|---|---|
| long edge traversal within otherwise independent candidates | warp-per-candidate fp64 refinement |
| serial surviving parts dominate | bounded candidate-part descriptors plus fp64 reduction |
| one or a few extreme descriptors dominate | edge-chunk descriptors plus segmented parity/boundary reduction |
| repeated classification dominates | a separate classification-once consumer project |
| fp64 arithmetic dominates without structural underutilization | stop; open a separate precision-certification project |

Sorting, adaptive bin construction, a block variant, consumer fusion, and
staged fp32 are not bundled with the selected alternative.

Before implementation, freeze:

- exact input and output contract
- physical work unit
- scratch and peak-live byte formula
- launch mapping and resource limits
- expected winning and losing shapes
- measured go/no-go threshold

Exit: the alternative can be falsified by a focused benchmark and does not
require a generic planner.

### E3. Implement The Exact Alternative

- scaffold the owned GPU path using the new-kernel checklist
- preserve fp64 storage and authoritative fp64 classification
- use structure-of-arrays work descriptors when staging is required
- keep count/scan/scatter bounded to one admitted tile
- account for every stage's scratch lifetime and peak-live bytes
- validate launch limits through the existing CUDA runtime
- retain outputs and temporaries through their completion dependency
- register the variant in the existing kernel registry

If the selected implementation has several stages, the point-region executor
owns their ordered local contract. This is not generalized into a library-wide
execution-graph object during E3.

Exit: GPU oracle tests pass with the baseline forced, alternative forced, and
normal automatic selection. No private benchmark-only entry point exists.

### E4. Add Minimal Operation-Private Selection

- extend the existing workload profile with only the measured discriminator
- derive a static crossover from the E1/E3 evidence
- select baseline or alternative at a declared chunk boundary
- use current runtime device facts only if the evidence shows they change the
  crossover materially
- record variant and reason through existing dispatch observability
- preserve explicit execution and precision requests

There is no online comparison, persistent cache, confidence score, expiry,
hysteresis, or automatic demotion. Unknown or unsupported shapes choose the
baseline.

Exit: the operation contains no product, query, dataset, or partition name and
does not duplicate precision, allocation, occupancy, or fallback ownership.

### E5. Validate Public Value

- compare exact results with the existing host oracle and upstream contracts
- rerun the shape corpus on consumer and datacenter devices
- rerun public 10K, 1M, and SF100 shootouts
- rerun the full end-to-end profile and inspect every 1M stage
- verify strict-native selection, zero unexpected compute D2H, bounded memory,
  and no silent materialization or fallback
- publish per-device selected variant, physical work, stage time, end-to-end
  time, memory, synchronization, and transfer evidence

Minimum go/no-go rails:

- zero correctness differences
- no public-path fallback or materialization regression
- no more than 5% regression on simple/uniform protected cases after noise is
  bounded
- at least 20% complete-stage improvement on the targeted skew shape after all
  staging and reduction overhead
- a measurable end-to-end improvement in the motivating public workflow

If the alternative misses these rails, remove it or return to E1. Do not tune a
structurally losing shape.

### E6. Generalization Decision Gate

Do not extract a generic device planner merely because E5 succeeds. Reconsider
shared infrastructure only when a second kernel family has:

- at least two independently exact implementations
- a measured cross-device or cross-shape selection problem
- the same physical inputs and decision boundary as point-region execution
- evidence that existing `AdaptivePlan`, `PrecisionPlan`, kernel registry, and
  CUDA runtime ownership cannot express the choice cleanly
- enough repeated local code that extraction removes rather than adds policy

If that gate is met, write a new ADR. It must address the hardware review's
minimum safety requirements:

- represent an ordered multi-stage execution graph, not one launch
- separate required launch-safety facts from optional cost facts
- rank coherent precision, robustness, and variant tuples
- calculate checked 64-bit peak-live memory across stage lifetimes
- use allocator-backed reservations or preallocated arenas where concurrency
  requires authoritative admission
- bind plans to device, context generation, allocator epoch, compiled function
  hashes, and stream dependencies
- treat occupancy as an input to launch candidates, not the performance answer
- bound calibration by launched primitive work, not a host timeout
- restrict fail-closed recovery to pre-submission failures

Until those conditions exist, the correct architecture is no new planner.

## Cross-Device Policy

Portability does not require different variants on every GPU. One exact
implementation and one physical cutoff are preferable if they remain fast on
both tested device classes.

Use hardware capability only when measurements prove it changes the decision.
Never infer performance from a product name or compute capability alone.
Required launch limits are always queried and validated. Optional performance
facts may be absent without making a launch unsafe.

Real-device evidence is required on RTX 4090 and H100-class systems for the
first decision. A100 and RTX 3090 must have safety-contract coverage and should
join periodic real performance rails before any future claim of mature support
across all four named targets.

## Correctness Matrix

Baseline, forced alternative, and automatic selection cover:

- Polygon and MultiPolygon
- holes and nested rings
- boundary, interior, and exterior points
- empty, null, and invalid inputs under existing public contracts
- duplicate candidates and stable pair ordering
- `within`, `covered_by`, `contains`, `covers`, `intersects`, and `disjoint`
  where currently admitted
- pair, existential, count, and paired-count output semantics
- tile boundaries and integer-limit canaries
- adversarial coordinate scale and non-finite handling according to current
  precision policy

The existing Shapely oracle fixture remains the mechanical host reference.
The host oracle is a test boundary, never part of the GPU execution path.

## Required Handoff Evidence

Each completed milestone records:

- commit and environment identity
- public command used
- input shape and physical-work counters
- selected implementation and reason
- stage and end-to-end times
- peak persistent and transient device bytes
- launches, synchronization, D2H, materialization, and fallback totals
- correctness digest and row counts
- conclusion, including falsified hypotheses

Performance claims without raw current-revision artifacts are not acceptance
evidence.

Current execution evidence and falsified alternatives are recorded in
`docs/dev/point-region-execution-evidence.md`.

## Completion

This plan is complete when one evidence-selected exact alternative improves a
general skewed point-region workload through public APIs on consumer and
datacenter hardware, protected simple workloads and the public 10K/1M/SF100
suites do not regress, and the repository has made an explicit evidence-based
decision either to keep selection operation-private or open a new ADR after a
second kernel family satisfies E6.
