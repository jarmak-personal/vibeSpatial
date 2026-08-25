# Work-Amplification Research Plan

<!-- DOC_HEADER:START
Scope: Evidence plan for finding spatial workflows that create substantially more pairs, topology, capacity, or repeated preparation work than their public results retain.
Read If: You are profiling broad performance opportunities, adding amplification counters, prioritizing physical-shape work, or deciding whether a local optimization generalizes.
STOP IF: You already have a measured amplification finding and only need its operation-specific implementation plan.
Source Of Truth: Research methodology, metric contracts, experiment matrix, and graduation gates for work-amplification studies.
Body Budget: 316/340 lines
Document: docs/dev/work-amplification-research-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-14 | Intent |
| 15-29 | Request Signals |
| 30-43 | Open First |
| 44-56 | Verify |
| 57-75 | Risks |
| 76-103 | Research Thesis |
| 104-182 | Metric Contract |
| 183-212 | Instrumentation Contract |
| 213-243 | Experiment Matrix |
| 244-267 | Investigation Protocol |
| 268-294 | Milestones |
| 295-316 | Graduation And Stop Rules |
DOC_HEADER:END -->

## Intent

Find broad performance opportunities by measuring physical work that is later
discarded, collapsed, or rebuilt. The program studies public vibeSpatial
workflows and reusable physical shapes; it does not optimize benchmark scripts
or treat high GPU utilization as proof of efficient execution.

The first output is a ranked evidence map. Each entry must identify the
amplified physical object, the terminal information actually retained, the wall
time and memory attributable to the gap, and a counterfactual execution shape.
Only then may an operation-specific implementation plan begin.

## Request Signals

- work amplification
- cardinality reduction
- intermediate explosion
- relation-to-output ratio
- fragment amplification
- capacity utilization
- repeated index build
- repeated metadata classification
- grouped compression
- exact-refinement waste
- performance opportunity map
- working smarter

## Open First

- `docs/dev/work-amplification-research-plan.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/testing/profiling-rails.md`
- `docs/dev/native-physical-shape-ledger.md`
- `docs/dev/native-100ms-physical-shape-plan.md`
- `docs/dev/grouped-constructive-distributive-execution-plan.md`
- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/architecture/adaptive-runtime.md`
- `src/vibespatial/bench/profiling.py`
- `src/vibespatial/bench/shootout.py`
- `src/vibespatial/bench/pipeline.py`

## Verify

- `uv run ruff check`
- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_pipeline_benchmarks.py tests/test_bench_shootout.py -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- run current vibeSpatial 10K and 1M public shootouts against validated,
  reusable comparator artifacts
- run SF100 in isolated query processes and validate all twelve same-data
  fingerprints before comparing totals
- run focused synthetic shape canaries only to explain a public-workflow signal,
  never as the sole acceleration claim

## Risks

- Instrumentation can become the bottleneck if it adds per-candidate global
  atomics, synchronization, or row-sized transfers.
- A large ratio can be harmless when its numerator is cheap; wall time and
  peak-live bytes remain mandatory prioritization inputs.
- A low output count does not make reduction legal when fragment attributes,
  ordering, multiplicity, or lower-dimensional results remain observable.
- Device-specific timings can turn a research correlation into a brittle
  selector. Amplification evidence is not dispatch policy.
- Capacity slack is often intentional safety headroom. Treating all unused
  capacity as waste can reintroduce allocation fences or retry paths.
- Reusing a spatial index or certificate after row, coordinate, CRS, precision,
  or validity changes can silently corrupt results.
- Whole-script speedups can hide regressions in small/simple shapes, terminal
  export, or another workflow family.
- Rerunning an unchanged GeoPandas comparator wastes time and weakens evidence
  provenance. Reuse validated static baselines.

## Research Thesis

The strongest GPU gains often come from changing what is computed rather than
making the same intermediate faster. The diagnostic question is:

> What information does the next sanctioned consumer retain, and how much
> physical work is created before everything else is discarded?

The main loss patterns are:

1. A candidate or exact relation is allocated before existence, count, grouped
   aggregate, argmin, or top-k reduction.
2. Constructive fragments are assembled before a terminal union or dissolve
   erases fragment identity.
3. Dynamic capacity is sized for a conservative upper bound while the logical
   result is consistently tiny.
4. Exact predicates visit parts, rings, edges, or ambiguity lanes that cannot
   affect the consumer's final answer.
5. Identical source geometry is repeatedly indexed, classified, or certified
   across identity-safe public operations.
6. Grouped work preserves source-row topology even though only one set result
   per group survives.

The program measures these patterns without adding a broad lazy dataframe
planner. Eager public APIs remain the semantic boundary. Local lowerings may use
native carriers and explicit public reducers; cross-operation fusion requires
independent evidence that the same admissible shape recurs.

## Metric Contract

Every amplification record identifies `operation`, `stage`, `physical_shape`,
`source_lineage`, `consumer_kind`, `device`, and measurement boundary. Counts
must describe physical work, not merely public rows.

### Relation Amplification

Record coarse candidates, exact surviving pairs, unique left rows, unique right
rows, terminal rows, pair bytes, and terminal bytes.

- `coarse_to_exact = coarse_candidates / max(exact_pairs, 1)`
- `relation_to_terminal = exact_pairs / max(terminal_rows, 1)`
- `relation_byte_amplification = pair_bytes / max(terminal_bytes, 1)`

Zero-result stages retain their raw numerators and an explicit zero-denominator
flag; they are not represented by a misleading finite ratio.

### Constructive Amplification

Record source segments, split/intersection events, emitted fragments, fragment
coordinates, retained output parts, output coordinates, constructive bytes, and
peak-live bytes.

- `event_to_output = split_events / max(output_coordinates, 1)`
- `fragment_to_output = fragment_coordinates / max(output_coordinates, 1)`
- `peak_live_to_output = peak_live_bytes / max(output_bytes, 1)`

Dimension, keep-geometry-type policy, and attribute lineage accompany the
record so an attractive ratio cannot imply an illegal set-algebra rewrite.

### Capacity Amplification

Record admitted slots/bytes, allocated slots/bytes, logical used slots/bytes,
peak simultaneously-live bytes, and the allocation fence count.

- `slot_utilization = logical_slots / max(capacity_slots, 1)`
- `byte_utilization = logical_bytes / max(allocated_bytes, 1)`

Capacity is considered suspicious only when low utilization combines with
material wall time or memory pressure. Capacity-backed execution that removes a
synchronization boundary remains a valid winning shape.

### Refinement Amplification

Record bbox candidates, candidate parts, active bins, edge/orientation
evaluations, ambiguous lanes, exact refinements, survivors, and consumer early
terminations. Include p50, p95, p99, and maximum work per descriptor when skew
matters.

- `exact_work_per_survivor = exact_evaluations / max(survivors, 1)`
- `ambiguity_fraction = ambiguous_lanes / max(candidate_lanes, 1)`
- `early_exit_fraction = early_terminated / max(candidate_lanes, 1)`

### Rebuild Amplification

For indexes, bounds, validity, convexity, regularity, coverage, precision
summaries, and prepared topology, record source lineage, build count, build
time, persistent bytes, cache hits/misses, consumer count, and invalidation
reason.

- `avoidable_rebuild_seconds` is the sum of repeated compatible builds after
  the first build for the same lineage and contract.
- `consumer_reuse = consumers / max(builds, 1)`

This is evidence only. Reuse is legal only when lineage, row mapping, geometry,
CRS, precision, readiness, and cache-generation contracts validate.

### Group Compression

Record input rows, groups, maximum group size, input segments/coordinates,
pre-reduction fragments, output parts/coordinates, and group-skew percentiles.

- `rows_per_output_group = input_rows / max(output_groups, 1)`
- `group_geometry_amplification = pre_reduction_coordinates / max(output_coordinates, 1)`

Preserved attributes, ordering, multiplicity, null policy, and dimensional
policy are mandatory fields.

## Instrumentation Contract

Add amplification evidence to the benchmark/profiling substrate, not the
production adaptive planner. `ProfileStageTrace.metadata` and shootout timed
stages remain the machine-readable envelope.

Use three measurement levels:

- Level 0 reuses existing counters, carrier sizes, allocator telemetry, runtime
  events, and stage row flow. It adds no device work.
- Level 1 adds fixed-size device reductions for missing aggregate counts. Use
  register-, warp-, or block-local accumulation and at most one reduced update
  per block or an existing CCCL reduction. One bounded packet may cross at an
  explicit profiling boundary.
- Level 2 uses NVTX and Nsight for selected mysteries after Levels 0-1 identify
  a stage; it is not required for every workflow run.

Disabled Level 1 instrumentation must compile out. No global atomic belongs in
an inner edge, pair, fragment, or coordinate loop. No counter may introduce a
new production synchronization, retry path, fallback, or candidate-sized D2H.

Each metric records whether it is exact, sampled, derived, unavailable, or
invalid. Do not silently substitute public row counts for unavailable physical
counts.

Before accepting an instrumented rail, compare it with instrumentation disabled
using interleaved repeats. A consistent wall-time shift larger than ordinary
run noise requires redesign or an explicit correction experiment; the
instrumented time cannot become the performance claim.

## Experiment Matrix

Run four layers in order:

1. Focused deterministic shape corpus: uniform, sparse, dense, skewed,
   multipart, holed, empty, degenerate, and boundary-heavy inputs.
2. Public 10K shootouts: launch/composition regression and correctness floor.
3. Public 1M shootouts and the full pipeline profile: capacity, memory, and
   intermediate-shape evidence.
4. SF100 isolated queries: real geometry distributions, repeated batches, IO,
   and end-to-end balance.

Escalate selected canaries to 10M only after 1M identifies a scaling hypothesis
and the peak-live formula is safe. A larger run is evidence about scaling, not
a substitute for the public 10K regression floor.

For relation studies, cover pair-preserving, existence, count, grouped
aggregate, nearest, and top-k consumers. For constructive studies, cover
intersection, clip, difference, union, and dissolve with both retained and
discarded fragment attributes. For reuse studies, compare one consumer, many
consumers, identity projection, row take/reorder, mutation, and CRS/precision
changes.

Capture the same exact current vibeSpatial revision on RTX 4090 and at least one
datacenter GPU when a proposed selector depends on work distribution or device
throughput. Record GPU, driver, CUDA, allocator mode, memory cap, storage,
warmup, repeat, fixture hash, public source hash, and correctness fingerprint.

Reuse static comparator timings only when their complete identity packet and
fingerprint validate. Always rerun the current vibeSpatial candidate.

## Investigation Protocol

Each deep dive answers the same questions:

1. Which physical object is amplified, and which public result information
   survives?
2. What share of wall time and peak-live memory is attributable to it?
3. Does the ratio predict cost across at least two shapes or scales?
4. What counterfactual shape avoids the work without changing semantics?
5. Can users express that shape through public APIs today?
6. What admissibility certificate is required?
7. Does a forced A/B experiment improve the complete public stage after
   preparation, reduction, assembly, and export?
8. Does the alternative protect small/simple and skewed inputs?

Triage a finding for deep investigation when the stage is at least 5% of
workflow wall time or exceeds one second at 1M, and it also shows substantial
physical amplification, repeated compatible preparation, or memory pressure.
These are research queue thresholds, never production selector constants.

Prioritize by recoverable wall time first, then peak-live memory, recurrence
across workflows, semantic confidence, and implementation risk. A 1000x ratio
inside a 2ms stage ranks below an 8x ratio consuming 100 seconds.

## Milestones

### R0. Freeze Evidence And Schema

Define the versioned JSON fields, denominator rules, lineage identifiers, and
instrumentation levels. Capture uninstrumented 10K, 1M, full-profile, and
SF100 current-revision evidence.

### R1. Collect The Amplification Map

Add Level 0 coverage across existing rails, then the minimum Level 1 counters
needed for relation, constructive, capacity, refinement, rebuild, and group
compression studies. Publish raw artifacts and a tracked ranked summary.

### R2. Run Counterfactual Experiments

Take the top three independent findings. Force baseline and alternative shapes
over identical public batches, measure every stage, and validate exact outputs.
Do not add automatic selection.

### R3. Select The First General Workstream

Choose the finding with the largest demonstrated recoverable public wall time
and at least one independent supporting workload. Write its own implementation
plan with physical-shape, precision, memory, readiness, public API, and
regression contracts.

## Graduation And Stop Rules

A finding graduates from research only when:

- exact public fingerprints pass at all measured scales;
- the amplified work materially contributes to wall time or memory pressure;
- a forced counterfactual reduces that work and improves end-to-end public wall
  time, not only a private kernel;
- the shape is useful to two independent workloads or one public operation
  family with several materially different shapes;
- admissibility and invalidation rules are explicit;
- small/simple cases, unsupported inputs, and terminal export are protected;
- no hidden fallback, materialization, D2H, or post-launch retry is introduced.

Stop or archive a hypothesis when the numerator is cheap, the counterfactual
only moves cost elsewhere, the gain disappears outside one fixture, semantics
require the discarded information, or instrumentation cannot measure the work
without materially perturbing it.

Automatic fusion, new adaptive-runtime policy, and generic planner
infrastructure remain deferred. Reopen them only when two completed operation
families require the same proven contract.
