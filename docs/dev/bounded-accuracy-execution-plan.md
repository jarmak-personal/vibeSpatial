# Bounded-Accuracy Execution Plan

<!-- DOC_HEADER:START
Scope: Research, ADR, API, and implementation plan for explicit user-authorized spatial error budgets that may admit lower-precision GPU execution while preserving exact behavior by default.
Read If: You are designing approximate predicates or metrics, user accuracy thresholds, fp32/fp16 execution without exact refinement, or tolerance-aware dispatch.
STOP IF: You only need exact convex polygon certification or existing exact staged-fp32 behavior.
Source Of Truth: Active plan for defining and proving a public bounded-accuracy execution contract.
Body Budget: 386/390 lines
Document: docs/dev/bounded-accuracy-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-26 | Intent |
| 27-40 | Request Signals |
| 41-63 | Open First |
| 64-74 | Verify |
| 75-91 | Risks |
| 92-124 | Relationship To Existing Precision |
| 125-180 | Proposed Accuracy Contract |
| 181-233 | V1 Scope And Semantics |
| 234-275 | Runtime And Public API Shape |
| 276-368 | Milestones And Exit Gates |
| 369-379 | Workspace And Evidence Isolation |
| 380-386 | Completion Condition |
DOC_HEADER:END -->

## Intent

Implementation status (2026-08-23): ADR-0048 accepts the internal numerical
error-envelope contract while deferring the public accuracy policy. Exact
point-region execution now uses a conservative centered-fp32 orientation
envelope with adaptive exact fallback, and nearest metrics carry their existing
distance bound through the shared native envelope carrier. No public bounded
mode is authorized yet.

Allow users to exchange a declared, bounded amount of spatial accuracy for GPU
performance when exact double-precision-like answers exceed the fidelity of the
source data or application.

Exact GeoPandas-compatible behavior remains the default and the compatibility
contract. Approximation is explicit, scoped, observable, and expressed as an
error budget. Users do not select `fp16` or disable refinement as a semantic
request; they declare acceptable error and the runtime chooses an implementation
that proves it fits that budget.

The first research target is point-region predicates and point-family distance,
where error can be defined relative to a boundary or numeric threshold. General
constructive topology is out of scope until deterministic snap-grid semantics
are separately designed and accepted.

## Request Signals

- coarse accuracy mode
- spatial tolerance
- bounded error
- approximate predicate
- fp32 without refinement
- fp16 spatial kernel
- accuracy threshold
- noisy geospatial data
- boundary uncertainty
- tolerance-aware dispatch
- snap grid semantics

## Open First

- `docs/dev/bounded-accuracy-execution-plan.md`
- `docs/architecture/precision.md`
- `docs/architecture/robustness.md`
- `docs/decisions/0002-dual-precision-dispatch.md`
- `docs/decisions/0004-robustness-strategy.md`
- `docs/decisions/0031-determinism-and-reproducibility.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0048-bounded-accuracy-spatial-execution.md`
- `docs/dev/convex-region-predicate-execution-plan.md`
- `src/vibespatial/runtime/precision.py`
- `src/vibespatial/runtime/robustness.py`
- `src/vibespatial/runtime/adaptive.py`
- `src/vibespatial/kernels/predicates/point_in_polygon.py`
- `src/vibespatial/spatial/point_distance.py`
- `src/vibespatial/spatial/distance_metrics.py`

This plan proposes a new public semantic contract. It requires an accepted ADR
and an explicit lift or exception to the active Native* feature hold before the
public API milestone begins.

## Verify

- `uv run ruff check`
- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_precision_policy.py -q`
- `uv run pytest tests/test_point_in_polygon.py tests/test_binary_predicates.py -q`
- `uv run pytest tests/test_distance_owned.py tests/test_point_distance.py tests/test_spatial_query.py -q`
- run exact and budgeted numerical corpora on RTX 4090 and H100/H200-class GPUs
- run public 10K, 1M, and SF100 exact regression gates
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- A raw `fp32=True` switch gives no meaningful accuracy guarantee.
- Boolean predicates have no numeric result error; their contract must refer to
  distance from the exact decision boundary.
- CRS coordinate units may be angular, unknown, or unsuitable for a physical
  tolerance.
- Chunk-relative centering or quantization origins can make results depend on
  partitioning.
- Approximate bounds that round inward can create uncontrolled false negatives.
- Lower precision may not accelerate branch-heavy kernels or datacenter GPUs.
- Approximation can silently accumulate if canonical geometry storage changes.
- Constructive topology can change discontinuously under tiny perturbations.
- A context-wide policy without clear scope and dispatch evidence can surprise
  users.
- Measured disagreement rates are not a proof that an error bound is respected.

## Relationship To Existing Precision

`PrecisionMode` and the proposed accuracy contract answer different questions.

| Contract | Question | Current default |
|---|---|---|
| `PrecisionPlan` | Which arithmetic and compensation implement the operation? | device- and kernel-aware |
| `RobustnessPlan` | How are ambiguous numerical decisions resolved? | exact predicates/topology |
| proposed `AccuracyBudget` | Which deviations has the user authorized? | none |

Current consumer-GPU predicate execution may use centered fp32 for coarse work,
then selectively refine ambiguous cases in fp64 or stronger arithmetic. An
explicit `precision="fp32"` request still does not authorize an incorrect
predicate result. That behavior remains unchanged.

The accuracy budget constrains planning; it does not directly choose compute
precision. An implementation may use fp64 and still be fastest. An fp32 or fp16
variant is admissible only when its conservative error envelope fits the active
budget and complete-stage evidence shows a win.

Canonical owned geometry storage remains fp64. Centered fp32, fp16, quantized
coordinates, and packed index bounds are execution-local derivatives tied to
source lineage. Outputs must not become the unmarked canonical input to later
operations.

This plan is independent of the exact convex-region plan:

- exact convex certification never consumes an accuracy budget
- exact execution supplies the oracle and fallback for this work
- bounded execution may reuse exact shape metadata after that metadata lands
- each track uses a separate branch/worktree and evidence directory
- GPU profiling and performance runs execute sequentially, not concurrently

## Proposed Accuracy Contract

The public concept is tentatively named `AccuracyBudget`; the ADR may choose a
different final name. Avoid `coarse`, which already names a kernel class, and
avoid treating data type as user-visible accuracy.

The default is equivalent to:

```python
AccuracyBudget.exact()
```

The first bounded form should express operation-relevant tolerances:

```python
AccuracyBudget(
    boundary_tolerance=0.25,
    metric_absolute_tolerance=0.01,
    units="crs",
)
```

Semantics:

- `boundary_tolerance=t`: a predicate may differ from the exact result only
  when the decisive geometry is within distance `t` of the exact topological
  boundary or decision threshold.
- `metric_absolute_tolerance=t`: a finite metric result must differ from the
  exact result by no more than `t`.
- null, empty, index, ordering, and type semantics are never approximate.
- a missing tolerance for an operation class means exact behavior for that
  class.

A percentage disagreement rate is not part of the contract. Ten million
correct rows do not excuse one error outside the declared tolerance.

CRS rules for V1:

- projected CRS: tolerance is expressed in CRS coordinate units
- geographic CRS: bounded mode declines until angular versus geodesic semantics
  are explicitly requested and implemented
- missing CRS: bounded mode requires an explicit acknowledgement that values
  are in unnamed coordinate units
- negative, NaN, infinite, or unrepresentable tolerances are rejected

The ADR must decide whether `units="crs"` is sufficiently explicit or whether
a units object is required. It must not silently interpret degrees as meters.

Determinism rules:

- coordinate centering and quantization use a stable origin independent of
  chunk size, stream, or device
- the same device/variant/input/budget must be repeatable
- cross-device results may differ only inside the authorized uncertainty band
- dispatch records enough policy and variant data to reproduce a result

## V1 Scope And Semantics

### Point-Region Predicates

The classifier produces `DEFINITE_TRUE`, `DEFINITE_FALSE`, or `AMBIGUOUS` with
a conservative numerical envelope.

- Exact mode refines every ambiguous result through the existing exact path.
- Bounded mode may resolve an ambiguous result without exact refinement only
  when it proves that any disagreement is confined to the requested boundary
  tolerance.
- If the proof is missing or wider than the budget, refine exactly.

For point-in-region, the semantic distance is the minimum point-to-region
boundary distance. The experimental kernel may accumulate a conservative lower
or interval bound while traversing active edges. If computing that proof costs
more than exact refinement, the variant is rejected rather than weakening the
contract.

### Distance And `dwithin`

Distance kernels return an interval or a conservative error bound around the
computed value.

- A metric value is admissible when the interval width fits
  `metric_absolute_tolerance`.
- `dwithin(d)` is exact when the interval lies wholly on one side of `d`.
- A budgeted `dwithin` result may differ only when the exact distance is within
  the authorized tolerance of `d`.
- Otherwise the pair is selectively refined.

### Index And Candidate Bounds

Index derivatives may use fp32 or lower precision only with conservative
outward rounding plus the active error envelope. Candidate generation may add
false positives; it must not introduce false negatives outside the authorized
boundary band.

### Deferred Surfaces

The first version excludes:

- overlay, union, intersection, difference, buffer, clip, and make-valid
- validity certification or repair under approximate topology
- area/centroid reductions until their error models are separately proven
- geographic/geodesic tolerance
- persistent canonical fp32/fp16 geometry storage
- implicit approximation selected only from device capability or dataset name

Constructive work requires a later snap-grid contract defining grid size,
origin, rounding, topology, provenance, and output guarantees. It is not an
automatic extension of point predicate success.

## Runtime And Public API Shape

`AccuracyBudget` should be a separate immutable runtime policy passed into
precision and robustness selection. It does not replace or duplicate
`PrecisionPlan`, `RobustnessPlan`, `AdaptivePlan`, or `KernelVariantSpec`.

Planning order:

1. Resolve public semantics, CRS, and the explicit accuracy budget.
2. Inspect residency, shape metadata, and physical work estimates.
3. Enumerate variants whose proven error envelope fits the budget.
4. Rank complete execution shapes using existing adaptive ownership.
5. Admit memory immediately before allocation and launch.
6. Record precision, robustness, accuracy, variant, and reason.
7. Export only at the public terminal boundary.

The ADR must compare two public API options:

1. Per-call `accuracy=` on vS-specific public functions such as
   `evaluate_binary_predicate`, while GeoPandas-compatible methods remain exact.
2. An explicit task-local context manager that preserves GeoPandas signatures:

   ```python
   with vibespatial.execution(
       accuracy=vibespatial.AccuracyBudget(boundary_tolerance=0.25)
   ):
       result = points.within(regions)
   ```

A process-global mutable option is rejected. If a context manager is selected,
it must use task/thread-local state, nest correctly, restore on exceptions, and
be visible in every dispatch event. Exact mode must remain obvious outside the
scope.

Public pandas results cannot reliably retain custom approximation metadata.
Therefore explicit opt-in and runtime observability are required; result attrs
may be supplemental but cannot be the sole record.

Strict-native behavior remains orthogonal. A budgeted GPU path may decline
observably before submission; it may not switch silently to CPU, exceed the
budget, or retry after a CUDA fault.

## Milestones And Exit Gates

### A0. Freeze Exact Behavior And Write The ADR

- preserve exact precision, robustness, 10K, 1M, SF100, and full-profile
  baselines
- specify error terms, CRS units, determinism, nesting, observability, and
  unsupported surfaces
- compare per-call and scoped-context public APIs
- obtain explicit approval for the semantic/API change and feature-hold lift or
  exception

Exit: an accepted ADR defines what bounded accuracy means before production
kernels or public options are added.

### A1. Build The Numerical Corpus And Proof Harness

- generate points at logarithmic distances from polygon edges and vertices
- cover convex, concave, holed, multipart, thin, near-degenerate, and huge-offset
  regions
- generate distance pairs around `dwithin` thresholds
- sweep projected coordinate magnitudes, spans, tolerance values, centering,
  and chunk boundaries
- compute exact host or authoritative fp64 references and exact boundary
  distances outside timed regions

Exit: the harness identifies every disagreement outside the allowed band and
cannot reduce correctness to aggregate error rate.

### A2. Prototype Point-Region Variants

- instrument current ambiguity and refinement rates
- implement one centered-fp32 no-refine candidate with a conservative envelope
- consider fp16 or quantized coordinates only if fp32 leaves a measured
  bandwidth/compute opportunity and representability is proven
- keep bounds outward-conservative
- measure 10K, 1M, 10M, sparse, dense, boundary-heavy, and real-data shapes
- run on RTX 4090 and H100/H200-class hardware

Exit: at least one variant has zero differences outside the requested boundary
tolerance and improves complete-stage time by at least 20% in a general shape.
Otherwise reject approximate PIP as a production feature.

### A3. Prototype Metric And `dwithin` Variants

- produce conservative distance error intervals
- selectively refine pairs whose interval exceeds the budget or overlaps an
  exact decision threshold
- retain distances and masks as native expressions
- measure arithmetic, memory bandwidth, compaction, refinement, and reduction
  separately and end to end

Exit: every admitted metric respects the absolute error bound and every
`dwithin` difference is confined to the threshold band.

### A4. Implement The Public Accuracy Policy

- add the accepted immutable policy and scoped propagation mechanism
- keep exact as the zero-configuration default
- wire only proven V1 operations
- reject unsupported CRS and operations explicitly
- record budget, units, error envelope, precision, refinement, and variant
- add nesting, exception restoration, concurrency, and strict-native tests

Exit: no approximate execution is reachable without explicit public opt-in and
no opt-in leaks beyond its declared scope.

### A5. Add Minimal Automatic Selection

- register only variants with proved envelopes and benchmark floors
- choose from current device capabilities and physical work, never product name
- use the requested tolerance as an admissibility constraint, not a performance
  promise
- exact-refine when uncertainty, memory, readiness, or metadata is insufficient
- propagate post-submission faults

Exit: selection cannot exceed the user's budget and has no dataset/query rule.

### A6. Validate Public Value

- exact mode remains oracle-identical and performance-neutral within 5% noise
- bounded predicates have zero disagreements outside the requested band
- bounded metrics have zero errors above the requested absolute tolerance
- admitted bounded shapes improve complete-stage time by at least 20%
- publish refinement fraction, interval widths, candidate counts, memory,
  launches, synchronization, D2H, and end-to-end time by device
- rerun public 10K, 1M, SF100, upstream suites, and the mandatory full profile
- inspect all 1M sparkline stages for hidden CPU work

An implementation that merely uses fp32 but does not improve the complete
public operation is removed. An implementation that is faster but cannot prove
the error contract is rejected.

## Workspace And Evidence Isolation

Use a dedicated branch or worktree after the ADR gate. Store raw artifacts
under a track-specific ignored directory such as
`benchmark_results/experiments/bounded-accuracy/`. Do not reuse mutable
allocator, warmup, or timing state from convex-region or SF100 runs.

Run GPU experiments serially on the single workstation. A clean experiment
sequence records active processes, device clocks, driver, allocator policy,
compiled-kernel warmup, and the exact source revision before each comparison.

## Completion Condition

The plan is complete when exact remains the default, users can explicitly state
a meaningful spatial error budget through a public API, every admitted variant
proves it stays inside that budget, lower precision produces a complete-stage
win on consumer and datacenter-class evidence where selected, unsupported
surfaces decline clearly, and protected exact workflows do not regress.
