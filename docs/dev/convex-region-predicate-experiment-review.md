# Exact Convex-Region Predicate Experiment Review

<!-- DOC_HEADER:START
Scope: RTX 4090 experiment findings and production recommendation for exact convex containing-region predicate lowering.
Read If: You are deciding whether to implement grouped vertex containment, choose a point-partition provider, or reuse convex certification.
STOP IF: You need the forward implementation sequence; use the convex-region predicate execution plan.
Source Of Truth: Reviewed summary of the 2026-08-23 convex-region experiment capsule.
Body Budget: 210/210 lines
Document: docs/dev/convex-region-predicate-experiment-review.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-22 | Intent |
| 23-28 | Request Signals |
| 29-34 | Open First |
| 35-46 | Verify |
| 47-52 | Risks |
| 53-74 | Verdict |
| 75-91 | Measurement Contract |
| 92-108 | Certification And Semantics |
| 109-131 | Complete Containment Results |
| 132-155 | Provider Evidence |
| 156-183 | Resolved Production Findings |
| 184-189 | Production Recommendation |
| 190-210 | Exit Conditions |
DOC_HEADER:END -->

Status: **IMPLEMENTED.** This preserves the experiment that returned REVISE. Production follow-up replaced the fixed boundary tolerance,
added bounded grouped reduction and conservative certification, and now selects
the measured shape through existing public predicates.

## Intent

Record the measured value, correctness limits, and production prerequisites
without converting private benchmark wiring into a public performance claim.

Production closure on 2026-08-23:

- the 10M boundary corpus has zero differences after exact selective
  orientation refinement
- offset-native grouped `ALL` uses output-sized scratch and complete-stage
  admission
- cached exact-sign certification rejects concave, holed, nonfinite, and
  self-intersecting targets
- public 16-vertex `within` measures 0.598 ms at 10K, 3.013 ms at 1M, and
  28.017 ms at 10M, or 4.77x, 7.94x, and 8.85x over the archived general path

## Request Signals

- convex containment experiment
- grouped vertex PIP results
- convex fast-path verdict

## Open First

- `docs/dev/convex-region-predicate-experiment-review.md`
- `docs/dev/convex-region-predicate-execution-plan.md`
- `docs/dev/bounded-accuracy-experiment-review.md`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run ruff check`
- rerun the ignored capsule commands recorded in its raw JSON artifacts

Focused precision, PIP, binary-predicate, native-grouped, robustness, and distance
verification passes 587/587. The shared full 1M profile completes
with zero fallback, zero compute D2H, and zero compute materialization. No
stage exceeds one second; the largest compute stage is
`read_geojson` at 70.733 ms.

## Risks

- Private experiment calls can be mistaken for already-public acceleration.
- Low disagreement counts can obscure an exactness violation.
- Provider timings from a different physical shape can be overgeneralized.

## Verdict

For one reused convex containing polygon, direct aligned point classification
plus a fixed-width grouped `ALL` reduction is the right first physical shape.
It is 7.57x to 10.16x faster than the current public polygon `within` baseline
at 1M source polygons, depending on target complexity. At 10M it remains 7.33x
to 12.37x faster before exact refinement.

Three blockers prevented the experiment from entering production:

1. The current point-region primitive classifies a narrow outside-boundary band
   as inside. The 10M containment corpus exposes two or three false positives,
   up to 8.88e-8 CRS units outside the convex target.
2. Generic `NativeGrouped.all` requests 28.61 GiB of histogram scratch at 10M.
   The fixed-width experiment avoids that shape, but no bounded general native
   grouped reducer is wired for production.
3. Certification is an experiment-only structural and fp64-turn check, not a
   lineage-bound, readiness-safe `NativeGeometryMetadata` certificate.

The production follow-up closed all three without weakening exact semantics or
exposing a provider control.

## Measurement Contract

Measurements were collected on 2026-08-23 at Git revision
`38f0de78a9431dee0170b75dc1ef43aafbe49d78` on an RTX 4090 with 24,564 MiB
VRAM, driver 580.173.02, and strict-native execution enabled.

The public baseline is `evaluate_binary_predicate("within", ...)` with an
N-row device-resident source and a single broadcast target. The experimental
lane uses the existing private exact point-region expression followed by either
`NativeGrouped.all` or an experiment-only fixed-width device reduction. Final
host export is included in both timed lanes; fixture construction and
certification are reported separately.

Raw evidence and the reproducible runner are under the ignored capsule
`benchmark_results/experiments/2026-08-23-convex-region/`. These private calls
measure a prospective production shape; they are not a public API claim.

## Certification And Semantics

The eight-case certification corpus produced zero false-positive convex
certificates across convex, reversed-winding, collinear, concave, holed,
self-intersecting, near-collinear, and 1e9-offset inputs. The first call paid
about 105 ms of compilation; warm turn checks were 0.23-0.37 ms.

Production certification now uses exact orientation signs, rejects nonfinite
and self-intersecting rings, is cached on the immutable source owner, and keeps
unknown or general masks on the existing exact path.

The 5,000-row semantic corpus matched the public polygon `within` result for
all 5,000 rows. It includes boundary contact, source holes, source
MultiPolygons, partially outside polygons, fully outside polygons, and random
rectangles. This supports the convex-set vertex theorem and exterior-vertex
provenance, but it does not clear the numerical boundary blocker found at 10M.

## Complete Containment Results

Times are medians in milliseconds. `Alternative` includes point classification,
grouped reduction, and final host export.

| Source rows | Target vertices | Public baseline | Alternative | Speedup | Oracle result |
|---:|---:|---:|---:|---:|---|
| 10K | 4 | 1.517 | 0.280 | 5.43x | exact |
| 10K | 16 | 2.850 | 0.245 | 11.62x | exact |
| 10K | 64 | 7.914 | 0.311 | 25.44x | exact |
| 1M | 4 | 14.872 | 1.464 | 10.16x | exact |
| 1M | 16 | 23.912 | 2.975 | 8.04x | exact |
| 1M | 64 | 64.294 | 8.489 | 7.57x | exact |
| 10M | 4 | 153.877 | 12.435 | 12.37x | 2 mismatches |
| 10M | 16 | 247.988 | 28.466 | 8.71x | 3 mismatches |
| 10M | 64 | 598.585 | 81.614 | 7.33x | 2 mismatches |

The archived three 16-vertex mismatches are source rectangles crossing the target by
2.99e-8, 3.34e-8, and 8.88e-8 CRS units. The public polygon predicate and
Shapely both return false; grouped point classification returns true. These are
not acceptable in exact mode. The production selective-refinement run removes
all three and has zero differences at 10M.

## Provider Evidence

The convex broadcast workload has no useful candidate-pair search space: every
source vertex is aligned against the same target. Direct classification avoids
index build, traversal, candidate materialization, and selector overhead.
Grid, quadtree, and Morton remain relevant to general many-point/many-region
work, not to this first convex broadcast lowering.

A fresh production point-region control sweep reinforces why provider choice
must remain shape-aware:

| Shape | Points | Auto | Grid | Morton | Observation |
|---|---:|---:|---:|---:|---|
| simple short polygon | 1M | 3.404 ms | 3.366 ms | 590.302 ms | grid wins |
| simple short polygon | 10M | 21.840 ms | 21.914 ms | 7,298.027 ms | grid wins |
| clustered extent skew | 1M | 370.857 ms | 370.792 ms | 14.338 ms | Morton wins 25.9x |
| clustered extent skew | 10M | capacity error | capacity error | 127.425 ms | Morton is feasible |

Automatic selection currently follows grid on the clustered shape and can fail
complete candidate admission at 10M. That is separate adaptive-runtime evidence,
not a reason to force Morton or quadtree into broadcast containment. Archived
true-hierarchy evidence remains useful for reused clustered point indexes, but
it did not establish a complete public Q11 win.

## Resolved Production Findings

### Exact boundary classification

The fixed epsilon was removed. Centered fp32 orientation now uses a conservative
envelope and calls adaptive exact orientation for every ambiguous edge before
grouped reduction.

### Bounded grouped reduction

The generic histogram-shaped `ALL` attempted a 28.61 GiB allocation for 50M
vertex results. Production now consumes authoritative offsets in a segmented
early-exit reducer with output-sized scratch and complete-stage admission. One
warp cooperates on each segment, including the million-value skew regression;
group counts fail closed above int32 capacity. Device guards require monotonic,
in-bounds offsets with dense endpoints before any value or sorted-order read.

### Conservative reusable certification

Positive target certification is now a typed device-state-owned derivative
with exact turn and intersection checks. A separate typed source carrier proves
one simple, non-empty, positive-area ring per admitted row. Both validate
source token, row mapping, residency, readiness, and immutable buffer lineage;
their bounded planning packets cross D2H once per source generation. Unknown,
holed, multipart, invalid, or degenerate source batches remain on the general
exact path. Selection caps each source ring at 65 coordinates; an acceptable
average cannot hide one quadratic skewed ring in an otherwise short batch.

## Production Recommendation

Ship exact refinement, bounded grouped reduction, conservative certificates,
and automatic public containment lowering. Keep provider controls private and
require many-region evidence before adding a hierarchy to this broadcast shape.

## Exit Conditions

Production acceptance satisfied these gates:

- the 10M boundary-stress corpus has zero oracle differences
- grouped reduction remains within admitted memory at 10M
- certification has zero false positives and native lifetime/readiness tests
- the automatic public path improves a stable region by at least 20%
- exact 10K and 1M protected workflows remain within their performance rails
- SF100 is 12/12 exact with zero fallback and remains within the protected
  single-run performance rail
- upstream predicates pass and the full profile has zero compute D2H,
  compute materialization, fallback, or stage above one second

Final source SHA-256 is `7dd29f6f053d672054e1fda3b75345da7e8e308e35ce21c2a7b18b7e1196e30c`.
10K is 14/14 exact and 2.663 s; 1M is 14/14 exact and 494.36 s versus the prior
accepted 576.50 s. SF100 is 12/12 exact with zero fallback at 469.01 s; Q11 is
227.30 s versus the prior 237.65 s. Eleven outputs are byte-exact; Q6 passes at
`rtol=1e-12`. The 11-pipeline full profile has zero compute
D2H/materialization/fallback and a maximum stage of 70.73 ms; one intentional
terminal GeoArrow export materializes 128 rows without runtime D2H.
