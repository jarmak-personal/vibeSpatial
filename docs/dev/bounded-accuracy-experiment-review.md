# Bounded-Accuracy Experiment Review

<!-- DOC_HEADER:START
Scope: RTX 4090 numerical and performance findings for explicit user-authorized point-region predicate and distance error budgets.
Read If: You are deciding whether to write the accuracy-budget ADR, expose approximate execution, or implement fp32 refinement envelopes.
STOP IF: You need exact convex containment design; use the convex-region predicate documents.
Source Of Truth: Reviewed summary of the 2026-08-23 bounded-accuracy experiment capsule.
Body Budget: 206/210 lines
Document: docs/dev/bounded-accuracy-experiment-review.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-7 | Preamble |
| 8-13 | Intent |
| 14-19 | Request Signals |
| 20-25 | Open First |
| 26-37 | Verify |
| 38-43 | Risks |
| 44-66 | Verdict |
| 67-83 | Measurement Contract |
| 84-110 | Point-Region Findings |
| 111-138 | Distance And Dwithin Findings |
| 139-159 | Interpretation |
| 160-181 | ADR Recommendation |
| 182-192 | Exact-Mode Integration Evidence |
| 193-206 | Exit Conditions |
DOC_HEADER:END -->

Status: **ADR WRITTEN; PUBLIC MODE DEFERRED.** ADR-0048 accepts the internal
error-envelope substrate. Exact point-region predicates now refine conservative
fp32 orientation ambiguity with adaptive exact arithmetic, but the empirical
metric results below still do not authorize a public accuracy budget.

## Intent

Record the measured performance and numerical envelope of unrefined fp32
point-region work, and decide whether it supports an accuracy-policy ADR or a
public implementation.

## Request Signals

- bounded-accuracy experiment
- fp32 spatial error results
- accuracy-budget verdict

## Open First

- `docs/dev/bounded-accuracy-experiment-review.md`
- `docs/dev/bounded-accuracy-execution-plan.md`
- `docs/architecture/precision.md`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run ruff check`
- rerun the ignored capsule commands recorded in its raw JSON artifacts

Focused precision, PIP, binary-predicate, native-grouped, robustness, and
distance verification passes 587/587. The shared full 1M profile completes
with zero fallback, zero compute D2H, and zero compute materialization. Its
largest setup, reduce, and emit stages are 70.733 ms, 69.790 ms, and
17.678 ms respectively; no stage exceeds one second.

## Risks

- Empirical maximum error is not a conservative proof.
- Shared fp64/fp32 behavior can hide a common boundary defect.
- CRS units and representability can make a numeric tolerance misleading.

## Verdict

Pure unrefined fp32 point-region classification improves complete device-return
and host-export time by roughly 1.12x-1.32x at 1M on the tested shapes. Pure
fp32 point-to-region distance improves 1M time by 1.76x-2.05x. At 10M the gains
narrow to 1.08x-1.37x for PIP and 1.11x-1.30x for distance.

The largest measured fp32 distance error is approximately 1.20e-7 CRS units.
However, no kernel computes a conservative per-row error interval, so this is
not yet an enforceable `AccuracyBudget`.

The boundary corpus also exposes that the current fp64 PIP primitive shares an
existing near-boundary inclusion tolerance with fp32. On sloped, concave, and
hole boundaries, both modes disagree with the analytic oracle inside a band as
wide as 1e-7. An approximation contract could authorize such a band; exact mode
must instead refine it.

That exact-mode finding is resolved in the production follow-up: the fixed
boundary tolerance was removed, and the former 10M convex-containment failures
now match the exact oracle with selective orientation refinement. The remaining
public-mode block is a conservative metric/budget proof across the supported
operation domain and device classes.

## Measurement Contract

Measurements were collected on 2026-08-23 at Git revision
`38f0de78a9431dee0170b75dc1ef43aafbe49d78` on an RTX 4090 with 24,564 MiB
VRAM, driver 580.173.02, and strict-native execution enabled.

The ignored capsule is
`benchmark_results/experiments/2026-08-23-bounded-accuracy/`. It deliberately
constructs private `PrecisionPlan` variants with fp32 compute and no refinement.
No public option reaches these variants and exact public behavior is unchanged.

Points are placed at logarithmic distances from 1e-12 through 1e-1 on both
sides of selected straight edges. The corpus covers axis-aligned convex,
sloped convex, concave, holed, and multipart regions at coordinate offsets zero
and 1e9. Boundary and distance oracles are analytic and computed outside timing.
Every timed result includes final host export.

## Point-Region Findings

At 1M rows and zero coordinate offset:

| Region shape | fp32 speedup | fp32 vs fp64 | fp64 vs analytic | Maximum oracle disagreement distance |
|---|---:|---:|---:|---:|
| square | 1.20x | 0 | 0 | 0 |
| diamond | 1.12x | 0 | 250,001 | 1.00e-7 |
| concave | 1.27x | 0 | 166,668 | 1.00e-9 |
| hole | 1.32x | 0 | 250,001 | 1.00e-7 |
| multipart | 1.29x | 0 | 0 | 0 |

This corpus intentionally overweights the decision boundary; disagreement
counts are not representative production rates. The meaningful contract fact
is the maximum distance from the exact boundary.

At 10M, diamond fp32 is 1.08x faster and hole fp32 is 1.29x faster. Both fp32
and fp64 disagree with the analytic boundary oracle for 2,500,001 rows, all no
farther than 1e-7 from the boundary. The shared result means unrefined fp32 did
not widen the observed PIP decision band, but it does not prove that another
shape cannot do so.

At a 1e9 coordinate offset the tiny requested displacements below fp64
representability collapse onto the actual boundary. Accuracy is evaluated from
the realized fp64 coordinates, not the pre-rounding requested displacement.
This is necessary to avoid claiming error that is absent from the stored input.

## Distance And Dwithin Findings

At 1M rows and zero coordinate offset:

| Region shape | fp32 speedup | Maximum absolute error |
|---|---:|---:|
| square | 1.76x | 3.37e-8 |
| diamond | 1.78x | 6.66e-8 |
| concave | 1.92x | 3.54e-9 |
| hole | 2.02x | 3.33e-8 |
| multipart | 2.05x | 1.00e-7 |

The fp64 analytic error is at floating-point noise scale for these fixtures.
At offset 1e9, maximum fp32 error is 1.68e-9 to 1.19e-7 after evaluating the
realized coordinates. No tested metric error exceeds 1.20e-7.

At 10M, complete distance speedups are 1.17x for square, 1.11x for diamond,
and 1.29x for the holed region at offset zero. The reduced gain relative to 1M
shows that arithmetic throughput is only part of the complete physical shape;
index arrays, dispatch, memory traffic, and export remain fp64/row shaped.

The capsule also evaluates `dwithin` decisions at thresholds from 1e-10 through
1e-2. At threshold 1e-8, the 1M diamond corpus has 33,140 fp32 decision
differences within 1e-8 of the threshold. Multipart has 83,333 differences no
farther than 9e-8 from the threshold. A budget of 1e-7 contains those observed
differences; a smaller budget does not. This remains empirical because the
kernel does not return a conservative interval.

## Interpretation

The work supports an accuracy-budget abstraction, not a user-visible dtype
switch:

- a 1e-7 CRS-unit budget appears sufficient for this corpus
- fp32 can materially accelerate metric work at 1M
- PIP benefit is smaller and shape dependent
- the current exact boundary ambiguity must be exposed rather than hidden
- complete-stage gains shrink at 10M despite greater arithmetic work
- fp16 has no proven representability or error model and was not prototyped

`precision="fp32"` must continue to mean an implementation preference under
exact semantics. It cannot double as authorization for wrong answers. Users
must declare acceptable spatial error, and the runtime may still choose fp64
when it is faster or required to prove the budget.

The experiment does not cover geographic/geodesic units, curved coordinates,
constructive topology, thin degeneracies beyond selected edges, or H100/H200
behavior. Those are explicit limits, not assumed generalizations.

## ADR Recommendation

Write an ADR that accepts the `AccuracyBudget` concept with these constraints:

1. Exact remains the zero-configuration default.
2. Predicate budgets refer to distance from the exact boundary or decision
   threshold; metric budgets are absolute result error.
3. V1 accepts projected CRS units only. Geographic and missing-CRS behavior
   must decline unless explicitly specified by the ADR.
4. Every admitted kernel produces or derives a conservative error interval.
   Empirical corpus maxima may test the proof but cannot replace it.
5. Exact mode refines every ambiguous row. Budgeted mode may skip refinement
   only when the proven interval fits the user budget.
6. Opt-in is per call or task-local context, never process-global mutable state.
7. Dispatch records budget, units, precision, refinement, variant, and reason.
8. Approximate outputs never become unmarked canonical geometry inputs.

Prefer a task-local execution context for GeoPandas-compatible methods and an
explicit per-call parameter for vS-specific functions, if both can share one
immutable policy owner. The ADR must resolve nesting and concurrency before API
implementation.

## Exact-Mode Integration Evidence

Exact refinement is production-ready; bounded accuracy remains deferred. Source
SHA-256 is `7dd29f6f053d672054e1fda3b75345da7e8e308e35ce21c2a7b18b7e1196e30c`.
10K is 14/14 exact at 2.663 s versus GeoPandas at 3.542 s; 1M is 14/14
exact at 494.36 s versus the prior accepted 576.50 s. SF100 is 12/12 exact with
zero fallback at 469.01 s. The full 1M profile has zero compute
D2H/materialization/fallback and a 70.73 ms maximum stage. Convex public
lowering is exact at 0.598/3.013/28.017 ms for 10K/1M/10M. These gates do not
authorize skipping refinement; that still requires a conservative proof.

## Exit Conditions

Public implementation remains blocked until:

- the ADR is accepted and the Native* feature hold is explicitly lifted or
  excepted for this surface
- point-region and distance kernels return conservative error envelopes
- every corpus difference lies inside the requested tolerance by proof and test
- exact mode refines the current 1e-7 PIP ambiguity band
- RTX 4090 and H100/H200 results show a complete-stage winning region
- exact 10K, 1M, SF100, upstream, and full-profile gates remain green

The correct next step is the ADR and exact ambiguity machinery, not a public
`coarse=True`, `fp32=True`, or undocumented tolerance flag.
