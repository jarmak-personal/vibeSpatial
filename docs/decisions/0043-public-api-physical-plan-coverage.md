---
id: ADR-0043
status: accepted
date: 2026-04-21
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - performance
  - public-api
  - benchmarks
---

# Public API Physical Plan Coverage

## Context

vibeSpatial's public API compatibility and GPU dispatch coverage have improved
enough that correctness now generalizes to new real-world workflows better than
performance does.

The April 2026 public shootout expansion exposed the gap. New workflows matched
GeoPandas fingerprints at 10K scale, but several were still slower than
GeoPandas. Profiles showed a mix of causes: many-few overlay, repeated
semijoins, anti-semijoin filtering, mask construction, grouped geometry
reduction, public composition overhead, and explicit compatibility fallbacks.

This means GPU/API coverage is not enough evidence that performance
generalizes. A workflow can use GPU-capable public operations and still lose if
the chain repeatedly crosses host boundaries, rebuilds intermediate public
objects, or only optimizes a narrow benchmark shape.

ADR-0009 established staged fusion. ADR-0042 moved the architectural boundary
toward device-native results. A follow-up staged-planner experiment then tested
whether broad eager-chain interception should become the next step. It reduced
some host/device movement and produced a small speedup on still-correct
workflows, but dropped the 10K public shootout sweep from 14/14 correct
workflows to 9/14. That trade is unacceptable.

The failed experiment is the important lesson: the repo needs physical-plan
coverage, but not an unbounded planner that rewrites public execution before
the relevant shapes have explicit semantic contracts.

## How To Read This ADR

This ADR is a warning and measurement guardrail, not a prescription to build a
broad public lazy planner.

Do not cite this decision as authority for eager-chain interception, whole-frame
lazy execution, or speculative pandas operation rewriting. The planner
experiment described above is the cautionary example: it improved a narrow
performance signal while breaking public workflow correctness.

The positive mandate is narrower:

- classify slow public workflows by reusable physical shape
- require explicit semantic contracts before native lowering expands
- measure composition overhead and export overhead as first-class regressions
- prefer local admissible lowering over broad cross-method rewrites
- decline safely when a workflow falls outside the declared contract

ADR-0044 defines the follow-up architecture: private native execution state
under exact public GeoPandas APIs. That architecture is the replacement path for
the rejected broad planner shape.

## Decision

Adopt public physical-plan coverage as a first-class performance target, but
treat it as a measurement and review discipline rather than a mandate to build
a broad planner.

Public performance evidence must distinguish:

- correctness coverage: output matches the GeoPandas contract
- API coverage: public operations dispatch to supported implementations
- physical-plan coverage: reusable workflow shapes perform well end to end

Workflow shootouts are canaries. They are not the optimization unit. A slow
workflow should first be classified by root cause:

- missing algorithm or kernel coverage
- kernel-floor throughput
- public composition overhead
- compatibility or export overhead
- external I/O or GeoPandas-denominator effects

Fixes should improve a reusable physical shape, add a shape-level profiling
rail, or document a measured external bound. Benchmark-specific private paths,
script detection, and one-off workflow shortcuts are rejected.

Reusable physical shapes may move from diagnostics to public execution only
when their semantic contract is explicit. At minimum, the contract must cover
row flow, geometry family, dimensional behavior, index/attribute semantics,
fallback behavior, and export behavior. Shape work should also have a
shape-floor benchmark, admissibility tests, and a full public shootout sweep
showing no correctness regression.

Broad staged execution or eager-chain interception is not the default answer.
It is appropriate only when evidence shows composition or export overhead is
the limiting factor and the affected shape has a strong enough semantic
contract to decline safely outside its admissible boundary.

Detailed shape vocabulary, profiling fields, and review checklists belong in
the benchmark/profiling roadmap docs. This ADR only sets the architecture
guardrail.

## Consequences

- GPU acceleration rate remains useful, but it is not sufficient evidence that
  performance generalizes.
- Public performance work must track physical-plan gaps separately from API
  coverage gaps.
- Operation-level benchmark wins are insufficient when the surrounding public
  workflow shape remains slow.
- `vsbench` shootout profiling is part of the performance contract, not an
  optional debugging aid.
- Public API compatibility work should prefer reusable execution shapes over
  narrow workflow patches.
- Staged execution should be introduced one admissible shape at a time, not as
  a cross-cutting runtime rewrite.

The April 2026 staged-planner experiment should remain documented as a rejected
path so future work does not repeat it under a different name.

## Alternatives Considered

- Continue fixing slow shootouts one by one.
  Rejected because it would recreate private-path benchmark thinking at the
  workflow level and would not make performance generalize.
- Introduce a broad staged planner or eager-chain interceptor immediately.
  Rejected because the April 2026 experiment improved a narrow timing metric
  while regressing public correctness across multiple workflow families.
- Treat this as only a `vsbench` reporting issue.
  Rejected because better telemetry is necessary but not sufficient; the
  architecture also needs reusable physical operator shapes.
- Fold this into ADR-0042.
  Rejected because ADR-0042 decides the result boundary. This decision governs
  how public workflows are measured, reviewed, and optimized across boundaries.
