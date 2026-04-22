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

vibeSpatial's GPU coverage and public API compatibility have improved enough
that correctness now generalizes to new real-world workflows better than
performance does. Four new public shootouts added after the 95% coverage push
matched GeoPandas fingerprints at 10K scale, but two exposed severe performance
gaps:

- emergency response catchments: correct, but dominated by many-few overlay and
  grouped geometry reduction
- retail trade-area screening: correct, but dominated by repeated semijoins,
  anti-join filtering, mask construction, and dissolve
- insurance flood screening and habitat compliance: correct, but still slower
  than GeoPandas at 10K

This proves that API coverage is not the same as physical-plan performance
coverage. A workflow can dispatch to GPU-capable public operations and still
behave like CPU code if the chain repeatedly crosses host boundaries, rebuilds
intermediate public objects, or optimizes only a narrow benchmark shape.

ADR-0009 established staged fusion. ADR-0042 moved the architectural boundary
toward device-native results. This decision adds the missing performance
coverage rule: public workflows must be understood and optimized through
reusable physical operator shapes, not one benchmark script at a time.

## Decision

Adopt public API physical-plan coverage as a first-class performance target.

### Physical shapes

Benchmarks, profiles, and remediation work must classify public workflows by
the reusable physical shapes they exercise. The initial shape vocabulary is:

- semijoin: `sjoin(...).index.unique()` followed by `loc` or `take`
- anti-semijoin: spatial exclusion joins followed by inverse filtering
- many-few overlay: many source geometries against a small mask/catchment set
- mask clip: buffer or dissolve output reused as a clip/intersection mask
- grouped geometry reduce: dissolve, grouped union, or grouped bounds/area
- area-filter-after-overlay: exact overlay followed by scalar area predicates
- chained device pipeline: public calls that should remain device-resident
  across intermediate table, index, and geometry operations

This vocabulary is allowed to grow, but new benchmark fixes must name the
physical shape they improve or explicitly explain why the case is unique.

### Benchmark and profiling contract

`vsbench` shootout and pipeline artifacts should report enough information to
answer physical-plan questions, not only wall-clock comparisons:

- actual backend by stage, not only planner intent
- fallback events and reasons
- host materialization and transfer counts
- top hotpath stages by elapsed time
- row-flow counts through joins, overlays, filters, and grouped reductions
- public script name and physical shape tags

The artifact should distinguish:

- correctness coverage: fingerprint or oracle agreement
- API coverage: public operation dispatch support
- physical-plan coverage: reusable workflow shape performance

### Remediation rule

Workflow shootouts are canaries. They are not the optimization unit.

A workflow-specific fix is incomplete unless it does at least one of the
following:

- improves a named reusable physical shape
- adds a profiling rail that isolates the named shape
- documents a measured external-bound limit that prevents further improvement

Benchmark-specific private paths, special-case script detection, or
workflow-only shortcuts are rejected.

### Review rule

Changes touching public dispatch, joins, overlay, dissolve, IO, constructive
operations, `vsbench`, or workflow shootouts require a physical-plan review.
The review must answer:

- What reusable physical shape changed?
- Did already-device-resident data stay on device?
- Are semijoin, anti-join, groupby, dissolve, and materialization steps visible?
- Does the evidence report actual backend and host/device movement?
- Did the change improve a real workflow canary or a shape-level benchmark?

## Consequences

- GPU acceleration rate remains useful, but it is no longer sufficient evidence
  that performance generalizes.
- The roadmap must track physical-plan gaps separately from API coverage gaps.
- `vsbench` shootout profiling becomes part of the performance story, not an
  optional debugging aid.
- Public API compatibility work should prefer reusable execution shapes over
  narrow workflow patches.
- Some previously acceptable "operation is fast in isolation" evidence is now
  insufficient when the surrounding workflow shape is slow.

## Alternatives Considered

- Continue fixing slow shootouts one by one.
  Rejected because it would recreate private-path benchmark thinking at the
  workflow level and would not make performance generalize.
- Treat this as only a `vsbench` reporting issue.
  Rejected because better telemetry is necessary but not sufficient; the
  architecture also needs reusable physical operator shapes.
- Fold this into ADR-0042.
  Rejected because ADR-0042 decides the result boundary. This decision governs
  how public workflows are measured, reviewed, and optimized across boundaries.
