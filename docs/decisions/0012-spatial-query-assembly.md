---
id: ADR-0012
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - sindex
  - joins
---

# Spatial Query And Join Assembly

## Context

`o17.4.3` needs to turn the Phase-3 coarse filters and Phase-4 predicate engine
into GeoPandas-compatible `sindex.query`, `sjoin`, `dwithin`, and nearest
results without losing pandas join semantics.

## Decision

Keep pandas/DataFrame join assembly in the vendored GeoPandas helpers and land a
repo-owned spatial query engine as the future GPU/performance seam.

The owned engine is responsible for:

- flat spatial index candidate generation
- exact predicate refine on compacted candidate rows
- distance-expanded candidate generation for `dwithin`
- bounded nearest reduction when `max_distance` is provided

The vendored layer remains responsible for:

- join `how` semantics
- suffix and overlapping-column handling
- index restoration
- geometry-column retention

Unsupported geometry inputs continue to fall back to the original STRtree path.
Current host benchmarks show the owned query path is not yet competitive with
STRtree, so the vendored `sindex` adapter remains on STRtree for default host
execution today.

## Consequences

- targeted upstream `sindex` and `sjoin` tests keep assembly semantics stable
- join/result semantics stay aligned with GeoPandas instead of being reimplemented
- future GPU/CCCL work can replace candidate generation and bounded nearest internals
  without changing the public assembly behavior

## Alternatives Considered

- Reimplement full pandas/DataFrame join assembly in repo-owned code.
  Rejected because it would duplicate a lot of mature GeoPandas semantics.
- Keep all query logic on STRtree and only patch `sjoin`.
  Rejected because it would leave no owned query engine for GPU-oriented work.
- Attempt an all-in nearest implementation now.
  Rejected because unbounded nearest still benefits from the existing STRtree path,
  while bounded nearest is the GPU-friendly path worth specializing first.

## Acceptance Notes

This decision lands the owned query engine plus vendored assembly integration and
validates them against targeted upstream `sindex` and `sjoin` families. Unbounded
nearest still uses STRtree today; bounded nearest now has an owned staged path.
