---
id: ADR-0015
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - constructive
  - cccl
---

# Rectangle Clip As First Constructive Fast Path

## Context

Phase 5 needed the first owned constructive surface that could exercise real
geometry-producing logic without taking on the full complexity of arbitrary
overlay. The repo now has segment primitives and a degeneracy corpus, but it
still needed a constructive operation that maps cleanly onto GPU-style stage
boundaries.

## Decision

Choose rectangle clip first.

The owned implementation:

- converts to owned buffers
- filters rows by rectangle bounds
- clips candidate rows with direct line or ring logic
- falls back row by row for unsupported or invalid geometries

The GeoPandas adapter seam is landed at `GeometryArray.clip_by_rect`, but the
current host path still uses direct Shapely after recording an explicit fallback
event.

## Consequences

- the repo now has a real constructive owned path before full overlay
- the future GPU implementation has a clear CCCL-oriented dataflow
- public GeoPandas behavior does not regress on CPU while the owned path is
  still host-slower than Shapely

## Alternatives Considered

- Arbitrary polygon intersection first.
  Rejected because it is too broad and would bury the first constructive kernel
  inside too much assembly logic.
- Full host reroute to the owned rectangle clip path.
  Rejected for now because benchmarks show the owned CPU path is much slower
  than direct Shapely.
- Delay all constructive work until a GPU kernel exists.
  Rejected because Phase 5 still needs a validated owned execution seam and
  benchmark surface now.

## Acceptance Notes

This decision lands the owned rectangle-clip engine, GeoPandas adapter seam,
benchmark script, and correctness tests. The benchmark currently justifies
keeping the public host path on Shapely while the owned path remains an explicit
optimization target.
