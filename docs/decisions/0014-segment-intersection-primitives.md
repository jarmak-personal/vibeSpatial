---
id: ADR-0014
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - overlay
  - cccl
---

# Staged Segment Intersection Primitives

## Context

Phase 5 needs a segment primitive that can support overlay, clipping, and
constructive topology work. The repo already has coarse segment MBR filters and
a robustness contract that requires exact decisions for ambiguous cases, but it
did not yet have a direct owned-buffer segment path or a reusable classifier for
degenerate intersections.

## Decision

Adopt a staged segment-intersection primitive:

- extract segments directly from owned buffers
- generate candidate pairs from segment MBR overlap
- classify clear pairs in vectorized fast-path arithmetic
- compact ambiguous rows and run exact-style fallback only on that subset

The current landing uses exact rational arithmetic on the compacted ambiguity
set for CPU correctness. That is not the final GPU implementation; it is the
reference contract for the future GPU path.

## Consequences

- constructive kernels now have a reusable primitive surface before overlay
  assembly lands
- collinear overlap, shared vertices, zero-length segments, and ring-edge
  corner cases are covered explicitly
- the future GPU implementation can map onto CCCL compaction/restoration steps
  instead of starting from a monolithic scalar kernel

## Alternatives Considered

- Delegate intersection classification to Shapely/GEOS.
  Rejected because it hides the primitive graph and anchors Phase 5 to host
  materialization.
- Use exact scalar arithmetic for every candidate pair.
  Rejected because it is fundamentally at odds with the repo's GPU throughput
  goals.
- Keep using the earlier segment MBR path that materialized Shapely segments.
  Rejected because Phase 5 should move toward owned-buffer execution, not away
  from it.

## Acceptance Notes

This decision lands the direct segment extractor, staged classifier, degeneracy
tests, and a benchmark surface. A real GPU variant remains future work, but the
dataflow and exact-fallback structure are now fixed.
