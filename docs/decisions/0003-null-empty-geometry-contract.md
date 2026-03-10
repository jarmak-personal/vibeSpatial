---
id: ADR-0003
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - nulls
  - geometry
  - runtime
---

# Null And Empty Geometry Contract

## Context

Real-world GeoDataFrames commonly contain both missing geometries and valid
empty geometries. GPU kernels need a stable state model before owned buffers,
bounds kernels, and predicate pipelines expand.

## Decision

Treat null and empty as distinct geometry states.

- nulls follow Arrow validity semantics and propagate through unary and predicate outputs
- empties remain valid geometries represented by zero-length spans
- empty measurements return defined values (`NaN` bounds, zero area/length)
- joins and aggregations exclude null rows from candidate generation
- batch kernels should prefer masks and predication over scalar slow-path branches

## Consequences

- Owned buffers need validity bitmaps in addition to offset-based empty representation.
- Kernel docs and tests must state null and empty behavior explicitly.
- Oracle and scaffolded tests need separate null and empty coverage.

## Alternatives Considered

- Collapse empty geometries to nulls
- Treat both states as generic “missing”
- Handle null and empty only in a host-side fallback layer

## Acceptance Notes

The first landed contract exposes explicit state classification helpers and a
doc-level policy for unary, measurement, and predicate behavior.
