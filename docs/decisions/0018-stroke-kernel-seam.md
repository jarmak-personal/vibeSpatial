---
id: ADR-0018
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - buffer
  - offset-curve
  - constructive
  - cccl
---

# Stroke Kernel Seam

## Context

`buffer` and `offset_curve` are classic constructive operations with variable
output sizes. A GPU-first implementation needs a clear output-allocation and
vertex-emission contract before a full CUDA kernel exists.

## Decision

Adopt a stroke-kernel seam built around:

- bulk distance expansion
- segment-frame generation
- join and cap classification
- prefix-sum allocation
- scatter-based vertex emission

The first owned implementations are intentionally narrow:

- positive-distance `Point` buffer rows
- simple `LineString` offset curves with non-round joins

The public GeoPandas adapters still fall back explicitly to Shapely because the
current host prototypes are not yet host-competitive.

## Consequences

- The repo now has a real stroke-style constructive kernel seam instead of
  treating these methods as opaque Shapely calls.
- Future GPU work can replace the math stages without changing the public
  GeoPandas adapter shape.
- Host performance is not broadly better than Shapely yet, so the public path
  remains on explicit Shapely fallback for now.

## Alternatives Considered

- keep both methods entirely on Shapely until a full GPU implementation exists
- build a monolithic custom constructive engine before proving the staging model
- try to route all geometry families through one premature host prototype

## Acceptance Notes

The landed implementation adds a stroke planner, a point-buffer prototype, a
simple offset-curve prototype, kernel registrations, benchmarks, and explicit
fallback reporting on the GeoPandas-facing array methods instead of routing the
slower host prototypes by default.
