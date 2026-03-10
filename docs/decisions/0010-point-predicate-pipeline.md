---
id: ADR-0010
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - predicates
  - cccl
---

# Staged Point Predicate Pipeline

## Context

`o17.4.1` is the first real predicate decision after the owned geometry buffers,
coarse filters, precision policy, and robustness policy landed. The repo needs a
point-versus-bounds predicate and a point-in-polygon predicate that can back
later `sindex`, `sjoin`, and exact-refine work without forcing a redesign when
GPU kernels arrive.

## Decision

Adopt a staged point-predicate pipeline:

- coarse aligned bounds filtering first
- candidate compaction second
- exact refine only on surviving rows

The landed CPU path preserves this structure:

- `point_within_bounds` evaluates the coarse predicate directly on owned point
  coordinates and aligned bounds
- `point_in_polygon` reuses the coarse predicate, then refines candidate rows
  with exact host `covers` semantics
- `auto` mode records explicit CPU fallback when no GPU predicate variant is
  registered
- explicit `gpu` requests fail loudly until a real GPU implementation exists

## Consequences

- the public predicate contract is stable before CUDA work lands
- correctness is preserved now, especially for boundary and empty cases
- the future GPU implementation can target stage boundaries instead of a single
  opaque kernel
- CCCL remains the preferred building block for compaction and reduction work

## Alternatives Considered

- Monolithic raw ray-casting kernels.
  Rejected because they entangle compaction, traversal, and reduction in one
  custom kernel.
- Polygon triangulation as the primary point-location structure.
  Rejected because the preprocessing cost and storage expansion are a bad fit
  for the first predicate landing.
- Keeping point predicates as Shapely-only host helpers with no staged model.
  Rejected because it would hide the eventual GPU execution shape and bypass the
  coarse-filter/index work that already landed.

## Acceptance Notes

The current environment does not include a real GPU predicate implementation, so
this decision lands the exact CPU path, explicit fallback behavior, parity tests,
and the architecture contract the later CCCL-backed GPU refine path should
follow.
