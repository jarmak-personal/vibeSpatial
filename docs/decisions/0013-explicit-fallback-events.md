---
id: ADR-0013
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - architecture
  - fallback
---

# Explicit CPU Fallback Events

## Context

Phase 4 now has real predicate and spatial-query surfaces, but several of those
paths still execute on host-only Shapely or STRtree implementations. The repo’s
runtime policy rejects silent CPU fallback, so those surfaces need an explicit
observability mechanism.

## Decision

Record explicit fallback events at host-only GeoPandas-facing surfaces and
expose them through the GeoPandas shim.

Each event records:

- the surface
- requested runtime
- selected runtime
- reason
- detail

Warnings are not the primary mechanism because they would risk changing
upstream-visible behavior. Explicit GPU requests in repo-owned kernels still
raise, while `auto` / host-only public paths log fallback events.

## Consequences

- fallback behavior is now machine-readable in repo-local tests
- host-only execution is observable without changing GeoPandas results
- future GPU rollouts can remove specific fallback events as surfaces gain real
  GPU support

## Alternatives Considered

- Rely on warnings only.
  Rejected because it is noisy and can disturb upstream expectations.
- Leave fallback visibility on owned arrays only.
  Rejected because GeoPandas-facing paths would still be effectively silent.
- Raise for all host-only public paths.
  Rejected because correctness-preserving CPU execution is still needed.

## Acceptance Notes

This decision lands explicit fallback observability for current host-only predicate
and spatial-query surfaces and validates it with repo-local tests.
