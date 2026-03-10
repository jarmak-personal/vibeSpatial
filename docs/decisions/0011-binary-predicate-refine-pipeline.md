---
id: ADR-0011
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

# Staged Binary Predicate Refine Pipeline

## Context

`o17.4.2` must provide exact binary-predicate refine kernels that later
`sindex`, `sjoin`, and overlay work can reuse. The repo already has coarse
bounds filters, exactness policy, and a first point-predicate pipeline. The
missing piece is a shared exact-refine engine for aligned binary predicates.

## Decision

Adopt a shared staged exact-refine engine for binary predicates:

- choose a coarse bounding-box relation per predicate family
- compact only the rows that survive the coarse relation
- run the exact predicate on the compacted candidate set
- scatter exact results back to the full output shape

Route the vendored GeoPandas `GeometryArray` binary predicate path through this
engine boundary for the supported predicate set, while keeping current host
execution on direct vectorized Shapely until performance data justifies moving
the adapter onto the staged refine path.

## Consequences

- exact predicate semantics now have one stable internal API
- targeted upstream GeoPandas predicate tests exercise repo-owned routing
- the future GPU implementation can swap in CCCL-based compaction and device-side
  exact kernels without changing the GeoPandas adapter contract
- current host workloads avoid a measured regression from coarse-filter staging

## Alternatives Considered

- Leave the vendored GeoPandas path on direct Shapely calls and build a separate
  internal engine later.
  Rejected because it would duplicate semantics and defer adapter integration.
- Build one monolithic raw CUDA kernel per predicate.
  Rejected because it would hide compaction boundaries and fight the repo’s CCCL-first direction.
- Use direct Shapely for every row forever.
  Rejected because it gives no reusable GPU path and no place to hang exact-refine diagnostics.

## Acceptance Notes

This decision lands CPU variants only. Exact binary-predicate semantics, vendored
GeoPandas routing, targeted upstream validation, and the staged engine contract
are in place now. GPU predicate variants remain future work on top of this
pipeline.
