---
id: ADR-0017
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - dissolve
  - groupby
  - union
  - cccl
---

# Dissolve Grouped Union Pipeline

## Context

`GeoDataFrame.dissolve` mixes pandas grouping semantics with grouped geometry
union. The public contract is groupby-driven, but the geometry core should be
stageable as GPU-friendly grouped constructive work later.

## Decision

Use a grouped-union pipeline:

- encode group keys once
- stable-sort rows by group
- derive group spans with run-length encode
- aggregate attributes separately from geometry
- union each group independently
- assemble the final frame at the end

The current host implementation now routes through this pipeline. Future GPU
work should replace the grouped-union stage with a CUDA implementation built on
sorting, segmented execution, and compaction primitives rather than redesigning
the full public dissolve surface.

## Consequences

- Public dissolve behavior stays aligned with upstream pandas and GeoPandas
  semantics.
- The geometry core has a clean seam for CUDA and CCCL-driven work later.
- Deterministic group order is explicit instead of incidental.
- This does not yet provide a GPU union kernel; it provides the correct staging
  contract for one.

## Alternatives Considered

- keep dissolve as a pure pandas groupby callback around `union_all`
- global union followed by post-hoc regrouping
- a geometry-only pipeline that ignores pandas aggregation boundaries

## Acceptance Notes

The landed implementation exposes a dissolve planner, grouped union executor,
benchmark surface, and a repo-owned `GeoDataFrame.dissolve` path that preserves
upstream test behavior.
