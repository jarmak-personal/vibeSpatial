---
id: ADR-0022
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geoarrow
  - geoparquet
  - wkb
  - gpu
---

# GeoArrow, GeoParquet, and WKB Bridges

## Context

Phase 6b needs a first-class IO stack that matches the owned geometry buffer
design instead of routing all interchange through ad hoc shapely-heavy paths.

## Decision

- Make GeoArrow the canonical in-memory interchange seam for owned buffers.
- Put GeoPandas Arrow and GeoParquet entry points behind repo-owned adapters.
- Use `pylibcudf` as the preferred GeoParquet scan engine when it is available
  and the request shape fits the current GPU path.
- Keep covering-based bbox pushdown policy at the adapter layer so host and GPU
  readers share one planning contract.
- Treat WKB as an explicit compatibility bridge with owned-buffer wrappers and
  visible fallback events.

## Consequences

- The repo now has one stable IO surface for later device-side codecs.
- Host Arrow and WKB fallbacks remain honest and observable.
- Future GPU work can replace the internals of the adapters without changing
  the public GeoPandas-facing methods.
