---
id: ADR-0021
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geoarrow
  - geoparquet
  - gpu
---

# IO Support Matrix

## Context

Phase 6b needs one consistent answer to which formats deserve GPU-native
implementation effort and which formats should remain hybrid or fallback paths.

## Decision

- GeoArrow is the canonical GPU-native interchange target.
- GeoParquet is the canonical GPU-native persisted format.
- WKB is a hybrid compatibility bridge.
- GeoJSON and Shapefile are explicit hybrid pipelines.
- Untargeted GDAL formats remain explicit fallback adapters.

## Consequences

- GPU-first implementation effort stays concentrated on the right formats.
- Hybrid and fallback formats can still be supported without dictating the core
  architecture.
- Later IO changes can share one planning surface instead of making per-format
  ad hoc decisions.
