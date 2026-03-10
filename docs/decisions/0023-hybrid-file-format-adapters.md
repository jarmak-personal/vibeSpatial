---
id: ADR-0023
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geojson
  - shapefile
  - gdal
  - gpu
---

# Hybrid File Format Adapters

## Context

Phase 6b needs public file-format entry points that acknowledge the difference
between targeted hybrid formats and untargeted legacy GDAL formats.

## Decision

- Route GeoJSON through a repo-owned hybrid adapter.
- Route Shapefile through a repo-owned hybrid adapter.
- Keep other GDAL formats behind an explicit legacy fallback adapter.
- Expose all of those choices through dispatch and fallback events.

## Consequences

- GeoJSON and Shapefile now have stable, observable repo-owned entry points.
- Legacy formats still work, but the repo no longer implies they are part of
  the GPU-oriented core stack.
- Later GPU-text or Arrow-backed file work can upgrade the adapters without
  changing the public GeoPandas surface.
