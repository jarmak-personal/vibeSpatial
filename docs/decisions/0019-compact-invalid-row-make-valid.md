---
id: ADR-0019
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - make-valid
  - constructive
  - topology
  - cccl
---

# Compact Invalid Row Make Valid

## Context

`make_valid` is expensive constructive topology repair, but many real datasets
are mostly valid. Running topology repair on every row is the wrong baseline for
a GPU-first system.

## Decision

Use a compact-invalid-row pipeline:

- compute validity for all rows
- compact only invalid rows
- repair the invalid subset
- scatter repaired rows back into original row order

The current host implementation already follows this contract using Shapely for
the repair stage. Future GPU work should replace the repair stage while keeping
the same compaction and scatter structure.

## Consequences

- Valid rows avoid unnecessary repair work.
- Overlay preprocessing and direct `GeoSeries.make_valid()` share one repair
  seam.
- The GPU path has a clear place to use `DeviceSelect` before topology repair.

## Alternatives Considered

- call `make_valid` on every row unconditionally
- postpone all repo-owned work until a full GPU topology repair kernel exists
- special-case overlay only and leave `GeoSeries.make_valid()` unchanged

## Acceptance Notes

The landed implementation adds a make-valid planner, compact-invalid-row
executor, benchmark surface, kernel registration, and a GeoPandas array adapter
that routes through the compacted repair path.
