---
id: ADR-0008
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - buffers
  - geoarrow
  - offsets
  - layout
---

# Owned Geometry Buffer Schema

## Context

The repo already settled mixed-geometry storage, precision, null handling,
residency, and adaptive planning. Phase 2 now needs one concrete payload shape
for points, lines, polygons, and multipart geometries so adapters and kernels
can share a single authoritative layout.

## Decision

Use separated fp64 coordinate buffers plus hierarchical prefix offsets.

- all families store authoritative `x` and `y` coordinates in contiguous fp64 arrays
- nulls use a validity bitmap
- empties remain valid rows with zero-length spans
- multipart and polygon nesting use prefix-offset buffers rather than nested objects
- mixed arrays reference family payloads by coarse family tag plus family-relative row offset

The concrete family layouts are:

- Point: row -> coordinate
- LineString: row -> coordinate via `geometry_offsets`
- Polygon: row -> ring -> coordinate via `geometry_offsets` and `ring_offsets`
- MultiPoint: row -> coordinate via `geometry_offsets`
- MultiLineString: row -> part -> coordinate via `geometry_offsets` and `part_offsets`
- MultiPolygon: row -> polygon part -> ring -> coordinate via `geometry_offsets`, `part_offsets`, and `ring_offsets`

## Consequences

- Adapters can target one predictable memory shape across CPU and GPU paths.
- Execution-local staging such as centered fp32 buffers or partition permutations stays out of the canonical contract.
- Later kernels can share offset traversal logic instead of re-encoding topology per operation.

## Alternatives Considered

- array-of-objects geometry storage
- interleaved xy coordinate storage as the canonical layout
- canonical dual fp32/fp64 coordinate buffers
- permanent split-by-family storage without a mixed-array contract

## Acceptance Notes

The landed schema is a contract module plus docs. Actual adapters, device
buffers, and traversal kernels remain later Phase 2 work.
