---
id: ADR-0025
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geoarrow
  - zero-copy
  - materialization
---

# Zero-Copy GeoArrow Adoption

## Context

`o17.6.19` now prunes GeoParquet work before decode, but the surviving GeoArrow
bridge still paid avoidable copy costs on both ingress and egress. That leaves
the best-case columnar path slower than it should be and weakens the handoff to
future GPU-native codecs.

## Decision

Adopt aligned GeoArrow buffers zero-copy and normalize only when required.

- owned geometry buffers remain canonical
- GeoArrow export should expose shared buffer views whenever the caller accepts
  a shared view
- GeoArrow import should default to `auto` adoption:
  - share aligned buffers directly
  - normalize mismatched dtypes or shapes once into canonical owned buffers
- host geometry objects remain lazily materialized and must not be created by
  the GeoArrow adoption path itself

Aligned buffers are those that already match the canonical owned schema:

- `x` and `y` are contiguous `float64`
- offsets are contiguous `int32`
- validity and empty masks are contiguous boolean vectors
- bounds, when present, are contiguous `float64[:, 4]`

## Consequences

- aligned GeoArrow import and export become wrapper-cost operations instead of
  copy-heavy adapters
- lazy Shapely materialization remains intact at the owned-buffer boundary
- `o17.6.20` and `o17.6.21` can build on one stable adoption contract instead
  of inventing separate Arrow staging rules

## Alternatives Considered

- always copy and normalize GeoArrow buffers
- keep opaque Arrow objects alive and defer all decode decisions until later
- introduce a second canonical Arrow-native buffer layout alongside owned
  buffers

## Acceptance Notes

The landed implementation provides explicit `copy`, `auto`, and `share` modes,
plus benchmarks showing the zero-copy path is the correct default for aligned
input. Public GeoPandas Arrow IO still has host-visible steps until the
device-side codecs land, but the owned-buffer boundary no longer forces copies
when layouts already match.
