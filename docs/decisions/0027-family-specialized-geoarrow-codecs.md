---
id: ADR-0027
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geoarrow
  - codecs
  - performance
---

# Family-Specialized GeoArrow Codecs

## Context

`o17.6.18` made aligned GeoArrow adoption nearly free and `o17.6.20` added a
real GeoParquet scan engine, but the family codec layer still paid too much
generic bridge overhead. The next performance lever is to encode and decode
native geometry columns by family so points, lines, polygons, and multiparts do
not all route through one mixed host reconstruction path.

## Decision

Adopt family-specialized GeoArrow encode and decode paths as the default native
codec layer behind owned geometry IO.

- homogeneous families encode directly to GeoArrow extension arrays with
  family-local offsets and shared metadata
- decode dispatches by GeoArrow extension type into family-specific owned-buffer
  builders
- malformed or unsupported inputs remain isolated through explicit fallback to
  WKB or host bridge paths instead of polluting the native fast path
- mixed-family GeoArrow export keeps using an explicit compatibility fallback
  until a partition-and-restore mixed codec lands
- the public GeoPandas Arrow export surface should not record a fallback event
  when the repo-owned native family codec succeeds

## Consequences

- native point export is now a lightweight owned-buffer to Arrow assembly step
  instead of a generic host bridge
- polygon and multipart decode stay on typed offset assembly paths that match
  the owned buffer schema
- `pylibcudf` GeoParquet scans now have a cleaner seam for replacing the current
  Arrow bridge with device-local family kernels later
- strict-native coverage can credit native GeoArrow export success instead of
  treating it as a fallback

## Alternatives Considered

- keep one generic mixed GeoArrow codec for all families
- export all unsupported or mixed cases through WKB only
- wait for full device-side CCCL kernels before landing any family-specialized
  codec structure

## Acceptance Notes

This landing establishes the typed family codec structure, native homogeneous
export fast paths, explicit mixed fallback, and benchmark surface. The current
implementation still uses `pyarrow` for final array/table assembly on the host;
the next step is replacing the remaining bridge cost with device-local kernels
behind the same family-specialized contract.
