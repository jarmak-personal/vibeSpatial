---
id: ADR-0026
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - geoparquet
  - pylibcudf
  - chunking
---

# GeoParquet Scan Engine

## Context

`o17.6.19` added metadata-first row-group pruning and `o17.6.18` made aligned
GeoArrow adoption cheap. The remaining gap is the scan engine itself: one
contract that can use `pylibcudf` when available, plan chunk boundaries from
row-group metadata, and assemble owned buffers without routing geometry through
Shapely.

## Decision

Adopt a backend-neutral GeoParquet scan engine with chunk planning and direct
Arrow-to-owned geometry decode.

- row-group selection remains the planner's responsibility
- scan execution chooses `pylibcudf` when available and otherwise uses `pyarrow`
- supported geometry encodings decode directly into owned buffers after scan
- chunked scans concatenate owned-buffer batches instead of rebuilding geometry
  objects between chunks
- WKB remains supported, but as an explicit slower bridge relative to native
  GeoArrow encodings

## Consequences

- the fast path is now `parquet scan -> Arrow geometry decode -> owned buffers`
  instead of `parquet scan -> Shapely objects -> owned buffers`
- `o17.6.21` can replace the host Arrow family decoders with device kernels
  without changing the scan-engine boundary
- chunked out-of-core execution now has a stable owned-buffer contract

## Alternatives Considered

- keep `pylibcudf` as an unfiltered table reader that immediately converts back
  to host GeoPandas objects
- wait for full device-side family decoders before adding any owned-buffer scan
  engine
- keep WKB as the default decode path for all scanned GeoParquet datasets

## Acceptance Notes

This landing provides the scan-engine boundary, chunk planning, and direct
GeoArrow-family owned decode on the host validation path. `pylibcudf` is still
an optional backend and device-side family decode remains follow-on work in
`o17.6.21`.
