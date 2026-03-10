---
id: ADR-0028
status: accepted
date: 2026-03-11
deciders:
  - codex
  - vibeSpatial maintainers
tags:
  - io
  - wkb
  - cccl
  - performance
---

# Staged WKB Bridge

## Context

WKB is not the canonical runtime layout, but it is still the widest geometry
compatibility bridge in the stack. The old repo-owned path was only an honest
host fallback through Shapely. That kept behavior visible, but it gave up the
biggest performance lever available for WKB: staged byte-stream processing
before any full geometry materialization.

## Decision

Adopt a staged WKB bridge built around the same phases the future GPU path will
use:

- one header scan to identify native rows and fallback rows
- family partitioning before decode or encode work
- family-local size and offset handling
- family-specialized native decode and encode for supported 2D little-endian
  point, linestring, polygon, multipoint, multilinestring, and multipolygon
  records
- explicit fallback pools for malformed, unsupported, or non-little-endian rows

The current landing validates this structure on the host. CCCL remains the
target substrate for the scan, partition, size, and scatter stages when the
device path lands.

## Consequences

- the WKB bridge now has a real fast path instead of always routing through
  Shapely
- mixed and malformed inputs no longer contaminate homogeneous fast paths
- the public WKB surface only records fallback events when rows actually enter
  the fallback pool
- the bridge contract now matches the staged IO model instead of fighting it

## Alternatives Considered

- keep WKB as pure Shapely bridge work until a full device implementation exists
- use one generic WKB parser for every family without partitioning
- optimize only encode or only decode and leave the other direction untouched

## Acceptance Notes

This landing establishes the staged bridge shape and validated host-native fast
paths. The next step is replacing the current host scan and family-local
assembly with CCCL-backed device passes and kernels behind the same API.
